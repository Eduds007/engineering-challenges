"""Minimal LLM client: no SDK, just requests + retry/backoff, because the whole
pipeline only needs a handful of calls (see extract_events.py's budget notes).

Supports two OpenAI-compatible providers, selected by LLM_PROVIDER in .env:
  - "groq" (default): fast inference, generous free tier, GROQ_API_KEY/GROQ_MODEL.
  - "openrouter": broader free-model catalog but slower/rate-limited,
    OPENROUTER_API_KEY/OPENROUTER_MODEL.

Responses are cached to disk keyed by a hash of (provider, model, system, user), so
iterating on the pipeline during development doesn't re-spend a free-tier quota, and
switching providers naturally produces fresh cache entries rather than stale hits.

Last-resort fallback: if every model on the configured HTTP provider fails (both free
tiers have hit us with hard rate limits during this project — OpenRouter's 50/day free
quota is what triggered this), call_llm() shells out to this machine's own `claude -p`
(Claude Code CLI, non-interactive print mode). That uses the local Claude Code
subscription's own auth, not a paid Anthropic API key — the user doing this build has
Claude Code access but no API credits. Kept as pure text-in/text-out (every tool
disallowed, --effort low) so it behaves like any other provider call here, not like an
agent making decisions. Anthropic's automatic prompt-cache reuse (same system-prompt
text = cache hit, confirmed empirically: ~50% cheaper from the second call on, no
--resume/session-continuity needed) keeps the per-call cost down across the many calls
this pipeline makes.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv(override=True)  # this project's .env wins over any same-named var already in the shell

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".llm_cache")

# Each provider's OpenAI-compatible chat-completions endpoint, its API-key env var,
# its default-model env var, and a small fallback chain to try if the configured
# model is unavailable — both providers' free catalogs shift without notice (this
# happened once already on OpenRouter while building this pipeline).
PROVIDERS = {
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key_env": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL",
        "fallback_models": ["openai/gpt-oss-120b", "groq/compound-mini", "qwen/qwen3.6-27b"],
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "key_env": "OPENROUTER_API_KEY",
        "model_env": "OPENROUTER_MODEL",
        "fallback_models": [
            "poolside/laguna-s-2.1:free",
            "nex-agi/nex-n2.5-pro:free",
            "nvidia/nemotron-3-super-120b-a12b:free",
        ],
    },
}


CLAUDE_CLI_MODEL = os.environ.get("CLAUDE_CLI_MODEL", "haiku")
# Every tool denied: this call must behave like a plain chat completion, never an
# agent with side effects. (Agent is included so it can't spawn a sub-agent either.)
CLAUDE_CLI_DISALLOWED_TOOLS = (
    "Bash Edit Write Read NotebookEdit Agent WebFetch WebSearch "
    "Artifact ArtifactComments ArtifactData ArtifactCheck ExitPlanMode"
)


def _cache_path(key: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{key}.json")


def _call_claude_cli(system: str, user: str, max_retries: int = 2, timeout: int = 150) -> str:
    cmd = [
        "claude", "-p", user,
        "--model", CLAUDE_CLI_MODEL,
        "--output-format", "json",
        "--system-prompt", system,
        "--effort", "low",
        "--disallowed-tools", CLAUDE_CLI_DISALLOWED_TOOLS,
    ]
    last_err: Exception | str | None = None
    for attempt in range(max_retries):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            last_err = e
            continue
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            last_err = f"unparseable output (exit {proc.returncode}): {proc.stdout[:300]!r} stderr: {proc.stderr[:300]!r}"
            continue
        if data.get("is_error"):
            last_err = data.get("result") or f"claude CLI error, exit {proc.returncode}"
            time.sleep(min(2 ** attempt * 2, 20))
            continue
        return data["result"]
    raise RuntimeError(f"claude CLI: exhausted retries, last error: {last_err}")


def _post(url: str, system: str, user: str, model: str, temperature: float, max_tokens: int,
          api_key: str, max_retries: int, reasoning_effort: str | None = None) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if reasoning_effort:
        # Groq's gpt-oss models spend part of max_tokens on a separate hidden
        # "reasoning" field before writing `content` — at the default effort a short
        # max_tokens budget can be entirely consumed by reasoning, leaving `content`
        # empty (finish_reason "length") even though the call "succeeded". Capping
        # effort keeps that overhead small and predictable for our classification/
        # extraction prompts, which don't need deep reasoning anyway.
        payload["reasoning_effort"] = reasoning_effort
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for attempt in range(max_retries):
        resp = requests.post(url, headers=headers, json=payload, timeout=90)
        if resp.status_code == 200:
            body = resp.json()
            content = body["choices"][0]["message"]["content"]
            if content is None:
                # A 200 with null content happens (seen live from an OpenRouter free
                # model) — treat it as this model's failure so the caller falls
                # through to the next candidate/provider instead of a None reaching
                # extract_json() and crashing on .strip() three frames downstream.
                finish_reason = body["choices"][0].get("finish_reason")
                raise RuntimeError(f"{url}: model {model} returned null content (finish_reason={finish_reason})")
            return content
        if resp.status_code in (429, 502, 503):
            time.sleep(min(2 ** attempt * 2, 60))
            continue
        raise RuntimeError(f"{url} error {resp.status_code}: {resp.text[:500]}")
    raise RuntimeError(f"{url}: exhausted retries on model {model}")


def call_llm(
    system: str,
    user: str,
    model: str | None = None,
    provider: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1500,
    use_cache: bool = True,
    max_retries: int | None = None,
    allow_claude_cli_fallback: bool = True,
) -> str:
    """Return the assistant's raw text content for one chat completion.

    Tries `model` (or <PROVIDER>_MODEL) first, then falls through the provider's
    fallback_models if the configured model is unavailable. If every one of those
    fails too (e.g. the free-tier daily quota is exhausted), and
    allow_claude_cli_fallback is left on, falls through one more time to this
    machine's own `claude -p` — see the module docstring.

    max_retries defaults to LLM_MAX_RETRIES in .env (5 if unset). Turn it down there
    (e.g. to 1) when the configured provider is *known* dead for the day (a hard daily
    quota, not a transient error) — each retry sleeps up to 60s, so with 3 fallback
    models the default backoff wastes minutes per call reconfirming something already
    known, before ever reaching the claude-cli fallback below.
    """
    if max_retries is None:
        max_retries = int(os.environ.get("LLM_MAX_RETRIES", "5"))
    provider = provider or os.environ.get("LLM_PROVIDER", "groq")
    if provider not in PROVIDERS:
        raise RuntimeError(f"unknown LLM_PROVIDER {provider!r} — expected one of {list(PROVIDERS)}")
    cfg = PROVIDERS[provider]

    api_key = os.environ.get(cfg["key_env"])
    if not api_key:
        raise RuntimeError(f"{cfg['key_env']} not set (check .env)")
    primary = model or os.environ.get(cfg["model_env"]) or cfg["fallback_models"][0]
    candidates = [primary] + [m for m in cfg["fallback_models"] if m != primary]

    cache_key = hashlib.sha256(f"{provider}|{primary}|{system}|{user}".encode("utf-8")).hexdigest()
    cpath = _cache_path(cache_key)
    if use_cache and os.path.exists(cpath):
        return json.load(open(cpath, encoding="utf-8"))["content"]

    last_err = None
    for candidate_model in candidates:
        # `reasoning_effort` support/accepted values differ per Groq model (gpt-oss
        # takes "low"/"medium"/"high", others reject the field outright or only take
        # "none"/"default") — only pass it for the family known to accept "low".
        reasoning_effort = "low" if "gpt-oss" in candidate_model else None
        try:
            content = _post(cfg["url"], system, user, candidate_model, temperature, max_tokens, api_key,
                             max_retries, reasoning_effort)
            if use_cache:
                json.dump({"content": content}, open(cpath, "w", encoding="utf-8"))
            return content
        except RuntimeError as e:
            last_err = e
            continue

    if allow_claude_cli_fallback:
        cli_cache_key = hashlib.sha256(f"claude_cli|{CLAUDE_CLI_MODEL}|{system}|{user}".encode("utf-8")).hexdigest()
        cli_cpath = _cache_path(cli_cache_key)
        if use_cache and os.path.exists(cli_cpath):
            return json.load(open(cli_cpath, encoding="utf-8"))["content"]
        try:
            content = _call_claude_cli(system, user)
            if use_cache:
                json.dump({"content": content}, open(cli_cpath, "w", encoding="utf-8"))
            print(f"  ({provider} exhausted ({last_err}) — fell back to claude CLI)", file=sys.stderr)
            return content
        except RuntimeError as cli_err:
            last_err = f"{last_err}; claude_cli also failed: {cli_err}"

    raise RuntimeError(f"{provider}: all candidate models failed, last error: {last_err}")


def extract_json(text: str) -> dict | list:
    """Best-effort JSON parse of an LLM response that may be wrapped in ```json fences,
    have leading/trailing prose, or (a reasoning model with leaked chain-of-thought)
    have a <think>...</think> block before the actual answer."""
    text = text.strip()
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = min((i for i in (text.find("["), text.find("{")) if i != -1), default=-1)
        end = max(text.rfind("]"), text.rfind("}"))
        if start != -1 and end != -1:
            return json.loads(text[start : end + 1])
        raise
