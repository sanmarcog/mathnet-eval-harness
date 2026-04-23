"""Unified inference client across frontier APIs and local HuggingFace models.

All backends return the same `Response` dataclass so graders / analysis code
don't care which provider generated a completion. A disk-backed cache keyed
on (model, params, prompt-hash) makes re-runs free — important for rapid
iteration on grading / analysis logic without re-paying API costs.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)

load_dotenv()  # pulls ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY from .env


def _is_retryable(exc: BaseException) -> bool:
    """Retry on rate-limit, timeout, and transient 5xx; fail fast on 4xx
    auth/validation. Providers raise different exception classes; match by
    class name and HTTP status to avoid importing every SDK at module load."""
    cls = type(exc).__name__
    # Common transient names across Anthropic / OpenAI / google-genai.
    transient_names = {
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "APIStatusError",
        "InternalServerError",
        "ServiceUnavailableError",
        "DeadlineExceeded",
        "TooManyRequests",
        "ResourceExhausted",
        "ServerError",      # google-genai 5xx
    }
    if cls in transient_names:
        return True
    # Fallback: some SDKs raise a generic ClientError with .code or .status_code.
    for attr in ("status_code", "code"):
        sc = getattr(exc, attr, None)
        if isinstance(sc, int) and (sc == 429 or 500 <= sc < 600):
            return True
    return False


_api_retry = retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)


# ---- Model registry ---------------------------------------------------------

# Maps short aliases and full IDs to (provider, provider_model_id).
# Update when frontier versions bump.
MODELS: dict[str, tuple[str, str]] = {
    # Anthropic
    "sonnet-4-6":         ("anthropic", "claude-sonnet-4-6"),
    "claude-sonnet-4-6":  ("anthropic", "claude-sonnet-4-6"),
    "opus-4-7":           ("anthropic", "claude-opus-4-7"),
    "claude-opus-4-7":    ("anthropic", "claude-opus-4-7"),
    # OpenAI
    "gpt-5.4":            ("openai", "gpt-5.4"),
    "gpt-5-4":            ("openai", "gpt-5.4"),
    "gpt-5.4-mini":       ("openai", "gpt-5.4-mini"),
    "gpt-5-4-mini":       ("openai", "gpt-5.4-mini"),
    # Google. The canonical 3.0-pro alias maps to the preview ID because
    # that is the only `gemini-3-pro-*` currently served over generateContent.
    # `gemini-3.1-pro-preview` is also available if we decide to switch up.
    "gemini-3-pro":       ("google", "gemini-3-pro-preview"),
    "gemini-3-pro-preview":   ("google", "gemini-3-pro-preview"),
    "gemini-3.1-pro":     ("google", "gemini-3.1-pro-preview"),
    "gemini-3.1-pro-preview": ("google", "gemini-3.1-pro-preview"),
    # Local HF (Day 3+)
    "qwen-base":          ("hf", "Qwen/Qwen2.5-1.5B-Instruct"),
}


# ---- Response structure -----------------------------------------------------

@dataclass
class Response:
    model: str
    provider_model_id: str
    prompt: str
    text: str
    raw: dict           # full provider payload, for debug / re-grading
    usage: dict         # input_tokens, output_tokens, etc.
    latency_s: float
    cached: bool = False
    params: dict = field(default_factory=dict)

    def to_json(self) -> str:
        def _default(o):
            # Some SDKs (seen with google-genai) emit bytes in their raw
            # payload (inline_data, safety metadata). Base64 them so the
            # cache JSON survives round-trips.
            if isinstance(o, (bytes, bytearray)):
                import base64
                return {"__bytes_b64__": base64.b64encode(bytes(o)).decode("ascii")}
            raise TypeError(f"not JSON serializable: {type(o).__name__}")
        return json.dumps(asdict(self), ensure_ascii=False, default=_default)


# ---- Disk cache -------------------------------------------------------------

DEFAULT_CACHE_DIR = Path(os.environ.get("MATHNET_CACHE_DIR", "results/cache"))


def _cache_key(model: str, prompt: str, params: dict) -> str:
    payload = json.dumps(
        {"model": model, "prompt": prompt, "params": params},
        sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cache_path(cache_dir: Path, key: str) -> Path:
    # 2-char prefix keeps any single dir from ballooning past ~thousands of files.
    return cache_dir / key[:2] / f"{key}.json"


def _load_cached(cache_dir: Path, key: str) -> Response | None:
    p = _cache_path(cache_dir, key)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    d["cached"] = True
    return Response(**d)


def _save_cached(cache_dir: Path, key: str, resp: Response) -> None:
    p = _cache_path(cache_dir, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(resp.to_json())


# ---- Providers --------------------------------------------------------------

@_api_retry
def _generate_anthropic(provider_model_id: str, prompt: str, params: dict) -> Response:
    import anthropic  # lazy import so this module imports fine without the SDK

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    t0 = time.perf_counter()
    msg = client.messages.create(
        model=provider_model_id,
        max_tokens=params.get("max_tokens", 4096),
        temperature=params.get("temperature", 0.0),
        system=params.get("system", "You are an expert mathematician solving olympiad problems."),
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.perf_counter() - t0

    text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    usage = {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}
    # Extended-thinking usage fields (appear when `thinking` is enabled in the
    # request — we currently don't enable it, but keep the hook so future runs
    # surface thinking tokens consistently with GPT-5 reasoning and Gemini
    # thoughts.
    thinking = getattr(msg.usage, "cache_creation_input_tokens", None)
    if thinking is None:
        thinking = getattr(msg.usage, "thinking_tokens", None)
    if thinking is not None:
        usage["thinking_tokens"] = thinking

    return Response(
        model=provider_model_id,
        provider_model_id=provider_model_id,
        prompt=prompt,
        text=text,
        raw=msg.model_dump(),
        usage=usage,
        latency_s=latency,
        params=params,
    )


@_api_retry
def _generate_openai(provider_model_id: str, prompt: str, params: dict) -> Response:
    import openai  # lazy

    client = openai.OpenAI()  # reads OPENAI_API_KEY from env
    t0 = time.perf_counter()

    # GPT-5.x reasoning models use `max_completion_tokens` (not `max_tokens`)
    # and reject custom `temperature` — the SDK accepts `1.0` only. Skip it.
    kwargs: dict = {
        "model": provider_model_id,
        "messages": [
            {"role": "system", "content": params.get("system", "You are an expert mathematician solving olympiad problems.")},
            {"role": "user", "content": prompt},
        ],
        "max_completion_tokens": params.get("max_tokens", 8192),
    }
    resp = client.chat.completions.create(**kwargs)
    latency = time.perf_counter() - t0

    choice = resp.choices[0]
    text = choice.message.content or ""
    usage = {
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
    }
    # GPT-5 reasoning tokens are a sub-field of completion_tokens; surface
    # them separately for cost analysis.
    reasoning = getattr(resp.usage, "completion_tokens_details", None)
    if reasoning is not None:
        rt = getattr(reasoning, "reasoning_tokens", None)
        if rt is not None:
            usage["reasoning_tokens"] = rt

    return Response(
        model=provider_model_id,
        provider_model_id=provider_model_id,
        prompt=prompt,
        text=text,
        raw=resp.model_dump(),
        usage=usage,
        latency_s=latency,
        params=params,
    )


@_api_retry
def _generate_google(provider_model_id: str, prompt: str, params: dict) -> Response:
    from google import genai  # lazy
    from google.genai import types

    client = genai.Client()  # reads GOOGLE_API_KEY from env

    # Gemini 3 Pro thinks unboundedly by default. A 15-problem smoke showed
    # median 5,500 thoughts/problem with a tail out to 15,730, which blows
    # the per-model budget. Cap at 4,096 to bound both cost and wall time.
    # Pass -1 via `thinking_budget` for the default (uncapped) behavior;
    # pass 0 to disable thinking.
    thinking_budget = params.get("thinking_budget", 4096)

    cfg = types.GenerateContentConfig(
        system_instruction=params.get("system", "You are an expert mathematician solving olympiad problems."),
        max_output_tokens=params.get("max_tokens", 8192),
        temperature=params.get("temperature", 0.0),
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
    )
    t0 = time.perf_counter()
    resp = client.models.generate_content(
        model=provider_model_id,
        contents=prompt,
        config=cfg,
    )
    latency = time.perf_counter() - t0

    text = resp.text or ""
    um = resp.usage_metadata
    usage = {
        "input_tokens": getattr(um, "prompt_token_count", 0) or 0,
        "output_tokens": getattr(um, "candidates_token_count", 0) or 0,
    }
    thoughts = getattr(um, "thoughts_token_count", None)
    if thoughts is not None:
        usage["thoughts_tokens"] = thoughts

    return Response(
        model=provider_model_id,
        provider_model_id=provider_model_id,
        prompt=prompt,
        text=text,
        # mode='json' tells pydantic to encode bytes etc. as JSON-safe types
        # up front; avoids the bytes-in-safety-metadata failure we hit on the
        # first Gemini smoke test.
        raw=resp.model_dump(mode="json") if hasattr(resp, "model_dump") else {},
        usage=usage,
        latency_s=latency,
        params=params,
    )


def _generate_hf(provider_model_id: str, prompt: str, params: dict) -> Response:
    raise NotImplementedError("Local HF backend lands on Day 3+.")


_DISPATCH = {
    "anthropic": _generate_anthropic,
    "openai": _generate_openai,
    "google": _generate_google,
    "hf": _generate_hf,
}


def _normalize_params(provider: str, params: dict) -> dict:
    """Apply provider-specific defaults *before* the cache key is computed, so
    two calls that resolve to the same effective request always share a cache
    entry, and two calls with different effective requests (e.g. different
    `thinking_budget`) never collide."""
    out = dict(params)
    if provider == "google":
        out.setdefault("thinking_budget", 4096)
    return out


# ---- Public API -------------------------------------------------------------

def generate(
    prompt: str,
    model: str,
    *,
    use_cache: bool = True,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    **params,
) -> Response:
    """Call `model` on `prompt`. Reads ANTHROPIC_API_KEY etc. from .env via dotenv.

    params are provider-agnostic hints (max_tokens, temperature, system).
    Each backend picks out what it needs.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model alias {model!r}. Known: {sorted(MODELS)}")
    provider, provider_model_id = MODELS[model]
    cache_dir = Path(cache_dir)

    params = _normalize_params(provider, params)
    key = _cache_key(provider_model_id, prompt, params)
    if use_cache:
        cached = _load_cached(cache_dir, key)
        if cached is not None:
            return cached

    resp = _DISPATCH[provider](provider_model_id, prompt, params)
    if use_cache:
        _save_cached(cache_dir, key, resp)
    return resp
