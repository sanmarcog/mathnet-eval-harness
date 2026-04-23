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

load_dotenv()  # pulls ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY from .env


# ---- Model registry ---------------------------------------------------------

# Maps short aliases and full IDs to (provider, provider_model_id).
# Update when frontier versions bump.
MODELS: dict[str, tuple[str, str]] = {
    # Anthropic
    "sonnet-4-6":         ("anthropic", "claude-sonnet-4-6"),
    "claude-sonnet-4-6":  ("anthropic", "claude-sonnet-4-6"),
    "opus-4-7":           ("anthropic", "claude-opus-4-7"),
    "claude-opus-4-7":    ("anthropic", "claude-opus-4-7"),
    # OpenAI (Day 2)
    "gpt-5":              ("openai", "gpt-5"),
    # Google (Day 2)
    "gemini-3-pro":       ("google", "gemini-3-pro"),
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
        return json.dumps(asdict(self), ensure_ascii=False)


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


def _generate_openai(provider_model_id: str, prompt: str, params: dict) -> Response:
    raise NotImplementedError("OpenAI backend lands on Day 2.")


def _generate_google(provider_model_id: str, prompt: str, params: dict) -> Response:
    raise NotImplementedError("Google Gemini backend lands on Day 2.")


def _generate_hf(provider_model_id: str, prompt: str, params: dict) -> Response:
    raise NotImplementedError("Local HF backend lands on Day 3+.")


_DISPATCH = {
    "anthropic": _generate_anthropic,
    "openai": _generate_openai,
    "google": _generate_google,
    "hf": _generate_hf,
}


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

    key = _cache_key(provider_model_id, prompt, params)
    if use_cache:
        cached = _load_cached(cache_dir, key)
        if cached is not None:
            return cached

    resp = _DISPATCH[provider](provider_model_id, prompt, params)
    if use_cache:
        _save_cached(cache_dir, key, resp)
    return resp
