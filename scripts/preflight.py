"""Pre-flight check for the full eval run.

Verifies environment, splits, and every backend end-to-end with a minimal
call before committing to hours of API spend. Run from the repo root:

    export HF_HOME=/gscratch/scrubbed/sanmarco/hf_cache
    export PYTHONPATH=src
    /gscratch/scrubbed/sanmarco/conda/envs/qlora/bin/python scripts/preflight.py

Exits 0 if everything is green, 1 if anything fails — useful as an sbatch
step-gate.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def check_env() -> bool:
    ok = True
    load_dotenv()
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        v = os.environ.get(var, "")
        has = bool(v) and not v.endswith("...") and len(v) > 20
        print(f"  {var}: {'OK' if has else 'MISSING/placeholder'} (len={len(v)})")
        ok &= has
    for p in ("data/splits/eval.jsonl", "data/splits/train.jsonl", ".env"):
        exists = Path(p).exists()
        print(f"  {p}: {'OK' if exists else 'MISSING'}")
        ok &= exists
    for var in ("HF_HOME", "PYTHONPATH"):
        v = os.environ.get(var)
        print(f"  ${var}: {v!r}")
    return ok


def check_backend(model: str) -> bool:
    # Lazy import so auth / env failure above surfaces first.
    from mathnet_eval.inference import generate
    try:
        r = generate(
            "What is 2 + 2? Reply with just the number.",
            model,
            max_tokens=32,
            temperature=0.0,
            use_cache=False,
        )
        out_tok = r.usage.get("output_tokens", "?")
        extra = ""
        for k in ("reasoning_tokens", "thoughts_tokens", "thinking_tokens"):
            if r.usage.get(k):
                extra += f" {k}={r.usage[k]}"
        print(f"  {model}: OK ({r.latency_s:.1f}s, out={out_tok}{extra})")
        return True
    except Exception as e:
        print(f"  {model}: FAIL -- {type(e).__name__}: {e}")
        return False


def main() -> int:
    section("env / files")
    env_ok = check_env()

    section("backend ping (1 tiny call each, not cached)")
    models = ["sonnet-4-6", "opus-4-7", "gpt-5.4", "gpt-5.4-mini", "gemini-3-pro"]
    backend_results = {m: check_backend(m) for m in models}

    section("summary")
    all_ok = env_ok and all(backend_results.values())
    print(f"  env+files: {'OK' if env_ok else 'FAIL'}")
    for m, ok in backend_results.items():
        print(f"  {m}: {'OK' if ok else 'FAIL'}")
    print(f"\n  >>> {'READY' if all_ok else 'NOT READY'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
