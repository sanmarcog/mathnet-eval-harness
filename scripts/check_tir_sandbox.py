"""Run the TIR sandbox sentinel set and assert each captured stdout matches
its hand-graded expected value. Run this before eval_tir.py --smoke (and
ideally pin it as a precondition in the smoke sbatch) so a sympy / numpy
/ Python version bump that silently breaks sandbox semantics gets caught
before any model rollouts.

Exits non-zero on the first mismatch.

Usage:
    python scripts/check_tir_sandbox.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mathnet_eval.tir import PythonSandbox  # noqa: E402


SENTINEL_PATH = Path("tests/tir_sandbox_sentinels.json")


def main() -> int:
    data = json.loads(SENTINEL_PATH.read_text())
    sandbox = PythonSandbox()
    fails: list[str] = []
    for s in data["sentinels"]:
        result = sandbox.run(s["code"])
        actual = result.stdout.strip()
        expected = s["expected_stdout"].strip()
        ok = (actual == expected) and not result.timed_out and result.returncode == 0
        status = "OK " if ok else "FAIL"
        print(f"  [{status}] {s['id']:40s} got={actual!r}  expected={expected!r}", flush=True)
        if not ok:
            fails.append(s["id"])
            if result.stderr:
                print(f"         stderr: {result.stderr.strip()[:300]}", flush=True)
    if fails:
        print(f"\n{len(fails)} sentinel(s) failed: {fails}", file=sys.stderr)
        return 1
    print(f"\nall {len(data['sentinels'])} sentinels passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
