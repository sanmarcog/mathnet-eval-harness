"""18-cell retrieval ablation for the Week-2 TIR-RAG sub-grid.

Sweeps `(index ∈ {bm25, dense, topic}) × (k ∈ {1, 3, 5}) × (exemplar_type ∈
{tir, cot})` on the dev N=100 split and picks the single winner under the
locked tie-break rule from the pre-reg (smaller k > BM25 over dense > TIR-
over CoT-exemplar). The winner advances to the headline n=500 main eval.

The model is loaded **once**; only the retriever (and the bank it points
at) swaps across cells. With vLLM hot, this keeps the 18-cell sweep at
~2.5h on an L40s rather than ~5h with re-load-per-cell.

Bank paths are required args so the smoke can swap in the fixture bank
(``tests/tir_smoke_exemplar_bank.jsonl``) without waiting on the real
bank to be built. Per the user's guidance: "make sure it accepts the bank
path as an arg (so smoke-bank can be swapped in for code-test runs)."

Output layout::

    results/tir/ablation/
        ablation_summary.json     # one-row-per-cell + the locked winner
        bm25_k1_tir/<id>.json     # per-cell, per-problem
        bm25_k1_tir/summary.json
        bm25_k3_tir/...
        ...

Usage:
    # smoke (CPU, 4 dev problems, fixture bank, all 18 cells exercised):
    python scripts/run_retrieval_ablation.py --smoke

    # production (Hyak):
    python scripts/run_retrieval_ablation.py \\
        --dev-jsonl data/splits/dev_100.jsonl \\
        --bank-tir results/tir/exemplar_bank.jsonl \\
        --bank-cot results/tir/exemplar_bank_cot.jsonl \\
        --model Qwen/Qwen2.5-Math-1.5B-Instruct \\
        --backend vllm \\
        --out results/tir/ablation
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Re-use eval_tir.py's per-problem evaluator + backend loaders so there's
# one source of truth for "what does TIR-RAG eval mean for one problem".
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_tir import (  # noqa: E402
    apply_smoke_defaults as _eval_smoke_defaults,  # not used directly; re-exported for parity
    eval_one,
    make_hf_backend,
    make_vllm_backend,
    with_smoke_tool_canary,
)

from mathnet_eval.retrieval import build_retriever, load_bank  # noqa: E402
from mathnet_eval.tir import PythonSandbox  # noqa: E402

del _eval_smoke_defaults  # silence unused-import linter


# ---------------------------------------------------------------------------
# Cell grid + tie-break
# ---------------------------------------------------------------------------

INDEXES = ["bm25", "dense", "topic"]
KS = [1, 3, 5]
EXEMPLAR_TYPES = ["tir", "cot"]


def all_cells() -> list[tuple[str, int, str]]:
    return [
        (idx, k, ex) for idx, k, ex in itertools.product(INDEXES, KS, EXEMPLAR_TYPES)
    ]


def cell_id(idx: str, k: int, ex: str) -> str:
    return f"{idx}_k{k}_{ex}"


def pick_winner(per_cell: list[dict]) -> dict:
    """Apply the locked tie-break from the pre-reg:
       1. highest accuracy
       2. within 1 pp of the leader: smaller k
       3. still tied: BM25 > dense > topic? (pre-reg: BM25 over dense)
       4. still tied: TIR-exemplar over CoT-exemplar
    """
    if not per_cell:
        raise ValueError("no cells to pick from")
    leader_acc = max(c["accuracy"] for c in per_cell)
    contenders = [c for c in per_cell if c["accuracy"] >= leader_acc - 0.01]

    index_pref = {"bm25": 0, "dense": 1, "topic": 2}  # BM25 over dense (per pre-reg)
    extype_pref = {"tir": 0, "cot": 1}                # TIR over CoT (per pre-reg)

    contenders.sort(key=lambda c: (
        -c["accuracy"],         # highest accuracy first
        c["k"],                  # smaller k first
        index_pref.get(c["index"], 99),
        extype_pref.get(c["exemplar_type"], 99),
    ))
    return contenders[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true",
                   help="CPU-only, fixture bank for both TIR + CoT, dev N=4, "
                        "Qwen2.5-0.5B + smoke canary. Validates the cell-loop, "
                        "retriever-swap, and summary-aggregation paths.")
    p.add_argument("--dev-jsonl", default="data/splits/dev_100.jsonl")
    p.add_argument("--bank-tir", default=None,
                   help="JSONL bank with TIR-shaped exemplars (problem + code + output + answer). "
                        "Smoke default: tests/tir_smoke_exemplar_bank.jsonl.")
    p.add_argument("--bank-cot", default=None,
                   help="JSONL bank with CoT-shaped exemplars (problem + reasoning + answer). "
                        "Smoke default: same fixture as --bank-tir (the fixture has both fields).")
    p.add_argument("--model", default=None)
    p.add_argument("--backend", choices=["hf", "vllm"], default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n", type=int, default=None,
                   help="dev problems to score; default 4 in smoke, all in production")
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--max-tool-calls", type=int, default=4)
    p.add_argument("--use-judge", action="store_true",
                   help="Pre-reg specifies cheap-grader-only on the 18 ablation cells.")
    p.add_argument("--out", default="results/tir/ablation")
    p.add_argument("--cells", default="all",
                   help="Comma-separated list of <idx>_k<k>_<extype> to run; default: all 18.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip per-cell directories that already have a summary.json.")
    return p.parse_args()


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    if args.bank_tir is None:
        args.bank_tir = "tests/tir_smoke_exemplar_bank.jsonl"
    if args.bank_cot is None:
        args.bank_cot = "tests/tir_smoke_exemplar_bank.jsonl"
    if args.model is None:
        args.model = "Qwen/Qwen2.5-0.5B-Instruct"
    if args.backend is None:
        args.backend = "hf"
    if args.dev_jsonl == "data/splits/dev_100.jsonl" and not Path(args.dev_jsonl).exists():
        # Dev split may not yet exist on a fresh checkout; fall back to eval.
        args.dev_jsonl = "data/splits/eval.jsonl"
    if args.n is None:
        args.n = 4
    if args.out == "results/tir/ablation":
        args.out = "results/tir/ablation_smoke"
    args.max_new_tokens = min(args.max_new_tokens, 256)
    args.max_tool_calls = min(args.max_tool_calls, 1)
    args.use_judge = False


def apply_production_defaults(args: argparse.Namespace) -> None:
    if args.smoke:
        return
    if args.bank_tir is None:
        args.bank_tir = "results/tir/exemplar_bank.jsonl"
    if args.bank_cot is None:
        args.bank_cot = "results/tir/exemplar_bank_cot.jsonl"
    if args.model is None:
        args.model = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    if args.backend is None:
        args.backend = "vllm"


def selected_cells(args) -> list[tuple[str, int, str]]:
    if args.cells == "all":
        return all_cells()
    out: list[tuple[str, int, str]] = []
    for spec in args.cells.split(","):
        spec = spec.strip()
        if not spec:
            continue
        try:
            idx, k_part, ex = spec.split("_")
            k = int(k_part.lstrip("k"))
            assert idx in INDEXES and k in KS and ex in EXEMPLAR_TYPES
        except (ValueError, AssertionError) as e:
            raise SystemExit(f"bad --cells entry: {spec!r}: {e}") from e
        out.append((idx, k, ex))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    apply_smoke_defaults(args)
    apply_production_defaults(args)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    cells = selected_cells(args)
    print(f"[config] cells={len(cells)}/18  smoke={args.smoke}  model={args.model}  "
          f"backend={args.backend}  dev={args.dev_jsonl}", flush=True)
    print(f"[banks] tir={args.bank_tir}  cot={args.bank_cot}", flush=True)

    # ---- Load dev rows ----
    dev_rows: list[dict] = []
    with open(args.dev_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                dev_rows.append(json.loads(line))
    if args.n is not None and args.n > 0:
        dev_rows = dev_rows[: args.n]
    print(f"[data] {len(dev_rows)} dev rows", flush=True)

    # ---- Backend (loaded once) ----
    if args.backend == "hf":
        generate_fn, tokenizer = make_hf_backend(args.model, device=args.device)
    elif args.backend == "vllm":
        generate_fn, tokenizer = make_vllm_backend(args.model)
    else:
        raise ValueError(args.backend)
    if args.smoke:
        generate_fn = with_smoke_tool_canary(generate_fn)
    sandbox = PythonSandbox()

    # ---- Banks (loaded once) ----
    bank_tir = load_bank(args.bank_tir)
    bank_cot = load_bank(args.bank_cot)
    print(f"[banks] tir_size={len(bank_tir)}  cot_size={len(bank_cot)}", flush=True)

    # ---- Cell loop ----
    per_cell: list[dict] = []
    cell_t0 = time.time()
    for cell_i, (idx, k, ex) in enumerate(cells, start=1):
        cid = cell_id(idx, k, ex)
        cell_dir = out_root / cid
        cell_dir.mkdir(parents=True, exist_ok=True)

        cell_summary_path = cell_dir / "summary.json"
        if args.skip_existing and cell_summary_path.exists():
            print(f"[{cell_i}/{len(cells)}] {cid} SKIP (exists)", flush=True)
            per_cell.append(json.loads(cell_summary_path.read_text()))
            continue

        bank = bank_tir if ex == "tir" else bank_cot
        retriever = build_retriever(idx, bank)
        cell_start = time.time()
        n_correct = 0
        n_scored = 0
        n_saturated = 0
        method_counts: dict[str, int] = {}

        for row in dev_rows:
            try:
                result = eval_one(
                    row=row,
                    mode="tir_rag",
                    tokenizer=tokenizer,
                    generate_fn=generate_fn,
                    sandbox=sandbox,
                    retriever=retriever,
                    k=k,
                    max_new_tokens_total=args.max_new_tokens,
                    max_tool_calls=args.max_tool_calls,
                    use_judge=args.use_judge,
                    exemplar_type=ex,
                )
            except Exception as e:
                if args.smoke:
                    raise
                print(f"  [error] {cid} {row['id']}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                continue

            (cell_dir / f"{row['id']}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2)
            )
            n_scored += 1
            if result["grade"]["correct"]:
                n_correct += 1
            method_counts[result["grade"]["method"]] = method_counts.get(result["grade"]["method"], 0) + 1
            if result["saturated"]:
                n_saturated += 1

        cell_summary = {
            "cell_id": cid,
            "index": idx,
            "k": k,
            "exemplar_type": ex,
            "n_problems": len(dev_rows),
            "n_scored": n_scored,
            "n_correct": n_correct,
            "accuracy": n_correct / n_scored if n_scored else 0.0,
            "n_saturated": n_saturated,
            "method_counts": method_counts,
            "elapsed_s": round(time.time() - cell_start, 2),
        }
        cell_summary_path.write_text(json.dumps(cell_summary, ensure_ascii=False, indent=2))
        per_cell.append(cell_summary)
        print(
            f"[{cell_i}/{len(cells)}] {cid:20s} "
            f"acc={cell_summary['accuracy']:.3f}  "
            f"saturated={n_saturated}/{n_scored}  "
            f"{cell_summary['elapsed_s']:.1f}s",
            flush=True,
        )

    # ---- Aggregate ablation summary ----
    winner = pick_winner(per_cell) if per_cell else None
    ablation_summary = {
        "smoke": args.smoke,
        "model": args.model,
        "backend": args.backend,
        "dev_jsonl": args.dev_jsonl,
        "n_problems": len(dev_rows),
        "bank_tir": args.bank_tir,
        "bank_cot": args.bank_cot,
        "n_cells": len(per_cell),
        "elapsed_s": round(time.time() - cell_t0, 2),
        "cells": per_cell,
        "winner": winner,
    }
    (out_root / "ablation_summary.json").write_text(
        json.dumps(ablation_summary, ensure_ascii=False, indent=2)
    )
    print(f"\n[ablation_summary] cells={len(per_cell)}  elapsed={ablation_summary['elapsed_s']}s", flush=True)
    if winner:
        print(f"[winner] {winner['cell_id']}  acc={winner['accuracy']:.3f}", flush=True)

    # Success criterion: every requested cell ran AND every cell produced
    # at least one scored output. Acc=0 is fine — that's a real possible
    # outcome on hard math at 1.5B (and the expected outcome in smoke,
    # where Qwen2.5-0.5B + canary cannot solve olympiad problems). What
    # we're guarding against is silent breakage where cells produce
    # zero outputs (sandbox down, retriever broken, etc.).
    n_expected = len(cells)
    empty_cells = [c["cell_id"] for c in per_cell if c.get("n_scored", 0) == 0]
    missing = n_expected - len(per_cell)
    if missing > 0:
        print(f"[FAIL] {missing} cell(s) did not run", file=sys.stderr, flush=True)
        return 4
    if empty_cells:
        print(f"[FAIL] {len(empty_cells)} cell(s) produced zero scored outputs: {empty_cells}",
              file=sys.stderr, flush=True)
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
