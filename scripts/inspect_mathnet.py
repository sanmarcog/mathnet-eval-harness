"""Inspect the ShadenA/MathNet dataset — schema, size, strata distribution.

Run once when onboarding so we know the field layout before writing splits /
prompts / graders. Prints to stdout, no files written.

Uses pyarrow directly (not `datasets.load_dataset`) because the dataset's
parquet metadata carries a feature type the qlora env's datasets 3.0.0
cannot parse. Projects only small columns + `images.list.element.path`
so image bytes are never loaded.

Usage (Hyak login node is fine — no GPU needed):
    export HF_HOME=/gscratch/scrubbed/sanmarco/hf_cache
    /gscratch/scrubbed/sanmarco/conda/envs/qlora/bin/python -u scripts/inspect_mathnet.py
"""

from __future__ import annotations

import sys
from collections import Counter

from huggingface_hub import HfApi, hf_hub_download
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


REPO = "ShadenA/MathNet"


def is_english(lang_arr):
    """Case-insensitive match for English; the dataset also uses native names
    like 'español' / 'français', so we can't assume any particular spelling
    for other languages, but 'english' (in lower case) appears to be the
    canonical value for English rows."""
    return pc.equal(
        pc.utf8_lower(pc.coalesce(lang_arr, pa.scalar(""))),
        pa.scalar("english"),
    )


def has_final_answer(fa_arr):
    trimmed = pc.utf8_trim_whitespace(pc.coalesce(fa_arr, pa.scalar("")))
    return pc.greater(pc.utf8_length(trimmed), 0)


def image_lens(local: str):
    """Get per-row image list length without loading image bytes."""
    try:
        # Sub-field projection skips the `bytes` column entirely — cheapest path.
        tbl = pq.read_table(local, columns=["images.list.element.path"])
    except Exception:
        tbl = pq.read_table(local, columns=["images"])
    return pc.list_value_length(tbl["images"])


def main() -> int:
    api = HfApi()
    info = api.dataset_info(REPO)
    parquets = sorted(s.rfilename for s in info.siblings if s.rfilename.endswith(".parquet"))
    print(f"parquet files: {len(parquets)}", flush=True)

    # Pass 1: row counts per partition (parquet metadata only).
    total_rows = 0
    rows_per_dir = Counter()
    paths = []
    for p in parquets:
        local = hf_hub_download(repo_id=REPO, filename=p, repo_type="dataset")
        paths.append(local)
        n = pq.ParquetFile(local).metadata.num_rows
        total_rows += n
        rows_per_dir[p.split("/")[1]] += n
    print(f"TOTAL_ROWS={total_rows}  partitions={len(rows_per_dir)}", flush=True)

    # Pass 2: distributions + funnel, single loop.
    lang_chunks = []
    ptype_chunks = []
    has_img_true = 0
    has_img_total = 0
    fa_nonempty = 0

    text_only = english_any = english_text = english_text_answered = 0

    for i, local in enumerate(paths):
        tbl = pq.read_table(local, columns=["language", "problem_type", "final_answer"])
        lang = tbl["language"]
        ptype = tbl["problem_type"]
        fa = tbl["final_answer"]

        lang_chunks.extend(lang.chunks)
        ptype_chunks.extend(ptype.chunks)

        fa_nonempty += pc.sum(has_final_answer(fa)).as_py() or 0

        img_lens = image_lens(local)
        is_text = pc.equal(img_lens, 0)
        is_en = is_english(lang)
        has_ans = has_final_answer(fa)

        n_rows = len(img_lens)
        n_has_img = pc.sum(pc.greater(img_lens, 0)).as_py() or 0
        has_img_true += n_has_img
        has_img_total += n_rows

        text_only += pc.sum(is_text).as_py() or 0
        english_any += pc.sum(is_en).as_py() or 0
        english_text += pc.sum(pc.and_(is_en, is_text)).as_py() or 0
        english_text_answered += pc.sum(pc.and_(pc.and_(is_en, is_text), has_ans)).as_py() or 0

        if (i + 1) % 25 == 0:
            print(f"  scanned {i + 1}/{len(paths)} files", flush=True)

    lang_all = pa.chunked_array(lang_chunks)
    ptype_all = pa.chunked_array(ptype_chunks)

    lang_counts = pc.value_counts(lang_all).to_pylist()
    ptype_counts = pc.value_counts(ptype_all).to_pylist()

    print("\n=== language (all values) ===", flush=True)
    for d in sorted(lang_counts, key=lambda x: -x["counts"]):
        v, c = d["values"], d["counts"]
        print(f"  {v!r}: {c}")

    print("\n=== problem_type ===", flush=True)
    for d in sorted(ptype_counts, key=lambda x: -x["counts"]):
        v, c = d["values"], d["counts"]
        print(f"  {v!r}: {c}")

    print("\n=== has_image ===", flush=True)
    print(f"  True:  {has_img_true}")
    print(f"  False: {has_img_total - has_img_true}")

    print("\n=== final_answer presence ===", flush=True)
    print(f"  non-empty: {fa_nonempty}")
    print(f"  empty:     {total_rows - fa_nonempty}")

    print("\n=== Week-1 funnel ===", flush=True)
    print(f"  total rows:                     {total_rows}")
    print(f"  text-only:                      {text_only}")
    print(f"  English (any):                  {english_any}")
    print(f"  English + text-only:            {english_text}")
    print(f"  English + text-only + answered: {english_text_answered}")

    print("\n=== top 20 partitions by row count ===", flush=True)
    for k, v in rows_per_dir.most_common(20):
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
