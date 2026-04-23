"""MathNet loading, Week-1 filters, stratified splits, and prompt formatting.

Notes on the dataset (ShadenA/MathNet):
    * The HF repo has both a unified `data/all/` partition (~27.8K unique
      problems, 56 parquet files) and per-country/competition partitions that
      are slices of the same data. We always load from `data/all/` to avoid
      double-counting.
    * The parquet metadata carries a feature type the qlora env's datasets
      3.0.0 can't parse, so we skip `datasets.load_dataset` and use pyarrow
      directly.
    * `language` values are inconsistent (e.g. 'English', 'español', 'English;
      Russian'). `is_english` below matches the canonical 'english' spelling
      (case-insensitive) only; multi-language rows like 'Chinese; English'
      are intentionally excluded from Week-1 to keep the comparison clean.
    * `problem_type` is one of: 'proof and answer', 'proof only', 'final
      answer only', 'MCQ'. Eval-set problems must have a non-empty
      `final_answer`, which rules out 'proof only'.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download


REPO = "ShadenA/MathNet"


# ---- Loading ----------------------------------------------------------------

def _all_parquet_files() -> list[str]:
    info = HfApi().dataset_info(REPO)
    return sorted(
        s.rfilename for s in info.siblings
        if s.rfilename.startswith("data/all/") and s.rfilename.endswith(".parquet")
    )


def load_mathnet(columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Load all of `data/all/` into a pandas DataFrame. ~28K rows.

    Pass `columns` to skip heavy fields — notably `images` (contains binary
    image bytes and is the largest column).
    """
    remote_files = _all_parquet_files()
    if columns is not None:
        columns = list(columns)
    tables = []
    for f in remote_files:
        local = hf_hub_download(repo_id=REPO, filename=f, repo_type="dataset")
        tables.append(pq.read_table(local, columns=columns))
    return pa.concat_tables(tables).to_pandas()


# ---- Filters ----------------------------------------------------------------

def is_english(lang: str | None) -> bool:
    """Only match rows whose language is exactly English (not bilingual)."""
    if not isinstance(lang, str):
        return False
    return lang.strip().lower() == "english"


def is_text_only(images) -> bool:
    """Row has no embedded images. pyarrow returns the list column as a
    numpy object array — length 0 means text-only."""
    if images is None:
        return True
    try:
        return len(images) == 0
    except TypeError:
        return False


def has_final_answer(fa: str | None) -> bool:
    return isinstance(fa, str) and fa.strip() != ""


def apply_week1_filters(df: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
    """Apply the Week-1 scope filters and print the funnel.

    Scope: English-only, text-only, has a non-empty `final_answer` (so we can
    grade it). Rule for language is strict — 'English' case-insensitive only,
    no bilingual rows like 'Chinese; English' — to keep the eval clean.
    """
    def _print(label: str, n: int) -> None:
        if verbose:
            print(f"  {label:38s} {n:>6d}")

    _print("total rows:", len(df))

    df = df[df["images"].map(is_text_only)]
    _print("after text-only filter:", len(df))

    df = df[df["language"].map(is_english)]
    _print("after English-only filter:", len(df))

    df = df[df["final_answer"].map(has_final_answer)]
    _print("after has-final-answer filter:", len(df))

    return df.reset_index(drop=True)


# ---- Stratified split -------------------------------------------------------

def stratified_split(
    df: pd.DataFrame,
    *,
    eval_size: int = 500,
    train_size: int | None = None,
    strata_col: str = "competition",
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split `df` into (eval, train) so that each stratum (competition) is
    represented proportionally in both. If `train_size` is None, train
    becomes everything not in eval.

    Small strata: if a competition has < 2 rows, its single row goes to
    train (can't be split). We never oversample.
    """
    rng = pd.Series(range(len(df))).sample(frac=1, random_state=seed).values
    df = df.iloc[rng].reset_index(drop=True)

    total = len(df)
    if eval_size > total:
        raise ValueError(f"eval_size {eval_size} > available rows {total}")

    eval_frac = eval_size / total

    eval_idx: list[int] = []
    for _, group in df.groupby(strata_col, sort=False):
        g = group.index.tolist()
        k = max(1, round(len(g) * eval_frac)) if len(g) >= 2 else 0
        eval_idx.extend(g[:k])

    # Trim / top up to hit eval_size exactly.
    eval_idx = eval_idx[:eval_size] if len(eval_idx) >= eval_size else (
        eval_idx + [i for i in range(total) if i not in set(eval_idx)][: eval_size - len(eval_idx)]
    )
    eval_set = set(eval_idx)

    eval_df = df.loc[sorted(eval_set)].reset_index(drop=True)
    train_df = df.drop(index=list(eval_set)).reset_index(drop=True)
    if train_size is not None and train_size < len(train_df):
        train_df = train_df.sample(n=train_size, random_state=seed).reset_index(drop=True)
    return eval_df, train_df


# ---- Prompt formatting ------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert mathematician. Solve the following olympiad problem. "
    "Show your work briefly, then put the final answer on its own line in the form: "
    "Final answer: <answer>."
)


def format_prompt(row: pd.Series | dict) -> str:
    """Turn a dataset row into a single user-message string.

    Chat-style role separation is applied by the inference backend; this
    function returns the user content only.
    """
    r = row if isinstance(row, dict) else row.to_dict()
    problem = r["problem_markdown"].strip()
    return problem


# ---- Serialization ----------------------------------------------------------

EVAL_COLUMNS = [
    "id", "country", "competition", "language", "problem_type",
    "problem_markdown", "final_answer", "topics_flat",
]


def to_jsonl(df: pd.DataFrame, path: str | Path, columns: Iterable[str] = EVAL_COLUMNS) -> None:
    """Write a subset of columns to JSONL at `path`. Lists (e.g. topics_flat)
    are preserved as JSON arrays."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[list(columns)].to_json(path, orient="records", lines=True, force_ascii=False)
