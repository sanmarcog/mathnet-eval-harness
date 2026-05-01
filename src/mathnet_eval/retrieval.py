"""Retrieval over a TIR exemplar bank.

Three policies, all returning ``list[dict]`` of exemplars in descending
relevance order:

* ``BM25Retriever`` — rank_bm25 over tokenized problem text.
* ``DenseRetriever`` — sentence-transformers dense embeddings (BGE-small).
  Lazily imports its deps; raises if ``--mode tir_rag`` is run without
  them.
* ``TopicRetriever`` — exact-match on `topics_flat`. Falls back to BM25
  within the matched topic for tiebreaking.

Bank format: JSONL, one row per exemplar, with at least these keys:
``id, problem, code, output, final_answer, topics_flat``. Built by
``scripts/build_tir_exemplar_bank.py``; smoke uses
``tests/tir_smoke_exemplar_bank.jsonl``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Sequence


# ---------------------------------------------------------------------------
# Bank loading
# ---------------------------------------------------------------------------

def load_bank(path: str | Path) -> list[dict]:
    """Load a JSONL exemplar bank. Required minimal fields: ``id, problem,
    final_answer``. Format-specific fields (``code/output`` for TIR,
    ``reasoning`` for CoT) are validated downstream by the formatter."""
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty exemplar bank: {path}")
    required = {"id", "problem", "final_answer"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"bank rows missing required keys: {missing}  ({path})")
    return rows


# ---------------------------------------------------------------------------
# Tokenization (shared by BM25 + topic-fallback BM25)
# ---------------------------------------------------------------------------

_TOK_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOK_RE.findall(text)]


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

class BM25Retriever:
    """rank_bm25-based retrieval. Falls back to a simple TF overlap if
    ``rank_bm25`` is not installed (smoke-friendly)."""

    def __init__(self, bank: Sequence[dict]) -> None:
        self.bank = list(bank)
        self.tokens = [_tokenize(r["problem"]) for r in self.bank]
        try:
            from rank_bm25 import BM25Okapi  # type: ignore

            self._bm25 = BM25Okapi(self.tokens)
            self._mode = "bm25"
        except ImportError:
            self._bm25 = None
            self._mode = "tf-fallback"

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        q_tokens = _tokenize(query)
        if self._bm25 is not None:
            scores = self._bm25.get_scores(q_tokens)
        else:
            # TF overlap fallback. Not as good as BM25 but lets smoke run
            # without rank_bm25 installed.
            q_set = set(q_tokens)
            scores = [
                sum(1 for t in toks if t in q_set) / max(1, len(toks))
                for toks in self.tokens
            ]
        ranked = sorted(
            range(len(self.bank)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]
        return [self.bank[i] for i in ranked]


# ---------------------------------------------------------------------------
# Dense (BGE) — lazy
# ---------------------------------------------------------------------------

class DenseRetriever:
    """Sentence-transformers dense retrieval. Imports the heavy deps lazily
    so smoke runs that only use BM25/topic don't pay for them."""

    def __init__(
        self,
        bank: Sequence[dict],
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            import numpy as np  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "DenseRetriever requires sentence-transformers + numpy. "
                "Install with `pip install sentence-transformers`."
            ) from e
        self.bank = list(bank)
        self.model = SentenceTransformer(model_name, device=device)
        problems = [r["problem"] for r in self.bank]
        self.embeddings = self.model.encode(
            problems, normalize_embeddings=True, convert_to_numpy=True
        )

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        import numpy as np  # type: ignore

        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        scores = self.embeddings @ q
        ranked = np.argsort(-scores)[:k]
        return [self.bank[int(i)] for i in ranked]


# ---------------------------------------------------------------------------
# Topic
# ---------------------------------------------------------------------------

def _top_topic(topics_flat: Iterable[str] | None) -> str | None:
    if not topics_flat:
        return None
    first = next(iter(topics_flat))
    return first.split(">")[0].strip() if first else None


class TopicRetriever:
    """Match exemplars whose top-level `topics_flat` prefix equals the
    query's. Within the matched group, rank by BM25. If a query has no
    topic or no exemplars share its top topic, fall back to global BM25."""

    def __init__(self, bank: Sequence[dict]) -> None:
        self.bank = list(bank)
        self._global_bm25 = BM25Retriever(self.bank)
        self._by_topic: dict[str, list[dict]] = {}
        for row in self.bank:
            t = _top_topic(row.get("topics_flat"))
            if t is None:
                continue
            self._by_topic.setdefault(t, []).append(row)

    def retrieve(self, query: str, k: int = 3, query_topics_flat: Iterable[str] | None = None) -> list[dict]:
        topic = _top_topic(query_topics_flat)
        if topic is None or topic not in self._by_topic:
            return self._global_bm25.retrieve(query, k=k)
        sub = self._by_topic[topic]
        sub_bm25 = BM25Retriever(sub)
        return sub_bm25.retrieve(query, k=k)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_retriever(policy: str, bank: Sequence[dict], **kwargs):
    """Dispatch by policy name. ``policy`` is one of: ``bm25``, ``dense``,
    ``topic``."""
    if policy == "bm25":
        return BM25Retriever(bank)
    if policy == "dense":
        return DenseRetriever(bank, **kwargs)
    if policy == "topic":
        return TopicRetriever(bank)
    raise ValueError(f"unknown retrieval policy: {policy}")
