"""Aggregate eval results, generate comparison tables and figures."""

from __future__ import annotations


def load_results(results_dir: str):
    """Load all eval JSON outputs under `results/` into a DataFrame."""
    raise NotImplementedError


def accuracy_by_model(df):
    """Overall and per-category accuracy for each model."""
    raise NotImplementedError


def cost_accuracy_pareto(df):
    """Cost vs. accuracy scatter for the blog post headline figure."""
    raise NotImplementedError
