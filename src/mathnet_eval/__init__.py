"""mathnet-eval: evaluation harness + QLoRA pipeline for MathNet."""

__version__ = "0.1.0"

# Single source of truth for the per-output token ceiling. Sbatch
# `--max-new-tokens` flags MUST match this for figure A's saturated /
# boxed labels to be comparable across runs. Analysis code imports
# this constant rather than hardcoding 16384.
SATURATION_CUTOFF = 16384
