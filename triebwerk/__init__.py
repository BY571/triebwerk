"""Triebwerk: Fast GRPO fine-tuning for LLMs."""

VERSION = "0.1.0"
__version__ = VERSION

from triebwerk.trainers.grpo import GRPOTrainer
from triebwerk.trainers.dg import DGTrainer

__all__ = ["GRPOTrainer", "DGTrainer", "VERSION", "__version__"]
