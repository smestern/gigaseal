"""
gigaseal.analysis.core — framework internals.

Holds the base class, result containers, the global registry, and the batch
runner. Import these through the public ``gigaseal.analysis`` namespace; this
subpackage is the implementation home, not the user-facing API.
"""

from .base import AnalysisBase
from .result import AnalysisResult, SummarySpec, FirstSpikeSpec
from .registry import register, get, list_modules, get_all, clear
from .runner import run_batch, save_results

__all__ = [
    "AnalysisBase",
    "AnalysisResult",
    "SummarySpec",
    "FirstSpikeSpec",
    "register",
    "get",
    "list_modules",
    "get_all",
    "clear",
    "run_batch",
    "save_results",
]
