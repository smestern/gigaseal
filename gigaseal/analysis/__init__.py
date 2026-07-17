"""
gigaseal.analysis — modular electrophysiology analysis framework
=====================================================================

Write a class, implement ``analyze()``, and you're done.

Quickstart
----------
::

    from gigaseal.analysis import AnalysisBase, register, run_batch

    class MyAnalysis(AnalysisBase):
        name = "my_analysis"
        sweep_mode = "per_sweep"          # or "per_file"

        threshold: float = -20.0          # parameters are class attributes

        def analyze(self, x, y, c, **kwargs):
            peak = float(max(y))
            return {"peak": peak, "above_threshold": peak > self.threshold}

    register(MyAnalysis)

    # Run on a single file
    result = MyAnalysis().run(file="recording.abf")
    print(result.to_dataframe())

    # Run on a folder
    combined = run_batch(MyAnalysis(), "data_folder/")
    combined.to_dataframe().to_csv("results.csv")

Public API
----------
- ``AnalysisBase``  — base class to subclass
- ``AnalysisResult`` — result container
- ``register``       — register a module with the global registry
- ``get``            — retrieve a registered module by name
- ``list_modules``   — list all registered module names
- ``run_batch``      — batch-process a folder of ABF files
- ``save_results``   — save an AnalysisResult to CSV/Excel
"""

from .core import (
    AnalysisBase,
    AnalysisResult,
    SummarySpec,
    FirstSpikeSpec,
    register,
    get,
    list_modules,
    get_all,
    clear,
    run_batch,
    save_results,
)

# Import the concrete analysis modules so they can be re-exported directly,
# e.g. ``from gigaseal.analysis import SpikeAnalysis``.
from .spike import SpikeAnalysis, LegacySpikeAnalysis
from .subthreshold import SubthresholdAnalysis
from .example import PeakDetector
from .qc import QcAnalysis
from .rmp import RmpAnalysis
from .membrane_fit import MembraneAnalysis
from .growth_factor import GrowthFactorAnalysis

# Register the built-in modules with the global registry on import.
register(SpikeAnalysis)
register(SubthresholdAnalysis)
register(LegacySpikeAnalysis)
register(QcAnalysis)
register(RmpAnalysis)
register(MembraneAnalysis)
register(GrowthFactorAnalysis)
# PeakDetector is a demo module (hidden=True): registered so it is reachable by
# name, but it never surfaces in the end-user GUI.
register(PeakDetector)


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
    # Built-in analysis modules
    "SpikeAnalysis",
    "LegacySpikeAnalysis",
    "SubthresholdAnalysis",
    "PeakDetector",
    "QcAnalysis",
    "RmpAnalysis",
    "MembraneAnalysis",
    "GrowthFactorAnalysis",
]


# ======================================================================
# Backward-compatibility shims for GUI code that imports old names.
# These will be removed once the GUI is updated.
# ======================================================================

class _RegistryCompat:
    """
    Minimal shim so ``from gigaseal.analysis import registry``
    followed by ``registry.get_module("spike")`` still works.
    """
    def get_module(self, name):
        return get(name)

    def get_module_by_tab(self, tab_index):
        # Legacy GUI tab mapping — return None so GUI falls back to legacy
        return None

    def get_analyzer(self, name):
        return get(name)

    def list_modules(self):
        return list_modules()

    def list_modules_detailed(self):
        return {name: mod.display_name for name, mod in get_all().items()}

    def register_module(self, module):
        register(module)

    def get_registry_info(self):
        return {"modules": self.list_modules_detailed(),
                "module_count": len(list_modules())}

    def get_analyzer_info(self, name):
        m = get(name)
        if m is None:
            return None
        return {"name": m.name, "display_name": m.display_name}


# Expose as a singleton so ``from gigaseal.analysis import registry``
# gives the shim object (matching the old global instance pattern).
registry = _RegistryCompat()

# Legacy name aliases
AnalysisModule = AnalysisBase          # old base class name
AnalysisParameters = None              # placeholder — GUI checks will still
                                       # need updating but won't crash on import
AnalysisRunner = None                  # same: placeholder
