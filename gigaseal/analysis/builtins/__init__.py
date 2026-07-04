"""
Built-in analysis modules.

Importing this package auto-registers the standard analysis modules
(spike, subthreshold, peak_detector example).
"""

from ..registry import register

from .spike import SpikeAnalysis, LegacySpikeAnalysis
from .subthreshold import SubthresholdAnalysis
from .example import PeakDetector
from .qc import QcAnalysis
from .rmp import RmpAnalysis
from .membrane_fit import MembraneAnalysis
from .growth_factor import GrowthFactorAnalysis

# Register all built-in modules
register(SpikeAnalysis)
register(SubthresholdAnalysis)
register(LegacySpikeAnalysis)
register(QcAnalysis)
register(RmpAnalysis)
register(MembraneAnalysis)
register(GrowthFactorAnalysis)

# PeakDetector is a demo/dummy module. It is registered so the test suite and
# programmatic users can reach it by name, but it sets ``hidden = True`` so it
# never shows up in the end-user GUI.
register(PeakDetector)