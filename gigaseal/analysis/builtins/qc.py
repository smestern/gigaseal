"""
Quality-control analysis module — wraps ``gigaseal.QC``.

Computes recording-quality metrics (RMS noise and Vm drift over the
baseline / zero-current region) across all sweeps of a file.

Migrated from the legacy ``gigaseal/bin/run_QC.py`` interactive script.
The framework plumbing (registration, parameters, batching) is scaffolded
here; the ``analyze()`` body is left for a human to author/port so the
lab-specific QC logic is reviewed rather than machine-generated.
"""

import logging

import numpy as np

from ..base import AnalysisBase

logger = logging.getLogger(__name__)


class QcAnalysis(AnalysisBase):
    """
    Compute per-file recording quality-control metrics.

    Wraps :func:`gigaseal.QC.run_qc`, which measures RMS noise and
    membrane-voltage drift over the baseline (zero-current) region using
    the full ``(sweeps × samples)`` response and command arrays.

    Runs in ``per_file`` mode because ``run_qc`` needs the 2-D response /
    command matrices to locate the shared baseline window and compute
    sweep-wise drift.

    Parameters
    ----------
    filter : int
        Allen/ipfx Gaussian filter frequency (kHz). ``0`` disables
        filtering (recommended for QC).

    Output keys
    -----------
    ``mean_rms``, ``max_rms``, ``mean_vm_drift``, ``max_vm_drift`` — the
    four values returned by :func:`gigaseal.QC.run_qc`.
    """

    name = "qc"
    display_name = "Quality Control"
    sweep_mode = "per_file"

    # Parameters — typed class attributes only
    filter: int = 0

    def analyze(self, x, y, c, **kwargs) -> dict:
        """
        Compute QC metrics for one file.

        Parameters
        ----------
        x, y, c : np.ndarray
            2-D ``(sweeps × samples)`` time, response, and command arrays.

        Returns
        -------
        dict
            Keys: ``mean_rms``, ``max_rms``, ``mean_vm_drift``,
            ``max_vm_drift``.
        """
        # TODO(human): port QC computation from gigaseal/bin/run_QC.py.
        # Wrap gigaseal.QC.run_qc(realY=y, realC=c) which returns
        # [mean_rms, max_rms, mean_drift, max_drift]. Apply the optional
        # ipfx Gaussian `filter` first if self.filter > 0, then map the
        # returned list onto the documented output keys. Import ipfx/QC
        # lazily inside this method (never at module top).
        raise NotImplementedError(
            "QcAnalysis.analyze() body pending human authoring — "
            "port from gigaseal/bin/run_QC.py (wraps gigaseal.QC.run_qc)."
        )
