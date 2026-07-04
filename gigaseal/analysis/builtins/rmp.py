"""
Resting-membrane-potential (RMP) analysis module.

Computes per-sweep resting membrane potential statistics (overall / windowed
mean, median, mode, and drift) with optional action-potential cropping, plus a
time-resolved running bin of Vm.

Migrated from the legacy ``gigaseal/bin/run_rmp.py`` interactive script.
The framework plumbing (registration, parameters, batching) is scaffolded
here; the ``analyze()`` body is left for a human to author/port so the
lab-specific RMP logic is reviewed rather than machine-generated.
"""

import logging

import numpy as np

from ..base import AnalysisBase

logger = logging.getLogger(__name__)


class RmpAnalysis(AnalysisBase):
    """
    Compute resting-membrane-potential statistics per sweep.

    Reproduces ``rmp_abf`` from the legacy ``run_rmp.py`` script: for each
    sweep it measures the overall mean/STD Vm, the mean/median/mode Vm over
    the first and last ``window`` seconds, and the delta between them.
    Action potentials can optionally be masked before averaging via the
    shared :func:`gigaseal.patch_utils.crop_spikes` helper.

    Running-bin aggregation should reuse
    :func:`gigaseal.patch_utils.build_running_bin` rather than the local
    ``running_bin`` in the legacy script.

    Parameters
    ----------
    window : float
        Length (s) of the leading/trailing window used for the
        first/last Vm statistics (legacy ``lowerlim``, default 10 s).
    bin_time : float
        Running-bin width in milliseconds (default 100 ms).
    crop_spikes : bool
        If ``True``, mask action potentials before computing Vm
        (legacy experimental ``crop`` option).
    filter : int
        Lowpass filter frequency (kHz); ``0`` disables filtering.

    Output keys
    -----------
    ``overall_mean_vm``, ``overall_std_vm``, ``first_window_mean_vm``,
    ``first_window_median_vm``, ``first_window_mode_vm``,
    ``end_window_mean_vm``, ``end_window_median_vm``,
    ``end_window_mode_vm``, ``delta_vm``, ``length_s``.
    """

    name = "rmp"
    display_name = "Resting Membrane Potential"
    sweep_mode = "per_sweep"

    # Parameters — typed class attributes only
    window: float = 10.0
    bin_time: float = 100.0
    crop_spikes: bool = False
    filter: int = 0

    def analyze(self, x, y, c, **kwargs) -> dict:
        """
        Compute RMP statistics for one sweep.

        Parameters
        ----------
        x, y, c : np.ndarray
            1-D time, response (Vm), and command arrays for one sweep.

        Returns
        -------
        dict
            The RMP statistics described in the class docstring.
        """
        # TODO(human): port per-sweep RMP computation from
        # gigaseal/bin/run_rmp.py::rmp_abf. Key points:
        #  * Optionally mask spikes via patch_utils.crop_spikes(x, y, c)
        #    when self.crop_spikes is True (replaces legacy crop_ap).
        #  * Use np.nanmean/np.nanstd/np.nanmedian and scipy.stats.mode
        #    over the first/last window (self.window seconds) of the sweep.
        #  * Reuse patch_utils.build_running_bin for the running-bin frame
        #    instead of the legacy local running_bin().
        #  * Import scipy/ipfx lazily inside this method.
        raise NotImplementedError(
            "RmpAnalysis.analyze() body pending human authoring — "
            "port from gigaseal/bin/run_rmp.py::rmp_abf."
        )
