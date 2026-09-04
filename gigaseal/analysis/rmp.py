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

from .core.base import AnalysisBase

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
    window: float = 10.0  # seconds
    bin_time: float = 100.0  # milliseconds
    crop_spikes: bool = False
    filter: int = 0
    method: str = 'mode'

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
        from ..patch_utils import crop_spikes
        #if the end user kwargs override the default parameters, update the instance attributes accordingly.
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.debug("Overriding parameter %s with value %s", key, value)
        # Proceed with the RMP computation using the updated parameters.
        dt = x[1] - x[0]  # time step between samples
        f10 = int(self.window * (1 / dt))  # number of samples in the first/last window
        if f10 > y.shape[0]:
            f10 = y.shape[0] -1 #prevent overstepping the array bounds

        if self.crop_spikes:
            y = crop_spikes(x, y, c)  # mask action potentials before computing Vm

        #full mean
        overall_mean_vm = np.nanmean(y)
        overall_std_vm = np.nanstd(y)
        overall_mode_vm = rmp_mode(y, method=self.method)

        #first window statistics
        first_window = y[:f10]
        first_window_mean_vm = np.nanmean(first_window)
        first_window_median_vm = np.nanmedian(first_window)
        first_window_mode_vm = rmp_mode(first_window, method=self.method)

        #end window statistics
        end_window = y[-f10:]
        end_window_mean_vm = np.nanmean(end_window)
        end_window_median_vm = np.nanmedian(end_window)
        end_window_mode_vm = rmp_mode(end_window, method=self.method)

        delta_vm = end_window_mean_vm - first_window_mean_vm
        length_s = x[-1] - x[0]

        return {
            "overall_mean_vm": overall_mean_vm,
            "overall_std_vm": overall_std_vm,
            "first_window_mean_vm": first_window_mean_vm,
            "first_window_median_vm": first_window_median_vm,
            "first_window_mode_vm": first_window_mode_vm,
            "end_window_mean_vm": end_window_mean_vm,
            "end_window_median_vm": end_window_median_vm,
            "end_window_mode_vm": end_window_mode_vm,
            "delta_vm": delta_vm,
            "length_s": length_s,
        }
        
def half_sample_mode(data):
    """Estimate the mode of continuous data via the half-sample mode (HSM) algorithm.

    Robertson & Cryer (1974) / Bickel & Fruhwirth (2006): recursively narrow to the
    shortest contiguous half of the sorted data, then return the mean of the survivors.
    Resolution-free (no binning) and robust to outliers/skew, so it does not suffer the
    bin-edge sensitivity of a coarse-to-fine binned mode.

    Args:
        data (np.array): 1-D sample of values (NaNs are dropped).
    Returns:
        float: the estimated mode, or nan if there is no finite data.
    """
    x = np.sort(np.ravel(data))
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return np.nan
    while n > 2:
        half = (n + 1) // 2  # ceil(n/2): smallest window holding at least half the points
        widths = x[half - 1:] - x[:n - half + 1]
        j = np.argmin(widths)
        x = x[j:j + half]
        n = x.size
    return float(np.mean(x))


def rmp_mode(dataV, dataC=None, round_factor=10, method='mode'):
    """Compute the resting membrane potential from the voltage trace before the stimulus.

    Args:
        dataV (np.array): the voltage data
        dataC (np.array): the current data
        round_factor (int, optional): rounding factor for the binned mode. Defaults to 10
            (0.1mV resolution). Set to 1 for 1mV resolution. Ignored when method='hsm'.
        method (str, optional): 'mode' (default) uses the binned/rounded mode; 'hsm' uses
            the resolution-free half-sample mode, which is more precise and avoids the
            bin-edge sensitivity of the rounded mode. Defaults to 'mode'.
    Returns:
        float: the resting membrane potential
    """
    import scipy.stats
    #take upto the first non zero in the current trace
    if dataV.ndim == 1:
        dataV = dataV[np.newaxis, :]

    if dataC is not None:
        #if data C or DataV is 1-D, make them 2-D for consistency
        if dataC.ndim == 1:
            dataC = dataC[np.newaxis, :]
        pre = np.ravel(np.where(dataC[0]>0))
        if len(pre) == 0:
            pre = 1
        else:
            pre = pre[0]
        if pre==0:
            #skip the nonzeros at the begining 
            pre = np.where(np.round(dataC[0])>0)[0][0]
    else:
        pre = dataV.shape[1] #if no current trace is provided, use the entire voltage trace before the stimulus
    if method == 'hsm':
        return half_sample_mode(dataV[:, :pre])
    mode_vm = scipy.stats.mode(np.ravel(np.round(dataV[:, :pre]*round_factor)/round_factor), axis=None, nan_policy='omit')[0]
    if not np.isscalar(mode_vm):
        mode_vm = mode_vm[0] #depending on the version of scipy, mode returns an array or a single value (here we make sure it is a single value)
    return mode_vm