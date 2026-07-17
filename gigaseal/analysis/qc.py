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

from .core.base import AnalysisBase

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
        #x is not used in the QC analysis, but is included in the signature for consistency with other analyses
        zero_ind = find_zero_qc(c[0,:])
        zero_ind = find_baseline(zero_ind)
        mean_rms, max_rms = compute_rms(y, zero_ind)
        mean_drift, max_drift = compute_vm_drift(y, zero_ind)
        return {
            "mean_rms": mean_rms,
            "max_rms": max_rms,
            "mean_vm_drift": mean_drift,
            "max_vm_drift": max_drift,
        }


def find_zero_qc(realC):
    #expects 1d array
    zero_ind = np.where(realC == 0)[0] #in this case we take the first sweep to find the zero current region, as it is assumed that all sweeps have the same zero current region
    ##Account for time constant?
    diff = np.diff(zero_ind) #zeros
    if np.amax(diff) > 1: #in this case we just want the zeros within the seweep
        diff_jump = np.where(diff>2)[0][0]
        if diff_jump + 3000 > realC.shape[0]:
            _hop = diff_jump
        else:
            _hop = diff_jump + 3000

        zero_ind_crop = np.hstack((zero_ind[:diff_jump], zero_ind[_hop:]))
    else: 
        zero_ind_crop = zero_ind
    return zero_ind_crop

def find_baseline(zero_ind):
    #the baseline will be the first continious set of zeros
    baseline_idx = np.where(np.diff(zero_ind) > 1)[0]
    if len(baseline_idx) == 0:
        baseline_idx = len(zero_ind)
    else:
        baseline_idx = baseline_idx[0]
    return zero_ind[0:baseline_idx+1]

def compute_vm_drift(realY, zero_ind):
    sweep_wise_mean = np.mean(realY[:,zero_ind], axis=1)
    mean_drift = np.abs(np.amax(sweep_wise_mean) - np.amin(sweep_wise_mean))
    abs_drift = np.abs(np.amax(realY[:,zero_ind]) - np.amin(realY[:,zero_ind]))
    return mean_drift, abs_drift

def compute_rms(realY, zero_ind):
    mean = np.mean(realY[:,zero_ind], axis=1)
    rms = []
    for x in np.arange(mean.shape[0]):
        temp = np.sqrt(np.mean(np.square(realY[x,zero_ind] - mean[x])))
        rms = np.hstack((rms, temp))
    full_mean = np.mean(rms)
    return full_mean, np.amax(rms)

#legacy wrapper for the QC analysis, to be removed once the QC class is fully integrated into the analysis framework.
#Returns the 4-item list [mean_rms, max_rms, mean_vm_drift, max_vm_drift] expected by the frozen featureExtractor call sites.
def run_qc(realY, realC):
    #spawn a object of the QC class and run the analysis
    #analyze(x, y, c): y=response (realY), c=command (realC); x is unused for QC
    qc = QcAnalysis()
    result = qc.analyze(None, realY, realC)
    return [
        result["mean_rms"],
        result["max_rms"],
        result["mean_vm_drift"],
        result["max_vm_drift"],
    ]






