"""
Subthreshold analysis module — wraps legacy patch_subthres functions.

Extracts passive membrane properties (sag, time constant, voltage
deflection) from hyperpolarizing sweeps.
"""

import logging
import numpy as np

#ipfx
from ipfx import subthresh_features as subt

#local
from ..patchUtils import find_hyperpolarization_segment
from .core.base import AnalysisBase
from .. import utils


logger = logging.getLogger(__name__)


class SubthresholdAnalysis(AnalysisBase):
    """
    Compute subthreshold membrane features per sweep.

    Wraps ``gigaseal.patch_subthres.subthres_allen`` to extract sag ratio,
    membrane time constant (tau_m), and voltage deflection from
    hyperpolarizing current injections.

    Parameters
    ----------
    start : float
        Start time (s) of the analysis window.
    end : float
        End time (s) of the analysis window (0 = end of sweep).
    """


    name = "subthreshold"
    display_name = "Subthreshold Analysis"
    sweep_mode = "per_sweep"

    start: float = 0.0
    end: float = 0.0

    def analyze(self, x, y, c, **kwargs) -> dict:
        """
        Compute sag, tau_m, and voltage deflection for one sweep.

        Returns
        -------
        dict
            Keys: ``sag_ratio``, ``taum``, ``voltage_deflection``.
            Values are ``np.nan`` if the sweep has no hyperpolarizing
            current or the fit fails.
        """
        from ..patch_subthres import subthres_allen

        end = self.end if self.end > 0 else x[-1]
        sag, taum, voltage_deflection = subthres_allen(x, y, c, self.start, end)

        return {
            "sag_ratio": sag,
            "taum": taum,
            "voltage_deflection": voltage_deflection,
        }


# Note: This module relies on the legacy ``gigaseal.patch_subthres`` functions. Which are now being ported here


## Models for fitting exponential growth and decay to voltage responses
def exp_grow(t, a, b, alpha):
    """Exponential growth model function.

    Parameters
    ----------
    t : array-like
        Time points.
    a : float
        Baseline voltage.
    b : float
        Amplitude of the exponential growth.
    alpha : float
        Growth rate.

    Returns
    -------
    array-like
        Modeled voltage response.
    """
    return a - b * np.exp(-alpha * t)

def exp_decay_2p(t, a, b1, alphaFast, b2, alphaSlow):
    """Double-exponential decay model function.

    Parameters
    ----------
    t : array-like
        Time points.
    a : float
        Baseline voltage.
    b1 : float
        Amplitude of the fast exponential component.
    alphaFast : float
        Decay rate of the fast component.
    b2 : float
        Amplitude of the slow exponential component.
    alphaSlow : float
        Decay rate of the slow component.

    Returns
    -------
    array-like
        Modeled voltage response.
    """
    return a + b1*np.exp(-alphaFast*t) + b2*np.exp(-alphaSlow*t)

def exp_decay_1p(t, a, b1, alphaFast):
    """Single-exponential decay model function.

    Parameters
    ----------
    t : array-like
        Time points.
    a : float
        Baseline voltage.
    b1 : float
        Amplitude of the exponential component.
    alphaFast : float
        Decay rate of the exponential component.

    Returns
    -------
    array-like
        Modeled voltage response.
    """
    return a + b1*np.exp(-alphaFast*t)

def rm_decay_2p(t, Iinj, Rm, alphaFast, Re, alphaSlow, a):
    """Double-exponential decay model for membrane potential with injected current.

    Parameters
    ----------
    t : array-like
        Time points.
    Iinj : array-like
        Injected current.
    Rm : float
        Membrane resistance for the fast component.
    alphaFast : float
        Time constant for the fast component.
    Re : float
        Membrane resistance for the slow component.
    alphaSlow : float
        Time constant for the slow component.
    a : float
        Baseline voltage.

    Returns
    -------
    array-like
        Modeled voltage response in mV.
    """
    v = a + (Iinj * (Rm * (1 - np.exp(-t/alphaFast)) + Re * (1 - np.exp(-t/alphaSlow))))
    v = v * 1000#in Volts to mV
    return v

def exp_growth_factor(dataT,dataV,dataI, alpha, end_index=1):
    """ Fits a exponential 'growth' model to the voltage response data.

    Parameters
    ----------
    dataT : array-like
        Time points of the data.
    dataV : array-like
        Voltage response data.
    dataI : array-like
        Injected current data.
    alpha : float
        Initial guess for the growth rate parameter.
    end_index : int, optional
        Index to end the fitting window (default is 1).

    Returns
    -------
    array
        Fitted parameters and the time shift.
    """
    try:
        dt = dataT[1] - dataT[0]
        end_index = int(end_index / dt)
        diff_I = np.diff(dataI)
        upwardinfl = np.argmax(diff_I)
        t1 = dataT[upwardinfl:end_index] - dataT[upwardinfl]
        curve = curve_fit(exp_grow, t1, dataV[upwardinfl:end_index], maxfev=50000, bounds=([-np.inf, -np.inf, alpha-0.05], [np.inf, np.inf, alpha+0.05]), xtol=None)[0]
        
        _, deriv_ar = deriv(t1, exp_grow(t1, *curve))
        diff = np.abs(deriv_ar - 2)
        minpoint = np.argmin(diff)
        return np.hstack((curve, (t1[0]-t1[minpoint])))
    except:
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan])


def deriv(x,y):
    """Compute the numerical derivative of y with respect to x.

    Parameters
    ----------
    x : array-like
        Independent variable.
    y : array-like
        Dependent variable.

    Returns
    -------
    xfirst : array-like
        Midpoints of x where the derivative is computed.
    yfirst : array-like
        Numerical derivative dy/dx at the midpoints.
    """
    dy=np.diff(y,1)
    dx=np.diff(x,1)
    yfirst=dy/dx
    xfirst=0.5*(x[:-1]+x[1:])
    return xfirst, yfirst

@utils.debug_wrap
def subthres_allen(dataT, dataV, dataI, lowerlim, upperlim) -> tuple[float, float, tuple[float, float]]:
    """Analyze the subthreshold features of the voltage response using allen institute's method.

    Args:
        dataT (array-like): Time points of the recording.
        dataV (array-like): Voltage response data.
        dataI (array-like): Injected current data.
        lowerlim (float): Lower limit for the fitting window.
        upperlim (float): Upper limit for the fitting window.

    Returns:
        tuple[float, float, tuple[float, float]]: Sag ratio, time constant, and voltage deflection.
    """
    if dataI[np.argmin(dataI)] < 0: #if the current is negative check, this is a hyperpolarizing current injection, we can compute the sag and the time constant of the decay
        if lowerlim < 0.1:
            b_lowerlim = 0.1
        else:
            b_lowerlim = lowerlim

        #get only the hyperpor segments
        dwninf, upinf = find_hyperpolarization_segment(dataT, dataI, lowerlim, upperlim)
        lowerlim_t = np.clip(dataT[dwninf]  - 0.1, 0, 1e9) #handles floating point precision issues
        upperlim_t = dataT[upinf]

        #temp_spike_df['baseline voltage' + real_sweep_number] = subt.baseline_voltage(dataT, dataV, start=b_lowerlim)
        sag = subt.sag(dataT,dataV,dataI, start=lowerlim_t, end=upperlim_t)
        taum = subt.time_constant(dataT,dataV,dataI, start=lowerlim_t, end=upperlim_t)
        
        voltage_deflection = subt.voltage_deflection(dataT,dataV,dataI, start=lowerlim_t, end=upperlim_t)
        return sag, taum, voltage_deflection
                        