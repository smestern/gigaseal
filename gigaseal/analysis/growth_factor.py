"""
Growth-factor analysis module (experimental).

Fits an exponential *growth* curve to the rising phase of a response to
estimate fast/slow growth time constants and curvature — used for
exploratory characterization of activation kinetics.

Migrated from the legacy ``gigaseal/bin/run_GROW_SPCA.py`` experimental
script (which contained Jupyter cell markers and prototype code). This
module is marked ``hidden = True`` so it is excluded from default GUI
pickers while it remains experimental. The framework plumbing is scaffolded
here; the ``analyze()`` body is left for a human to author/port.
"""

import logging

import numpy as np

from .core.base import AnalysisBase

logger = logging.getLogger(__name__)


class GrowthFactorAnalysis(AnalysisBase):
    """
    Fit exponential growth kinetics to the rising phase of a response.

    Reproduces ``exp_growth_factor`` from the legacy ``run_GROW_SPCA.py``
    script: locates the upward inflection, fits one- and two-phase
    exponential-growth curves, and reports fast/slow growth time constants
    plus the minimum of the smoothed derivative. Relatedly,
    :func:`gigaseal.patch_subthres.exp_growth_factor` implements a
    lab-reviewed variant that should be preferred where equivalent.

    Marked experimental (``hidden = True``); registered but excluded from
    default GUI module pickers.

    Parameters
    ----------
    end_index : int
        Sample index marking the end of the growth-fit window (legacy
        default 300).

    Output keys
    -----------
    ``growth_a``, ``growth_b_fast``, ``growth_tau_fast``,
    ``growth_b_slow``, ``growth_tau_slow``, ``min_derivative``.
    """

    name = "growth_factor"
    display_name = "Growth Factor (experimental)"
    sweep_mode = "per_file"
    hidden = True

    # Parameters — typed class attributes only
    end_index: int = 300

    def analyze(self, x, y, c, **kwargs) -> dict:
        """
        Fit growth kinetics for one file.

        Parameters
        ----------
        x, y, c : np.ndarray
            2-D ``(sweeps × samples)`` time, response, and command arrays.

        Returns
        -------
        dict
            Growth-fit features (see class docstring).
        """
        # TODO(human): port growth-factor fitting from
        # gigaseal/bin/run_GROW_SPCA.py::exp_growth_factor (or reuse
        # gigaseal.patch_subthres.exp_growth_factor where equivalent).
        #  * Find upward inflection via np.argmax(np.diff(c)).
        #  * curve_fit exp_grow / exp_grow_2p over the growth window.
        #  * Report fast/slow tau, coefficients, and min smoothed derivative.
        #  * Strip the legacy Jupyter cell markers and plotting side effects;
        #    import scipy/sklearn lazily inside this method.
        raise NotImplementedError(
            "GrowthFactorAnalysis.analyze() body pending human authoring — "
            "port from gigaseal/bin/run_GROW_SPCA.py::exp_growth_factor."
        )
