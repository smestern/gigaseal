"""Long square analysis on heterogeneous current clamp files.

select_long_square_run picks the long square sweeps out of a loaded file, groups them into
protocol runs (same step duration) and returns the chosen run aligned on a common onset.
long_square_features runs ipfx's LongSquareAnalysis on such a run and flattens the result
into one row of cell features. Which run to analyze is a policy choice, so the ranking is
a parameter; the default just takes the largest run.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ipfx.bin import run_feature_collection
from ipfx.sweep import SweepSet, Sweep
import ipfx.stimulus_protocol_analysis as spa
import ipfx.data_set_features as dsf
import ipfx.feature_vectors as fv

from gigaseal.database.protocol_detect import (
    _iter_current_clamp_traces, _group_long_square_runs, match_protocol,
    get_stim_characteristics, QC_voltage_data,
)

log = logging.getLogger(__name__)


@dataclass
class LongSquareRun:
    """Result of select_long_square_run. Candidate lists are parallel and in file order;
    groups and primary hold indexes into them."""
    sweeps: SweepSet  # the chosen run, each sweep shifted onto the latest onset; empty if none
    start: Optional[float] = None
    end: Optional[float] = None
    candidates: list = field(default_factory=list)  # every long square Sweep that passed QC, unaligned
    start_times: list = field(default_factory=list)
    end_times: list = field(default_factory=list)
    amps: list = field(default_factory=list)
    volts_range: list = field(default_factory=list)  # [v.min(), v.max()] per candidate
    groups: list = field(default_factory=list)  # candidate indexes per protocol run
    primary: Optional[list] = None  # the group in sweeps
    log: dict = field(default_factory=dict)  # sweep id -> why it was skipped or kept
    traces: dict = field(default_factory=dict)  # sweep id -> (t, v) as analyzed, for QC plots

    @property
    def amps_used(self):
        return [] if self.primary is None else [self.amps[k] for k in self.primary]


def _largest_run(g, start_times, end_times, amps, volts):
    #default ranking: most sweeps, then earliest in the recording
    return (-len(g), g[0])


def select_long_square_run(data_set, rank=None, sweep_ids=None, time_tol=0.005):
    """Pick the long square run to analyze from a data_set loaded with loadNWB(..., return_obj=True).

    rank(group, start_times, end_times, amps, volts_range) returns a sort key for a group of
    candidate indexes, lowest wins; defaults to the largest run. If sweep_ids is given, only
    those sweeps are considered and they are taken as the run as-is, without regrouping
    (for re-deriving a run picked earlier). Returns a LongSquareRun."""
    rank = rank or _largest_run
    run = LongSquareRun(sweeps=SweepSet([]))
    for sweep, t, v, i, skip_reason in _iter_current_clamp_traces(data_set):
        if skip_reason is not None:
            log.debug(f"sweep {sweep} skipped: {skip_reason}")
            run.log[sweep] = skip_reason
            continue
        if sweep_ids is not None and sweep not in sweep_ids:
            continue
        run.traces[sweep] = (t, v)
        dt = t[1] - t[0]

        #try to figure out if this is a long square
        if match_protocol(i, t) != "Long Square":
            log.debug(f"skipping sweep {sweep} because it is not a long square")
            run.log[sweep] = "likely not a long square"
            continue

        start_time, duration, amplitude, start_idx, end_idx = get_stim_characteristics(i, t)

        if QC_voltage_data(t, v, i) == 0:
            log.debug(f"skipping sweep {sweep} because it failed QC")
            run.log[sweep] = "failed QC"
            continue

        #floating point errors can cause issues here, so round to the nearest 0.1 pA
        #since we are doing long square; if this was ramp or noise or something this is bad.
        i = np.round(i, decimals=1)
        run.log[sweep] = f"passed QC {start_time:.3f}-{start_time+duration:.3f}s"
        run.start_times.append(start_time)
        run.end_times.append(start_time+duration)
        run.candidates.append(Sweep(t, v, i, clamp_mode="CurrentClamp", sampling_rate=int(1/dt), sweep_number=sweep))
        run.amps.append(amplitude)
        run.volts_range.append([v.min(), v.max()])

    #split the passing sweeps into runs of the same protocol, then keep only the best run
    if sweep_ids is not None:
        run.groups = [list(range(len(run.candidates)))] if run.candidates else []
    else:
        run.groups = _group_long_square_runs(run.start_times, run.end_times, time_tol=time_tol)
    if not run.groups:
        log.error("No long square sweeps passed QC")
        return run

    starts, ends, cands = run.start_times, run.end_times, run.candidates
    run.primary = min(run.groups, key=lambda g: rank(g, starts, ends, run.amps, run.volts_range))
    #onsets can differ within a group, so shift each sweep's time axis to line up on the latest onset
    run.start = max(starts[k] for k in run.primary)
    run.end = run.start + float(np.median([ends[k] - starts[k] for k in run.primary]))
    for n, g in enumerate(run.groups):
        if g is not run.primary:
            for k in g:
                run.log[cands[k].sweep_number] += f", not used (group {n})"
    run.sweeps = SweepSet([Sweep(cands[k].t + (run.start - starts[k]), cands[k].v, cands[k].i,
                                 clamp_mode="CurrentClamp", sampling_rate=int(round(1 / (cands[k].t[1] - cands[k].t[0]))),
                                 sweep_number=cands[k].sweep_number) for k in run.primary])
    return run


def long_square_features(sweeps, start, end, amp_interval=20, max_above_rheo=None,
                         min_peak=-25, subthresh_min_amp=-200.0, max_filter_hz=10000):
    """ipfx LongSquareAnalysis on an aligned SweepSet with a common step window (start, end), in s.

    Returns (features, sweep_indexes): features is a flat dict of passive properties, rheobase,
    spike and sweep features per amp_interval step above rheobase (keyed by step number) and the
    FI fit; sweep_indexes are the suprathreshold sweeps used. Raises on failure (e.g. no spiking
    sweeps), so the caller decides how to record it."""
    result = {}
    lsq_spx, lsq_spfx = dsf.extractors_for_sweeps(
        sweeps,
        start=start,
        end=end,
        min_peak=min_peak,
    )

    #ipfx defaults to a filter @10kHz, this breaks for signals sampled at or below 20kHz, so cap it below Nyquist.
    #Do NOT put the cutoff right at Nyquist (we used to use 0.99999*Nyquist): the 4-pole bessel then has a pole
    #just outside the unit circle, filtfilt grows ~1.0001^N, and long sweeps give dv/dt up to ~1e23 mV/ms
    min_sample_rate = min(sweeps.sampling_rate)
    filter_cutoff = min(max_filter_hz, 0.8 * min_sample_rate / 2)
    if filter_cutoff < max_filter_hz:
        log.warning(f"Minimum sample rate is {min_sample_rate} Hz, lowering the dv/dt filter cutoff to {filter_cutoff:.0f} Hz.")

    lsq_spx.filter = filter_cutoff / 1000 #in kHz

    lsq_an = spa.LongSquareAnalysis(lsq_spx, lsq_spfx,
        subthresh_min_amp=subthresh_min_amp)
    if start < 0.01:
       lsq_an.sptx.baseline_interval = start*0.1
       lsq_an.sptx.sag_baseline_interval = start*0.1
    lsq_features = lsq_an.analyze(sweeps)

    result.update({
            "input_resistance": lsq_features["input_resistance"],
            "tau": lsq_features["tau"],
            "v_baseline": lsq_features["v_baseline"],
            "sag_nearest_minus_100": lsq_features["sag"],
            "sag_measured_at": lsq_features["vm_for_sag"],
            "rheobase_i": int(lsq_features["rheobase_i"]),
            "fi_linear_fit_slope": lsq_features["fi_fit_slope"],
        })
    # Identify suprathreshold set for analysis
    sweep_table = lsq_features["spiking_sweeps"]
    mask_supra = sweep_table["stim_amp"] >= lsq_features["rheobase_i"]
    sweep_indexes = fv._consolidated_long_square_indexes(sweep_table.loc[mask_supra, :])
    amps = np.rint(sweep_table.loc[sweep_indexes, "stim_amp"].values - lsq_features["rheobase_i"])
    spike_data = lsq_features["spikes_set"]

    # Assign each sweep to the nearest amp_interval step above rheobase, keeping the
    # closest sweep per step. Requiring an exact multiple of amp_interval left cells
    # recorded with other step sizes (e.g. 10, 25, 100 pA) with only the rheobase sweep.
    best_by_label = {}
    for amp, swp_ind in zip(amps, sweep_indexes):
        amp_label = int(np.floor(amp / amp_interval + 0.5))
        if (amp < 0) or (max_above_rheo is not None and amp_label * amp_interval > max_above_rheo):
            continue
        offset = abs(amp - amp_label * amp_interval)
        if amp_label not in best_by_label or offset < best_by_label[amp_label][0]:
            best_by_label[amp_label] = (offset, amp, swp_ind)

    sweep_feature_list = [
        "first_isi",
        "avg_rate",
        "isi_cv",
        "latency",
        "median_isi",
        "adapt",
    ]
    for amp_label, (_, amp, swp_ind) in sorted(best_by_label.items()):

        first_spike_lsq_sweep_features = run_feature_collection.first_spike_lsq(spike_data[swp_ind])
        result.update({"ap_1_{:s}_{:d}_long_square".format(f, amp_label): v
                            for f, v in first_spike_lsq_sweep_features.items()})

        mean_spike_lsq_sweep_features = run_feature_collection.mean_spike_lsq(spike_data[swp_ind])
        result.update({"ap_mean_{:s}_{:d}_long_square".format(f, amp_label): v
                            for f, v in mean_spike_lsq_sweep_features.items()})

        result.update({"{:s}_{:d}_long_square".format(f, amp_label): sweep_table.at[swp_ind, f]
                            for f in sweep_feature_list})
        result["stimulus_amplitude_{:d}_long_square".format(amp_label)] = int(amp + lsq_features["rheobase_i"])

    rates = sweep_table.loc[sweep_indexes, "avg_rate"].values
    result.update(run_feature_collection.fi_curve_fit(amps, rates))
    return result, sweep_indexes
