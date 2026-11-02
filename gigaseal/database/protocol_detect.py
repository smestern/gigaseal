"""Stimulus and sweep primitives for heterogeneous current clamp recordings.

Noise-robust step detection, test pulse handling, volt/amp unit scaling, splitting of
back-to-back step families, protocol matching, and sweep QC. Importing this module also
patches the matching ipfx functions (get_stim_characteristics, find_time_index,
voltage_deflection, _step_stim_amp) with the versions here.
"""
import copy
import logging

import numpy as np
import scipy.stats
import scipy.ndimage

import ipfx
import ipfx.json_utilities as ju
from ipfx.stimulus import StimulusOntology
import ipfx.stim_features
import ipfx.subthresh_features
import ipfx.time_series_utils as tsu

from gigaseal.loadFile.loadNWB import loadNWB, GLOBAL_STIM_NAMES
try:
    from gigaseal.dev import stim_classifier as sc
except:
    print("Could not import stim_classifier")

# ==== GLOBALS =====
_ONTOLOGY = ju.read(StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE)
_UNIT_ONTOLOGY = {'amp': ['amp', 'ampere', 'amps', 'amperes', 'A'],
                  'volt': ['volt', 'v', 'volts', 'V'], 
                  'sec': ['sec', 's', 'second', 'seconds', 'secs', 'sec']}
log = logging.getLogger(__name__)




#======== IPFX functions ===========
#clone the function from ipfx/stimulus_protocol_analysis.py
# here we will modify it to handle test pulses intelligently, then overwrite the function in ipfx for this session

def find_step_transitions(i, noise_k=8.0, abs_floor_frac=0.005, min_gap=5):
    """Find square-step edges in a (possibly noisy) command waveform.

    The naive ``np.flatnonzero(np.diff(i))`` treats every sample that differs
    from its neighbour as an edge, so command currents that carry line noise
    produce an edge on essentially every sample and the real steps are lost.
    Instead we threshold the first difference against a robust estimate of the
    noise floor (MAD of the diff) and collapse each ramped transition -- which
    spans several samples -- into a single representative index.

    Returns a tuple ``(transition_indices, threshold)``.
    """
    i = np.asarray(i, dtype=float)
    di = np.diff(i)
    if di.size == 0:
        return np.array([], dtype=int), 0.0
    # robust noise estimate from successive differences (MAD -> sigma)
    med = np.median(di)
    sigma = 1.4826 * np.median(np.abs(di - med))
    # absolute floor keeps the threshold > 0 on perfectly clean signals
    rng = float(np.nanmax(i) - np.nanmin(i))
    abs_floor = max(abs_floor_frac * rng, 1e-9)
    thr = max(noise_k * sigma, abs_floor)
    idx = np.flatnonzero(np.abs(di) > thr)
    if idx.size == 0:
        return idx, thr
    # collapse runs of consecutive indices (a single ramped edge) into one
    splits = np.flatnonzero(np.diff(idx) > min_gap) + 1
    groups = np.split(idx, splits)
    collapsed = np.array([g[0] for g in groups], dtype=int)
    return collapsed, thr


def _holding_current(i, start_idx):
    """Holding (bias) current: median of the command before the stimulus starts.
    The median ignores a short leading test pulse. Returns 0 if there is no pre-stimulus data."""
    if start_idx is None or start_idx < 1:
        return 0.0
    return float(np.median(i[:start_idx]))


def _longest_stim_epoch(i, di_idx, thr):
    """Split a command waveform at its step edges and return (start_idx, end_idx) of the
    longest excursion away from the pre-stimulus baseline, or None if none returns to baseline.

    Consecutive off-baseline segments (e.g. a staircase) are merged into a single excursion.
    Indices follow the get_stim_characteristics convention: start_idx is the first sample of
    the excursion and end_idx is the diff index of the edge that returns to baseline.
    """
    bounds = np.concatenate(([0], np.asarray(di_idx) + 1, [len(i)]))
    levels = np.array([np.median(i[a:b]) for a, b in zip(bounds[:-1], bounds[1:])])
    off = np.abs(levels - levels[0]) > max(thr, 1e-9)

    best = None
    seg = 1
    while seg < len(levels):
        if not off[seg]:
            seg += 1
            continue
        first = seg
        while seg < len(levels) and off[seg]:
            seg += 1
        if seg == len(levels): # never returns to baseline, sweep cut off
            break
        start_idx, end_idx = bounds[first], bounds[seg] - 1
        if best is None or (end_idx - start_idx) > (best[1] - best[0]):
            best = (start_idx, end_idx)
    return best


def get_stim_characteristics(i, t, test_pulse=True, start_epoch=None, end_epoch=None, test_pulse_length=0.250):
    """
    Identify the start time, duration, amplitude, start index, and end index of a general stimulus.
    """
    # noise-robust edge detection instead of flagging every nonzero diff
    di_idx, thr = find_step_transitions(i)

    if len(di_idx) == 0:    # if no stimulus is found
        return None, None, 0.0, None, None

    start_idx = di_idx[0] + 1   # shift by one to compensate for diff()
    end_idx = di_idx[-1]

    #here we will check for test pulses and skip them if present.
    #test pulses can come before OR after the main stimulus (e.g. dandiset 001261 has a
    #trailing -20pA test pulse), so we split the command into excursions away from baseline
    #and keep the longest one as the stimulus
    if test_pulse and len(di_idx) > 3: # if there are more than 3 up/down transitions, there is probably a test pulse
        epoch = _longest_stim_epoch(i, di_idx, thr)
        if epoch is not None:
            start_idx, end_idx = epoch
    if start_idx >= end_idx: # sweep has been cut off before stimulus end
        return None, None, 0.0, None, None

    start_time = float(t[start_idx])
    duration = float(t[end_idx] - t[start_idx-1])

    #measure amplitude relative to the holding current, not raw command
    stim = i[start_idx:end_idx+1] - _holding_current(i, start_idx)

    peak_high = max(stim)
    peak_low = min(stim)

    if abs(peak_high) > abs(peak_low):
        amplitude = float(peak_high)
    else:
        amplitude = float(peak_low)

    return start_time, duration, amplitude, start_idx, end_idx





ipfx.stim_features.get_stim_characteristics = get_stim_characteristics


def _group_long_square_runs(start_times, end_times, time_tol=0.005):
    """Group long square sweeps by step duration: sweeps whose durations match (within time_tol, s)
    belong to the same protocol, even if the step onset differs (some labs run the hyperpolarizing
    and depolarizing steps as separate protocols with different onsets). Returns lists of indexes."""
    durations = np.asarray(end_times) - np.asarray(start_times)
    groups = []
    for k, d in enumerate(durations):
        for g in groups:
            if abs(d - durations[g[0]]) <= time_tol:
                g.append(k)
                break
        else:
            groups.append([k])
    return groups


def _plausible_vm(v):
    """True if v (mV) looks like a membrane potential trace: its floor (5th percentile, the resting/AHP
    level, robust to sweeps that sit depolarized for most of the step) is physiological, it is bounded,
    and not flat. Exact zeros are ignored, they are padding from aborted sweeps (QC handles those)."""
    v = v[v != 0]
    if v.size == 0:
        return False
    floor = np.percentile(v, 5)
    return -150 < floor < -10 and np.max(v) < 150 and np.min(v) > -250 and np.ptp(v) > 0.5


def _choose_volt_scale(v, conversion=None):
    """Scale a response logged as volts to mV, picking how to apply the NWB conversion by plausibility.
    The conversion attr is inconsistent across dandisets: sometimes a multiplier (per the NWB spec),
    sometimes a divisor, sometimes bogus. loadNWB has already multiplied by it, so we try each
    interpretation (with V, mV and uV units) and keep the first that looks like a membrane potential.
    Returns (v_mV, description) or (None, reason)."""
    try:
        c = float(conversion)
    except (TypeError, ValueError):
        c = 1.0
    if not np.isfinite(c) or c == 0:
        c = 1.0
    candidates = [("conversion as multiplier", 0)]
    if c != 1.0:
        candidates += [("conversion as divisor", -2), ("conversion ignored", -1)]
    for name, power in candidates:
        base = v * c**power
        for unit_factor in (1e3, 1.0, 1e-3):
            cand = base * unit_factor
            if _plausible_vm(cand[::10]): #subsample, stats only
                return cand, f"{name}, x{unit_factor:g}"
    return None, "implausible voltage scale"


def split_concatenated_steps(t, i, min_step=0.25, min_steps=3, min_total=5.0, dur_tol=0.005):
    """Detect a long recording that holds a whole step family played back-to-back (e.g. dandiset
    001454 records each cell as one ~70s sweep with a 0.5s step every 2s), and return the
    (start_idx, end_idx) window of each step so it can be analyzed as its own sweep. Each window
    holds the same baseline before the step onset. Returns None if the trace is not such a recording.
    i should be in pA."""
    t = np.asarray(t, dtype=float)
    i = np.asarray(i, dtype=float)
    if len(t) < 2 or (t[-1] - t[0]) < min_total:
        return None
    dt = t[1] - t[0]
    #median filter knocks out line noise and spikes in the command, ~5ms window
    i_f = scipy.ndimage.median_filter(i, size=max(int(round(0.005 / dt)), 1) | 1)
    #holding current: median of the baseline, refined once excluding the steps
    hold = np.median(i_f)
    off = None
    for _ in range(2):
        base = i_f if off is None else i_f[~off]
        thr = max(5.0, 5 * 1.4826 * np.median(np.abs(base - hold)))
        off = np.abs(i_f - hold) > thr
        if off.all():
            return None
        hold = np.median(i_f[~off])
    off = np.abs(i_f - hold) > thr

    bounds = np.concatenate(([0], np.flatnonzero(np.diff(off.astype(np.int8))) + 1, [len(i)]))
    #step excursions that start after and return to baseline
    runs = np.array([(a, b) for a, b in zip(bounds[:-1], bounds[1:])
                     if off[a] and a > 0 and b < len(i) and (b - a) * dt > min_step])
    if len(runs) < min_steps:
        return None
    #keep the steps sharing the most common duration
    durs = (runs[:, 1] - runs[:, 0]) * dt
    runs = runs[np.abs(durs - np.median(durs)) <= dur_tol]
    if len(runs) < min_steps:
        return None
    onsets, ends = runs[:, 0], runs[:, 1]
    pre = min(onsets[0], np.min(onsets[1:] - ends[:-1]) // 2)
    width = np.min(np.diff(onsets))
    return [(o - pre, min(o - pre + width, len(i))) for o in onsets]


def _iter_current_clamp_traces(data_set):
    """Yield (sweep_id, t, v, i, skip_reason) for each sweep, with v in mV and i in pA, as ipfx wants.
    For skipped sweeps t, v, i are None and skip_reason says why. Recordings holding a whole step family
    back-to-back are split into one trace per step, with sweep_id = sweep*1000 + step index."""
    for sweep in np.arange(len(data_set.dataY)):
        meta = data_set.sweepMetadata[sweep]
        stim_unit = match_unit(meta['stim_dict']["unit"])
        #if its not current clamp
        if stim_unit != "amp":
            yield sweep, None, None, None, f"not current clamp, found units of {stim_unit}"
            continue
        i = np.nan_to_num(data_set.dataC[sweep]*1)
        t = data_set.dataX[sweep]
        v = np.nan_to_num(data_set.dataY[sweep])
        #if the sweep v is in volts, convert to mV
        if match_unit(meta['resp_dict']["unit"]) == "volt":
            v, how = _choose_volt_scale(v, meta['resp_dict'].get("conversion"))
            if v is None:
                yield sweep, None, None, None, how
                continue
            log.debug(f"sweep {sweep} voltage scaled with {how}")
        #if the sweep i is in amps, convert to pA
        if np.max(i) < 0.1 and np.min(i) > -0.1:
            i = i*1000000000000
        windows = split_concatenated_steps(t, i)
        if windows is None:
            yield sweep, t, v, i, None
            continue
        log.debug(f"sweep {sweep} split into {len(windows)} concatenated steps")
        for k, (a, b) in enumerate(windows):
            yield sweep*1000 + k, t[a:b] - t[a], v[a:b], i[a:b], None


def find_time_index(t, t_0):#
    """ Find the index value of a given time (t_0) in a time series (t).


    Parameters
    ----------
    t   : time array
    t_0 : time point to find an index

    Returns
    -------
    idx: index of t closest to t_0
    """
    if not (t[0] <= t_0 <= t[-1]):
        f"Given time ({t_0}) is outside of time range ({t[0]}, {t[-1]})"
        
    if t_0 < t[0]:
        t_0 = t[0]
    if t_0 > t[-1]:
        t_0 = t[-1]
    
    idx = np.argmin(abs(t - t_0))
    return idx


def voltage_deflection(t, v, i, start, end, deflect_type=None):
    """Measure deflection (min or max, between start and end if specified).

    Parameters
    ----------
    deflect_type : measure minimal ('min') or maximal ('max') voltage deflection
        If not specified, it will check to see if the current (i) is positive or negative
        between start and end, then choose 'max' or 'min', respectively
        If the current is not defined, it will default to 'min'.

    Returns
    -------
    deflect_v : peak
    deflect_index : index of peak deflection
    """

    deflect_dispatch = {
        "min": np.argmin,
        "max": np.argmax,
    }

    start_index = tsu.find_time_index(t, start)
    end_index = tsu.find_time_index(t, end)

    if deflect_type is None:
        if i is not None:
            halfway_index = tsu.find_time_index(t, (end - start) / 2. + start)
            if i[halfway_index] >= 0:
                deflect_type = "max"
            else:
                deflect_type = "min"
        else:
            deflect_type = "min"

    deflect_func = deflect_dispatch[deflect_type]

    #if the start_and end times are the same, just return the voltage at that time
    if start_index == end_index:
        #also ensure 
        return v[start_index], start_index

    v_window = v[start_index:end_index]
    deflect_index = deflect_func(v_window) + start_index

    return v[deflect_index], deflect_index


def _step_stim_amp(t, i, start):
    #stim amp is relative to the holding current, so cells with a bias current get the true step size
    t_index = tsu.find_time_index(t, start)
    holding = _holding_current(i, t_index)
    if t_index + 1 >= len(i):
        return i[t_index] - holding
    #take the median over the step (up to the next edge) rather than a single sample,
    #recorded/filtered commands have ramped edges that are still settling just after onset
    di_idx, _ = find_step_transitions(i)
    later = di_idx[di_idx > t_index + 1]
    end_index = later[0] + 1 if len(later) else len(i)
    return float(np.median(i[t_index + 1:end_index])) - holding


# monkey patch time
tsu.find_time_index = find_time_index


ipfx.subthresh_features.voltage_deflection = voltage_deflection


ipfx.stim_features._step_stim_amp = _step_stim_amp


#stimulus protocol analysis functions, here we will guess what stimulus protocol was used, and affix that to the stimulus ontology later
def get_stimulus_protocols(files, ext="nwb", method='random'):
    #this function is going to take a folder and a file extension, and return a list of stimulus protocols, then guess what type of stimulus protocol was used
    #method can be random, first, or all
    #random will choose 10 random files and try to guess the stimulus protocol
    if method == 'random':
        files = np.random.choice(files, min(100, len(files)))
    elif method == 'first':
        files = files[0]
    elif method == 'all':
        pass
    #here we are gonna set the GLOBAL_STIM_NAMES filter to blank, so that we can get all the stimulus names
    GLOBAL_STIM_NAMES.stim_inc = ['']
    GLOBAL_STIM_NAMES.stim_exc = []
    classifier = sc.stimClassifier()
    stim_to_use = []
    for i, f in enumerate(files):
        _, _, _, _, data_set = loadNWB(f, return_obj=True)
        #
        #[plt.plot(x) for x in data_set.dataY]
        #plt.show()
        for j in np.arange(len(data_set.dataY)):
            sweep_meta = data_set.sweepMetadata[j]
            i = data_set.dataC[j]
            t = data_set.dataX[j]
            #stim_protocol = match_protocol(i, t) #stimulus protocol is the matching protocol 
            stim_protocol = classifier.predict(i)
            #reference mapped to the allen protocol names
            if stim_protocol is not None:
                #add stim_protocol to ontology
                stim_name_1 = sweep_meta['description']
                stim_name_2 = sweep_meta['stimulus_description']
                for stim_name in [stim_name_1, stim_name_2]:
                    if stim_name not in GLOBAL_STIM_NAMES.stim_inc:
                        if stim_name != '' and stim_name != 'N//A' and stim_name != 'NA' and stim_name != 'N/A':
                            stim_to_use.append(stim_name)
                
    GLOBAL_STIM_NAMES.stim_inc = stim_to_use   
    return copy.deepcopy(GLOBAL_STIM_NAMES)


def match_protocol(i, t, test_pulse=True, start_epoch=None, end_epoch=None, test_pulse_length=0.25):
    #this function will take a stimulus and return the stimulus protocol at least it will try
    #first we will try to match the stimulus protocol to a known protocol
    #classifier = sc.stimClassifier()
    try:
        start_time, duration, amplitude, start_idx, end_idx = get_stim_characteristics(i, t, test_pulse=test_pulse, start_epoch=start_epoch, end_epoch=end_epoch, test_pulse_length=test_pulse_length)
    except:
        return None
    #pred = classifier.decode(classifier.predict(i.reshape(1, -1)))[0]
    #if pred=="long_square":
     #   return "Long Square"
    if start_time is None:
        #if we can't find the start time, then we can't identify the stimulus protocol
        return None
    if duration > 0.25:
        #if the stimulus is longer than 500ms, then it is probably a long square
        return match_long_square_protocol(i, t, start_idx, end_idx)
    elif duration < 0.1:
        #if the stimulus is less than 100ms, then it is probably a short square
        return match_short_square_protocol(i, t)
    else:
        #check if ramp
        return match_ramp_protocol(i, t)

    

def match_long_square_protocol(i, t, start_idx, end_idx):
    #here we will do some analysis to determine if the stimulus is a long square, and if so, what the parameters are
    i = np.asarray(i, dtype=float)

    # noise-robust edge detection instead of flagging every nonzero diff
    di_idx, thr = find_step_transitions(i)

    if len(di_idx) < 2:
        #zero or one transition -> not a (returning) square step
        return None

    # A long square holds a flat step and then returns to its pre-step baseline.
    # We compare robust (median) levels rather than requiring the command to
    # start/end at exactly 0, because real recordings carry a holding current
    # and line noise means no sample is ever exactly the baseline value.
    pre = i[:start_idx]
    post = i[end_idx + 1:]
    step = i[start_idx:end_idx + 1]
    if len(pre) == 0 or len(post) == 0 or len(step) == 0:
        return None
    base_pre = np.median(pre)
    base_post = np.median(post)
    step_level = np.median(step)
    amp = step_level - base_pre
    noise = max(thr, 1e-9)
    if abs(amp) < noise:
        #no real step relative to baseline
        return None

    # Reject ramps: a square step is flat, so a straight-line fit across the
    # step region must not drift by a meaningful fraction of the step height.
    # This guard is unconditional so a ramp can never be flagged as a square,
    # no matter how many noise crossings its edge detection produced.
    x_step = t[start_idx:end_idx + 1]
    if len(x_step) > 2:
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x_step, step)
        if abs(slope * (x_step[-1] - x_step[0])) > 0.3 * abs(amp):
            #significant drift across the step -> ramp-like, not a square
            return None

    # The command must return to (near) its pre-step baseline after the step.
    if abs(base_post - base_pre) > 0.5 * abs(amp):
        return None

    return "Long Square"


def match_short_square_protocol(stimulus_protocol, ontology):
    #TODO: implement this function
    pass


def match_ramp_protocol(stimulus_protocol, ontology):
    #TODO: implement this function
    pass


def match_unit(unit, ontology=_UNIT_ONTOLOGY):
    #unit should be a string, if its bytes or something else, convert it to a string
    if isinstance(unit, bytes) or isinstance(unit, np.bytes_):
        unit = unit.decode('utf-8')
    elif isinstance(unit, str) != True:
        unit = str(unit)
    #this function will take a unit and return the unit ontology
    for unit_name in ontology:
        check = [unit.upper() in x.upper() for x in ontology[unit_name]]
        if np.any(check):
            return unit_name
    return None



def QC_voltage_data(t,v,i, zero_threshold=0.2, noise_threshold=10):
    #this function will take a voltage trace and return a QC score
    #Sometimes the voltage trace is not a voltage trace, but rather a current trace
    #or with IGOR / MIES generated NWB files, the sweep was aborted halfway through, and there is a large jump in the voltage trace, and a bunhc of zeros
    #this function will check for these things and return a QC score
    #if the QC score is 0, then the sweep is bad
    #if the QC score is 1, then the sweep is good
    if v is None:
        return 0
    if i is None:
        return 0
    if len(v) == 0:
        return 0
    if np.any(v > 500) or np.any(v < -500): #membrane voltages are very very unlikely to be this large, this threshold could be lowered
        return 0
    

    #check for extended periods of 0
    if np.sum(v == 0) > zero_threshold*len(v): #if more than 10% of the trace is 0, then it was probably aborted
        #this is only a problem if the current is not 0
        #check if while the voltage is 0, the current is 0
        idx_zero = np.flatnonzero(np.isclose(v, 0))
        if np.sum(i[idx_zero] != 0) > (zero_threshold/2)*len(idx_zero):
            return 0
        else:
            return 1 
        

    #check for large jumps in the voltage trace
    #dv = np.diff(v)
    #if np.any(np.abs(dv) > 1e9):
        #return 0

    #todo, more qc checks
    return 1

iter_current_clamp_traces = _iter_current_clamp_traces
