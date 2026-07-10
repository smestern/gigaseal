import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from numpy.char import zfill
import pyabf
from scipy import interpolate
from scipy.optimize import curve_fit
import scipy.signal as signal
from .dataset import cellData

logger = logging.getLogger(__name__)

def build_running_bin(array, time, start, end, bin=20, time_units='s', kind='nearest'):
    """ Builds a running bin of the data. The data is binned into bins of size 'bin' and the mean of the data in each bin is calculated. 
    If there are any NaN values in the binned data, they are replaced with the mean of the data in that bin.
    If there are no NaN values, the binned data is returned as is.
    Parameters
    ----------
    array : np.ndarray
        The data to be binned.
    time : np.ndarray
        The time values corresponding to the data.
    start : float
        The start time of the binning.
    end : float
        The end time of the binning.
    bin : float, optional
        The size of the bins, by default 20
    time_units : str, optional
        The units of the time values, by default 's'
    kind : str, optional
        The kind of interpolation to use for filling NaN values, by default 'nearest'

    returns
    -------
    binned_ : np.ndarray
        The binned data.
    time_bins : np.ndarray
        The time values corresponding to the binned data.
    """
    if time_units == 's':
        start = start * 1000
        end = end* 1000
        time = time*1000
    time_bins = np.arange(start, end+bin, bin)
    binned_ = np.full(time_bins.shape[0], np.nan, dtype=np.float64)
    index_ = np.digitize(time, time_bins)
    uni_index_ = np.unique(index_)
    for time_ind in uni_index_:
        data = np.asarray(array[index_==time_ind])
        data = np.nanmean(data)
        binned_[time_ind] = data
    nans = np.isnan(binned_)
    if np.any(nans):
        if time.shape[0] > 1:
            f = interpolate.interp1d(time, array, kind=kind, fill_value="extrapolate")
            new_data = f(time_bins)
            binned_[nans] = new_data[nans]
        else:
            binned_[nans] = np.nanmean(array)
    return binned_, time_bins

def crop_spikes(dataT, dataV, dataI, dv_cutoff=20.0, thresh_frac=0.2, pad=500):
    """Mask out action potentials from a voltage trace.

    Shared replacement for the duplicated ``crop_ap`` helpers in the legacy
    ``run_rmp.py`` / ``run_QC.py`` scripts. Detects spikes with the ipfx
    feature extractor and returns a copy of ``dataV`` with the samples
    spanning each spike (threshold − ``pad`` to trough + ``pad``) set to
    ``np.nan`` so downstream ``np.nan*`` statistics ignore them.

    Parameters
    ----------
    dataT, dataV, dataI : np.ndarray
        1-D time, voltage, and current arrays for a single sweep.
    dv_cutoff : float
        Minimum dV/dt (mV/ms) for ipfx spike detection.
    thresh_frac : float
        Fraction of spike height used for the threshold.
    pad : int
        Number of samples to crop before threshold and after trough.

    Returns
    -------
    np.ndarray
        Copy of ``dataV`` with spike regions replaced by ``np.nan``. If no
        spikes are detected the array is returned unmodified.
    """
    # TODO(human): port spike-cropping logic from the legacy crop_ap()
    # functions in gigaseal/bin/run_rmp.py and run_QC.py. Use
    # ipfx.feature_extractor.SpikeFeatureExtractor (import lazily), guard the
    # empty-spike case, clamp indices to the trace length, and replace the
    # deprecated ``np.int`` casts with ``int`` / ``np.int64``.
    raise NotImplementedError(
        "crop_spikes() body pending human authoring — consolidate the "
        "crop_ap() helpers from run_rmp.py and run_QC.py."
    )


def create_dir(fp):
    if os.path.exists(fp):
        pass
    else:
        os.makedirs(fp)
    return fp


def df_select_by_col(df, string_to_find):
    columns = df.columns.values
    out = []
    for col in columns:
        string_found = [x in col for x in string_to_find]
        if np.any(string_found):
            out.append(col)
    return df[out]

def time_to_idx(dataX, time):
    if dataX.nDim > 1:
        dataX = dataX[0, :]
    
    idx = np.argmin(np.abs(time-dataX))
    return idx

def idx_to_time(dataX, idx):
    if dataX.nDim > 1:
        dataX = dataX[0, :]
    return dataX[idx] #probably?

def find_stim_changes(dataI):
    diff_I = np.diff(dataI)
    infl = np.nonzero(diff_I)[0]
    return infl

def find_downward(dataI):
    diff_I = np.diff(dataI)
    downwardinfl = np.nonzero(np.where(diff_I<0, diff_I, 0))[0][0]
    return downwardinfl

def find_non_zero_range(dataT, dataI):
    non_zero_points = np.nonzero(dataI)[0]
    if len(non_zero_points) == 0:
        return (0, 0)
    return (dataT[non_zero_points[0]], dataT[non_zero_points[-1]])

def filter_bessel(data_V, fs, cutoff):
    """ Internal bessel filter function. This function is used to filter the data using a bessel filter.
      The cutoff frequency is set to 5 kHz by default. If the cutoff frequency is lower than the critical frequency, the data is filtered. Otherwise, the data is returned unfiltered.
    Parameters
    ----------
    data_V : np.ndarray
        The data to be filtered.
    fs : float
        The sampling frequency of the data.
    cutoff : float
        The cutoff frequency of the filter.
    """
    #filter the abf with 5 khz lowpass
    #if the cutoff is lower than critical frequency, filter the data
    try:
        b, a = signal.bessel(4, cutoff, 'low', norm='phase', fs=fs)
        dataV = signal.filtfilt(b, a, data_V)
    except:
        dataV = data_V
    return dataV

def parse_user_input(x=None, y=None, c=None, file=None):
    """ Try to parse the user input and return the parsed values. The user may pass in a single sweep, a list of sweeps, or a range of sweeps. or a file containing the sweeps. 
    The function will return the parsed input as a cellData object.
    Parameters
    ----------
    x : np.ndarray, optional
        The x values of the sweeps, by default None
    y : np.ndarray, optional
        The y values of the sweeps, by default None
    c : np.ndarray, optional
        The c values of the sweeps, by default None
    file : str, optional
        The file containing the sweeps, by default None
    
    Returns
    -------
    cellData
        The parsed input as a cellData object.
    """
    #check if any of the inputs are not None
    for val in [x, y, c, file]:
        if val is not None:
            if isinstance(val, cellData):
                return val
            
    if file is not None:
        #if file is a cellData object, return it
        if isinstance(file, cellData):
            return file
        logger.info(f"Loading data from file {file}")
        data = cellData(file)
        return data
    elif x is not None and y is not None and c is not None:
        #try to figure out if its a single sweep, a list of sweeps, or a range of sweeps
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and isinstance(c, np.ndarray):
            #check if its a single sweep
            if x.ndim == 1 and y.ndim == 1 and c.ndim == 1:
                logger.info("User passed in a single sweep")
                data = cellData(dataX=x.reshape(1, -1), dataY=y.reshape(1, -1), dataC=c.reshape(1, -1))
            else:
                logger.info("User passed in ndarray")
                data = cellData(dataX=x, dataY=y, dataC=c)
            return data
        elif isinstance(x, list) and isinstance(y, list) and isinstance(c, list):
            logger.info("User passed in a list of sweeps")
            data = cellData(dataX=x, dataY=y, dataC=c)
            return data
        else:
            raise ValueError("No valid input was passed to the function. Please pass in a file or the dataX, dataY, and dataC arrays")
    else:
        raise ValueError("No valid input was passed to the function. Please pass in a file or the dataX, dataY, and dataC arrays")

def sweepNumber_to_real_sweep_number(sweepNumber):
    """Convert a sweep number to a real sweep number. Internally the sweep number is zero indexed, but the real sweep number is one indexed. 
    This function converts the zero indexed sweep number to a one indexed sweep number. For users used to Clampex conventions
    The real sweep number is the sweep number + 1, and is zero padded to 3 digits.
    For example, sweep number 0 will be converted to 001, sweep number 1 will be converted to 002, etc.
    Parameters
    ----------
    sweepNumber : int
        The zero indexed sweep number.
    Returns
    -------
    str
        The one indexed sweep number, zero padded to 3 digits.
    """
    return str(zfill(str(sweepNumber + 1), 3))

####
# Some legacy functions that should be removed or replaced with the above functions. These are here for legacy reasons and should not be used in new code.


def plotabf(abf, spiketimes, lowerlim, upperlim, sweep_plots):
   """
   Very legacy function to plot sweeps from an abf file. 
   This function is used to plot the sweeps from an abf file. 
   The user can specify which sweeps to plot, and the time range to plot. The function will save the plot as a png file in the current working directory.
   Probably should not be used by anyone, but is here for legacy reasons.
   Parameters
    ----------
    abf : pyabf.ABF
          The abf file to plot.
   
   """
   try:
    if sweep_plots[0] == -1:
        pass
    else:
        plt.figure(num=2, figsize=(16,6))
        plt.clf()
        cm = plt.get_cmap("Set1") #Changes colour based on sweep number
        if sweep_plots[0] == 0:
            sweepList = abf.sweepList
        else:
            sweepList = sweep_plots - 1
        colors = [cm(x/np.asarray(sweepList).shape[0]) for x,_ in enumerate(sweepList)]
        
        plt.autoscale(True)
        plt.grid(alpha=0)

        plt.xlabel(abf.sweepLabelX)
        plt.ylabel(abf.sweepLabelY)
        plt.title(abf.abfID)

        for c, sweepNumber in enumerate(sweepList):
            abf.setSweep(sweepNumber)
            
            spike_in_sweep = (spiketimes[spiketimes[:,1]==int(sweepNumber+1)])[:,0]
            i1, i2 = int(abf.dataRate * lowerlim), int(abf.dataRate * upperlim) # plot part of the sweep
            dataX = abf.sweepX
            dataY = abf.sweepY
            colour = colors[c]
            sweepname = 'Sweep ' + str(sweepNumber)
            plt.plot(dataX, dataY, color=colour, alpha=1, lw=1, label=sweepname)
            
            plt.scatter(dataX[spike_in_sweep[:]], dataY[spike_in_sweep[:]], color=colour, marker='x')
           
        

        plt.xlim(abf.sweepX[i1], abf.sweepX[i2])
        plt.legend()
        
        plt.savefig(abf.abfID +'.png', dpi=600)
        plt.pause(0.05)
   except:
        print('plot failed')

def load_protocols(path):
    """Load all protocols from abf files in a given directory.
    Parameters
    ----------
    path : str
        The path to the directory containing the abf files.
    Returns
    -------
        np.ndarray
            Array of unique protocols found in the directory.
    """
    protocol = []
    for root,dir,fileList in os.walk(path):
        for filename in fileList:
            if filename.endswith(".abf"):
                try:
                    file_path = os.path.join(root,filename)
                    abf = pyabf.ABF(file_path, loadData=False)
                    protocol = np.hstack((protocol, abf.protocol))
                except:
                    print('error processing file ' + file_path)
    return np.unique(protocol)