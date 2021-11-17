import os
import numpy as np

from collections import defaultdict
from matplotlib import pyplot as plt
from gwpy.timeseries import TimeSeries
from gwpy.time import tconvert
from tqdm.auto import tqdm

from ts_outlier_detection import *
from ts_outlier_detection.plotting import *


VMIN = 0
VMAX = 25.5


def _scale_width_asymptotic(width, base=4, maximum=100):
    return maximum * (1 - np.exp(-base * (width + 1) / maximum))


def _generate_windows(start, stop, level, offset, wrap=False):
    step = offset * level
    left = np.arange(start, stop, step)
    right = left + level
    overflow = right > stop
    if wrap:
        right[overflow] -= (stop - start)
    else:
        right[overflow] = stop
    return np.stack((left, right), axis=-1)


def _setup_main_plot(width, levels, height=12, level_height=1):
    total_height = height + levels*level_height
    fig = plt.figure(figsize=(width,total_height))
    gs = fig.add_gridspec(total_height, 1)
    qax = fig.add_subplot(gs[:height])
    axs = [fig.add_subplot(gs[height+i]) for i in range(levels)]
    return fig, qax, axs


def tof_detections_from_file(target, start, stop, **kwargs):
    '''
    Wrapper for tof_detections. Reads detector data from file on disk.
    :param str target: path to a data file to read
    :param float start: GPS time for start of time of interest
    :param float stop: GPS time for end of time of interest

    See tof_detections docstring for description of other parameters.
    '''
    rdata = TimeSeries.read(target)
    return tof_detections(rdata, start, stop, **kwargs)


def tof_detections_open_data(det, start, stop, padding=2, **kwargs):
    '''
    Wrapper for tof_detections. Pulls detector data from GWOSC.
    :param str det: String naming target detector (ex. H1, L1, V1)
    :param float start: GPS time for start of time of interest
    :param float stop: GPS time for end of time of interest
    :param float padding: (Optional) Number of seconds to pad to the beginning and end of time windows
                                     to reduce edge effects in q-transforms

    See tof_detections docstring for description of other parameters.
    '''
    rdata = TimeSeries.fetch_open_data(det, start-padding, stop+padding)
    return tof_detections(rdata, start, stop, **kwargs)


def tof_detections(
    rdata, start, stop,
    levels=[4, 30, 120], base_level=None, sample_rate=None, offset=1, bandpass=(20,500),
    plot_summary=True, plot_detection_windows=True, plot_all_windows=False, flatten_output=True,
    colors=['red', 'blue', 'grey'], desc='tof', q_kwargs={}, save_dir=None, **kwargs
):
    '''
    Plot and return detections made by TOF within a time frame in open data from a GW detector

    :param gwpy.TimeSeries rdata: Raw data to parse for anomalies
    :param float start: GPS time for start of time of interest
    :param float stop: GPS time for end of time of interest
    
    :param [float] levels: (Optional) List of window lengths to slide over data (default: 4, 30, 120)
    :param float base_level: (Optional) Window length to use as the sample rate standard.
                                        Windows of different lengths will be resampled to match the number of samples in base_level.
                                        If None is provided, the first window in levels is used as the base_level.
    :param float offset: (Optional) Number of window lengths to shift window for each sliding step (default: 1)
    :param (float, float) bandpass: (Optional) Bandpass to apply to detector data before TOF is run (default: (20, 500))

    :param bool plot_summary: (Optional) Plots a q-transform of the entire time of interest and TOF detections made at all window sizes
    :param bool plot_detection_windows: (Optional) Plots a q-transform and TOF plot of every window where a TOF detection was made
    :param bool plot_all_windows: (Optional) Plots a q-transform and TOF plot of every window
    :param bool flatten_output: (Optional) Whether or not to return a 1D array of detections at every level.
                                           If False, returns a dictionary of {level: [detection times]}
    :param [str] colors: (Optional) List of strings representing the colors corresponding to each window level.
                                    If len(colors) < len(levels), the colors will wrap back to the beginning.
    :param str desc: (Optional) Custom identifier for this particular pipeline run to use in plot and file names (default: 'tof')
    :param dict q_kwargs: (Optional) Keyword arguments for ts.q_transform
    :param str save_dir: (Optional) Directory to save generated plots (default: plots are not saved)

    Remaining parameters are passed to TemporalOutlierFactor constructor

    :return: Array of all times where TOF detected anomalies
    :rtype: numpy.ndarray (or dict if flatten_output is False)
    '''
    if save_dir and not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if base_level is None:
        base_level = levels[0]

    if plot_summary:
        width = round((stop - start) / base_level)
        scaled_width = _scale_width_asymptotic(width)
        scaled_base_level = base_level * width / scaled_width
        fig, qax, axs = _setup_main_plot(scaled_width, len(levels))
        qax.set_title(f'TOF detections on {desc} between {tconvert(start)} and {tconvert(stop)}')
        base_windows = _generate_windows(start, stop, scaled_base_level, 1)
        plot_high_resolution_qscan(
            qax, rdata if sample_rate is None else rdata.resample(sample_rate),
            tqdm(base_windows, desc=f'{desc} - Main plot'), q_kwargs=q_kwargs, vmin=VMIN, vmax=VMAX)
    
    level_windows = {level: _generate_windows(start, stop, level, offset, wrap=True) for level in levels}
    
    outlier_times = defaultdict(lambda: np.array([]))
    ctof = TemporalOutlierFactor(**kwargs)
    hdata = rdata.whiten().bandpass(*bandpass)
    
    for level, windows in level_windows.items():
        resampled_data = hdata.resample(
            base_level * (rdata.sample_rate if sample_rate is None else sample_rate) / level)
        for left, right in tqdm(windows, desc=f'{desc} - Windows of length {level}'):
            wdata = resampled_data[resampled_data.times.value >= left]
            if right > left:
                wdata = wdata[wdata.times.value < right]
                data = wdata.value
                times = wdata.times.value
            else:
                wdata = wdata[wdata.times.value < stop]
                data = np.concatenate((
                    wdata.value,
                    resampled_data[(resampled_data.times.value >= start) & (resampled_data.times.value < right)].value,
                ), axis=-1)
                times = np.concatenate((
                    wdata.times.value,
                    resampled_data[(resampled_data.times.value >= start) & (resampled_data.times.value < right)].times.value
                ), axis=-1)
            
            ctof.fit(data, times)
            outliers = times[ctof.get_outlier_indices()]
            outliers = outliers[(outliers >= start) & (outliers < stop)]
            outlier_times[level] = np.append(outlier_times[level], outliers)

            if plot_all_windows or (outliers.size > 0 and plot_detection_windows):
                fig1 = plt.figure(figsize=(9, 9), constrained_layout=True)
                gs = fig1.add_gridspec(4, 1)
                bax = fig1.add_subplot(gs[3])
                bax.set_ylabel('TOF')
                bax.set_xlabel('GPS Time (s)')
                tax = fig1.add_subplot(gs[2], sharex=bax)
                tax.set_ylabel('Strain')
                plot_ts_outliers(ctof, (tax, bax))
                oax = fig1.add_subplot(gs[0:2], sharex=bax)
                oax.set_ylabel('Frequency (Hz)')
                oax.set_title(f'TOF detections on {desc} between {tconvert(left)} and {tconvert(right)}')
                plot_qscan(oax, rdata, q_kwargs={'outseg': (left, right), **q_kwargs}, vmin=VMIN, vmax=VMAX)
                oax.set_yscale('log', base=2)
                if save_dir:
                    fig1.savefig(os.path.join(save_dir, f'{desc}_{left}_{right}.png'), bbox_inches='tight')
                plt.close(fig1)
    
    if plot_summary:
        for i, (level, outliers) in enumerate(outlier_times.items()):
            ax = axs[i]
            c = colors[i % len(colors)]
            ax.vlines(outliers, -1, 1, color=c, linewidth=3)
            ax.text(0.0, 0.1, f'{level}s window size', transform=ax.transAxes)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(start, stop)
        
        xticks = np.arange(start, stop, scaled_base_level)
        ax.set_xticks(xticks)
        ax.set_xlabel('GPS Time (s)')
        qax.set_xticks([])
        qax.set_xlim(start, stop)
        qax.set_yscale('log', base=2)
        qax.set_ylabel('Frequency (Hz)')
        qax.grid(False)
        fig.tight_layout(h_pad=0, w_pad=0)

        if save_dir:
            fig.savefig(os.path.join(save_dir, f'multi_window_{desc}_{left}_{right}.jpg'), bbox_inches='tight')

    return np.unique(np.concatenate(list(outlier_times.values()), axis=None)) if flatten_output else outlier_times


def stdev_loss(t0, detected_times, t1=None):
    '''
    Loss function to quantitatively determine the "goodness"
    of a TOF detection on a known event

    :param float t0: GPS start time of event
    :param numpy.ndarray detected_times: 1D array of TOF detection times
    :param float t1: (Optional) GPS end time of event

    :return: loss score (lower is better)
    :rtype: float
    '''
    if t1 is None:
        t1 = t0
    if detected_times.size < 2:
        return 0
    outside = (detected_times < t0) | (detected_times > t1)
    return np.mean(np.abs(detected_times[outside] - t0)) - detected_times.size / np.std(detected_times)
