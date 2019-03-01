"""
Created on Sun Feb 24 019

Reads trials one-by-one
Determines signals that will be used in regression process
Calls "feature axtraction" and "regression" modules

@author: sinan
"""

## Import
from pathlib import Path
import glob
import os
import time
import matplotlib
import gc

import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import fftpack
from scipy import signal
from scipy import io
from scipy import stats
from itertools import compress
from sklearn.preprocessing import normalize

from my_functions.loading import *
from my_functions.processing import *
from my_functions.plotting import *

## Initial setup
maindir = Path.home() / 'Documents' / 'DATA' / 'CCI'
# maindir = Path.home() / 'Documents' / 'MATLAB' / 'ESMA'
subject = 'CCI5'
ses_names = ['112718', '120318', '120718', '121718', '121918']

force_threshold = 0.05
saturation_threshold = 1000

offsets_file_name = maindir / subject / 'baseline-offsets-CCI5.xlsx'
offsets_all = load_excel_file(offsets_file_name,cell_range='B2:F4')

## Read all trials
trials_file_name = maindir / subject / 'selected-trials.xlsx'
trials_all = load_excel_file(trials_file_name, sheet_name='Sheet', cell_range='B2:E54')

## Process
extra = 0.25  # Extra data length to cut (in seconds)
CHANS = []
RAW = []
EMG = []
trial_lengths = []
cut_times = []

for count, row in enumerate(trials_all):
    if count >10:
        break
    # Unpack trial's info
    session = int(row[0])
    trial = str(row[1])
    start_time = int(float(row[2]))
    end_time = int(float(row[3]))

    file_name = maindir / subject / ses_names[session] / trial
    video_name = maindir / subject / ses_names[session] / str(trial[:-4] + '.mp4')

    print('Processing......', subject, '-', ses_names[session], '-', trial)

    # Load data
    neural_data, emg_data, force_data, time_data, fs = load_h5py_file(file_name, offsets_all[:,session])
    fs = int(fs)

    # Specify "good" channels
    Right1 = (9, 11, 19)
    Right2 = (3, 21, 29)
    Right3 = (15, 23, 31)
    Right4 = (7, 13, 25)  # successful at bubble test
    Left1 = (2, 22, 26)
    Left2 = (10, 18, 20, 28)
    Left3 = ()
    Left4 = (14, 16, 24)  # successful at bubble test
    Extras = (0, 1, 4, 5, 6, 8, 12, 17, 27, 30)
    positions = Right2 + Right3 + Left1 + Left2 + Left3
    neural_data = neural_data[:, positions]

    # Filter data
    fneural = apply_fancy_filter(neural_data, fs, 80, 1500, ftype='bandpass')
    femg = apply_fancy_filter(emg_data, fs, 20, 1000, ftype='bandpass')
    fforce = apply_fancy_filter(force_data, fs, 10)

    # Double check the start and end points if they include the EXTRA portion
    EXTRA = int(extra*fs)

    if start_time - EXTRA + 1 < 1:
        print('START TIME ADJUSTED!')
        st = EXTRA + 1
    if end_time + EXTRA > len(neural_data):
        print('END-TIME ADJUSTED!')
        end_time = len(neural_data) - EXTRA - 1


    def plot_lines(ax_handle, points):
        """ Plot vertical lines """
        y_axis = ax_handle.get_ylim()
        for pnt in points:
            x_axis = (pnt, pnt)
            ax_handle.plot(x_axis, y_axis, 'b', lw=2)

    """
    # Plot figure to examine. Will be reused over and over
    fig, axs = plt.subplots(3, 1, sharex=True)
    mngr = plt.get_current_fig_manager()
    # x, y, dx, dy = mngr.window.geometry().getRect()
    mngr.window.setGeometry(1000, 50, 600, 800)

    axs[0].plot(fforce); axs[0].grid(True); axs[0].set_title('Force')
    axs[1].plot(femg); axs[1].grid(True); axs[1].set_title('EMG')
    axs[2].plot(fneural); axs[2].grid(True); axs[2].set_title('Neural')

    plot_lines(axs[0], (start_time, end_time))
    plot_lines(axs[1], (start_time, end_time))
    plot_lines(axs[2], (start_time, end_time))

    fig.show()
    plt.pause(0.1)
    plt.close(fig) """

    # Detect EMG outlier (spike) and replace with median EMG
    for channel in femg.T:
        emg_std = np.std(channel)
        emg_mean = np.mean(channel)
        channel[channel > emg_mean + 6 * emg_std] = 6 * emg_std
        channel[channel < emg_mean - 6 * emg_std] = -6 * emg_std

    # Cut the desired portion from the data
    neural_data = neural_data[start_time - EXTRA + 1:end_time + EXTRA, :]
    emg_data = emg_data[start_time - EXTRA + 1:end_time + EXTRA, :]
    force_data = force_data[start_time - EXTRA + 1:end_time + EXTRA]

    # Bad channel detection based on saturation threshold
    fneural = fneural[start_time - EXTRA + 1:end_time + EXTRA, :]
    used_channels = np.arange(0,fneural.shape[1])
    boolind = np.any(np.abs(fneural) > saturation_threshold, axis=0)
    CHANS.append(used_channels[~boolind])

    # Prepare ouput
    RAW.append(neural_data)
    EMG.append(emg_data)
    trial_lengths.append(len(neural_data))
    cut_times.append([EXTRA+1, len(neural_data)-EXTRA])

    print('DONE!')

## Identify the commonly used channels within session
""" How many times a channel appears in all trials? Keep the tally in a vector. """
chan_vect = np.arange(RAW[0].shape[1])
used_vect = np.zeros(RAW[0].shape[1])
for chan in CHANS:
    for x in chan:
        used_vect[x] += 1

""" There is a number-of-trials vs number-of-channels tradeof. Check how many channels we can keep to use 
60 to 90 percent of all the trials """
#pp = [0.9, 0.8, 0.7, 0.6]
pp = [0.9]
for item in pp:
    common_chans = chan_vect[used_vect >= round(item * max(used_vect))]

    count = 0
    for chan in CHANS:
        if len(list(set(common_chans).intersection(chan))) >= len(common_chans):
            count += 1
        else:
            continue
    print('To include {} % of trials:'.format(item*100))
    print('{} channels will be used: '.format(len(common_chans)), common_chans)
    print('Number of trials retained: ', count)
    print('-------------------------------------')

## Pick the desired channels, and trials that have those channels
""" Construct a boolean list: If a trial's remaining channels include the desired/common channels (intersect),
place a True in the list. If not, place a False. Use this list to choose trials. """
#common_chans = [0,1,2,3,4,5,6,7,8,9,10,11,12]
common_chans = list(common_chans)
keep1 = []
for chan in CHANS:
    if len(list(set(common_chans).intersection(chan))) >= len(common_chans):
        keep1.append(True)
    else:
        keep1.append(False)

RAW = [item for i, item in enumerate(RAW) if keep1[i] is True]
EMG = [item for i, item in enumerate(EMG) if keep1[i] is True]
trial_lengths = [item for i, item in enumerate(trial_lengths) if keep1[i] is True]
cut_times = [item for i, item in enumerate(cut_times) if keep1[i] is True]

## Detect and eliminate abnormal EMG shape
""" Unusually high amplitudes result in large peaks in EMG envelope. These outliers hinder the performance of 
the regression. Eliminate those outliers (not just a part of the data, the whole trial itself) """
maxemg = []
for emg in EMG:
    maxemg.append(np.max(np.abs(emg), axis=0))
maxemg = np.array(maxemg)

""" Construct a boolean array. Place a false if the trial is an outlier"""
keep2 = np.ones((len(EMG)), dtype=bool)
for item in maxemg.T:
    q75, q50, q25 = np.percentile(item, [75, 50, 25])
    iqr = q75 - q25
    keep2 = keep2 & ~(item > q50 + 6 * iqr)
keep2 = list(keep2)

RAW = [item for i, item in enumerate(RAW) if keep2[i] & True]
EMG = [item for i, item in enumerate(EMG) if keep2[i] & True]
trial_lengths = [item for i, item in enumerate(trial_lengths) if keep2[i] & True]
cut_times = [item for i, item in enumerate(cut_times) if keep2[i] & True]

## Determine average trial length
avelen = list(map(lambda x:(x-EXTRA*2)/fs, trial_lengths))
print('Average trial length : {0:.2f} sec'.format(np.mean(avelen)))
print('STD                  : {0:.2f} sec'.format(np.std(avelen)))

## Prepare data set
data = {
    "raw" : np.concatenate(RAW, axis=0),
    "emg" : np.concatenate(EMG, axis=0),
    "cut-times" : cut_times,
    "trial-lengths" : trial_lengths,
    "channels" : common_chans,
    "fs" : fs
    }

del(RAW, EMG)

gc.collect()

## Feature extraction
from extract_features import extract_features
dataset = extract_features(data)

## Regression

