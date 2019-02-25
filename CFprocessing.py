#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 08:40:48 2019

@author: sinan
"""

## Import
from pathlib import Path
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


## Select processing
things_to_run = [0]

# 1. Play movie
# 2. Remove saturated channels
# 3. Plot raw data
# 4. Plot filtered data
# 5. Plot power spectrum
# 6. Spectrograms - big picture
# 7. Correlogram
# 8. Coherogram

## Initial setup
maindir = Path.home() / 'Documents' / 'DATA' / 'Cerebellum' / 'CCI'
#maindir = Path.home() / 'Documents' / 'MATLAB' / 'ESMA'
folder = ''
subject = 'CCI5'
ses_names = ['112718', '120318', '120718', '121718', '121918']

force_thresh = 0.05
satThresh = 200

session = 0
trial = 11

file_name = maindir / subject / ses_names[session] / folder / 'trial{}.mat'.format(trial)
video_name = maindir / subject / ses_names[session] / folder / 'trial{}.mp4'.format(trial)

print('Processing......', subject, '-', ses_names[session], '-', folder, '-', 'trial{}.mat'.format(trial))

## Load data
neural_data, emg_data, force_data, time, fs = load_h5py_file(file_name)
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
fneural = filter_data(neural_data, fs, 20, 2000, ftype='bandpass')
femg = filter_data(emg_data, fs, 20, 1000, ftype='bandpass')
fforce = filter_data(force_data, fs, 20)

## Play the video file
if 1 in things_to_run:
    play_video_file(str(video_name))

## Take care of "BAD" signals
if 2 in things_to_run:
    keep = remove_saturated_parts(neural_data, satThresh, fs)  # logical indices (row)
    # keep = remove_saturated_channels(neural_data, satThresh, fs) # good channels (column)
    neural_data = neural_data[keep, :]
    emg_data = emg_data[keep, :]
    force_data = force_data[keep]
    time = np.arange(0, len(neural_data) / fs, 1 / fs)

    fneural = filter_data(neural_data, fs, 20, 2000, ftype='bandpass')
    femg = filter_data(emg_data, fs, 20, 1000, ftype='bandpass')
    fforce = filter_data(force_data, fs, 20)

## Plot raw data
if 3 in things_to_run:
    simple_plot(time, neural_data, 'time(s)', '(uV)', 'Raw neural', True)
    simple_plot(time, emg_data, 'time(s)', '(uV)', 'Raw EMG', True)
    simple_plot(time, force_data, 'time(s)', '(uV)', 'Raw force', True)

## Plot filtered data
if 4 in things_to_run:
    simple_plot(time, fneural, 'time(s)', '(uV)', 'Filtered neural', True)
    simple_plot(time, femg, 'time(s)', '(uV)', 'Filtered EMG', True)
    simple_plot(time, fforce, 'time(s)', '(uV)', 'Filtered force', True)

## Plot the frequency spectrum of the neural signal
if 5 in things_to_run:
    n = len(neural_data)
    NEURAL = np.abs(fftpack.fft(neural_data, axis=0))
    freqs = fftpack.fftfreq(n, d=1 / fs)

    fig, ax = plt.subplots()
    ax.plot(freqs[:n // 2], NEURAL[:n // 2])
    ax.set_xlabel('Frequency in Hertz [Hz]')
    ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
    ax.set_xlim(0, 3000)
    ax.grid(True)

## Plot spectrograms BIG PICTURE
if 6 in things_to_run:
    f, ax = plt.subplots(2, 2, figsize=(8, 6))

    # Plot normalized signal envelopes
    ax[0, 0].plot(time, normalize(my.find_envelope(femg, fs), 'max', axis=0))
    ax[0, 0].legend(('dorsal', 'biceps'))
    ax[0, 0].plot(time, normalize(my.find_envelope(fneural, fs), 'max', axis=0) + 1)
    ax[0, 0].set_title('Signal Envelopes')
    ax[0, 0].set_ylabel('Normalized')

    M = 2048  # Window length
    L = 128  # Shift length
    overlap = M - L
    fmin = 20
    fmax = 2000
    fres = fs / M
    coefmin = int(fmin / fres + 1)
    coefmax = int(fmax / fres + 1)

    freqs, times, Sx = signal.spectrogram(emg_data[:, 0], fs=fs, window='hanning',
                                          nperseg=M, noverlap=overlap,
                                          detrend=False, scaling='spectrum')

    ax[0, 1].pcolormesh(times, freqs[coefmin:coefmax] / 1000, 10 * np.log10(Sx[coefmin:coefmax, :]), cmap='viridis')
    ax[0, 1].set_ylabel('Frequency [kHz]')

    freqs, times, Sx = signal.spectrogram(emg_data[:, 1], fs=fs, window='hanning',
                                          nperseg=M, noverlap=overlap,
                                          detrend=False, scaling='spectrum')

    ax[1, 1].pcolormesh(times, freqs[coefmin:coefmax] / 1000, 10 * np.log10(Sx[coefmin:coefmax, :]), cmap='viridis')
    ax[1, 1].set_ylabel('Frequency [kHz]')
    ax[1, 1].set_xlabel('Time [s]')

    # Find the average spectrogram of neural signals
    Sx = np.zeros(np.shape(Sx))
    for column in neural_data.T:
        freqs, times, temp = signal.spectrogram(column, fs=fs, window='hanning',
                                                nperseg=M, noverlap=overlap,
                                                detrend=False, scaling='spectrum')
        Sx = Sx + temp

    Sx = Sx / neural_data.shape[1]

    ax[1, 0].pcolormesh(times, freqs[coefmin:coefmax] / 1000, 10 * np.log10(Sx[coefmin:coefmax, :]), cmap='viridis')
    ax[1, 0].set_ylabel('Frequency [kHz]')
    ax[1, 0].set_xlabel('Time [s]')

## Correlogram
if 7 in things_to_run:
    # Enter window length and step size
    # Applying a los-pass or band-pass filter is optional
    # Enter only 1ow-cutoff for LFP
    correlogram_plot(neural_data, 1024, 512, fs)

