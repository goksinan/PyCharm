"""
Created on Sun Jan 13 08:40:48 2019

Reads trials in a particular session one-by-one
Plots the signals and allows user to specify a data segment by clicking on the screen
Also plays video
Saves information inside a .cvs file

@author: sinan
"""

## Import
from pathlib import Path
import glob
import os
import time

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

force_thresh = 0.05
satThresh = 200

session = 0

## Read all trials
path_to_folder = maindir / subject / ses_names[session]
os.chdir(path_to_folder)
trial_names = []

for file in glob.glob("*.mat"):
    trial_names.append(file)

def give_me_numbers(name):
    return int(name[5:-4])

trial_numbers = list(map(give_me_numbers, trial_names))
trial_names = [x for _, x in sorted(zip(trial_numbers, trial_names))]

## Process trials
offsets_file_name = maindir / subject / 'baseline-offsets-CCI5.xlsx'
offsets_all = load_excel_file(offsets_file_name,cell_range='B2:F4')
offsets = offsets_all[:,session]

get_times = []
count = 0
for trial in trial_names:
    count += 1
    if count < 1:
        continue
    file_name = maindir / subject / ses_names[session] / trial
    video_name = maindir / subject / ses_names[session] / str(trial[:-4] + '.mp4')

    print('Processing......', subject, '-', ses_names[session], '-', trial)

    # Load data
    neural_data, emg_data, force_data, time_data, fs = load_h5py_file(file_name, offsets)
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

    # Plot figure to examine and click
    fig, axs = plt.subplots(3, 1, sharex=False)
    mngr = plt.get_current_fig_manager()
    x, y, dx, dy = mngr.window.geometry().getRect()
    mngr.window.setGeometry(1087, 43, 814, 969)

    axs[0].plot(fforce); axs[0].grid(True)
    axs[1].plot(femg); axs[1].grid(True)
    axs[2].plot(fneural); axs[2].grid(True)
    #fig.suptitle(ses_names[session] + ' - ' + trial)

    # Play video
    play_video_file(str(video_name))

    def tellme(s):
        """ Update figure title and print text """
        print(s)
        fig.suptitle(s, fontsize=16)
        plt.pause(0.01)

    def plot_lines(ax_handle, points):
        """ Plot vertical lines """
        y_axis = ax_handle.get_ylim()
        for pnt in points:
            x_axis = (pnt, pnt)
            ax_handle.plot(x_axis, y_axis, 'r', lw=2)

    tellme('Select two points, middle mouse button to skip')
    pts = np.asarray(plt.ginput(2, timeout=-1))

    if len(pts) < 2:
        tellme('Skipped!')
        time.sleep(.5)
        plt.close(fig)
        #break
    else:
        tellme('Done!')
        pts = np.sort(pts[:,0], axis=0)
        plot_lines(axs[0], pts); plot_lines(axs[1], pts); plot_lines(axs[2], pts)
        plt.pause(0.01)
        time.sleep(1)
        plt.close(fig)
        #get_times.append(pts)
        get_times.append((session, trial, pts[0], pts[1]))


## Save Results
from openpyxl import Workbook

wb = Workbook()
sheet = wb.active
sheet['A1'] = 'SESSION'
sheet['B1'] = 'TRIAL'
sheet['C1'] = 'START'
sheet['D1'] = 'END'

for row in get_times:
    sheet.append(row)

os.chdir(maindir / subject)
wb.save('selected-trials-{}.xlsx'.format(session))

