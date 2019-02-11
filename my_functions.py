
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import fftpack
from scipy import signal
from scipy import io
from scipy import stats
from itertools import compress
import cv2

# ------------------------------------------------------------------------------------
# VARIOUS CUSTOM-WRITTEN FUNCTIONS FOR NEURAL SIGNAL PROCESSING ----------------------
# ------------------------------------------------------------------------------------

def load_data(fname):
    """
    Load data file. Organize and return data, time vector, and sample rate.
    Ex:
        neural, emg, force, time, fs = load_data('trial8.mat')
    """
    # Load the data
    f = h5py.File(fname, 'r')  # r for read only
    print("Available fields: ", list(f.keys()))  # f is a dictionary. Let's look at the keys

    # Create variables from loaded dictionary
    neural_data = f['ripple_data'][:,0:32]
    emg_data = f['ripple_data'][:,32:]
    force_data = f['data'][0:6,:].transpose()
    fs = f['mySampleRate'][:]

    # Transform matrix for force data
    TF = [[1.117	, -0.096747,	 1.7516, 0.03441, -0.88072, 0.042127, -0.89026],
          [0.3134, 0.0041349, 0.0045219, -0.055942, 1.5273, 0.037719,-1.5227],
          [0.135	, 1.4494, -0.061075, 1.6259, 0.083867, 1.5999, 0.0058155]]
    TF = np.array(TF)

    # Read force data
    force_data = np.concatenate((np.ones((len(force_data),1)), force_data), axis=1)
    force_data = force_data @ TF.transpose()

    # Use sent and received pulse signals to allign DAQ and RIPPLE data
    pulse_sent = f['data'][6,:].transpose()
    ps_ind, = np.nonzero(pulse_sent>1)
    ps_ind = ps_ind[0]

    pulse_received = f['ttl_data'][:,0]
    pr_ind, = np.nonzero(pulse_received>2000)
    pr_ind = pr_ind[0]

    p_diff = ps_ind - pr_ind

    # Align pulse data
    pulse_sent = np.concatenate((pulse_sent[p_diff+1:], np.zeros((p_diff,))), axis=0)

    # Align force data
    trailing = np.mean(force_data[:-int(fs*0.1)], axis=0) * np.ones((p_diff,1))
    force_data = np.concatenate((force_data[p_diff:,:], trailing))
    # Choose force channel for analysis
    force_data = force_data[:,1]
    force_data = -force_data # Invert the sign (increased as applied force increased)

    # Align EMG data
    emg_data = emg_data[:,(5,15)]-emg_data[:,(23,25)]

    # Re-order EMG data so that 1. Dorsal 2. Biceps 3. Ventral 4. Triceps
    positions3 = (0,1)
    emg_data = emg_data[:,positions3]

    # Corresponding time vectors
    time = f['ripple_time'][:]
    return neural_data, emg_data, force_data, time, fs


def filter_file(data, fs, lc, hc=None, order=4, ftype=None, flag=False):
    """
    Design a butterworth filter.
    At least 3 arguments needed: data, filter cutoff, sample rate
    Ex:
       y = filter_file(x, 30, 16000)
    """
    nyq = fs/2
    fdata = np.empty(0)

    if ftype is None:
        sos = signal.butter(order, lc/nyq, 'lowpass', output = 'sos')
        fdata = signal.sosfiltfilt(sos, data, axis=0)

    if ftype is 'bandpass':
        wn = [lc/nyq, hc/nyq]
        sos = signal.butter(order, wn, 'bandpass', output = 'sos')
        fdata = signal.sosfiltfilt(sos, data, axis=0)

    if ftype is 'bandstop':
        wn = [lc/nyq, hc/nyq]
        sos = signal.butter(order, wn, 'bandstop', output = 'sos')
        fdata = signal.sosfiltfilt(sos, data, axis=0)

    if ftype is 'high':
        sos = signal.butter(order, lc/nyq, 'highpass', output = 'sos')
        fdata = signal.sosfiltfilt(sos, data, axis=0)

    if flag is not False:
        w, h = signal.sosfreqz(sos)
        plt.figure(1)
        plt.plot(float(fs * 0.5 / np.pi) * w, abs(h), label='order={}'.format(order))
        plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],'--', label='sqrt(0.5)')
        plt.title('Butterworth filter frequency response')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Gain')
        plt.grid(True)
        plt.legend(loc='best')
        plt.show()

    return fdata


def remove_saturated_channels(data, threshold, fs):
    """
    REMOVES SATURATED CHANNELS ENTIRELY
    Regardless of the duration of the saturated signal, the entire signal
    is removed. 3-level saturation detection:
        1. Based on a user specified threshold
        2. Based on range
    """
    # Remove baseline wandering
    lc = 10 # cutoff
    nyq = fs/2
    sos = signal.butter(4, lc/nyq, 'highpass', output = 'sos')
    data = signal.sosfiltfilt(sos, data, axis=0)

    original_channels = [i for i in range(len(data[0]))]

    TH = 3  # Threshold for quartile elimination method. The smaller, the more consevative

    # Bad channel detection based on saturation
    incl = [not (column>threshold).any() for column in data.T]
    data = data[:,incl]
    remaining_channels = list(compress(original_channels,incl))

    # Bad channel detection based on abnormal range
    doneFlag = False

    while doneFlag is False:
        nfeat = np.max(data,axis=0) - np.min(data,axis=0)
        nfeat = nfeat.round()
        # sorted(nfeat, reverse=True)
        niqr = stats.iqr(nfeat)
        nfbig = nfeat > np.median(nfeat) + TH*niqr
        nfbig = ~nfbig
        nfsml = nfeat < np.median(nfeat) - TH*niqr
        nfsml = ~nfsml
        nfall = list(nfbig & nfsml)
        data = data[:,nfall]
        if not any(nfall):
            remaining_channels = list(compress(remaining_channels,nfall))
            continue
        else:
            doneFlag = True

    return remaining_channels


def remove_saturated_parts(neural, threshold, fs):
    """
    To remove saturated part plus x ms before & x ms after
    Returns a logical index vector
    To stitch the good parts back together
    """
    # Window length to remove
    rem = int(fs*0.01)

    # Remove baseline wandering
    lc = 0.1 # cutoff
    nyq = fs/2
    sos = signal.butter(4, lc/nyq, 'highpass', output = 'sos')
    neural = signal.sosfiltfilt(sos, neural, axis=0)

    for ind, column in enumerate(neural.T):
        satpnts = abs(column)>threshold
        if any(satpnts):
            pos_sat, = np.where(satpnts>0)
            for i in pos_sat:
                if i >= len(column)-rem:
                    column[i-rem:] = 0
                elif i <= rem:
                    column[:i+rem] = 0
                else:
                    column[i-rem:i+rem] = 0
            neural[:,ind] = column

    keep = [all(row) for row in neural]
    return keep


def play_video_file(fname : str):
    """
    Plays video file and closes the window when finished.
    :type fname: string
    """
    cap = cv2.VideoCapture(fname)
    fps = cap.get(5)
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (255, 0, 0)
    lineType = 2

    while cap.isOpened():
        ret, frame = cap.read()

        if ret is True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.putText(gray, 'Time: ' + str(round(cap.get(0) / 1000, 2)),
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)
            cv2.putText(gray, 'Frame: ' + str(int(cap.get(1))),
                        (10, 70),
                        font,
                        fontScale,
                        fontColor,
                        lineType)
            cv2.imshow('frame', gray)
            #cv2.waitKey(10)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def find_envelope(data, fs, N=512, hc=8):
    """
    Finds the envelope of signal using a windowing technique
    :param data: signal whose envelope to be found
    :param fs: sampling rate for low-pass filter
    :param N: window length (default=512)
    :param hc: corner frequency of smoothing (low pass) filter (default:8)
    :return: the envelope
    """
    data = np.array(data)
    shifts = int(np.floor(len(data)/N))
    for i in range(shifts):
        s = i * N
        e = s + N
        data[s:e] = np.ones(np.shape(data[s:e])) * np.max(data[s:e], axis=0)
        if len(data) - e < N:
            data[e:] = np.ones(np.shape(data[e:])) * np.max(data[e:], axis=0)

    env = filter_file(data, fs, hc)
    return env


def plot_correlogram(data, N, L, fs, lc=None, hc=None):
    """
    Compute all pairwise combinations of correlation coefficients for a multi-dimensional array.
    Plot them as a color mesh where x-axis is time, y-axis is pair numbers, z-axis is the CC value
    :param data: N-dimensional array
    :param N: Moving window length
    :param L: Moving window step
    :param fs: Sampling rate of original array
    :param lc: Lower cut-off of the band-pass filter
    :param hc: Higher cut-off of the band-pass filter
    :return: Displays a figure.
    """
    # How many times the window will shift?
    shifts = int(np.floor((len(data)-N) / L) + 1)

    fdata = np.array(data)

    if hc is not None:
        fdata = filter_file(fdata, fs, lc, hc, ftype='bandpass')
    elif lc is not None:
        fdata = filter_file(fdata, fs, lc)

    cols = int(shifts + 1)
    rows = int(fdata.shape[1] * (fdata.shape[1] - 1) * 0.5)
    CC = np.zeros((rows,cols))
    for i in range(shifts):
        s = i * L
        e = s + N
        cc = np.triu(np.corrcoef(fdata[s:e], rowvar=False), 1) # Extract the upper triangle of the correlation matrix
        cc = cc.flatten()
        cc = cc[np.nonzero(cc)]
        if len(cc) != rows: print('Warning! Missing value.')
        CC[:,i] = cc
        if len(fdata) - e < L:
            s = e - L
            e = len(fdata)
            cc = np.triu(np.corrcoef(fdata[s:e], rowvar=False), 1)
            cc = cc.flatten()
            cc = cc[np.nonzero(cc)]
            if len(cc) != rows: print('Warning! Missing value.')
            CC[:,i+1] = cc

    # Plot surface
    x = np.linspace(0, len(fdata)/fs, CC.shape[1])
    y = np.linspace(1, CC.shape[0], CC.shape[0])
    f, ax = plt.subplots(figsize=(8, 4))
    # If there are only two channels:
    if CC.shape[0] == 1:
        ax.plot(x, CC[0])
        ax.set_ylabel('Corr')
        ax.set_xlabel('Time [s]')
        ax.set_title('Between two channels')
    # If there are 3 or more channels:
    else:
        cax = ax.pcolormesh(x, y, CC, cmap='viridis')
        ax.set_ylabel('Channel pairs')
        ax.set_xlabel('Time [s]')
        ax.set_title('Correlogram')
        f.colorbar(cax, ticks=[-1, -0.5, 0, 0.5, 1])


def plot_figure(x, y=None, xlab='x-axis', ylab='y-axis', ttl='Title', grd=True):
    """
    Function for easy plotting jobs.
    :param x: x data
    :param y: y data
    :param xlab: x label
    :param ylab: y label
    :param ttl: title
    :param grd: grid flag
    :return: Figure handle
    """
    f = plt.figure()
    if y is None:
        plt.plot(x)
    else:
        plt.plot(x, y)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(ttl)
    # plt.axis([40, 160, 0, 0.03])
    plt.grid(grd)
    return f