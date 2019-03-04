
import matplotlib
import numpy as np

from my_functions.loading import *
from my_functions.processing import *
from my_functions.plotting import *

Fs = 3000

sos = design_fancy_filter(Fs, 5, order=5, ftype=None, flag=True)

b, a = butter_lowpass(Fs, 5, order=5, flag=False)

# Demonstrate the use of the filter.
# First make some data to be filtered.
T = 5.0         # seconds
n = int(T * Fs) # total number of samples
t = np.linspace(0, T, n, endpoint=False)
# "Noisy" data.  We want to recover the 1.2 Hz signal from this.
data = np.sin(2*2*np.pi*t) + 1.5*np.cos(100*2*np.pi*t) + 2*np.random.rand(len(t))

plt.subplot(2, 1, 1)
plt.plot(t, data, 'b-', label='data')
plt.plot(t, apply_fancy_filter(data, Fs, 5, order=4, ftype=None, flag=False))
plt.plot(t, apply_fancy_filter(data, Fs, 5, order=10, ftype=None, flag=False))
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, data, 'b-', label='data')
plt.plot(t, signal.filtfilt(b, a, data), 'g-', linewidth=2, label='filtered data')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()


plt.subplots_adjust(hspace=0.35)
plt.show()



from scipy import signal

x = np.linspace(0, 10, 100, endpoint=False)
y = np.cos(-x**2/6.0)
f = signal.resample(y, 50)
xnew = np.linspace(0, 10, 50, endpoint=False)

plt.plot(x, y, 'go-', xnew, f, '.-', 10, y[0], 'ro')
plt.legend(['data', 'resampled'], loc='best')
plt.show()



##
# Frequency magnitude response, impulse response

from __future__ import division, print_function
import numpy as np
from numpy.random import randn
from numpy.fft import rfft
from scipy import signal
import matplotlib.pyplot as plt

b, a = signal.butter(4, 0.03, analog=False)

# Show that frequency response is the same
impulse = np.zeros(1000)
impulse[500] = 1

# Applies filter forward and backward in time
imp_ff = signal.filtfilt(b, a, impulse)

# Applies filter forward in time twice (for same frequency response)
imp_lf = signal.lfilter(b, a, signal.lfilter(b, a, impulse))

plt.subplot(2, 2, 1)
plt.semilogx(20*np.log10(np.abs(rfft(imp_lf))))
plt.ylim(-100, 20)
plt.grid(True, which='both')
plt.title('lfilter')

plt.subplot(2, 2, 2)
plt.semilogx(20*np.log10(np.abs(rfft(imp_ff))))
plt.ylim(-100, 20)
plt.grid(True, which='both')
plt.title('filtfilt')

sig = np.cumsum(randn(800))  # Brownian noise
sig_ff = signal.filtfilt(b, a, sig)
sig_lf = signal.lfilter(b, a, signal.lfilter(b, a, sig))
plt.subplot(2, 1, 2)
plt.plot(sig, color='silver', label='Original')
plt.plot(sig_ff, color='#3465a4', label='filtfilt')
plt.plot(sig_lf, color='#cc0000', label='lfilter')
plt.grid(True, which='both')
plt.legend(loc="best")