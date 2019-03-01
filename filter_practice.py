
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