import matplotlib
import numpy as np
from importlib import reload
import util
from util import generate_frequency, plot_dft_and_fft

reload(util)

if __name__ == '__main__':
    T = 1.0
    Fs = 1000

    freq = generate_frequency(1.0, 20, 0.0, T, Fs)

    y_dft = util.dft(freq, Fs, T)
    y_fft = np.fft.fft(freq)

    plot_dft_and_fft(y_fft, T, Fs, "FFT")
    plot_dft_and_fft(y_fft, T, Fs, "DFT")
