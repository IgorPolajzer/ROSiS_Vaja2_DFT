from importlib import reload
import util

reload(util)

if __name__ == '__main__':
    MAX_FREQS = 500

    T = 1.0
    Fs = 5000

    freq = util.generate_frequency(1.0, 20, 0.0, T, Fs)

    util.fft_dft(freq, Fs, T, MAX_FREQS)