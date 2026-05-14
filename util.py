from pydub import AudioSegment
import numpy as np
from matplotlib import pyplot as plt


def read_signal_from_mp3(input_file):
    sound = AudioSegment.from_mp3(input_file)
    samples = np.frombuffer(sound.raw_data, dtype=np.int16).astype(np.float32) / 32767
    return samples, sound.frame_rate


def generate_frequency(A, freq, p, T, Fs):
    t = np.arange(0, T, (1.0 / Fs))
    return A * np.sin(2 * np.pi * freq * t + p)


def plot_frequency(A, y, p, t, Fs, freq):
    plt.plot(t, y)
    plt.title(f'Vzor: {Fs} Hz, Frek: {freq} Hz, Ampl: {A}, Faza: {p} rad')
    plt.xlabel('Čas [s]')
    plt.ylabel('Amplituda')
    plt.grid(True)
    plt.show()

def dft(y, Fs, max_freqs):
    T = len(y) / Fs
    N = np.arange(len(y)) # Frequency length.

    # All frequencies from 0 to Fs stepped by 1/T
    # Freqency resolution 1/T
    freqs = np.arange(0, Fs, 1/T)
    freqs = freqs[:max_freqs]
    result = []

    # Dot product over all frequencies [0, Fs].
    for f in freqs:
        result.append(np.dot(y, np.exp(-1j * 2 * np.pi * f * N / Fs)))

    return np.array(result), freqs


import numpy as np


def real_sin_projection(y, Fs, max_freq_bins):
    N = len(y)
    T = N / Fs
    t = np.arange(N) / Fs

    result = []
    freqs = []

    for k in range(max_freq_bins):
        f = k / T  # Izračun frekvence za trenutni bin
        reference_sin = np.cos(2 * np.pi * f * t) # Realna frekvenca

        result.append(np.dot(y, reference_sin))
        freqs.append(f)

    return np.array(result), np.array(freqs)


def plot_analasys(y, x, label):
    plt.plot(x, abs(y), label=label)
    plt.title('Frekvenčna vsebina')
    plt.xlabel('Frekvenca [Hz]')
    plt.ylabel('Amplituda')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_dft_and_fft(y_fft, y_dft, freqs_fft, freqs_dft, Fs):
    plot_analasys(y_fft, freqs_fft, 'FFT')
    plot_analasys(y_dft, freqs_dft, 'DFT')


def fft_dft(freq, Fs, max_freqs):
    y_dft, freqs_dft = dft(freq, Fs, max_freqs)
    y_fft = np.fft.fft(freq)[:max_freqs]

    T_eff = len(freq) / Fs
    freqs_fft = np.arange(0, Fs, 1/T_eff)[:max_freqs]

    plot_dft_and_fft(y_fft, y_dft, freqs_fft, freqs_dft, Fs)