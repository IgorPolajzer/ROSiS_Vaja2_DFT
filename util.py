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

def dft(y, Fs, T, max_freqs):
    N = np.arange(len(y)) # Frequency length.

    # All frequencies from 0 to Fs stepped by 1/T
    # Freqency resolution 1/T
    freqs = np.arange(0, Fs, 1/T)
    freqs = freqs[:max_freqs]
    result = []

    # Dot product over all frequencies [0, Fs].
    for f in freqs:
        result.append(np.dot(y, np.exp(-1j * 2 * np.pi * f * N / Fs)))

    return np.array(result)


def real_sin_dft(y, Fs, T, real_freq, max_freqs):
    # All frequencies from 0 to Fs stepped by 1/T
    # Freqency resolution 1/T
    freqs = np.arange(0, Fs, 1/T)[:max_freqs]
    result = []

    # Dot product over all frequencies [0, Fs].
    for f in freqs:
        result.append(np.dot(y, real_freq))

    return np.array(result)


def plot_analasys(y, Fs, T, label):
    N_total = int(T * Fs)
    max_freq = (len(y) / N_total) * Fs
    x = np.linspace(0, max_freq, len(y))

    plt.plot(x, abs(y), label=label)
    plt.title('Frekvenčna vsebina')
    plt.xlabel('Frekvenca [Hz]')
    plt.ylabel('Amplituda')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_dft_and_fft(y_fft, y_dft, T1, T2, Fs):
    plot_analasys(y_fft, Fs, T1, 'FFT')
    plot_analasys(y_dft, Fs, T2, 'DFT')


def fft_dft(freq, Fs, T, max_freqs):
    y_dft = dft(freq, Fs, T, max_freqs)
    y_fft = np.fft.fft(freq)[:max_freqs] # We take the first 'max_freqs' found through FFT.

    plot_dft_and_fft(y_fft, y_dft, T, T, Fs)