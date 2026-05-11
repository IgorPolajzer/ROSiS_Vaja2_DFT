import numpy as np
from matplotlib import pyplot as plt


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

def dft(x, Fs, T):
    N = len(x)
    n = np.arange(N)
    freqs = np.arange(0, Fs, 1/T)
    result = []

    for f in freqs:
        X_f = np.dot(x, np.exp(-1j * 2 * np.pi * f * n / Fs))
        result.append(X_f)

    return np.array(result)


def plot_dft_and_fft(y_fft, y_dft, T, Fs):
    t = np.arange(0, T, (1.0 / Fs))

    N = len(t)
    x = np.linspace(0, Fs, N)

    plt.plot(x, abs(y_dft), label="DFT")
    plt.title('Frekvenčna vsebina')
    plt.xlabel('Frekvenca [Hz]')
    plt.ylabel('Amplituda')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(x, abs(y_fft), label="FFT")
    plt.title('Frekvenčna vsebina')
    plt.xlabel('Frekvenca [Hz]')
    plt.ylabel('Amplituda')
    plt.legend()
    plt.grid(True)
    plt.show()