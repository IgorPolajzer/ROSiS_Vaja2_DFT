from importlib import reload
import util

reload(util)

if __name__ == '__main__':
    T = 1.0
    y_a, Fsa = util.read_signal_from_mp3("/home/igor/Desktop/MAG/1_LETNIK/2_SEMESTER/RACUNALNISKA_OBDELAVA_SIGNALOV_IN_SLIK/Vaja_2/recordings/a_medium_pitch.mp3")
    util.fft_dft(y_a, Fsa, T)