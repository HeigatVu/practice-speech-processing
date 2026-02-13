import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy
import numpy as np

def stft(data, fs, window_length_ms=30, window_step_ms=20, windowing_function=None):
    window_length = int(window_length_ms*fs/1000)
    window_step = int(window_step_ms*fs/1000)
    if windowing_function == None:
        windowing_function = np.sin(np.pi*np.arange(0.5, window_length, 1)/window_length)**2 # Hann windowing function

    total_length = len(data)
    window_count = int((total_length - window_length)/window_step) + 1

    spectrum_length = int(window_length/2) + 1
    spectrogram = np.zeros((window_count, spectrum_length))

    for k in range(window_count):
        starting_position = k*window_step

        data_vector = data[starting_position:(starting_position+window_length), ]
        window_spectrum = np.abs(scipy.fft.rfft(data_vector*windowing_function, n=window_length))

        spectrogram[k, :] = window_spectrum
    return spectrogram

if __name__ == "__main__":
    fs = 16000
    data = np.random.randn(fs) # 1 sec of noise
    try:
        s = stft(data, fs)
        print("Spectrogram shape:", s.shape)
    except NameError as e:
        print(f"NameError caught: {e}")
    except Exception as e:
        print(f"Error caught: {e}")
