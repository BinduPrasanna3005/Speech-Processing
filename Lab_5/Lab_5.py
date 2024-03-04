#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Load the speech signal
file_path = r"C:\Users\Bindu\Downloads\Amrita2.wav"
sample_rate, speech_signal = wavfile.read(file_path)

# Perform FFT to transform to frequency domain
spectral_components = np.fft.fft(speech_signal)

# Plot the amplitude part of the spectral components
plt.figure(figsize=(10, 5))
plt.plot(np.abs(spectral_components))
plt.title('Amplitude of Spectral Components')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.show()

# Inverse transform to time domain signal
time_domain_signal = np.fft.ifft(spectral_components)


# In[16]:


import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import windows

# Load the speech signal
file_path = r"C:\Users\Bindu\Downloads\Amrita2.wav"  # Replace this with the path to your speech file
speech_signal, sr = librosa.load(file_path, sr=None)

# Perform FFT
fft_result = np.fft.fft(speech_signal)

# Compute the frequency spectrum
freq_spectrum = np.abs(fft_result)

# Define the length of the FFT and frequency bins
N = len(freq_spectrum)
freq_bins = np.fft.fftfreq(N, d=1/sr)

# Apply rectangular window to select low frequency components (e.g., below 1000 Hz)
low_pass_cutoff = 1000
rectangular_window = np.zeros_like(freq_bins)
rectangular_window[freq_bins < low_pass_cutoff] = 1

# Filter the spectrum using the rectangular window
filtered_spectrum_rectangular = fft_result * rectangular_window

# Inverse FFT to get back the time-domain signal for rectangular window
filtered_signal_rectangular = np.fft.ifft(filtered_spectrum_rectangular)
filtered_signal_rectangular = np.real(filtered_signal_rectangular)

# Play the filtered signal for rectangular window
import sounddevice as sd
sd.play(filtered_signal_rectangular, sr)
sd.wait()

# Plot the amplitude part of the spectral components for the filtered signal (rectangular window)
plt.figure(figsize=(10, 5))
plt.plot(np.abs(filtered_spectrum_rectangular))
plt.title('Amplitude Spectrum of Filtered Speech Signal (Rectangular Window)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# Apply band-pass window to select frequencies within a specific range (e.g., between 500 Hz and 1500 Hz)
band_pass_cutoff_low = 500
band_pass_cutoff_high = 1500
band_pass_window = np.zeros_like(freq_bins)
band_pass_window[(freq_bins >= band_pass_cutoff_low) & (freq_bins <= band_pass_cutoff_high)] = 1

# Filter the spectrum using the band-pass window
filtered_spectrum_band_pass = fft_result * band_pass_window

# Inverse FFT to get back the time-domain signal for band-pass window
filtered_signal_band_pass = np.fft.ifft(filtered_spectrum_band_pass)
filtered_signal_band_pass = np.real(filtered_signal_band_pass)

# Play the filtered signal for band-pass window
sd.play(filtered_signal_band_pass, sr)
sd.wait()

# Plot the amplitude part of the spectral components for the filtered signal (band-pass window)
plt.figure(figsize=(10, 5))
plt.plot(np.abs(filtered_spectrum_band_pass))
plt.title('Amplitude Spectrum of Filtered Speech Signal (Band-Pass Window)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
# Apply high-pass window to select frequencies above a certain cutoff (e.g., above 2000 Hz)
high_pass_cutoff = 2000
high_pass_window = np.zeros_like(freq_bins)
high_pass_window[freq_bins >= high_pass_cutoff] = 1

# Filter the spectrum using the high-pass window
filtered_spectrum_high_pass = fft_result * high_pass_window

# Inverse FFT to get back the time-domain signal for high-pass window
filtered_signal_high_pass = np.fft.ifft(filtered_spectrum_high_pass)
filtered_signal_high_pass = np.real(filtered_signal_high_pass)

# Play the filtered signal for high-pass window
sd.play(filtered_signal_high_pass, sr)
sd.wait()

# Plot the amplitude part of the spectral components for the filtered signal (high-pass window)
plt.figure(figsize=(10, 5))
plt.plot(np.abs(filtered_spectrum_high_pass))
plt.title('Amplitude Spectrum of Filtered Speech Signal (High-Pass Window)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()



# In[7]:


# Apply Gaussian window to select low frequency components (e.g., below 1000 Hz)
sigma = 100  # Standard deviation of the Gaussian window
gaussian_window = windows.gaussian(N, std=sigma)
filtered_spectrum_gaussian = fft_result * gaussian_window

# Inverse FFT to get back the time-domain signal for Gaussian window
filtered_signal_gaussian = np.fft.ifft(filtered_spectrum_gaussian)
filtered_signal_gaussian = np.real(filtered_signal_gaussian)

# Play the filtered signal for Gaussian window
sd.play(filtered_signal_gaussian, sr)
sd.wait()

# Plot the amplitude part of the spectral components for the filtered signal (Gaussian window)
plt.figure(figsize=(10, 5))
plt.plot(np.abs(filtered_spectrum_gaussian))
plt.title('Amplitude Spectrum of Filtered Speech Signal (Gaussian Window)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()


# In[ ]:




