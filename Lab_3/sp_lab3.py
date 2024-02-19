#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load the audio file
audio_file = 'SP.wav'
y, sr = librosa.load(audio_file)

# Trim the silence from the beginning and end of the signal
y_trimmed, index = librosa.effects.trim(y)

# Plot the original and trimmed signals
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
librosa.display.waveshow(y, sr=sr)
plt.title('Original Signal')

plt.subplot(2, 1, 2)
librosa.display.waveshow(y_trimmed, sr=sr)
plt.title('Trimmed Signal')

plt.tight_layout()
plt.show()
import IPython.display as ipd

import soundfile as sf

# Save the trimmed audio signal to a WAV file
sf.write('trimmed_audio.wav', y_trimmed, sr)

# Play the trimmed audio file
ipd.Audio('trimmed_audio.wav')


# In[2]:


import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio

# Load the audio file
file_path = "SP.wav"
signal, sr = librosa.load(file_path, sr=None)

# Perform speech splitting with different top_db values
top_db_values = [20, 30, 40]  # Adjust these values as desired
split_signals = []

for top_db in top_db_values:
    split_signal = librosa.effects.split(signal, top_db=top_db)
    split_signals.append(split_signal)

# Plot the original and split audio signals for each top_db value
plt.figure(figsize=(12, 8))
plt.subplot(len(top_db_values) + 1, 1, 1)
librosa.display.waveshow(signal, sr=sr)
plt.title('Original Signal')

for i, split_signal in enumerate(split_signals):
    plt.subplot(len(top_db_values) + 1, 1, i + 2)
    split_signal_plot = np.zeros_like(signal)
    for interval in split_signal:
        split_signal_plot[interval[0]:interval[1]] = signal[interval[0]:interval[1]]
    librosa.display.waveshow(split_signal_plot, sr=sr)
    plt.title(f'Split Signal (top_db={top_db_values[i]})')

    # Listen to the split signal
    split_audio = np.concatenate([signal[interval[0]:interval[1]] for interval in split_signal])
    display(Audio(data=split_audio, rate=sr))

plt.tight_layout()
plt.show()


# In[ ]:




