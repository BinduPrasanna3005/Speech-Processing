#!/usr/bin/env python
# coding: utf-8

# In[2]:


import librosa
file='SP.wav'
y, sr=librosa.load(file)


# In[3]:


librosa.display.waveshow(y)


# In[4]:



from itertools import cycle
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from glob import glob
sns.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])


# In[5]:


import librosa

file_path = "SP.wav"  # Replace with the actual path to your audio file
y, sr = librosa.load(file_path)
y_trimmed, _ = librosa.effects.trim(y, top_db=20)
pd.Series(y_trimmed).plot(figsize=(10, 5),
                  lw=1,
                  title='Raw Audio Trimmed Example',
                 color=color_pal[1])
plt.show()


# In[15]:


import librosa
import sounddevice as sd
import time


filename = 'SP.wav'
y, sr = librosa.load(filename)


segment_start_time = 0
segment_end_time = 1.0


segment = y[int(segment_start_time * sr):int(segment_end_time * sr)]


sd.play(segment, sr)
time.sleep(2) 


# In[16]:


import librosa
import sounddevice as sd
import time

# Load the recorded speech file with a custom sampling rate
filename = 'SP.wav'
custom_sr = 3200 # Custom sampling rate in Hz
y, sr = librosa.load(filename, sr=custom_sr)

# Define segment start and end time
segment_start_time = 0
segment_end_time = 1.0

# Extract the segment from the loaded signal
segment = y[int(segment_start_time * custom_sr):int(segment_end_time * custom_sr)]

# Play the segment
sd.play(segment, custom_sr)
time.sleep(2)  # Wait for 2 seconds to allow playback to finish


# In[14]:


import librosa
import sounddevice as sd
import time
import matplotlib.pyplot as plt
import numpy as np

# Load the recorded speech file with a custom sampling rate
filename = 'SP.wav'
custom_sr = 4100 # Custom sampling rate in Hz
y, sr = librosa.load(filename, sr=custom_sr)

# Define segment start and end time
segment_start_time = 0
segment_end_time = 1.0

# Extract the segment from the loaded signal
segment = y[int(segment_start_time * custom_sr):int(segment_end_time * custom_sr)]

# Play the segment
sd.play(segment, custom_sr)
time.sleep(2)  # Wait for 2 seconds to allow playback to finish

# Plot the waveform
plt.figure(figsize=(10, 4))
time_axis = np.linspace(segment_start_time, segment_end_time, len(segment))
plt.plot(time_axis, segment, color='b')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Waveform of Speech Segment')
plt.show()


# In[11]:


import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Load the WAV file
sample_rate, signal = wavfile.read('SP.wav')

# Define the time step (sampling interval)
dt = 1.0 / sample_rate

# Calculate the first derivative using finite differences
derivative = np.diff(signal) / dt

# Plot the original and derivative signals
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(np.arange(len(signal)) / sample_rate, signal, label='Original Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Original Speech Signal')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(np.arange(len(derivative)) / sample_rate, derivative, label='First Derivative')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('First Derivative of Speech Signal')
plt.legend()

plt.tight_layout()
plt.show()

# Convert derivative back to audio and listen (optional)
# Note: You may need to adjust the scale of the derivative to avoid clipping
# wavfile.write('derivative_signal.wav', sample_rate, derivative.astype(np.int16))


# In[17]:


from scipy.io import wavfile

# Assuming 'derivative' is the calculated derivative signal and 'sample_rate' is the sampling rate
# Scale the derivative signal to avoid clipping
scaled_derivative = np.int16(derivative / np.max(np.abs(derivative)) * 32767)

# Write the derivative signal to a WAV file
wavfile.write('derivative_signal.wav', sample_rate, scaled_derivative)

print("Derivative signal saved as 'derivative_signal.wav'. You can now listen to it.")


# In[21]:


import numpy as np

# Assume 'derivative' is the first derivative signal

# Step 1: Determine a threshold value
threshold = 0.1  # Adjust as needed based on the characteristics of your derivative signal

# Step 2: Detect zero crossing points
zero_crossings = np.where(np.diff(np.sign(derivative)))[0]

# Step 3: Differentiate between speech and silence regions
speech_regions = []
silence_regions = []
current_region = []

for crossing_index in zero_crossings:
    if abs(derivative[crossing_index]) < threshold:
        # Silence region
        if current_region:
            silence_regions.append(current_region)
            current_region = []
    else:
        # Speech region
        current_region.append(crossing_index)

# Handle the last region
if current_region:
    silence_regions.append(current_region)

# Step 4: Calculate average length between two consecutive zero crossings
def calculate_average_length(regions):
    if regions:
        lengths = [len(region) for region in regions]
        return sum(lengths) / len(lengths)
    else:
        return 0

average_length_speech = calculate_average_length(silence_regions)
average_length_silence = calculate_average_length(speech_regions)

# Step 5: Compare average lengths
print("Average length between two consecutive zero crossings for speech regions:", average_length_speech)
print("Average length between two consecutive zero crossings for silence regions:", average_length_silence)


# In[33]:


import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile  # Import wavfile module

# Function to calculate duration from frames and frame rate
def calculate_duration(frames, frame_rate):
    return frames / frame_rate

# Load your audio file
your_audio_path = r"C:\Users\Bindu\Downloads\Amrita.wav"  # Replace with the actual file path
with wave.open(your_audio_path, 'rb') as wf:
    frames = wf.getnframes()
    frame_rate = wf.getframerate()
    your_duration = calculate_duration(frames, frame_rate)

print("Your speech duration:", your_duration, "seconds")

# Load your audio file using scipy.io.wavfile
your_rate, your_audio = wavfile.read(your_audio_path)

# Plot the waveform
plt.figure(figsize=(10, 4))
time = np.arange(len(your_audio)) / your_rate
plt.plot(time, your_audio)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Your Speech Signal")
plt.grid(True)
plt.show()


# In[37]:


import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile  # Import wavfile module

# Function to calculate duration from frames and frame rate
def calculate_duration(frames, frame_rate):
    return frames / frame_rate

# Load your audio file
your_audio_path = r"C:\Users\Bindu\Downloads\Amrita2.wav"  # Replace with the actual file path
with wave.open(your_audio_path, 'rb') as wf:
    frames = wf.getnframes()
    frame_rate = wf.getframerate()
    your_duration = calculate_duration(frames, frame_rate)

print("Your speech duration:", your_duration, "seconds")

# Load your audio file using scipy.io.wavfile
your_rate, your_audio = wavfile.read(your_audio_path)

# Plot the waveform
plt.figure(figsize=(10, 4))
time = np.arange(len(your_audio)) / your_rate
plt.plot(time, your_audio)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Your Speech Signal")
plt.grid(True)
plt.show()


# In[40]:


import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile  # Import wavfile module

# Function to calculate duration from frames and frame rate
def calculate_duration(frames, frame_rate):
    return frames / frame_rate

# Load your audio files
your_audio_path = r"C:\Users\Bindu\Downloads\Amrita.wav"  # Replace with the actual file path of your audio
teammate_audio_path = r"C:\Users\Bindu\Downloads\Amrita2.wav"  # Replace with the actual file path of your teammate's audio

# Load your audio file
try:
    with wave.open(your_audio_path, 'rb') as wf:
        frames = wf.getnframes()
        frame_rate = wf.getframerate()
        your_duration = calculate_duration(frames, frame_rate)
    
    print("Your speech duration:", your_duration, "seconds")
    
    # Load your audio file using scipy.io.wavfile
    your_rate, your_audio = wavfile.read(your_audio_path)
    
    # Plot the waveform
    plt.figure(figsize=(10, 4))
    time = np.arange(len(your_audio)) / your_rate
    plt.plot(time, your_audio)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Your Speech Signal")
    plt.grid(True)
    plt.show()
    
except FileNotFoundError:
    print("Your audio file not found. Please provide the correct file path.")
except Exception as e:
    print("An error occurred while processing your audio file:", e)

# Load your teammate's audio file
try:
    with wave.open(teammate_audio_path, 'rb') as wf:
        frames = wf.getnframes()
        frame_rate = wf.getframerate()
        teammate_duration = calculate_duration(frames, frame_rate)
    
    print("Teammate's speech duration:", teammate_duration, "seconds")
    
    # Load your teammate's audio file using scipy.io.wavfile
    teammate_rate, teammate_audio = wavfile.read(teammate_audio_path)
    
    # Plot the waveform
    plt.figure(figsize=(10, 4))
    time_teammate = np.arange(len(teammate_audio)) / teammate_rate
    plt.plot(time_teammate, teammate_audio)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Teammate's Speech Signal")
    plt.grid(True)
    plt.show()
    
except FileNotFoundError:
    print("Teammate's audio file not found. Please provide the correct file path.")
except Exception as e:
    print("An error occurred while processing teammate's audio file:", e)


# In[46]:


import librosa
import numpy as np
import matplotlib.pyplot as plt

def get_pitch(audio_file_path):
    # Loading the audio file
    y, sr = librosa.load(audio_file_path)

    # Compute the pitch using Harmonic-Percussive Source Separation
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    # Average pitch across time frames
    avg_pitch = np.mean(pitches)

    return avg_pitch, y, sr

# Replace with actual file paths
question_pitch, question_waveform, question_sr = get_pitch('Qstn.wav')
statement_pitch, statement_waveform, statement_sr = get_pitch('sentence.wav')

# Plot waveforms of the two signals
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(np.arange(len(question_waveform)) / question_sr, question_waveform, color='b')
plt.title('Waveform of Question')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(1, 2, 2)
plt.plot(np.arange(len(statement_waveform)) / statement_sr, statement_waveform, color='r')
plt.title('Waveform of Statement')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

# Compare pitch characteristics
print("Pitch analysis results:")
print(f"Question average pitch: {question_pitch:.2f} Hz")
print(f"Statement average pitch: {statement_pitch:.2f} Hz")


# In[ ]:




