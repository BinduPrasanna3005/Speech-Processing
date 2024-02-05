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


# In[6]:


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


# In[10]:


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


# In[ ]:




