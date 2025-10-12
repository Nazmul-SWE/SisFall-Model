# -*- coding: utf-8 -*-
"""
Create a Scalogram (CWT Image) from a multi-column TXT sensor file
Only uses the 1st column (Acceleration X from ADXL345)
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt
import pandas as pd

# === Step 1: Load the TXT file ===
# Replace this path with your actual file path
file_path = r"H:\SisFall Model\sensor_data\dataset1.txt"

# Read using pandas (handles commas, tabs, spaces, extra columns automatically)
data = pd.read_csv(file_path, header=None, sep=None, engine='python')

# Extract only the first column (Acceleration X)
signal = data.iloc[:, 0].values

# === Step 2: Preprocess Signal ===
# Remove NaN or non-numeric values if any
signal = signal[np.isfinite(signal)]

# Normalize the signal
signal = signal - np.mean(signal)
signal = signal / np.max(np.abs(signal))

# === Step 3: Define parameters ===
# Sampling frequency (Hz) — set based on your dataset info
fs = 1000  # adjust if you know actual sampling rate
t = np.arange(len(signal)) / fs

# === Step 4: Compute Continuous Wavelet Transform ===
scales = np.arange(1, 128)
coefficients, frequencies = pywt.cwt(signal, scales, 'morl', sampling_period=1/fs)

# === Step 5: Plot and Save Scalogram ===
plt.figure(figsize=(12, 6))

# Plot original signal
plt.subplot(2, 1, 1)
plt.plot(t, signal, color='black')
plt.title("Sensor Signal (1st Column - Acceleration X)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# Plot scalogram
plt.subplot(2, 1, 2)
plt.imshow(
    np.abs(coefficients),
    extent=[t[0], t[-1], max(scales), min(scales)],
    cmap='jet',
    aspect='auto'
)
plt.gca().invert_yaxis()
plt.colorbar(label='Magnitude')
plt.title("Scalogram (CWT of Acceleration X)")
plt.xlabel("Time (s)")
plt.ylabel("Scales")

plt.tight_layout()
plt.savefig("sensor_scalogram.png", bbox_inches='tight')
plt.show()

print(" Scalogram image generated and saved as 'sensor_scalogram.png'")
