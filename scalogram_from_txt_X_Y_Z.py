import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt

# --- Define paths ---
base_folder = r"D:\3.2\Thesis\SisFall Model"
data_folder = os.path.join(base_folder, "sensor_data")
output_folder = os.path.join(base_folder, "Scalogram images")


# Make sure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load your specific file
file_path = os.path.join(data_folder, "dataset1.txt")

# Check if file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# --- Load Data (handle commas or mixed separators) ---
data = pd.read_csv(file_path, header=None, sep=r'[,\s]+', engine='python')

# --- Process first three columns (X, Y, Z) ---
axes = ['X', 'Y', 'Z']
for i, axis in enumerate(axes[:3]):
    signal = data.iloc[:, i].values  # select column
    signal = signal - np.mean(signal)  # remove DC offset

    # Continuous Wavelet Transform
    scales = np.arange(1, 128)
    coef, freqs = pywt.cwt(signal, scales, 'cmor')

    # Plot scalogram
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(coef), cmap='jet', aspect='auto',
               extent=[0, len(signal), max(scales), min(scales)])
    plt.gca().invert_yaxis()
    plt.title(f"Scalogram - {os.path.basename(file_path).split('.')[0]}_{axis}")
    plt.xlabel("Time")
    plt.ylabel("Scales")
    plt.colorbar(label="Magnitude")

    # Save image
    output_name = f"{os.path.basename(file_path).split('.')[0]}_{axis}.png"
    output_path = os.path.join(output_folder, output_name)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

print(f"✅ Scalograms saved in: {output_folder}")
