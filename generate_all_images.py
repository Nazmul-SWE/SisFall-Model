# generate_all_images.py
"""
Generate Scalogram, Spectrogram and Kurtogram images for SisFall dataset.

Requirements:
  - Python 3.8+
  - numpy, scipy, pandas, matplotlib, pywt, pillow

Install once:
  pip install numpy scipy pandas matplotlib pywt pillow

Usage:
  - Put this script into your project root: D:\3.2\Thesis\SisFall Model\
  - Ensure raw data in: D:\3.2\Thesis\SisFall Model\SisFall_dataset\SA01, SA02, ..., SE01 ...
  - Run:
      python generate_all_images.py
  - Or supply base directory:
      python generate_all_images.py "D:\3.2\Thesis\SisFall Model"

Behavior:
  - Reads all subject folders that start with SA or SE.
  - Processes every .txt file inside each subject folder.
  - Detects activity type by file name prefix: D.. => Daily Living, F.. => Fall.
  - Extracts columns 0,1,2 (X,Y,Z) from each file (robust to separators and trailing commas).
  - Generates three image types per axis: Scalogram (CWT), Spectrogram (STFT), Kurtogram (kurtosis-based map).
  - Saves images to:
      Generated Images/<Type>/<Subject>/<Daily Living|Fall>/FILE_X.png

Notes about kurtogram implementation:
  - A full Kurtogram (Antoni) implementation is non-trivial; here we provide a practical kurtosis-based
    time-frequency map derived from the CWT: for each scale and sliding time window we compute
    kurtosis of the absolute CWT coefficients inside the window. This produces a scale-vs-time kurtosis map
    that is informative and actionable for fault/fall detection tasks.

The script is written to be robust and well-logged. Adjust FS, CWT scales, STFT params, and kurtosis
window length near the top of the script.
"""

import sys
from pathlib import Path
import os
import re
import traceback
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pywt
from scipy import signal, stats
from PIL import Image

# ---------------- USER CONFIGURATION ----------------
# Default base directory (change if you want)
DEFAULT_BASE_DIR = Path(r"D:\3.2\Thesis\SisFall Model").resolve()
RAW_DATA_DIR_NAME = 'SisFall_dataset'            # folder containing SA01, SE01 ...
OUTPUT_ROOT_NAME = 'Generated Images'            # root for all generated image types
IMAGE_TYPES = ['Scalogram', 'Spectrogram', 'Kurtogram']

# Sampling frequency in Hz (set to your dataset fs if known)
FS = 100.0

# CWT (Scalogram) params
CWT_WAVELET = 'morl'
CWT_SCALES = np.arange(1, 128)

# STFT (Spectrogram) params
STFT_NFFT = 256
STFT_NPERSEG = 256
STFT_NOOVERLAP = 128

# Kurtogram params (kurtosis over sliding windows on CWT magnitude)
KURTOGRAM_WINDOW_SAMPLES = 128  # length of sliding window in samples (time axis of signal)
KURTOGRAM_STEP = 32             # step between windows

# Image / save params
DPI = 150
RESIZE_TO = None  # e.g., (224,224) to normalize for ML, or None to keep generated size

# Subject folder regex
SUBJECT_PATTERN = re.compile(r'^(SA|SE)\d{2}$', re.IGNORECASE)
TXT_SUFFIX = '.txt'
MIN_COLS_REQUIRED = 3

# ----------------------------------------------------


def find_subject_folders(base_dir: Path):
    dataset_dir = base_dir / RAW_DATA_DIR_NAME
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Raw data folder not found: {dataset_dir}")

    folders = [p for p in sorted(dataset_dir.iterdir()) if p.is_dir() and SUBJECT_PATTERN.match(p.name)]
    return folders


def robust_read_file(file_path: Path):
    """Read a text file with unknown separator (comma, space, tab) and return DataFrame."""
    # Try regex sep first
    try:
        df = pd.read_csv(file_path, header=None, sep=r'[,\s]+', engine='python', comment='#')
    except Exception:
        try:
            df = pd.read_csv(file_path, header=None, delimiter=',', engine='python', comment='#')
        except Exception:
            try:
                df = pd.read_csv(file_path, header=None, delim_whitespace=True, engine='python', comment='#')
            except Exception as e:
                raise RuntimeError(f"Failed reading {file_path}: {e}")

    # drop fully empty columns that sometimes appear due trailing commas
    df = df.dropna(axis=1, how='all')
    return df


def preprocess_signal(col):
    arr = pd.to_numeric(col, errors='coerce').values
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return arr
    arr = arr - np.mean(arr)
    max_abs = np.max(np.abs(arr))
    if max_abs == 0 or not np.isfinite(max_abs):
        return arr
    return arr / max_abs


def make_output_path(base_output: Path, subject: str, cls: str, img_type: str):
    """Return output folder path for given subject and class (Daily Living / Fall) and image type."""
    folder = base_output / img_type / subject / cls
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# ---- Image generation functions ----

def save_figure_to_path(fig, out_path: Path, dpi=DPI, resize_to=RESIZE_TO):
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    if resize_to is not None:
        try:
            img = Image.open(out_path)
            img = img.resize(resize_to, resample=Image.BILINEAR)
            img.save(out_path)
        except Exception as e:
            warnings.warn(f"Could not resize {out_path}: {e}")


def generate_scalogram_image(signal1d, fs, scales, wavelet_name, out_path: Path, title='Scalogram'):
    coeffs, freqs = pywt.cwt(signal1d, scales, wavelet_name, sampling_period=1.0/fs)
    mag = np.abs(coeffs)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    extent = [0, len(signal1d)/fs if fs>0 else len(signal1d), max(scales), min(scales)]
    im = ax.imshow(mag, aspect='auto', extent=extent)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Scale')
    fig.colorbar(im, ax=ax, label='Magnitude')

    save_figure_to_path(fig, out_path)


def generate_spectrogram_image(signal1d, fs, nperseg, noverlap, nfft, out_path: Path, title='Spectrogram'):
    f, t_seg, Sxx = signal.spectrogram(signal1d, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=nfft, scaling='spectrum')
    # convert to dB
    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.pcolormesh(t_seg, f, Sxx_db, shading='gouraud')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='dB')

    save_figure_to_path(fig, out_path)


def generate_kurtogram_image(signal1d, fs, scales, window_samples, step, out_path: Path, title='Kurtogram'):
    # Compute CWT magnitude
    coeffs, freqs = pywt.cwt(signal1d, scales, CWT_WAVELET, sampling_period=1.0/fs)
    mag = np.abs(coeffs)  # shape: (n_scales, n_times)

    n_scales, n_times = mag.shape
    if n_times < 1:
        raise ValueError('Empty CWT result')

    # Sliding window kurtosis along time for each scale
    ws = int(window_samples)
    st = int(step)
    if ws < 3:
        ws = 3
    # number of output time positions
    positions = list(range(0, max(1, n_times - ws + 1), st))
    K = np.zeros((n_scales, len(positions)))

    for si in range(n_scales):
        for pi, p in enumerate(positions):
            window = mag[si, p:p+ws]
            if window.size < 3:
                K[si, pi] = 0.0
            else:
                # kurtosis: fisher=False to get Pearson definition (kurtosis of normal is 3)
                k = stats.kurtosis(window, fisher=False, bias=False)
                # shift to zero-mean (optional): subtract 3 to center normal at 0
                K[si, pi] = k - 3.0

    # Build time axis in seconds
    time_positions = np.array(positions) / fs

    fig, ax = plt.subplots(figsize=(8, 4.5))
    # extent: time start, time end, scale max, scale min
    extent = [time_positions[0] if time_positions.size>0 else 0,
              time_positions[-1] if time_positions.size>0 else (n_times/fs if fs>0 else n_times),
              max(scales), min(scales)]
    im = ax.imshow(K, aspect='auto', extent=extent)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Scale')
    fig.colorbar(im, ax=ax, label='Kurtosis-3')

    save_figure_to_path(fig, out_path)


# ---- Processing single file ----

def process_single_file(file_path: Path, base_output: Path, fs=FS):
    fname = file_path.name
    base = file_path.stem  # without extension

    # Determine class by filename prefix
    if fname.upper().startswith('D'):
        cls = 'Daily Living'
    elif fname.upper().startswith('F'):
        cls = 'Fall'
    else:
        # fallback: look at pattern <Dxx_ or Fxx_>
        if re.match(r'^[dD]\d{2}_', fname) is not None:
            cls = 'Daily Living'
        elif re.match(r'^[fF]\d{2}_', fname) is not None:
            cls = 'Fall'
        else:
            cls = 'Unknown'

    # Read data robustly
    try:
        df = robust_read_file(file_path)
    except Exception as e:
        print(f"[ERROR] Could not read {file_path}: {e}")
        return

    if df.shape[1] < MIN_COLS_REQUIRED:
        print(f"[SKIP] {file_path} has less than {MIN_COLS_REQUIRED} columns ({df.shape[1]})")
        return

    # Extract X,Y,Z columns (first three)
    for i, axis in enumerate(['X', 'Y', 'Z']):
        if i >= df.shape[1]:
            print(f"[SKIP] {base}_{axis}: missing column")
            continue

        raw_col = df.iloc[:, i]
        sig = preprocess_signal(raw_col)
        if sig.size < 8:
            print(f"[SKIP] {base}_{axis}: insufficient data after preprocessing (len={sig.size})")
            continue

        # Build output folder and filename
        subject = file_path.parent.name
        out_folder_scal = make_output_path(base_output, subject, cls, 'Scalogram')
        out_folder_spec = make_output_path(base_output, subject, cls, 'Spectrogram')
        out_folder_kurt = make_output_path(base_output, subject, cls, 'Kurtogram')

        out_name = f"{base}_{axis}.png"

        # Generate Scalogram
        try:
            out_path = out_folder_scal / out_name
            generate_scalogram_image(sig, fs, CWT_SCALES, CWT_WAVELET, out_path, title=f"{base}_{axis} Scalogram")
        except Exception as e:
            print(f"[ERROR] Scalogram {base}_{axis}: {e}")
            traceback.print_exc()

        # Generate Spectrogram
        try:
            out_path = out_folder_spec / out_name
            generate_spectrogram_image(sig, fs, STFT_NPERSEG, STFT_NOOVERLAP, STFT_NFFT, out_path, title=f"{base}_{axis} Spectrogram")
        except Exception as e:
            print(f"[ERROR] Spectrogram {base}_{axis}: {e}")
            traceback.print_exc()

        # Generate Kurtogram
        try:
            out_path = out_folder_kurt / out_name
            generate_kurtogram_image(sig, fs, CWT_SCALES, KURTOGRAM_WINDOW_SAMPLES, KURTOGRAM_STEP, out_path, title=f"{base}_{axis} Kurtogram")
        except Exception as e:
            print(f"[ERROR] Kurtogram {base}_{axis}: {e}")
            traceback.print_exc()

    print(f"[DONE] {file_path}")


# ---- Main pipeline ----

def run_pipeline(base_dir: Path):
    base_dir = base_dir.resolve()
    raw_dir = base_dir / RAW_DATA_DIR_NAME
    output_root = base_dir / OUTPUT_ROOT_NAME
    output_root.mkdir(parents=True, exist_ok=True)

    subjects = find_subject_folders(base_dir)
    if not subjects:
        print(f"No subject folders found in {raw_dir}. Make sure folder names are SAxx or SExx and data placed inside.")
        return

    print(f"Found {len(subjects)} subject folders. Processing...")

    for subj in subjects:
        print(f"Processing subject: {subj.name}")
        txt_files = sorted([p for p in subj.iterdir() if p.is_file() and p.suffix.lower() == TXT_SUFFIX])
        if not txt_files:
            print(f"  No .txt files found in {subj}")
            continue

        for f in txt_files:
            try:
                process_single_file(f, output_root, fs=FS)
            except Exception as e:
                print(f"Unexpected error processing {f}: {e}")
                traceback.print_exc()

    print("\nProcessing complete. Generated images saved under:", output_root)


# ---------------- Entry point ----------------
if __name__ == '__main__':
    # parse optional CLI arg for base dir
    base = DEFAULT_BASE_DIR
    if len(sys.argv) > 1:
        arg = Path(sys.argv[1])
        if arg.is_dir():
            base = arg.resolve()
        else:
            print(f"Argument is not a directory: {arg}. Using default: {base}")

    print("Base directory:", base)
    try:
        run_pipeline(base)
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
