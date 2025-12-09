# generate_all_images.py - FIXED VERSION
"""
Generate Scalogram, Spectrogram and Kurtogram images for UpFall dataset.
FIXES:
- Better error handling with detailed logging
- Improved parameter adaptation for short signals
- Verification that images were actually created
"""

import sys
from pathlib import Path
import os
import re
import traceback
import warnings
import io
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pywt
from scipy import signal, stats
from PIL import Image

# ---------------- USER CONFIGURATION ----------------
DEFAULT_BASE_DIR = Path(r"/Users/Fahim/Codes/ML/Model/UpFall_Model").resolve()
RAW_DATA_DIR_NAME = 'UpFall_dataset'
OUTPUT_ROOT_NAME = 'Generated Images'
IMAGE_TYPES = ['Scalogram', 'Spectrogram', 'Kurtogram']

# Sampling frequency in Hz
FS = 100.0

# Time window (in seconds) to process from each signal
TIME_START_S = None
TIME_END_S = None

# CWT (Scalogram) params
CWT_WAVELET = 'morl'
CWT_SCALES = np.arange(1, 128)

# STFT (Spectrogram) params - adjusted for 10s signals
STFT_NFFT = 256
STFT_NPERSEG = 64  # Reduced from 128 for better time resolution
STFT_NOVERLAP = 48  # Adjusted overlap (75%)

# Kurtogram params
KURTOGRAM_WINDOW_SAMPLES = 32  # Reduced from 64 for better resolution
KURTOGRAM_STEP = 8  # Reduced from 16 for smoother map

# Image / save params
DPI = 150
RESIZE_TO = None

# Subject folder regex
SUBJECT_PATTERN = re.compile(r'^A\d{2}$', re.IGNORECASE)
CSV_SUFFIX = '.csv'
MIN_COLS_REQUIRED = 3

# Track statistics
stats_counter = {
    'files_processed': 0,
    'scalogram_success': 0,
    'spectrogram_success': 0,
    'kurtogram_success': 0,
    'scalogram_fail': 0,
    'spectrogram_fail': 0,
    'kurtogram_fail': 0
}


# ----------------------------------------------------


def infer_sampling_rate(df: pd.DataFrame, default_fs: float) -> float:
    """Infer sampling rate (Hz) from a TIME column if present."""
    try:
        time_col_candidates = [c for c in df.columns if isinstance(c, str) and c.strip().lower() == 'time']
        if not time_col_candidates:
            return float(default_fs)
        tcol = time_col_candidates[0]
        tseries = pd.to_datetime(df[tcol], errors='coerce')
        if tseries.isna().all():
            return float(default_fs)
        sec = tseries.dt.floor('S')
        unique_secs = sec.nunique(dropna=True)
        if unique_secs and unique_secs > 0:
            total_samples = len(df)
            fs_est = total_samples / float(unique_secs)
            if np.isfinite(fs_est) and fs_est > 0.1:
                return float(fs_est)
        return float(default_fs)
    except Exception:
        return float(default_fs)


def find_subject_folders(base_dir: Path):
    dataset_dir = base_dir / RAW_DATA_DIR_NAME
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Raw data folder not found: {dataset_dir}")

    folders = [p for p in sorted(dataset_dir.iterdir()) if p.is_dir() and SUBJECT_PATTERN.match(p.name)]
    return folders


def robust_read_file(file_path: Path):
    """Read a CSV (or text) file and return a DataFrame."""
    try:
        df = pd.read_csv(file_path, engine='python', comment='#')
    except Exception:
        try:
            df = pd.read_csv(file_path, header=None, sep=r'[,\s]+', engine='python', comment='#')
        except Exception as e:
            raise RuntimeError(f"Failed reading {file_path}: {e}")

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


def normalize_array(arr: np.ndarray) -> np.ndarray:
    """Normalize a numeric 1D array."""
    if arr is None or arr.size == 0:
        return arr
    arr = arr.astype(float)
    arr = arr - np.nanmean(arr)
    max_abs = np.nanmax(np.abs(arr))
    if not np.isfinite(max_abs) or max_abs == 0:
        return arr
    return arr / max_abs


def slice_time_window(arr: np.ndarray, fs: float, t_start: Optional[float], t_end: Optional[float]) -> np.ndarray:
    """Return a slice of arr corresponding to [t_start, t_end) in seconds."""
    if arr is None or arr.size == 0:
        return arr
    n = len(arr)
    if not (fs and fs > 0):
        return arr

    start_idx = 0 if t_start is None else int(np.floor(t_start * fs))
    end_idx = n if t_end is None else int(np.ceil(t_end * fs))

    start_idx = max(0, min(start_idx, n))
    end_idx = max(0, min(end_idx, n))

    if end_idx <= start_idx:
        return np.array([], dtype=arr.dtype)
    return arr[start_idx:end_idx]


def make_output_path(base_output: Path, subject: str, device: str, img_type: str):
    """Return output folder path."""
    folder = base_output / device / img_type / subject
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# ---- Image generation functions with improved error handling ----

def save_figure_to_path(fig, out_path: Path, dpi=DPI, resize_to=RESIZE_TO):
    """Save figure and optionally resize."""
    try:
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

        # Verify file was created
        if not out_path.exists():
            raise RuntimeError(f"Image file was not created: {out_path}")

        if resize_to is not None:
            img = Image.open(out_path)
            img = img.resize(resize_to, resample=Image.BILINEAR)
            img.save(out_path)
    except Exception as e:
        plt.close(fig)
        raise RuntimeError(f"Failed to save figure to {out_path}: {e}")


def generate_scalogram_image(signal1d, fs, scales, wavelet_name, out_path: Path, title='Scalogram',
                             t_offset_s: float = 0.0):
    """Generate scalogram with improved error handling."""
    if len(signal1d) < 8:
        raise ValueError(f"Signal too short for scalogram: {len(signal1d)} samples")

    # Adjust scales if signal is very short
    max_scale = min(max(scales), len(signal1d) // 4)
    scales_used = scales[scales <= max_scale]
    if len(scales_used) < 2:
        scales_used = np.arange(1, min(max_scale + 1, 10))

    coeffs, freqs = pywt.cwt(signal1d, scales_used, wavelet_name, sampling_period=1.0 / fs)
    mag = np.abs(coeffs)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    duration = len(signal1d) / fs if fs > 0 else len(signal1d)
    extent = [t_offset_s, t_offset_s + duration, max(scales_used), min(scales_used)]
    im = ax.imshow(mag, aspect='auto', extent=extent, cmap='viridis')
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Scale')
    fig.colorbar(im, ax=ax, label='Magnitude')

    save_figure_to_path(fig, out_path)
    print(f"    ✓ Scalogram saved: {out_path.name}")


def generate_spectrogram_image(signal1d, fs, nperseg, noverlap, nfft, out_path: Path, title='Spectrogram',
                               t_offset_s: float = 0.0):
    """Generate spectrogram with better parameter adaptation."""
    n = len(signal1d)
    if n < 8:
        raise ValueError(f"Signal too short for spectrogram: {n} samples")

    # Adapt parameters based on signal length
    # For a 10s signal at 100Hz (1000 samples), use reasonable window
    npseg = min(nperseg, n)

    # Ensure window is not too large
    if npseg > n // 2:
        npseg = max(n // 4, 16)

    # Ensure overlap is valid
    nov = min(noverlap, npseg - 1)
    if nov < 0:
        nov = 0

    # Ensure nfft is at least as large as nperseg
    nfft_eff = max(nfft, npseg)

    print(f"    Spectrogram params: n={n}, nperseg={npseg}, noverlap={nov}, nfft={nfft_eff}")

    try:
        f, t_seg, Sxx = signal.spectrogram(
            signal1d,
            fs=fs,
            window='hann',
            nperseg=npseg,
            noverlap=nov,
            nfft=nfft_eff,
            scaling='spectrum',
            detrend=False
        )
    except Exception as e:
        raise RuntimeError(f"Spectrogram computation failed: {e}")

    # Convert to dB
    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    t_axis = t_seg + t_offset_s if fs > 0 else t_seg
    im = ax.pcolormesh(t_axis, f, Sxx_db, shading='gouraud', cmap='viridis')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='dB')

    save_figure_to_path(fig, out_path)
    print(f"    ✓ Spectrogram saved: {out_path.name}")


def generate_kurtogram_image(signal1d, fs, scales, window_samples, step, out_path: Path, title='Kurtogram',
                             t_offset_s: float = 0.0):
    """Generate kurtogram with improved error handling."""
    n = len(signal1d)
    if n < 8:
        raise ValueError(f"Signal too short for kurtogram: {n} samples")

    # Adjust scales if needed
    max_scale = min(max(scales), n // 4)
    scales_used = scales[scales <= max_scale]
    if len(scales_used) < 2:
        scales_used = np.arange(1, min(max_scale + 1, 10))

    # Compute CWT magnitude
    coeffs, freqs = pywt.cwt(signal1d, scales_used, CWT_WAVELET, sampling_period=1.0 / fs)
    mag = np.abs(coeffs)

    n_scales, n_times = mag.shape
    if n_times < 1:
        raise ValueError('Empty CWT result')

    # Adapt window size if needed
    ws = min(int(window_samples), n_times // 2)
    if ws < 3:
        ws = min(3, n_times)

    st = min(int(step), ws // 2)
    if st < 1:
        st = 1

    print(f"    Kurtogram params: n_times={n_times}, window={ws}, step={st}")

    # Sliding window kurtosis
    positions = list(range(0, max(1, n_times - ws + 1), st))
    K = np.zeros((n_scales, len(positions)))

    for si in range(n_scales):
        for pi, p in enumerate(positions):
            window = mag[si, p:p + ws]
            if window.size >= 3:
                k = stats.kurtosis(window, fisher=False, bias=False)
                K[si, pi] = k - 3.0

    # Build time axis
    time_positions = (np.array(positions) / fs) + t_offset_s if fs > 0 else np.array(positions)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if time_positions.size > 0:
        t_start = time_positions[0]
        t_end = time_positions[-1]
    else:
        t_start = t_offset_s
        t_end = t_offset_s + (n_times / fs if fs > 0 else n_times)

    extent = [t_start, t_end, max(scales_used), min(scales_used)]
    im = ax.imshow(K, aspect='auto', extent=extent, cmap='viridis')
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Scale')
    fig.colorbar(im, ax=ax, label='Kurtosis-3')

    save_figure_to_path(fig, out_path)
    print(f"    ✓ Kurtogram saved: {out_path.name}")


def generate_xyz_with_original_image(x_norm, y_norm, z_norm, original_signal, fs, out_path: Path,
                                     title='XYZ + Original', t_offset_s: float = 0.0):
    """Create unified plot with X, Y, Z and original signal."""
    n = min(len(x_norm), len(y_norm), len(z_norm), len(original_signal))
    if n < 8:
        raise ValueError("Insufficient data to plot unified image")

    x = x_norm[:n]
    y = y_norm[:n]
    z = z_norm[:n]
    orig = original_signal[:n]

    t = (np.arange(n) / fs + t_offset_s) if (fs and fs > 0) else np.arange(n)

    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)

    axes[3].plot(t, orig, color='black', linewidth=1.5)
    axes[3].set_xlabel('Time (s)' if fs and fs > 0 else 'Samples')
    axes[3].set_ylabel('Original')
    axes[3].grid(True, alpha=0.3)

    axes[0].plot(t, x, color='tab:blue', linewidth=1.5)
    axes[0].set_ylabel('X')
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, y, color='tab:orange', linewidth=1.5)
    axes[1].set_ylabel('Y')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, z, color='tab:green', linewidth=1.5)
    axes[2].set_ylabel('Z')
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure_to_path(fig, out_path)


def render_original_signal_image(original_signal, fs, width_px, height_px, title, t_offset_s: float = 0.0):
    """Render the original resultant signal into a PIL Image."""
    n = len(original_signal)
    t = (np.arange(n) / fs + t_offset_s) if (fs and fs > 0) else np.arange(n)

    fig, ax = plt.subplots(figsize=(width_px / 100.0, height_px / 100.0), dpi=100)
    ax.plot(t, original_signal, color='black', linewidth=0.9)
    ax.set_title(title)
    ax.set_xlabel('Time (s)' if fs and fs > 0 else 'Samples')
    ax.set_ylabel('Original')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    img = img.resize((width_px, height_px), resample=Image.BILINEAR)
    return img


def compose_single_type_image(original_signal, fs, base_output: Path, subject: str, device: str, base: str,
                              img_type: str, t_offset_s: float = 0.0):
    """Create a 2-row composite image for a single modality."""
    folder = base_output / device / img_type / subject
    paths = {
        'X': folder / f"{base}_X.png",
        'Y': folder / f"{base}_Y.png",
        'Z': folder / f"{base}_Z.png",
    }

    if not all(p.exists() for p in paths.values()):
        missing = [ax for ax, p in paths.items() if not p.exists()]
        print(f"[SKIP] {img_type} composite for {base}: missing axis images: {', '.join(missing)}")
        return None

    tile_w, tile_h = 600, 360
    original_h = 260
    pad = 10
    cols = 3

    canvas_w = cols * tile_w + (cols + 1) * pad
    canvas_h = original_h + 3 * pad + tile_h

    canvas = Image.new('RGB', (canvas_w, canvas_h), color=(255, 255, 255))

    orig_img = render_original_signal_image(original_signal, fs, canvas_w - 2 * pad, original_h,
                                            title=f"{base} Original Resultant", t_offset_s=t_offset_s)
    canvas.paste(orig_img, (pad, pad))

    y_row = pad + original_h + pad
    x = pad
    for ax in ['X', 'Y', 'Z']:
        im = Image.open(paths[ax]).convert('RGB')
        im = im.resize((tile_w, tile_h), resample=Image.BILINEAR)
        canvas.paste(im, (x, y_row))
        x += tile_w + pad

    out_folder = folder
    out_path = out_folder / f"{base}_{img_type}.png"
    canvas.save(out_path)
    print(f"[OK] {img_type} composite image: {out_path}")
    return out_path


# ---- Processing single file with detailed logging ----

def process_single_file(file_path: Path, base_output: Path, fs=FS):
    """Process a single file and generate all image types."""
    base = file_path.stem
    print(f"\n[Processing] {file_path.name}")

    try:
        df = robust_read_file(file_path)
    except Exception as e:
        print(f"[ERROR] Could not read {file_path}: {e}")
        return

    if df.shape[1] < MIN_COLS_REQUIRED:
        print(f"[SKIP] {file_path} has less than {MIN_COLS_REQUIRED} columns ({df.shape[1]})")
        return

    fs_used = infer_sampling_rate(df, fs)
    print(f"  Using sampling rate: {fs_used} Hz")

    t_offset_s = float(TIME_START_S) if (TIME_START_S is not None and fs_used and fs_used > 0) else 0.0

    subject = file_path.parent.name

    device_cols = {
        'Pocket': [16, 17, 18],
        'Wrist': [23, 24, 25],
    }

    for device, cols in device_cols.items():
        print(f"  Device: {device}")
        max_idx = max(cols)
        if df.shape[1] <= max_idx:
            print(f"    [SKIP] Not enough columns (needed index {max_idx}, have {df.shape[1] - 1})")
            continue

        norm_signals = {}
        raw_signals = {}
        axes = ['X', 'Y', 'Z']
        axis_t_offsets = {}

        for axis, col_idx in zip(axes, cols):
            raw_col = df.iloc[:, col_idx]
            raw_arr_full = pd.to_numeric(raw_col, errors='coerce').values
            raw_arr_full = raw_arr_full[np.isfinite(raw_arr_full)]

            raw_arr = slice_time_window(raw_arr_full, fs_used, TIME_START_S, TIME_END_S)
            t_off_local = t_offset_s

            if raw_arr.size < 8:
                raw_arr = raw_arr_full
                t_off_local = 0.0

            sig = normalize_array(raw_arr)

            if sig.size < 8 or raw_arr.size < 8:
                print(f"    [SKIP] {axis}: insufficient data (len={sig.size})")
                continue

            print(f"    {axis}: {sig.size} samples")
            norm_signals[axis] = sig
            raw_signals[axis] = raw_arr
            axis_t_offsets[axis] = t_off_local

            out_folder_scal = make_output_path(base_output, subject, device, 'Scalogram')
            out_folder_spec = make_output_path(base_output, subject, device, 'Spectrogram')
            out_folder_kurt = make_output_path(base_output, subject, device, 'Kurtogram')

            out_name = f"{base}_{axis}.png"

            # Generate Scalogram
            try:
                out_path = out_folder_scal / out_name
                generate_scalogram_image(sig, fs_used, CWT_SCALES, CWT_WAVELET, out_path,
                                         title=f"{base} {device} {axis} Scalogram", t_offset_s=t_off_local)
                stats_counter['scalogram_success'] += 1
            except Exception as e:
                print(f"    ✗ Scalogram {axis} FAILED: {e}")
                stats_counter['scalogram_fail'] += 1
                # Continue processing other types

            # Generate Spectrogram
            try:
                out_path = out_folder_spec / out_name
                generate_spectrogram_image(sig, fs_used, STFT_NPERSEG, STFT_NOVERLAP, STFT_NFFT, out_path,
                                           title=f"{base} {device} {axis} Spectrogram", t_offset_s=t_off_local)
                stats_counter['spectrogram_success'] += 1
            except Exception as e:
                print(f"    ✗ Spectrogram {axis} FAILED: {e}")
                stats_counter['spectrogram_fail'] += 1

            # Generate Kurtogram
            try:
                out_path = out_folder_kurt / out_name
                generate_kurtogram_image(sig, fs_used, CWT_SCALES, KURTOGRAM_WINDOW_SAMPLES, KURTOGRAM_STEP, out_path,
                                         title=f"{base} {device} {axis} Kurtogram", t_offset_s=t_off_local)
                stats_counter['kurtogram_success'] += 1
            except Exception as e:
                print(f"    ✗ Kurtogram {axis} FAILED: {e}")
                stats_counter['kurtogram_fail'] += 1

        # Generate unified images if all axes present
        if all(ax in norm_signals for ax in ['X', 'Y', 'Z']):
            try:
                n_min_raw = min(len(raw_signals['X']), len(raw_signals['Y']), len(raw_signals['Z']))
                if n_min_raw >= 8:
                    rx = raw_signals['X'][:n_min_raw]
                    ry = raw_signals['Y'][:n_min_raw]
                    rz = raw_signals['Z'][:n_min_raw]
                    original_resultant = np.sqrt(rx * rx + ry * ry + rz * rz)
                else:
                    original_resultant = norm_signals['X']

                out_folder_unified = make_output_path(base_output, subject, device, 'Plotting')
                out_name_unified = f"{base}_XYZ.png"
                out_path_unified = out_folder_unified / out_name_unified

                n_min_norm = min(len(norm_signals['X']), len(norm_signals['Y']), len(norm_signals['Z']))
                generate_xyz_with_original_image(
                    norm_signals['X'][:n_min_norm],
                    norm_signals['Y'][:n_min_norm],
                    norm_signals['Z'][:n_min_norm],
                    original_resultant[:n_min_norm],
                    fs_used,
                    out_path_unified,
                    title=f"{base} {device} XYZ + Original",
                    t_offset_s=t_offset_s
                )
            except Exception as e:
                print(f"    ✗ Unified XYZ image FAILED: {e}")

            # Build per-type composite images
            try:
                n_use = n_min_norm if 'n_min_norm' in locals() else min(len(norm_signals['X']), len(norm_signals['Y']),
                                                                        len(norm_signals['Z']))
                orig_for_comp = original_resultant[:n_use]
                for img_type in ['Scalogram', 'Spectrogram', 'Kurtogram']:
                    out = compose_single_type_image(orig_for_comp, fs_used, base_output, subject, device, base,
                                                    img_type, t_offset_s=t_offset_s)
                    if out is not None:
                        folder = make_output_path(base_output, subject, device, img_type)
                        for ax in ['X', 'Y', 'Z']:
                            try:
                                (folder / f"{base}_{ax}.png").unlink()
                            except FileNotFoundError:
                                pass
            except Exception as e:
                print(f"    ✗ Per-type composite/cleanup FAILED: {e}")

    stats_counter['files_processed'] += 1
    print(f"[DONE] {file_path.name}")


# ---- Main pipeline ----

def run_pipeline(base_dir: Path):
    """Run the complete image generation pipeline."""
    base_dir = base_dir.resolve()
    raw_dir = base_dir / RAW_DATA_DIR_NAME
    output_root = base_dir / OUTPUT_ROOT_NAME
    output_root.mkdir(parents=True, exist_ok=True)

    subjects = find_subject_folders(base_dir)
    if not subjects:
        print(f"No subject folders found in {raw_dir}. Make sure folder names are Axx and CSV files are placed inside.")
        return

    print(f"Found {len(subjects)} subject folders. Processing...")
    print(f"Output directory: {output_root}")
    print(f"Sampling rate: {FS} Hz")
    print(f"Time window: {TIME_START_S}s to {TIME_END_S}s" if TIME_START_S or TIME_END_S else "Full signal")
    print("=" * 80)

    for subj in subjects:
        print(f"\n{'=' * 80}")
        print(f"Processing subject: {subj.name}")
        print(f"{'=' * 80}")

        csv_files = sorted([p for p in subj.iterdir() if p.is_file() and p.suffix.lower() == CSV_SUFFIX])
        if not csv_files:
            print(f"  No .csv files found in {subj}")
            continue

        for f in csv_files:
            try:
                process_single_file(f, output_root, fs=FS)
            except Exception as e:
                print(f"Unexpected error processing {f}: {e}")
                traceback.print_exc()

    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Files processed: {stats_counter['files_processed']}")
    print(f"\nScalogram - Success: {stats_counter['scalogram_success']}, Failed: {stats_counter['scalogram_fail']}")
    print(f"Spectrogram - Success: {stats_counter['spectrogram_success']}, Failed: {stats_counter['spectrogram_fail']}")
    print(f"Kurtogram - Success: {stats_counter['kurtogram_success']}, Failed: {stats_counter['kurtogram_fail']}")
    print(f"\nGenerated images saved under: {output_root}")
    print("=" * 80)


# ---------------- Entry point ----------------
if __name__ == '__main__':
    # Parse optional CLI arg for base dir
    base = DEFAULT_BASE_DIR
    if len(sys.argv) > 1:
        arg = Path(sys.argv[1])
        if arg.is_dir():
            base = arg.resolve()
        else:
            print(f"Argument is not a directory: {arg}. Using default: {base}")

    print("=" * 80)
    print("UpFall Image Generation Pipeline - FIXED VERSION")
    print("=" * 80)
    print("Base directory:", base)
    print()

    try:
        run_pipeline(base)
    except Exception as e:
        print(f"\nFatal error: {e}")
        traceback.print_exc()

    print("\nDone!")