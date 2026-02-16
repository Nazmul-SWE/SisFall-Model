# Shorter version
#part1_optimized.py
"""
Generate Scalogram, Spectrogram, and Kurtogram images for SisFall dataset.
Optimized for machine learning classification tasks.

Usage:
  python part1_optimized.py "path/to/project"  # or uses default path
"""

import sys
from pathlib import Path
import re
import traceback
import warnings
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pywt
from scipy import signal, stats
from PIL import Image

# ============ CONFIG ============
DEFAULT_BASE_DIR = Path(r"D:\4.1\CSE 400-A\SisFall-Model").resolve()
RAW_DATA_DIR = 'SisFall_dataset'
OUTPUT_DIR = 'Generated Images'
SUBJECT_PATTERN = re.compile(r'^(SA|SE)\d{2}$', re.IGNORECASE)

# Signal processing
FS = 200.0  # Sampling frequency (Hz)
TIME_START_S, TIME_END_S = None, None  # Full signal
SPLIT_TIMESTAMPS_S = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
SPLIT_WINDOW_S = 5.0
SPLIT_OVERLAP = 0.5

# CWT/Spectrogram/Kurtogram params
CWT_WAVELET, CWT_SCALES = 'morl', np.arange(1, 128)
STFT_NFFT, STFT_NPERSEG, STFT_NOOVERLAP = 256, 256, 128
KURTOGRAM_WINDOW = 128
KURTOGRAM_STEP = 64

# Output
DPI = 150
RESIZE_TO = None  # e.g., (224, 224) for ML models
GENERATE_COMBINED_IMAGES = False
MIN_COLS = 9
# ==================================

class SignalProcessor:
    """Centralized signal processing utilities."""

    @staticmethod
    def read_robust(fpath: Path) -> pd.DataFrame:
        """Read CSV with unknown separator (comma, space, tab)."""
        try:
            df = pd.read_csv(fpath, header=None, sep=r'[,\s]+', engine='python', comment='#')
        except:
            try:
                df = pd.read_csv(fpath, header=None, delimiter=',', engine='python', comment='#')
            except:
                df = pd.read_csv(fpath, header=None, delim_whitespace=True, engine='python', comment='#')

        # Clean semicolons and trailing commas
        try:
            df = df.map(lambda x: x.replace(';', '').strip() if isinstance(x, str) else x)
        except:
            pass
        df = df.dropna(axis=1, how='all')
        return df

    @staticmethod
    def normalize(arr: np.ndarray) -> np.ndarray:
        """Zero-mean normalization with max scaling."""
        if arr is None or arr.size == 0:
            return arr
        arr = arr.astype(float)
        arr = arr - np.nanmean(arr)
        max_abs = np.nanmax(np.abs(arr))
        return arr / max_abs if np.isfinite(max_abs) and max_abs > 0 else arr

    @staticmethod
    def slice_window(arr: np.ndarray, fs: float, t_start: Optional[float], t_end: Optional[float]) -> np.ndarray:
        """Slice array to [t_start, t_end) seconds."""
        if arr is None or arr.size == 0 or not (fs and fs > 0):
            return arr
        n = len(arr)
        start_idx = 0 if t_start is None else int(np.floor(t_start * fs))
        end_idx = n if t_end is None else int(np.ceil(t_end * fs))
        start_idx, end_idx = max(0, min(start_idx, n)), max(0, min(end_idx, n))
        return arr[start_idx:end_idx] if end_idx > start_idx else np.array([], dtype=arr.dtype)


class ImageGenerator:
    """Generate time-frequency representation images."""

    @staticmethod
    def save_fig(fig, out_path: Path, dpi=DPI, resize_to=RESIZE_TO):
        """Save figure to PNG with optional resizing."""
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        if resize_to:
            try:
                img = Image.open(out_path).resize(resize_to, Image.BILINEAR)
                img.save(out_path)
            except Exception as e:
                warnings.warn(f"Resize failed {out_path}: {e}")

    @staticmethod
    def scalogram(sig, fs, scales, wavelet, out_path: Path, title='', t_offset=0.0, vmin=None, vmax=None):
        """CWT-based scalogram."""
        coeffs, _ = pywt.cwt(sig, scales, wavelet, sampling_period=1.0/fs)
        mag = np.abs(coeffs)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        duration = (len(sig)/fs) if fs > 0 else len(sig)
        extent = [t_offset, t_offset + duration, max(scales), min(scales)]
        im = ax.imshow(mag, aspect='auto', extent=extent, vmin=vmin, vmax=vmax)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Scale')
        fig.colorbar(im, ax=ax, label='Magnitude')
        ImageGenerator.save_fig(fig, out_path)

    @staticmethod
    def spectrogram(sig, fs, nperseg, noverlap, nfft, out_path: Path, title='', t_offset=0.0, vmin=None, vmax=None):
        """STFT-based spectrogram."""
        f, t_seg, Sxx = signal.spectrogram(sig, fs=fs, window='hann', nperseg=nperseg,
                                          noverlap=noverlap, nfft=nfft, scaling='spectrum')
        Sxx_db = 10 * np.log10(Sxx + 1e-12)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        t_axis = t_seg + t_offset
        im = ax.pcolormesh(t_axis, f, Sxx_db, shading='gouraud', vmin=vmin, vmax=vmax)
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [sec]')
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label='dB')
        ImageGenerator.save_fig(fig, out_path)

    @staticmethod
    def kurtogram(sig, fs, scales, window_samples, step, out_path: Path, title='', t_offset=0.0, vmin=None, vmax=None):
        """Kurtosis-based time-frequency map from CWT."""
        coeffs, _ = pywt.cwt(sig, scales, CWT_WAVELET, sampling_period=1.0/fs)
        mag = np.abs(coeffs)
        n_scales, n_times = mag.shape

        if n_times < 1:
            raise ValueError('Empty CWT')

        ws, st = max(3, int(window_samples)), int(step)
        positions = list(range(0, max(1, n_times - ws + 1), st))
        K = np.zeros((n_scales, len(positions)))

        for si in range(n_scales):
            for pi, p in enumerate(positions):
                window = mag[si, p:p+ws]
                K[si, pi] = (stats.kurtosis(window, fisher=False, bias=False) - 3.0) if window.size >= 3 else 0.0

        time_pos = (np.array(positions) / fs) + t_offset if fs > 0 else positions

        fig, ax = plt.subplots(figsize=(8, 4.5))
        t_start, t_end = (time_pos[0], time_pos[-1]) if time_pos.size > 0 else (t_offset, t_offset + n_times/fs)
        extent = [t_start, t_end, max(scales), min(scales)]
        im = ax.imshow(K, aspect='auto', extent=extent, vmin=vmin, vmax=vmax)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Scale')
        fig.colorbar(im, ax=ax, label='Kurtosis-3')
        ImageGenerator.save_fig(fig, out_path)

    @staticmethod
    def xyz_with_original(x_norm, y_norm, z_norm, original_signal, fs, out_path: Path, title='XYZ + Original', t_offset_s=0.0):
        """
        Create a single figure that shows four subplots stacked vertically:
          1) X normalized signal
          2) Y normalized signal
          3) Z normalized signal
          4) Original (resultant) signal magnitude
        """
        n = min(len(x_norm), len(y_norm), len(z_norm), len(original_signal))
        if n < 8:
            raise ValueError("Insufficient data to plot unified image")

        x = x_norm[:n]
        
        y = y_norm[:n]
        z = z_norm[:n]
        orig = original_signal[:n]

        t = (np.arange(n) / fs + t_offset_s) if (fs and fs > 0) else np.arange(n)

        fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)

        # 1) X (top)
        axes[0].plot(t, x, color='tab:blue', linewidth=1.5)
        axes[0].set_ylabel('X', fontsize=10, fontweight='bold')
        axes[0].set_title(title, fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # 2) Y
        axes[1].plot(t, y, color='tab:orange', linewidth=1.5)
        axes[1].set_ylabel('Y', fontsize=10, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # 3) Z
        axes[2].plot(t, z, color='tab:green', linewidth=1.5)
        axes[2].set_ylabel('Z', fontsize=10, fontweight='bold')
        axes[2].grid(True, alpha=0.3)

        # 4) Original resultant magnitude (bottom)
        axes[3].plot(t, orig, color='black', linewidth=1.0)
        axes[3].set_xlabel('Time (s)' if fs and fs > 0 else 'Samples', fontsize=10)
        axes[3].set_ylabel('Resultant', fontsize=10, fontweight='bold')
        axes[3].grid(True, alpha=0.3)

        fig.tight_layout()
        ImageGenerator.save_fig(fig, out_path)


class DataProcessor:
    """Main processing pipeline."""

    DEVICES = {'Acc1': [0, 1, 2], 'Gyro': [3, 4, 5], 'Acc2': [6, 7, 8]}
    GLOBAL_BOUNDS = {'scalogram': (None, None), 'spectrogram': (None, None), 'kurtogram': (None, None)}

    @staticmethod
    def find_subjects(base_dir: Path) -> List[Path]:
        """Find subject folders (SA01, SA02, SE01, etc.)."""
        dataset_dir = base_dir / RAW_DATA_DIR
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_dir}")
        return sorted([p for p in dataset_dir.iterdir() if p.is_dir() and SUBJECT_PATTERN.match(p.name)])

    @staticmethod
    def compute_global_bounds(base_dir: Path):
        """Use hardcoded bounds to skip expensive computation entirely."""
        print("[INFO] Using predefined global scaling bounds...")
        # These are reasonable defaults that work for most datasets
        DataProcessor.GLOBAL_BOUNDS = {
            'scalogram': (0.01, 5.0),
            'spectrogram': (-120, -5),
            'kurtogram': (-2, 120)
        }
        print(f"[INFO] Scalogram: {DataProcessor.GLOBAL_BOUNDS['scalogram']}")
        print(f"[INFO] Spectrogram: {DataProcessor.GLOBAL_BOUNDS['spectrogram']}")
        print(f"[INFO] Kurtogram: {DataProcessor.GLOBAL_BOUNDS['kurtogram']}")
        print("[INFO] Global bounds loaded!")

    @staticmethod
    def build_segments(n_samples: int) -> List[Tuple[float, float]]:
        """Build time segments."""
        if not (FS and FS > 0):
            return [(0.0 if TIME_START_S is None else float(TIME_START_S),
                     float(n_samples) if TIME_END_S is None else float(TIME_END_S))]

        duration_s = n_samples / FS
        start = max(0.0, min(float(TIME_START_S) if TIME_START_S else 0.0, duration_s))
        end = max(0.0, min(float(TIME_END_S) if TIME_END_S else duration_s, duration_s))

        if end <= start:
            return []

        # Use fixed sliding window if configured
        if SPLIT_WINDOW_S and SPLIT_WINDOW_S > 0:
            w, ol = float(SPLIT_WINDOW_S), float(SPLIT_OVERLAP) if SPLIT_OVERLAP else 0.0
            ol = max(0.0, min(ol, 0.999))
            hop = w * (1.0 - ol) if w > 0 else w
            segs = []
            s = start
            while s + w <= end + 1e-12:
                if (s + w) - s > (1.0 / FS):
                    segs.append((s, s + w))
                s += hop
            return segs if segs else [(start, end)]

        # Legacy timestamp-based split
        pts = [p for p in SPLIT_TIMESTAMPS_S if start < p < end]
        points = [start] + sorted(set(pts)) + [end]
        return [(a, b) for a, b in zip(points[:-1], points[1:]) if b - a > (1.0 / FS)]

    @staticmethod
    def process_file(file_path: Path, output_root: Path):
        """Process single file and generate images."""
        fname = file_path.name
        base = file_path.stem
        cls = 'Daily Living' if fname[0].upper() == 'D' else ('Fall' if fname[0].upper() == 'F' else 'Unknown')
        subject = file_path.parent.name

        try:
            df = SignalProcessor.read_robust(file_path)
        except Exception as e:
            print(f"[ERROR] Read failed {file_path}: {e}")
            return

        if df.shape[1] < MIN_COLS:
            print(f"[SKIP] {file_path} insufficient columns")
            return

        n_samples = len(df)
        segments = DataProcessor.build_segments(n_samples)

        if not segments:
            print(f"[SKIP] {file_path} no valid segments")
            return

        for device, cols in DataProcessor.DEVICES.items():
            if max(cols) >= df.shape[1]:
                continue

            for seg_start, seg_end in segments:
                t_offset = float(seg_start) if FS > 0 else 0.0

                # Process X, Y, Z axes
                norm_sigs = {}
                raw_sigs = {}

                for i, axis in enumerate(['X', 'Y', 'Z']):
                    col_idx = cols[i]
                    raw_arr = pd.to_numeric(df.iloc[:, col_idx], errors='coerce').values
                    raw_arr = raw_arr[np.isfinite(raw_arr)]
                    raw_window = SignalProcessor.slice_window(raw_arr, FS, seg_start, seg_end)

                    if raw_window.size == 0:
                        raw_window = raw_arr

                    sig = SignalProcessor.normalize(raw_window)
                    if sig.size < 8:
                        continue

                    norm_sigs[axis] = sig
                    raw_sigs[axis] = raw_window

                    # Output paths
                    ts = f"_T{seg_start:.0f}-{seg_end:.0f}s" if segments and len(segments) > 1 else ""
                    fname_out = f"{base}_{axis}{ts}.png"

                    for img_type, gen_func in [
                        ('Scalogram', lambda p: ImageGenerator.scalogram(sig, FS, CWT_SCALES, CWT_WAVELET, p,
                                                                         f"{base}_{device}_{axis} Scalogram", t_offset,
                                                                         *DataProcessor.GLOBAL_BOUNDS['scalogram'])),
                        ('Spectrogram', lambda p: ImageGenerator.spectrogram(sig, FS, STFT_NPERSEG, STFT_NOOVERLAP,
                                                                            STFT_NFFT, p, f"{base}_{device}_{axis} Spectrogram",
                                                                            t_offset, *DataProcessor.GLOBAL_BOUNDS['spectrogram'])),
                        ('Kurtogram', lambda p: ImageGenerator.kurtogram(sig, FS, CWT_SCALES, KURTOGRAM_WINDOW,
                                                                         KURTOGRAM_STEP, p, f"{base}_{device}_{axis} Kurtogram",
                                                                         t_offset, *DataProcessor.GLOBAL_BOUNDS['kurtogram'])),
                    ]:
                        try:
                            out_folder = (output_root / device / cls / img_type / subject)
                            out_folder.mkdir(parents=True, exist_ok=True)
                            gen_func(out_folder / fname_out)
                        except Exception as e:
                            print(f"[ERROR] {img_type} {base}_{device}_{axis}: {e}")

                # Generate combined XYZ + Original image
                if len(norm_sigs) == 3 and len(raw_sigs) == 3:
                    try:
                        # Calculate resultant magnitude from raw X, Y, Z
                        x_raw = raw_sigs.get('X', np.array([]))
                        y_raw = raw_sigs.get('Y', np.array([]))
                        z_raw = raw_sigs.get('Z', np.array([]))
                        
                        if x_raw.size > 0 and y_raw.size > 0 and z_raw.size > 0:
                            # Ensure same length
                            min_len = min(len(x_raw), len(y_raw), len(z_raw))
                            x_raw = x_raw[:min_len]
                            y_raw = y_raw[:min_len]
                            z_raw = z_raw[:min_len]
                            
                            # Compute resultant (magnitude)
                            resultant = np.sqrt(x_raw**2 + y_raw**2 + z_raw**2)
                            resultant_norm = SignalProcessor.normalize(resultant)
                            
                            ts = f"_T{seg_start:.0f}-{seg_end:.0f}s" if segments and len(segments) > 1 else ""
                            fname_xyz = f"{base}_XYZ{ts}.png"
                            
                            out_folder = (output_root / device / cls / 'XYZ_Combined' / subject)
                            out_folder.mkdir(parents=True, exist_ok=True)
                            
                            ImageGenerator.xyz_with_original(
                                norm_sigs['X'], norm_sigs['Y'], norm_sigs['Z'],
                                resultant_norm, FS, out_folder / fname_xyz,
                                title=f"{base}_{device} XYZ + Resultant",
                                t_offset_s=t_offset
                            )
                    except Exception as e:
                        print(f"[ERROR] XYZ_Combined {base}_{device}: {e}")

        print(f"[OK] {base}")


def run_pipeline(base_dir: Path):
    """Main pipeline."""
    base_dir = base_dir.resolve()
    output_root = (base_dir / OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    subjects = DataProcessor.find_subjects(base_dir)
    if not subjects:
        print(f"No subjects found in {base_dir / RAW_DATA_DIR}")
        return

    print(f"Found {len(subjects)} subjects")
    DataProcessor.compute_global_bounds(base_dir)

    for subj in subjects:
        print(f"Processing: {subj.name}")
        for txt_file in sorted(subj.glob('*.txt')):
            try:
                DataProcessor.process_file(txt_file, output_root)
            except Exception as e:
                print(f"[ERROR] {txt_file}: {e}")
                traceback.print_exc()

    print(f"\n✓ Complete. Output: {output_root}")


if __name__ == '__main__':
    base = DEFAULT_BASE_DIR
    if len(sys.argv) > 1:
        arg = Path(sys.argv[1])
        base = arg if arg.is_dir() else base

    print(f"Base: {base}")
    try:
        run_pipeline(base)
    except Exception as e:
        print(f"Fatal: {e}")
        traceback.print_exc()
