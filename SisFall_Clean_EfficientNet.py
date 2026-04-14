# SisFall_Clean_EfficientNet.py
"""
SisFall Dataset  →  Scalogram / Spectrogram / Kurtogram / XYZ images
                     for EfficientNet-B0 training.

═══════════════════════════════════════════════════════════════════════════════
  CLEAN IMAGE OUTPUT (new in this version)
═══════════════════════════════════════════════════════════════════════════════

  All previously present labels, axes ticks, titles, colourbars, axis spines,
  and white padding have been removed from every image generator.
  Every output image contains ONLY the raw signal visualisation pixels —
  no text, no borders, no margins.

  What changed vs SisFall_Final.py:
    • save_fig()        — pad_inches=0, no tight-layout override
    • gen_scalogram()   — ax.axis('off'), fig fills figure exactly
    • gen_spectrogram() — ax.axis('off'), fig fills figure exactly
    • gen_kurtogram()   — ax.axis('off'), fig fills figure exactly
    • gen_xyz()         — all 4 sub-axes stripped; grid / spines / ticks off
    • RESIZE_TO         — uses Image.LANCZOS (best quality downsampling)
    • DPI / figsize     — chosen so raw canvas ≥ 224 px before resize

  All bug fixes from SisFall_Final.py are preserved unchanged
  (BUG-1 … BUG-10 + COLOUR FIX).

Usage:
    python SisFall_Clean_EfficientNet.py
    python SisFall_Clean_EfficientNet.py  path/to/project
"""

import sys, re, traceback, warnings
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pywt
from scipy import signal, stats
from PIL import Image

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG  — edit these to match your setup
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULT_BASE_DIR = Path(r"D:\4.1\CSE 400-A\SisFall-Model").resolve()
RAW_DATA_DIR     = 'SisFall_dataset'
OUTPUT_DIR       = 'Generated_Images_Clean'       # separate output folder
SUBJECT_PATTERN  = re.compile(r'^(SA|SE)\d{2}$', re.IGNORECASE)

FS             = 200.0   # SisFall sensor rate (Hz)
TIME_START_S   = None    # None = start of recording
TIME_END_S     = None    # None = end of recording
SPLIT_WINDOW_S = 5.0     # window length (s)
SPLIT_OVERLAP  = 0.5     # hop = 5 × (1−0.5) = 2.5 s

CWT_WAVELET    = 'morl'
CWT_SCALES     = np.arange(1, 128)   # 127 scales
STFT_NPERSEG   = 256
STFT_NOVERLAP  = 128
STFT_NFFT      = 256
KURT_WINDOW    = 128
KURT_STEP      = 64

# Context extension — fixes wavelet bleedback (Bugs 1, 2, 4, 5)
CONTEXT_S      = 3.0

# ── Colormaps ────────────────────────────────────────────────────────────────
CMAP_SCALOGRAM   = 'plasma'    # purple → orange → yellow
CMAP_SPECTROGRAM = 'inferno'   # black  → dark-red → orange → yellow
CMAP_KURTOGRAM   = 'coolwarm'  # blue (sub-Gauss) → white (Gauss) → red (impulsive)

# ── Output image settings ─────────────────────────────────────────────────────
#   DPI=150 + figsize (8, 4.5)  →  1200 × 675 px raw canvas
#   Resized to RESIZE_TO with LANCZOS → perfect square for EfficientNet-B0
DPI        = 150
FIG_W      = 8.0          # inches — wide enough for good time resolution
FIG_H      = 4.5          # inches — height for scalogram / spectrogram / kurtogram
FIG_H_XYZ  = 8.0          # inches — taller for the 4-panel XYZ plot
RESIZE_TO  = (224, 224)   # EfficientNet-B0 native input

MIN_COLS   = 9

# After first run copy the printed bounds here and set True to skip re-scan
USE_HARDCODED_BOUNDS = False
HARDCODED_BOUNDS = {
    'scalogram':   (0.0,   3.0),
    'spectrogram': (-80.0, 0.0),
    'kurtogram':   (-15.0, 15.0),
}
# ═══════════════════════════════════════════════════════════════════════════════

DEVICES = {'Acc1': [0, 1, 2], 'Gyro': [3, 4, 5], 'Acc2': [6, 7, 8]}
#DEVICES = {'Acc1': [0, 1, 2]}
GLOBAL_BOUNDS = {
    'scalogram':   (None, None),
    'spectrogram': (None, None),
    'kurtogram':   (None, None),
}


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def read_df(fpath: Path) -> pd.DataFrame:
    """Try three common SisFall separators; clean semicolons."""
    df = None
    for kw in [dict(sep=r'[,\s]+', engine='python'),
               dict(sep=','),
               dict(sep=r'\s+', engine='python')]:
        try:
            df = pd.read_csv(fpath, header=None, comment='#', **kw)
            break
        except Exception:
            continue
    if df is None:
        raise IOError(f"Cannot parse {fpath}")
    try:
        df = df.map(lambda x: x.replace(';', '').strip() if isinstance(x, str) else x)
    except Exception:
        pass
    return df.dropna(axis=1, how='all')


def normalize(arr: np.ndarray) -> np.ndarray:
    """
    Zero-mean + peak-normalise to [−1, +1].
    Called ONCE on the FULL signal (Bug-1 fix).
    """
    arr = arr.astype(float) - np.nanmean(arr)
    m   = np.nanmax(np.abs(arr))
    return arr / m if (np.isfinite(m) and m > 0) else arr


def get_context_slice(full_arr: np.ndarray,
                      t0: float, t1: float) -> Tuple[np.ndarray, float]:
    """
    Return (ctx_array, ctx_t0).
    Bug-2/4/5 fix: both overlapping windows draw context from the SAME
    full_arr so CWT/STFT overlap zone is computed with identical neighbours.
    """
    n      = len(full_arr)
    ctx_t0 = max(0.0, t0 - CONTEXT_S)
    ctx_t1 = min(n / FS, t1 + CONTEXT_S)
    i0     = int(np.floor(ctx_t0 * FS))
    i1     = int(np.ceil (ctx_t1 * FS))
    i0, i1 = max(0, i0), min(n, i1)
    return full_arr[i0:i1], i0 / FS


def _clean_ax(ax: plt.Axes) -> None:
    """
    Strip EVERY visual decoration from an Axes object:
      • all tick marks and tick labels
      • all axis spine lines (top / bottom / left / right)
      • axis labels
      • title
      • background patch
    The data content (imshow / pcolormesh / plot) is left untouched.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.patch.set_visible(False)   # transparent axes background


def _make_clean_fig(w: float = FIG_W,
                    h: float = FIG_H) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a figure where the single Axes fills the entire canvas —
    zero margins, zero padding, no background.

    Returns (fig, ax) ready for imshow / pcolormesh.
    """
    fig = plt.figure(figsize=(w, h), facecolor='black')
    # Axes occupies 100 % of the figure canvas
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('black')
    return fig, ax


def save_clean_fig(fig: plt.Figure, path: Path) -> None:
    """
    Save with zero padding, then resize to RESIZE_TO using Lanczos
    (best quality for downsampling — preferred over BILINEAR for CNN inputs).
    """
    fig.savefig(
        path,
        dpi=DPI,
        bbox_inches=None,    # do NOT use tight; axes already fills canvas
        pad_inches=0,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    if RESIZE_TO:
        try:
            img = Image.open(path).convert('RGB')
            img = img.resize(RESIZE_TO, Image.LANCZOS)
            img.save(path)
        except Exception as e:
            warnings.warn(f"Resize failed {path}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  IMAGE GENERATORS  (pure-signal, label-free)
# ─────────────────────────────────────────────────────────────────────────────

def gen_scalogram(full_sig: np.ndarray, t0: float, t1: float,
                  out_path: Path, title: str,
                  vmin: Optional[float], vmax: Optional[float]) -> None:
    """
    CWT scalogram — pure pixels, no axes/labels/colourbar.
    Y: scale 1 (high freq) at TOP, scale 127 (low freq) at BOTTOM.
    Colourmap: plasma (purple/beguni → orange → yellow).
    """
    # ── compute ──────────────────────────────────────────────────────────────
    ctx, ctx_t0 = get_context_slice(full_sig, t0, t1)
    coeffs, _   = pywt.cwt(ctx, CWT_SCALES, CWT_WAVELET, sampling_period=1.0 / FS)
    mag         = np.abs(coeffs)

    t_ctx    = ctx_t0 + np.arange(mag.shape[1]) / FS
    col_mask = (t_ctx >= t0 - 0.5 / FS) & (t_ctx <= t1 + 0.5 / FS)
    mag_win  = mag[:, col_mask]

    # ── render ────────────────────────────────────────────────────────────────
    fig, ax = _make_clean_fig()
    ax.imshow(
        mag_win,
        aspect='auto',
        origin='upper',
        extent=[t0, t1, float(CWT_SCALES[-1]), float(CWT_SCALES[0])],
        vmin=vmin, vmax=vmax,
        cmap=CMAP_SCALOGRAM,
        interpolation='bilinear',   # smooth for CNN
    )
    ax.set_xlim(t0, t1)
    _clean_ax(ax)
    save_clean_fig(fig, out_path)


def gen_spectrogram(full_sig: np.ndarray, t0: float, t1: float,
                    out_path: Path, title: str,
                    vmin: Optional[float], vmax: Optional[float]) -> None:
    """
    STFT spectrogram — pure pixels, no axes/labels/colourbar.
    Context provides real signal on both sides (Bug-4/5 fix).
    Colourmap: inferno (black → dark-red → orange → yellow).
    """
    # ── compute ──────────────────────────────────────────────────────────────
    ctx, ctx_t0 = get_context_slice(full_sig, t0, t1)
    f, t_seg, Sxx = signal.spectrogram(
        ctx, fs=FS, window='hann',
        nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP,
        nfft=STFT_NFFT, scaling='spectrum')
    Sxx_db = 10.0 * np.log10(Sxx + 1e-12)
    t_abs  = t_seg + ctx_t0

    mask  = (t_abs >= t0 - 1e-9) & (t_abs <= t1 + 1e-9)
    t_win = t_abs[mask]
    S_win = Sxx_db[:, mask]

    # Pin left / right edges (Bug-4/5 fix)
    if t_win.size > 0 and t_win[0] > t0 + 1e-9:
        t_win = np.insert(t_win, 0, t0)
        S_win = np.hstack([S_win[:, [0]], S_win])
    if t_win.size > 0 and t_win[-1] < t1 - 1e-9:
        t_win = np.append(t_win, t1)
        S_win = np.hstack([S_win, S_win[:, [-1]]])
    if t_win.size < 2:
        t_win = np.array([t0, t1])
        S_win = np.hstack([S_win, S_win]) if S_win.size else np.zeros((len(f), 2))

    # ── render ────────────────────────────────────────────────────────────────
    fig, ax = _make_clean_fig()
    ax.pcolormesh(
        t_win, f, S_win,
        shading='gouraud',
        vmin=vmin, vmax=vmax,
        cmap=CMAP_SPECTROGRAM,
    )
    ax.set_xlim(t0, t1)
    ax.set_ylim(0, FS / 2)
    _clean_ax(ax)
    save_clean_fig(fig, out_path)


def gen_kurtogram(full_sig: np.ndarray, t0: float, t1: float,
                  out_path: Path, title: str,
                  vmin: Optional[float], vmax: Optional[float]) -> None:
    """
    CWT-based kurtogram — pure pixels, no axes/labels/colourbar.
    Y matches scalogram (scale 1 at TOP — Bug-7 fix; no invert_yaxis).
    extent right = t1 (Bug-6 fix).
    Colourmap: coolwarm centred at 0.
    """
    # ── compute ──────────────────────────────────────────────────────────────
    ctx, ctx_t0 = get_context_slice(full_sig, t0, t1)
    coeffs, _   = pywt.cwt(ctx, CWT_SCALES, CWT_WAVELET, sampling_period=1.0 / FS)
    mag         = np.abs(coeffs)

    t_ctx    = ctx_t0 + np.arange(mag.shape[1]) / FS
    col_mask = (t_ctx >= t0 - 0.5 / FS) & (t_ctx <= t1 + 0.5 / FS)
    mag_win  = mag[:, col_mask]

    n_scales, n_times = mag_win.shape
    ws        = max(3, KURT_WINDOW)
    positions = list(range(0, max(1, n_times - ws + 1), KURT_STEP))
    if not positions:
        positions = [0]

    K = np.zeros((n_scales, len(positions)))
    for si in range(n_scales):
        for pi, p in enumerate(positions):
            w = mag_win[si, p: p + ws]
            K[si, pi] = (stats.kurtosis(w, fisher=False, bias=False) - 3.0
                         if w.size >= 3 else 0.0)

    # ── render ────────────────────────────────────────────────────────────────
    fig, ax = _make_clean_fig()
    ax.imshow(
        K,
        aspect='auto',
        origin='upper',
        extent=[t0, t1, float(CWT_SCALES[-1]), float(CWT_SCALES[0])],
        vmin=vmin, vmax=vmax,
        cmap=CMAP_KURTOGRAM,
        interpolation='bilinear',
    )
    ax.set_xlim(t0, t1)
    _clean_ax(ax)
    save_clean_fig(fig, out_path)


def gen_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray,
            resultant: np.ndarray, t0: float,
            out_path: Path, title: str) -> None:
    """
    4-panel XYZ + Resultant signal plot — pure pixels, no axes/labels/grid.
    Each channel occupies an equal vertical strip; no whitespace between panels.
    """
    n = min(len(x), len(y), len(z), len(resultant))
    if n < 8:
        raise ValueError("Too short for XYZ plot")
    t_vec = np.arange(n) / FS + t0

    # 4 sub-axes stacked vertically with zero spacing
    fig = plt.figure(figsize=(FIG_W, FIG_H_XYZ), facecolor='black')
    n_panels = 4
    panel_h  = 1.0 / n_panels  # fraction of figure height per panel

    channels = [
        (x[:n],           'tab:blue'),
        (y[:n],           'tab:orange'),
        (z[:n],           'tab:green'),
        (resultant[:n],   'white'),
    ]

    for idx, (data, color) in enumerate(channels):
        # Each Axes fills its exact strip of the figure — no gap
        bottom = 1.0 - (idx + 1) * panel_h
        ax = fig.add_axes([0, bottom, 1, panel_h])
        ax.set_facecolor('black')

        ax.plot(t_vec, data, color=color, linewidth=1.0)
        ax.set_xlim(t0, t0 + n / FS)
        ax.set_ylim(-1.25, 1.25)
        _clean_ax(ax)

    save_clean_fig(fig, out_path)


# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL COLOUR BOUNDS  (Bug-8 fix — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def compute_global_bounds(base_dir: Path) -> None:
    """
    Compute vmin/vmax for each image type from a representative dataset sample.
    For every subject: up to 2 Fall + 2 Daily-Living files.
    Kurtogram bounds are CENTRED at 0 after aggregation.
    """
    global GLOBAL_BOUNDS
    if USE_HARDCODED_BOUNDS:
        GLOBAL_BOUNDS = dict(HARDCODED_BOUNDS)
        print("[INFO] Using hardcoded bounds.")
        return

    print("[INFO] Scanning dataset for global colour bounds …")
    sv, spv, kv = [], [], []

    subjects = sorted([p for p in (base_dir / RAW_DATA_DIR).iterdir()
                       if p.is_dir() and SUBJECT_PATTERN.match(p.name)])

    for subj in subjects:
        all_files = sorted(subj.glob('*.txt'))
        falls = [f for f in all_files if f.name[0].upper() == 'F'][:2]
        adls  = [f for f in all_files if f.name[0].upper() == 'D'][:2]
        sample = falls + adls or all_files[:4]

        for fpath in sample:
            try:
                df = read_df(fpath)
                if df.shape[1] < MIN_COLS:
                    continue
                for cols in DEVICES.values():
                    if max(cols) >= df.shape[1]:
                        continue
                    for ci in cols:
                        raw = pd.to_numeric(df.iloc[:, ci],
                                            errors='coerce').dropna().values
                        if raw.size < STFT_NPERSEG * 2:
                            continue
                        s = normalize(raw)

                        # Scalogram
                        c, _ = pywt.cwt(s, CWT_SCALES, CWT_WAVELET,
                                        sampling_period=1.0 / FS)
                        mag = np.abs(c)
                        sv.append((float(np.nanpercentile(mag, 1)),
                                   float(np.nanpercentile(mag, 99))))

                        # Spectrogram
                        _, _, Sxx = signal.spectrogram(
                            s, fs=FS, nperseg=STFT_NPERSEG,
                            noverlap=STFT_NOVERLAP, nfft=STFT_NFFT,
                            scaling='spectrum')
                        db = 10.0 * np.log10(Sxx + 1e-12)
                        spv.append((float(np.nanpercentile(db, 1)),
                                    float(np.nanpercentile(db, 99))))

                        # Kurtogram
                        n_t = mag.shape[1]
                        ws  = max(3, KURT_WINDOW)
                        pos = list(range(0, max(1, n_t - ws + 1), KURT_STEP))
                        K_flat = [
                            stats.kurtosis(mag[si, p: p + ws],
                                           fisher=False, bias=False) - 3.0
                            for si in range(mag.shape[0])
                            for p in pos
                            if mag[si, p: p + ws].size >= 3
                        ]
                        if K_flat:
                            kv.append((float(np.percentile(K_flat, 1)),
                                       float(np.percentile(K_flat, 99))))

            except Exception as e:
                warnings.warn(f"[BOUNDS] {fpath.name}: {e}")

    def agg(pairs, lo=2, hi=98):
        if not pairs:
            return (None, None)
        return (float(np.percentile([p[0] for p in pairs], lo)),
                float(np.percentile([p[1] for p in pairs], hi)))

    s_lo,  s_hi  = agg(sv)
    sp_lo, sp_hi = agg(spv)
    k_lo,  k_hi  = agg(kv)

    if k_lo is not None and k_hi is not None:
        k_abs = max(abs(k_lo), abs(k_hi))
        k_lo, k_hi = -k_abs, k_abs

    GLOBAL_BOUNDS = {
        'scalogram':   (s_lo,  s_hi),
        'spectrogram': (sp_lo, sp_hi),
        'kurtogram':   (k_lo,  k_hi),
    }
    for name, (lo, hi) in GLOBAL_BOUNDS.items():
        print(f"  {name:12s}:  vmin={lo:.4f}   vmax={hi:.4f}")
    print("[TIP] Copy above into HARDCODED_BOUNDS and set USE_HARDCODED_BOUNDS=True "
          "to skip this scan on future runs.")


# ─────────────────────────────────────────────────────────────────────────────
#  SEGMENTATION  (Bug-10 fix — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def build_segments(n_samples: int) -> List[Tuple[float, float]]:
    """
    Sliding window — emits ONLY full-length windows (Bug-10 fix).
    """
    dur   = n_samples / FS
    start = float(TIME_START_S) if TIME_START_S is not None else 0.0
    end   = min(float(TIME_END_S), dur) if TIME_END_S is not None else dur
    if end <= start:
        return []
    w   = float(SPLIT_WINDOW_S)
    hop = w * (1.0 - float(SPLIT_OVERLAP))
    segs, s = [], start
    while s + w <= end + 1e-9:
        seg_end = round(s + w, 6)
        seg_sta = round(s, 6)
        if seg_end - seg_sta >= w - 1e-9:
            segs.append((seg_sta, seg_end))
        s += hop
    return segs or []


# ─────────────────────────────────────────────────────────────────────────────
#  FILE PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

def process_file(file_path: Path, output_root: Path) -> None:
    fname   = file_path.name
    base    = file_path.stem
    cls     = ('Fall'         if fname[0].upper() == 'F' else
               'Daily Living' if fname[0].upper() == 'D' else
               'Unknown')
    subject = file_path.parent.name

    try:
        df = read_df(file_path)
    except Exception as e:
        print(f"[ERROR] Read {file_path}: {e}"); return
    if df.shape[1] < MIN_COLS:
        print(f"[SKIP]  {file_path.name}: only {df.shape[1]} columns"); return

    segments = build_segments(len(df))
    if not segments:
        print(f"[SKIP]  {file_path.name}: no full-length segments"); return

    for device, cols in DEVICES.items():
        if max(cols) >= df.shape[1]:
            continue

        # Bug-1 fix: normalise FULL signal once per axis
        full_norm: dict = {}
        for i, axis in enumerate('XYZ'):
            raw = pd.to_numeric(df.iloc[:, cols[i]], errors='coerce').values
            raw = raw[np.isfinite(raw)]
            if raw.size < 8:
                continue
            full_norm[axis] = normalize(raw)

        if not full_norm:
            continue

        for t0, t1 in segments:
            ts = f"_T{t0:.1f}-{t1:.1f}s" if len(segments) > 1 else ""

            for axis, fn_arr in full_norm.items():
                i0 = max(0, int(np.floor(t0 * FS)))
                i1 = min(len(fn_arr), int(np.ceil(t1 * FS)))
                if (i1 - i0) < 8:
                    continue

                pfx = f"{base}_{device}_{axis}"
                sb, sp, sk = (GLOBAL_BOUNDS['scalogram'],
                              GLOBAL_BOUNDS['spectrogram'],
                              GLOBAL_BOUNDS['kurtogram'])

                # Bug-3 fix: pass fn_arr directly — no lambda, no closure
                for img_type, gen_fn, bounds in [
                    ('Scalogram',   gen_scalogram,   sb),
                    ('Spectrogram', gen_spectrogram, sp),
                    ('Kurtogram',   gen_kurtogram,   sk),
                ]:
                    try:
                        folder = output_root / device / cls / img_type / subject
                        folder.mkdir(parents=True, exist_ok=True)
                        gen_fn(fn_arr, t0, t1,
                               folder / f"{pfx}{ts}.png",
                               "",        # title arg kept for API compat; ignored
                               bounds[0], bounds[1])
                    except Exception as e:
                        print(f"[ERROR] {img_type} {pfx} [{t0:.1f}-{t1:.1f}s]: {e}")

            # XYZ combined plot
            if len(full_norm) == 3:
                try:
                    wnorm = {ax: fn[max(0, int(np.floor(t0*FS))):
                                    min(len(fn), int(np.ceil(t1*FS)))]
                             for ax, fn in full_norm.items()}
                    ml  = min(len(wnorm[a]) for a in 'XYZ')
                    res = normalize(np.sqrt(sum(wnorm[a][:ml]**2 for a in 'XYZ')))
                    folder = output_root / device / cls / 'XYZ_Combined' / subject
                    folder.mkdir(parents=True, exist_ok=True)
                    gen_xyz(
                        wnorm['X'], wnorm['Y'], wnorm['Z'], res, t0,
                        folder / f"{base}_{device}_XYZ{ts}.png",
                        "",   # title arg kept for API compat; ignored
                    )
                except Exception as e:
                    print(f"[ERROR] XYZ {base}_{device} [{t0:.1f}-{t1:.1f}s]: {e}")

    print(f"[OK]   {base}")


# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(base_dir: Path) -> None:
    base_dir = base_dir.resolve()
    out_root = base_dir / OUTPUT_DIR
    out_root.mkdir(parents=True, exist_ok=True)

    subjects = sorted([p for p in (base_dir / RAW_DATA_DIR).iterdir()
                       if p.is_dir() and SUBJECT_PATTERN.match(p.name)])
    if not subjects:
        print(f"[ERROR] No subject folders in {base_dir / RAW_DATA_DIR}"); return

    print(f"Found {len(subjects)} subjects.")
    compute_global_bounds(base_dir)

    for subj in subjects:
        print(f"  Subject: {subj.name}")
        for fpath in sorted(subj.glob('*.txt')):
            try:
                process_file(fpath, out_root)
            except Exception as e:
                print(f"[ERROR] {fpath.name}: {e}")
                traceback.print_exc()

    print(f"\n✓  Done.  Output: {out_root}")


if __name__ == '__main__':
    base = DEFAULT_BASE_DIR
    if len(sys.argv) > 1:
        arg = Path(sys.argv[1])
        if arg.is_dir():
            base = arg
    print(f"Base directory: {base}\n")
    try:
        run_pipeline(base)
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
