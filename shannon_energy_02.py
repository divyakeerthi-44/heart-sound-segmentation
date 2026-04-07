#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 02_shannon_energy.py

import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# STEP 1: Compute Average Shannon Energy (ASE)
# ─────────────────────────────────────────

def compute_ase(x_norm, sr=4000, window_ms=20, overlap=0.5):
    """
    Compute the Average Shannon Energy (ASE) envelope of the
    normalized bandpass filtered heart sound signal.

    Formula (from paper - Equation 1):
        Es = (-1/N) * sum[ x_norm(j) * log(x_norm(j)^2) ]

    Parameters:
        x_norm     : np.array - normalized bandpass filtered signal
        sr         : int      - sampling rate (default 4000 Hz)
        window_ms  : float    - window size in milliseconds (default 20ms)
        overlap    : float    - overlap between windows (default 50% = 0.5)

    Returns:
        ase        : np.array - Average Shannon Energy envelope
        time_ase   : np.array - time axis corresponding to ASE values
    """
    # Convert window size from ms to samples
    window_size = int((window_ms / 1000) * sr)   # 20ms → 80 samples at 4000Hz
    hop_size    = int(window_size * (1 - overlap)) # 50% overlap → 40 samples

    print(f"Window size   : {window_ms} ms = {window_size} samples")
    print(f"Hop size      : {hop_size} samples (50% overlap)")

    ase_values = []
    time_ase   = []

    # Slide window across the signal
    start = 0
    while start + window_size <= len(x_norm):
        window = x_norm[start : start + window_size]
        N      = len(window)

        # Avoid log(0) by replacing zeros with very small number
        window_safe = np.where(window == 0, 1e-10, window)

        # Shannon Energy formula: Es = (-1/N) * sum[ x * log(x^2) ]
        shannon_energy = (-1.0 / N) * np.sum(
            window_safe * np.log(window_safe ** 2)
        )

        ase_values.append(shannon_energy)

        # Time stamp = center of window
        center_sample = start + window_size // 2
        time_ase.append(center_sample / sr)

        start += hop_size

    ase      = np.array(ase_values)
    time_ase = np.array(time_ase)

    # Replace any NaN values with 0 (as stated in paper Algorithm 1)
    ase = np.nan_to_num(ase, nan=0.0)

    print(f"ASE computed  : {len(ase)} frames")
    return ase, time_ase


# ─────────────────────────────────────────
# STEP 2: Normalize ASE → NASE
# ─────────────────────────────────────────

def compute_nase(ase):
    """
    Normalize the Average Shannon Energy (ASE) to obtain
    the Normalized Average Shannon Energy (NASE).

    Formula (from paper - Equation 2):
        NASE(t) = Es(t) - mean(Es(t))

    The sound lobe boundaries are localized by applying
    threshold = 0 on the NASE.

    Parameters:
        ase  : np.array - Average Shannon Energy values

    Returns:
        nase : np.array - Normalized Average Shannon Energy
    """
    mean_ase = np.mean(ase)
    nase     = ase - mean_ase

    print(f"ASE mean      : {mean_ase:.6f}")
    print(f"NASE range    : [{nase.min():.4f}, {nase.max():.4f}]")
    print(f"Threshold for lobe detection: 0.0")
    return nase


# ─────────────────────────────────────────
# STEP 3: Find Sound Lobes from NASE
# ─────────────────────────────────────────

def find_sound_lobes(nase, time_ase, threshold=0.0):
    """
    Find sound lobes (contiguous segments) where NASE > threshold.
    These are candidate regions containing heart sounds (S1 or S2).

    Parameters:
        nase      : np.array - Normalized ASE
        time_ase  : np.array - time axis for NASE
        threshold : float    - threshold value (default 0.0 as per paper)

    Returns:
        lobes : list of dicts, each containing:
                {
                  'start_idx'  : int   - start index in NASE array,
                  'end_idx'    : int   - end index in NASE array,
                  'start_time' : float - start time in seconds,
                  'end_time'   : float - end time in seconds,
                  'duration'   : float - duration in seconds,
                  'energy'     : np.array - NASE values within lobe
                }
    """
    lobes      = []
    in_lobe    = False
    lobe_start = 0

    for i in range(len(nase)):
        if nase[i] > threshold and not in_lobe:
            # Lobe starts
            in_lobe    = True
            lobe_start = i

        elif (nase[i] <= threshold or i == len(nase) - 1) and in_lobe:
            # Lobe ends
            in_lobe  = False
            lobe_end = i

            lobe = {
                'start_idx'  : lobe_start,
                'end_idx'    : lobe_end,
                'start_time' : time_ase[lobe_start],
                'end_time'   : time_ase[lobe_end],
                'duration'   : time_ase[lobe_end] - time_ase[lobe_start],
                'energy'     : nase[lobe_start:lobe_end]
            }
            lobes.append(lobe)

    print(f"Total sound lobes found: {len(lobes)}")
    return lobes


# ─────────────────────────────────────────
# STEP 4: Full Shannon Energy Pipeline
# ─────────────────────────────────────────

def compute_shannon_envelope(x_norm, sr=4000, window_ms=20, overlap=0.5):
    """
    Full pipeline:
        x_norm → ASE → NASE → Sound Lobes

    Parameters:
        x_norm    : np.array - preprocessed normalized signal
        sr        : int      - sampling rate
        window_ms : float    - sliding window size in ms
        overlap   : float    - window overlap (0 to 1)

    Returns:
        ase      : np.array - Average Shannon Energy
        nase     : np.array - Normalized ASE
        time_ase : np.array - time axis for ASE/NASE
        lobes    : list     - detected sound lobes
    """
    print("\n--- Computing Shannon Energy Envelope ---")

    # Compute ASE
    ase, time_ase = compute_ase(x_norm, sr, window_ms, overlap)

    # Compute NASE
    nase = compute_nase(ase)

    # Find sound lobes
    lobes = find_sound_lobes(nase, time_ase)

    print(f"Shannon envelope pipeline complete.")
    return ase, nase, time_ase, lobes


# ─────────────────────────────────────────
# STEP 5: Visualization
# ─────────────────────────────────────────

def plot_shannon_envelope(x_norm, ase, nase, time_ase, lobes, sr=4000):
    """
    Visualize:
        - Original normalized signal
        - ASE envelope
        - NASE envelope with threshold line
        - Detected sound lobes highlighted
    """
    time_signal = np.arange(len(x_norm)) / sr

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.suptitle("Shannon Energy Envelope", fontsize=14, fontweight='bold')

    # Plot 1: Normalized Signal
    axes[0].plot(time_signal, x_norm, color='steelblue', linewidth=0.6)
    axes[0].set_title("Normalized Bandpass Filtered Signal")
    axes[0].set_ylabel("Amplitude")

    # Plot 2: ASE
    axes[1].plot(time_ase, ase, color='darkorange', linewidth=1.0)
    axes[1].set_title("Average Shannon Energy (ASE)")
    axes[1].set_ylabel("Energy")

    # Plot 3: NASE with threshold and lobes
    axes[2].plot(time_ase, nase, color='green', linewidth=1.0, label='NASE')
    axes[2].axhline(y=0, color='red', linestyle='--',
                    linewidth=1.0, label='Threshold = 0')

    # Highlight detected lobes
    for i, lobe in enumerate(lobes):
        axes[2].axvspan(
            lobe['start_time'],
            lobe['end_time'],
            alpha=0.25,
            color='purple',
            label='Sound Lobe' if i == 0 else ""
        )

    axes[2].set_title("Normalized ASE (NASE) with Detected Sound Lobes")
    axes[2].set_ylabel("Energy")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    # Summary
    print(f"\nLobe Summary:")
    print(f"{'Lobe':<6} {'Start(s)':<12} {'End(s)':<12} {'Duration(ms)':<15}")
    print("-" * 45)
    for i, lobe in enumerate(lobes):
        print(f"{i+1:<6} "
              f"{lobe['start_time']:<12.3f} "
              f"{lobe['end_time']:<12.3f} "
              f"{lobe['duration']*1000:<15.1f}")


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────

if __name__ == "__main__":

    # Import preprocessor from Code 1
    from preprocess_01 import preprocess

    FILE_PATH = "heart_sound.wav"

    # Preprocess
    x_norm, sr, valid = preprocess(FILE_PATH)

    if valid:
        # Compute Shannon envelope
        ase, nase, time_ase, lobes = compute_shannon_envelope(x_norm, sr)

        # Visualize
        plot_shannon_envelope(x_norm, ase, nase, time_ase, lobes, sr)


# In[ ]:




