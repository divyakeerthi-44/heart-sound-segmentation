#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 03_noisy_lobe_detection.py

import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# STEP 1: Compute Area Under Each Lobe
# ─────────────────────────────────────────

def compute_lobe_areas(lobes, nase):
    """
    Compute the area under each detected sound lobe by
    integrating (summing) the energy envelope within
    the lobe boundaries.

    Formula (from paper - Equation 4):
        A(Li) = sum[ Es[n] ] for n = si to ei

    Parameters:
        lobes : list     - list of lobe dicts from find_sound_lobes()
        nase  : np.array - Normalized ASE envelope

    Returns:
        lobes : list - updated lobe dicts with 'area' key added
    """
    print("\n--- Computing Lobe Areas ---")

    for i, lobe in enumerate(lobes):
        start = lobe['start_idx']
        end   = lobe['end_idx']

        # Extract energy values within lobe
        lobe_energy = nase[start:end]

        # Only sum positive values (above threshold)
        lobe_energy_positive = np.where(lobe_energy > 0, lobe_energy, 0)

        # Area = sum of energy values (discrete integration)
        area = np.sum(lobe_energy_positive)

        # Add area to lobe dict
        lobes[i]['area'] = area

    areas = [lobe['area'] for lobe in lobes]
    print(f"Total lobes         : {len(lobes)}")
    print(f"Min lobe area       : {min(areas):.4f}")
    print(f"Max lobe area       : {max(areas):.4f}")
    print(f"Mean lobe area      : {np.mean(areas):.4f}")
    print(f"Std lobe area       : {np.std(areas):.4f}")

    return lobes


# ─────────────────────────────────────────
# STEP 2: Compute Z-Scores for Each Lobe
# ─────────────────────────────────────────

def compute_zscore(lobes):
    """
    Compute z-score for each lobe area to identify
    statistical outliers (noisy lobes).

    Formula (from paper - Equations 5, 6, 7):
        A_mean = (1/M) * sum[ A(Li) ]          (Eq. 5)
        sigma  = sqrt( (1/M) * sum[(Ai-A_mean)^2] )  (Eq. 6)
        zi     = (Ai - A_mean) / sigma          (Eq. 7)

    Parameters:
        lobes : list - lobe dicts with 'area' key

    Returns:
        lobes    : list  - updated with 'zscore' key
        A_mean   : float - mean lobe area
        sigma_A  : float - standard deviation of lobe areas
    """
    print("\n--- Computing Z-Scores ---")

    M      = len(lobes)
    areas  = np.array([lobe['area'] for lobe in lobes])

    # Equation 5: Mean area
    A_mean = (1.0 / M) * np.sum(areas)

    # Equation 6: Standard deviation
    sigma_A = np.sqrt((1.0 / M) * np.sum((areas - A_mean) ** 2))

    print(f"Number of lobes (M) : {M}")
    print(f"Mean area (A_mean)  : {A_mean:.4f}")
    print(f"Std deviation       : {sigma_A:.4f}")

    # Equation 7: Z-score for each lobe
    for i, lobe in enumerate(lobes):
        if sigma_A == 0:
            z = 0.0
        else:
            z = (lobe['area'] - A_mean) / sigma_A
        lobes[i]['zscore'] = z

    zscores = [lobe['zscore'] for lobe in lobes]
    print(f"Min z-score         : {min(zscores):.4f}")
    print(f"Max z-score         : {max(zscores):.4f}")

    return lobes, A_mean, sigma_A


# ─────────────────────────────────────────
# STEP 3: Identify Noisy Lobes
# ─────────────────────────────────────────

def identify_noisy_lobes(lobes, cutoff=2.75):
    """
    Identify noisy lobes as outliers based on z-score.

    Rule (from paper):
        If z_i > cutoff → lobe is NOISY (outlier)
        cutoff = 2.75 (as set in the paper)

    Parameters:
        lobes  : list  - lobe dicts with 'zscore' key
        cutoff : float - z-score threshold (default 2.75)

    Returns:
        clean_lobes : list - lobes with zscore <= cutoff
        noisy_lobes : list - lobes with zscore > cutoff
    """
    print(f"\n--- Identifying Noisy Lobes (cutoff = {cutoff}) ---")

    clean_lobes = []
    noisy_lobes = []

    for lobe in lobes:
        lobe['is_noisy'] = lobe['zscore'] > cutoff

        if lobe['is_noisy']:
            noisy_lobes.append(lobe)
        else:
            clean_lobes.append(lobe)

    print(f"Total lobes         : {len(lobes)}")
    print(f"Clean lobes         : {len(clean_lobes)}")
    print(f"Noisy lobes         : {len(noisy_lobes)}")

    if noisy_lobes:
        print(f"\nNoisy Lobe Details:")
        print(f"{'#':<5} {'Start(s)':<12} {'End(s)':<12} "
              f"{'Area':<12} {'Z-Score':<10}")
        print("-" * 51)
        for i, lobe in enumerate(noisy_lobes):
            print(f"{i+1:<5} "
                  f"{lobe['start_time']:<12.3f} "
                  f"{lobe['end_time']:<12.3f} "
                  f"{lobe['area']:<12.4f} "
                  f"{lobe['zscore']:<10.4f}")
    else:
        print("No noisy lobes detected.")

    return clean_lobes, noisy_lobes


# ─────────────────────────────────────────
# STEP 4: Extract Clean Signal Intervals
# ─────────────────────────────────────────

def extract_clean_intervals(x_norm, lobes, noisy_lobes,
                             time_ase, sr=4000, min_duration=3.0):
    """
    Extract clean (non-noisy) intervals from the signal by
    removing the time segments corresponding to noisy lobes.
    Only intervals longer than min_duration are kept for
    segmentation (as per paper Algorithm 1).

    Parameters:
        x_norm       : np.array - normalized signal
        lobes        : list     - all lobes
        noisy_lobes  : list     - identified noisy lobes
        time_ase     : np.array - time axis of ASE
        sr           : int      - sampling rate
        min_duration : float    - minimum clean interval duration (seconds)

    Returns:
        clean_intervals : list of dicts:
                {
                  'start_sample' : int,
                  'end_sample'   : int,
                  'start_time'   : float,
                  'end_time'     : float,
                  'duration'     : float,
                  'signal'       : np.array
                }
    """
    print(f"\n--- Extracting Clean Intervals "
          f"(min duration = {min_duration}s) ---")

    total_samples = len(x_norm)
    total_time    = total_samples / sr

    # Build list of noisy time ranges
    noisy_ranges = [
        (lobe['start_time'], lobe['end_time'])
        for lobe in noisy_lobes
    ]

    # Build clean time ranges by inverting noisy ranges
    # Start with the full signal as one clean range
    clean_ranges = [(0.0, total_time)]

    # Remove each noisy range from clean ranges
    for noisy_start, noisy_end in noisy_ranges:
        updated_ranges = []
        for c_start, c_end in clean_ranges:

            # No overlap
            if noisy_end <= c_start or noisy_start >= c_end:
                updated_ranges.append((c_start, c_end))

            # Noisy range overlaps start
            elif noisy_start <= c_start and noisy_end < c_end:
                updated_ranges.append((noisy_end, c_end))

            # Noisy range overlaps end
            elif noisy_start > c_start and noisy_end >= c_end:
                updated_ranges.append((c_start, noisy_start))

            # Noisy range is in the middle
            elif noisy_start > c_start and noisy_end < c_end:
                updated_ranges.append((c_start, noisy_start))
                updated_ranges.append((noisy_end, c_end))

            # Noisy range covers entire clean range - skip

        clean_ranges = updated_ranges

    # Convert time ranges to sample intervals
    clean_intervals = []

    for c_start, c_end in clean_ranges:
        duration = c_end - c_start

        # Only keep intervals >= min_duration seconds
        if duration >= min_duration:
            start_sample = int(c_start * sr)
            end_sample   = min(int(c_end * sr), total_samples)

            interval = {
                'start_sample' : start_sample,
                'end_sample'   : end_sample,
                'start_time'   : c_start,
                'end_time'     : c_end,
                'duration'     : duration,
                'signal'       : x_norm[start_sample:end_sample]
            }
            clean_intervals.append(interval)

    print(f"Clean intervals kept (>= {min_duration}s): "
          f"{len(clean_intervals)}")

    for i, interval in enumerate(clean_intervals):
        print(f"  Interval {i+1}: "
              f"{interval['start_time']:.2f}s → "
              f"{interval['end_time']:.2f}s "
              f"({interval['duration']:.2f}s)")

    return clean_intervals


# ─────────────────────────────────────────
# STEP 5: Full Noisy Lobe Detection Pipeline
# ─────────────────────────────────────────

def detect_and_remove_noise(x_norm, lobes, nase,
                             time_ase, sr=4000,
                             cutoff=2.75, min_duration=3.0):
    """
    Full noisy lobe detection pipeline:
        Compute Areas → Z-Scores → Identify Noisy Lobes
        → Extract Clean Intervals

    Parameters:
        x_norm       : np.array - normalized signal
        lobes        : list     - sound lobes from Shannon energy
        nase         : np.array - NASE envelope
        time_ase     : np.array - time axis
        sr           : int      - sampling rate
        cutoff       : float    - z-score cutoff (default 2.75)
        min_duration : float    - min clean interval in seconds

    Returns:
        clean_lobes     : list - non-noisy lobes
        noisy_lobes     : list - detected noisy lobes
        clean_intervals : list - clean signal intervals for segmentation
    """
    print("\n====== Noisy Lobe Detection Pipeline ======")

    # Step 1: Compute areas
    lobes = compute_lobe_areas(lobes, nase)

    # Step 2: Compute z-scores
    lobes, A_mean, sigma_A = compute_zscore(lobes)

    # Step 3: Identify noisy lobes
    clean_lobes, noisy_lobes = identify_noisy_lobes(lobes, cutoff)

    # Step 4: Extract clean intervals
    clean_intervals = extract_clean_intervals(
        x_norm, lobes, noisy_lobes, time_ase, sr, min_duration
    )

    return clean_lobes, noisy_lobes, clean_intervals


# ─────────────────────────────────────────
# STEP 6: Visualization
# ─────────────────────────────────────────

def plot_noisy_lobe_detection(x_norm, nase, time_ase,
                               clean_lobes, noisy_lobes,
                               clean_intervals, sr=4000):
    """
    Visualize the noisy lobe detection results showing:
        - Normalized signal with noisy regions highlighted
        - NASE with clean vs noisy lobes colored differently
        - Clean intervals marked for segmentation
    """
    time_signal = np.arange(len(x_norm)) / sr

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.suptitle("Noisy Lobe Detection", fontsize=14, fontweight='bold')

    # ── Plot 1: Signal with noisy regions ──
    axes[0].plot(time_signal, x_norm,
                 color='steelblue', linewidth=0.6, label='Signal')

    for lobe in noisy_lobes:
        axes[0].axvspan(lobe['start_time'], lobe['end_time'],
                        alpha=0.35, color='red', label='Noisy Region')

    axes[0].set_title("Normalized Signal (Red = Noisy Regions)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc='upper right')

    # ── Plot 2: NASE with lobe classification ──
    axes[1].plot(time_ase, nase,
                 color='gray', linewidth=0.8, label='NASE')
    axes[1].axhline(y=0, color='black',
                    linestyle='--', linewidth=0.8, label='Threshold')

    for i, lobe in enumerate(clean_lobes):
        axes[1].axvspan(lobe['start_time'], lobe['end_time'],
                        alpha=0.3, color='green',
                        label='Clean Lobe' if i == 0 else "")

    for i, lobe in enumerate(noisy_lobes):
        axes[1].axvspan(lobe['start_time'], lobe['end_time'],
                        alpha=0.4, color='red',
                        label='Noisy Lobe' if i == 0 else "")

    axes[1].set_title("NASE — Green: Clean Lobes | Red: Noisy Lobes")
    axes[1].set_ylabel("Energy")
    axes[1].legend(loc='upper right')

    # ── Plot 3: Clean intervals for segmentation ──
    axes[2].plot(time_signal, x_norm,
                 color='lightgray', linewidth=0.6, label='Full Signal')

    for i, interval in enumerate(clean_intervals):
        t_start = interval['start_time']
        t_end   = interval['end_time']
        axes[2].axvspan(t_start, t_end, alpha=0.3, color='blue',
                        label='Clean Interval' if i == 0 else "")
        axes[2].annotate(
            f"Interval {i+1}\n{interval['duration']:.1f}s",
            xy=((t_start + t_end) / 2, 0.7),
            ha='center', fontsize=8, color='darkblue'
        )

    axes[2].set_title("Clean Intervals Selected for Segmentation "
                      "(Blue = Valid, >= 3s)")
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    # ── Z-score bar chart ──
    all_lobes = clean_lobes + noisy_lobes
    all_lobes_sorted = sorted(all_lobes, key=lambda x: x['start_time'])

    zscores = [lobe['zscore'] for lobe in all_lobes_sorted]
    colors  = ['red' if lobe['is_noisy'] else 'green'
               for lobe in all_lobes_sorted]

    fig2, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(len(zscores)), zscores, color=colors, edgecolor='black',
           linewidth=0.5)
    ax.axhline(y=2.75, color='red', linestyle='--',
               linewidth=1.5, label='Cutoff = 2.75')
    ax.set_title("Z-Score per Lobe (Red = Noisy, Green = Clean)")
    ax.set_xlabel("Lobe Index")
    ax.set_ylabel("Z-Score")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────

if __name__ == "__main__":

    from preprocess_01 import preprocess
    from shannon_energy_02 import compute_shannon_envelope

    FILE_PATH = "heart_sound.wav"

    # Step 1: Preprocess
    x_norm, sr, valid = preprocess(FILE_PATH)

    if valid:
        # Step 2: Shannon energy envelope
        ase, nase, time_ase, lobes = compute_shannon_envelope(x_norm, sr)

        # Step 3: Detect and remove noise
        clean_lobes, noisy_lobes, clean_intervals = detect_and_remove_noise(
            x_norm, lobes, nase, time_ase, sr
        )

        # Step 4: Visualize
        plot_noisy_lobe_detection(
            x_norm, nase, time_ase,
            clean_lobes, noisy_lobes,
            clean_intervals, sr
        )

