#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 05_s1_s2_identification.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

# ─────────────────────────────────────────
# STEP 1: Compute Envelope of a Segment
# ─────────────────────────────────────────

def compute_segment_envelope(x_norm, start_time, end_time,
                               sr=4000, window_ms=20, overlap=0.5):
    """
    Compute the Average Shannon Energy envelope of a
    specific segment of the signal.
    Used for correlation-based S1/S2 matching.

    Parameters:
        x_norm     : np.array - normalized signal
        start_time : float    - segment start in seconds
        end_time   : float    - segment end in seconds
        sr         : int      - sampling rate
        window_ms  : float    - window size in ms
        overlap    : float    - window overlap

    Returns:
        envelope : np.array - ASE envelope of the segment
    """
    start_sample = int(start_time * sr)
    end_sample   = int(end_time   * sr)

    # Clamp to signal length
    start_sample = max(0, start_sample)
    end_sample   = min(len(x_norm), end_sample)

    segment     = x_norm[start_sample:end_sample]
    window_size = int((window_ms / 1000) * sr)
    hop_size    = int(window_size * (1 - overlap))

    envelope = []
    start    = 0

    while start + window_size <= len(segment):
        window      = segment[start: start + window_size]
        N           = len(window)
        window_safe = np.where(window == 0, 1e-10, window)
        ase_val     = (-1.0 / N) * np.sum(
            window_safe * np.log(window_safe ** 2)
        )
        envelope.append(ase_val)
        start += hop_size

    envelope = np.array(envelope)
    envelope = np.nan_to_num(envelope, nan=0.0)

    return envelope


# ─────────────────────────────────────────
# STEP 2: Compute Envelope Correlation
# ─────────────────────────────────────────

def compute_envelope_correlation(env1, env2):
    """
    Compute normalized cross-correlation between two
    Shannon energy envelopes.

    Key reason (from paper):
        Correlation is computed on the ENVELOPE instead of
        the original signal because heart sounds of two
        adjacent cycles can have slightly different
        frequencies, making direct signal correlation
        less reliable.

    Parameters:
        env1 : np.array - envelope of first segment
        env2 : np.array - envelope of second segment

    Returns:
        max_corr : float - maximum normalized correlation value
    """
    if len(env1) == 0 or len(env2) == 0:
        return 0.0

    # Normalize both envelopes to zero mean unit variance
    def normalize_env(env):
        std = np.std(env)
        if std == 0:
            return env - np.mean(env)
        return (env - np.mean(env)) / std

    env1_norm = normalize_env(env1)
    env2_norm = normalize_env(env2)

    # Pad shorter envelope to match longer one
    max_len = max(len(env1_norm), len(env2_norm))
    env1_pad = np.pad(env1_norm,
                      (0, max_len - len(env1_norm)))
    env2_pad = np.pad(env2_norm,
                      (0, max_len - len(env2_norm)))

    # Cross-correlation
    correlation = correlate(env1_pad, env2_pad, mode='full')
    max_corr    = np.max(np.abs(correlation)) / max_len

    return float(max_corr)


# ─────────────────────────────────────────
# STEP 3: Find Initial S1-S2 Pair
# ─────────────────────────────────────────

def find_initial_s1s2_pair(validated_lobes, intervals,
                            cardiac_cycle_s):
    """
    Find the first reliable S1-S2 pair to use as a
    reference for the rest of the recording.

    Strategy (from paper):
        Starting point = the sound pair with the LONGEST
        interval between them, since the diastolic interval
        (S2→S1) is generally greater than the systolic
        interval (S1→S2).

    Parameters:
        validated_lobes : list  - validated candidate lobes
        intervals       : list  - inter-lobe intervals
        cardiac_cycle_s : float - estimated cardiac cycle

    Returns:
        s1_idx : int   - index of S1 in validated_lobes
        s2_idx : int   - index of S2 in validated_lobes
        systolic_interval : float - S1→S2 interval in seconds
    """
    print(f"\n--- Finding Initial S1-S2 Pair ---")

    if len(validated_lobes) < 2 or len(intervals) < 1:
        print("Not enough lobes to find S1-S2 pair.")
        return None, None, None

    # Find the longest interval
    max_interval_idx = int(np.argmax(intervals))
    max_interval     = intervals[max_interval_idx]

    print(f"Longest interval    : {max_interval*1000:.1f}ms "
          f"at index {max_interval_idx}")

    # The pair BEFORE the longest interval is S1-S2
    # (since S2→S1 is the longest = diastole)
    # So we look for the pair just before the longest gap

    # The pair just before the longest gap
    if max_interval_idx >= 1:
        s1_idx = max_interval_idx - 1
        s2_idx = max_interval_idx
    else:
        s1_idx = 0
        s2_idx = 1

    s1 = validated_lobes[s1_idx]
    s2 = validated_lobes[s2_idx]

    # Systolic interval = time from end of S1 to start of S2
    systolic_interval = s2['start_time'] - s1['end_time']

    print(f"Initial S1          : {s1['start_time']:.3f}s → "
          f"{s1['end_time']:.3f}s")
    print(f"Initial S2          : {s2['start_time']:.3f}s → "
          f"{s2['end_time']:.3f}s")
    print(f"Systolic interval   : {systolic_interval*1000:.1f}ms")

    return s1_idx, s2_idx, systolic_interval


# ─────────────────────────────────────────
# STEP 4: Score a Candidate S1-S2 Pair
# ─────────────────────────────────────────

def score_candidate_pair(candidate_s1, candidate_s2,
                          ref_s1_env, ref_s2_env,
                          ref_systolic_interval,
                          ref_cardiac_cycle,
                          x_norm, sr=4000,
                          w_corr=0.5,
                          w_systolic=0.3,
                          w_cycle=0.2):
    """
    Score a candidate S1-S2 pair against the reference pair
    using three measurements (from paper):
        1. Correlation of envelope of systolic interval
        2. Cardiac cycle length match
        3. Systolic interval match

    Parameters:
        candidate_s1           : dict  - candidate S1 lobe
        candidate_s2           : dict  - candidate S2 lobe
        ref_s1_env             : array - reference S1 envelope
        ref_s2_env             : array - reference S2 envelope
        ref_systolic_interval  : float - reference systolic interval
        ref_cardiac_cycle      : float - reference cardiac cycle
        x_norm                 : array - normalized signal
        sr                     : int   - sampling rate
        w_corr/w_systolic/w_cycle : float - scoring weights

    Returns:
        score          : float - combined match score (higher = better)
        score_details  : dict  - breakdown of individual scores
    """
    # ── Measurement 1: Envelope Correlation ──
    cand_s1_env = compute_segment_envelope(
        x_norm,
        candidate_s1['start_time'],
        candidate_s1['end_time'],
        sr
    )
    cand_s2_env = compute_segment_envelope(
        x_norm,
        candidate_s2['start_time'],
        candidate_s2['end_time'],
        sr
    )

    corr_s1 = compute_envelope_correlation(ref_s1_env, cand_s1_env)
    corr_s2 = compute_envelope_correlation(ref_s2_env, cand_s2_env)
    corr_score = (corr_s1 + corr_s2) / 2.0

    # ── Measurement 2: Systolic Interval Match ──
    cand_systolic = (candidate_s2['start_time'] -
                     candidate_s1['end_time'])

    if ref_systolic_interval > 0:
        systolic_diff  = abs(cand_systolic - ref_systolic_interval)
        systolic_score = max(0, 1 - systolic_diff /
                             ref_systolic_interval)
    else:
        systolic_score = 0.0

    # ── Measurement 3: Cardiac Cycle Length Match ──
    cand_cycle = (
        (candidate_s1['start_time'] + candidate_s1['end_time']) / 2
        - 0  # placeholder for previous S1 midpoint
    )

    if ref_cardiac_cycle and ref_cardiac_cycle > 0:
        cand_cycle_length = (
            (candidate_s1['start_time'] + candidate_s1['end_time']) / 2
            - (candidate_s1['start_time'])
        )
        # Use systolic proportion as proxy
        if cand_systolic > 0 and ref_cardiac_cycle > 0:
            systolic_ratio      = cand_systolic / ref_cardiac_cycle
            ref_systolic_ratio  = (ref_systolic_interval /
                                   ref_cardiac_cycle)
            cycle_score = max(0, 1 - abs(
                systolic_ratio - ref_systolic_ratio
            ))
        else:
            cycle_score = 0.5
    else:
        cycle_score = 0.5

    # ── Combined Score ──
    score = (w_corr     * corr_score +
             w_systolic * systolic_score +
             w_cycle    * cycle_score)

    score_details = {
        'corr_s1'        : corr_s1,
        'corr_s2'        : corr_s2,
        'corr_score'     : corr_score,
        'systolic_score' : systolic_score,
        'cycle_score'    : cycle_score,
        'total_score'    : score
    }

    return score, score_details


# ─────────────────────────────────────────
# STEP 5: Propagate Forward
# ─────────────────────────────────────────

def propagate_forward(validated_lobes, start_s1_idx,
                       start_s2_idx, ref_systolic_interval,
                       ref_cardiac_cycle, x_norm,
                       sr=4000):
    """
    Starting from the initial S1-S2 pair, search forward
    through all remaining lobes to identify S1-S2 pairs.

    For each next position, all possible combinations of
    candidate lobes are assessed and the best matching
    pair is selected.

    Parameters:
        validated_lobes       : list  - all validated lobes
        start_s1_idx          : int   - starting S1 index
        start_s2_idx          : int   - starting S2 index
        ref_systolic_interval : float - reference systolic interval
        ref_cardiac_cycle     : float - reference cardiac cycle
        x_norm                : array - normalized signal
        sr                    : int   - sampling rate

    Returns:
        s1_sounds : list - identified S1 lobe dicts
        s2_sounds : list - identified S2 lobe dicts
    """
    print(f"\n--- Propagating Forward ---")

    s1_sounds = []
    s2_sounds = []

    # Add initial pair
    s1_sounds.append(validated_lobes[start_s1_idx])
    s2_sounds.append(validated_lobes[start_s2_idx])

    # Reference envelopes from initial pair
    ref_s1_env = compute_segment_envelope(
        x_norm,
        validated_lobes[start_s1_idx]['start_time'],
        validated_lobes[start_s1_idx]['end_time'],
        sr
    )
    ref_s2_env = compute_segment_envelope(
        x_norm,
        validated_lobes[start_s2_idx]['start_time'],
        validated_lobes[start_s2_idx]['end_time'],
        sr
    )

    # Search forward from start_s2_idx + 1
    current_idx = start_s2_idx + 1

    while current_idx < len(validated_lobes) - 1:

        best_score   = -1
        best_s1_idx  = None
        best_s2_idx  = None

        # Try all combinations of next two lobes as S1-S2
        for i in range(current_idx,
                       min(current_idx + 3,
                           len(validated_lobes) - 1)):
            for j in range(i + 1,
                           min(i + 3,
                               len(validated_lobes))):

                cand_s1 = validated_lobes[i]
                cand_s2 = validated_lobes[j]

                # S2 must come after S1
                if cand_s2['start_time'] <= cand_s1['end_time']:
                    continue

                score, _ = score_candidate_pair(
                    cand_s1, cand_s2,
                    ref_s1_env, ref_s2_env,
                    ref_systolic_interval,
                    ref_cardiac_cycle,
                    x_norm, sr
                )

                if score > best_score:
                    best_score  = score
                    best_s1_idx = i
                    best_s2_idx = j

        if best_s1_idx is not None:
            new_s1 = validated_lobes[best_s1_idx]
            new_s2 = validated_lobes[best_s2_idx]

            # Avoid duplicate entries
            if new_s1 not in s1_sounds:
                s1_sounds.append(new_s1)
            if new_s2 not in s2_sounds:
                s2_sounds.append(new_s2)

            # Update reference envelopes with running average
            new_s1_env = compute_segment_envelope(
                x_norm,
                new_s1['start_time'],
                new_s1['end_time'], sr
            )
            new_s2_env = compute_segment_envelope(
                x_norm,
                new_s2['start_time'],
                new_s2['end_time'], sr
            )

            if len(new_s1_env) == len(ref_s1_env):
                ref_s1_env = (ref_s1_env + new_s1_env) / 2
            if len(new_s2_env) == len(ref_s2_env):
                ref_s2_env = (ref_s2_env + new_s2_env) / 2

            current_idx = best_s2_idx + 1
        else:
            current_idx += 1

    print(f"Forward S1 found    : {len(s1_sounds)}")
    print(f"Forward S2 found    : {len(s2_sounds)}")

    return s1_sounds, s2_sounds


# ─────────────────────────────────────────
# STEP 6: Propagate Backward
# ─────────────────────────────────────────

def propagate_backward(validated_lobes, start_s1_idx,
                        ref_systolic_interval,
                        ref_cardiac_cycle,
                        x_norm, sr=4000):
    """
    Starting from the initial S1 pair, search backward
    through all preceding lobes to identify more S1-S2 pairs.

    Parameters:
        validated_lobes       : list  - all validated lobes
        start_s1_idx          : int   - starting S1 index
        ref_systolic_interval : float - reference systolic interval
        ref_cardiac_cycle     : float - reference cardiac cycle
        x_norm                : array - normalized signal
        sr                    : int   - sampling rate

    Returns:
        s1_backward : list - S1 lobes found going backward
        s2_backward : list - S2 lobes found going backward
    """
    print(f"\n--- Propagating Backward ---")

    s1_backward = []
    s2_backward = []

    # Reference envelopes
    ref_s1_env = compute_segment_envelope(
        x_norm,
        validated_lobes[start_s1_idx]['start_time'],
        validated_lobes[start_s1_idx]['end_time'],
        sr
    )

    current_idx = start_s1_idx - 2

    while current_idx >= 1:

        best_score  = -1
        best_s1_idx = None
        best_s2_idx = None

        for i in range(max(0, current_idx - 2),
                       current_idx + 1):
            for j in range(i + 1,
                           min(i + 3,
                               current_idx + 2)):

                if j >= len(validated_lobes):
                    continue

                cand_s1 = validated_lobes[i]
                cand_s2 = validated_lobes[j]

                if cand_s2['start_time'] <= cand_s1['end_time']:
                    continue

                score, _ = score_candidate_pair(
                    cand_s1, cand_s2,
                    ref_s1_env, ref_s1_env,
                    ref_systolic_interval,
                    ref_cardiac_cycle,
                    x_norm, sr
                )

                if score > best_score:
                    best_score  = score
                    best_s1_idx = i
                    best_s2_idx = j

        if best_s1_idx is not None:
            new_s1 = validated_lobes[best_s1_idx]
            new_s2 = validated_lobes[best_s2_idx]

            s1_backward.insert(0, new_s1)
            s2_backward.insert(0, new_s2)

            # Update reference envelope
            new_s1_env = compute_segment_envelope(
                x_norm,
                new_s1['start_time'],
                new_s1['end_time'], sr
            )
            if len(new_s1_env) == len(ref_s1_env):
                ref_s1_env = (ref_s1_env + new_s1_env) / 2

            current_idx = best_s1_idx - 2
        else:
            current_idx -= 1

    print(f"Backward S1 found   : {len(s1_backward)}")
    print(f"Backward S2 found   : {len(s2_backward)}")

    return s1_backward, s2_backward


# ─────────────────────────────────────────
# STEP 7: Full S1/S2 Identification Pipeline
# ─────────────────────────────────────────

def identify_s1_s2(validated_lobes, intervals,
                    cardiac_cycle_s, x_norm, sr=4000):
    """
    Full S1/S2 identification pipeline:
        Find initial pair → Propagate forward
        → Propagate backward → Merge results

    Parameters:
        validated_lobes : list     - validated candidate lobes
        intervals       : list     - inter-lobe intervals
        cardiac_cycle_s : float    - estimated cardiac cycle
        x_norm          : np.array - normalized signal
        sr              : int      - sampling rate

    Returns:
        s1_sounds : list - all identified S1 lobe dicts
        s2_sounds : list - all identified S2 lobe dicts
    """
    print("\n====== S1/S2 Identification Pipeline ======")

    # Step 1: Find initial pair
    s1_idx, s2_idx, systolic_interval = find_initial_s1s2_pair(
        validated_lobes, intervals, cardiac_cycle_s
    )

    if s1_idx is None:
        print("Could not find initial S1-S2 pair.")
        return [], []

    # Step 2: Propagate forward
    s1_fwd, s2_fwd = propagate_forward(
        validated_lobes, s1_idx, s2_idx,
        systolic_interval, cardiac_cycle_s,
        x_norm, sr
    )

    # Step 3: Propagate backward
    s1_bwd, s2_bwd = propagate_backward(
        validated_lobes, s1_idx,
        systolic_interval, cardiac_cycle_s,
        x_norm, sr
    )

    # Step 4: Merge backward + forward results
    # Remove duplicates by checking start times
    def merge_unique(list1, list2):
        merged     = list1 + list2
        seen_times = set()
        unique     = []
        for lobe in merged:
            key = round(lobe['start_time'], 4)
            if key not in seen_times:
                seen_times.add(key)
                unique.append(lobe)
        # Sort by time
        unique.sort(key=lambda x: x['start_time'])
        return unique

    s1_sounds = merge_unique(s1_bwd, s1_fwd)
    s2_sounds = merge_unique(s2_bwd, s2_fwd)

    print(f"\nFinal Results:")
    print(f"Total S1 identified : {len(s1_sounds)}")
    print(f"Total S2 identified : {len(s2_sounds)}")

    return s1_sounds, s2_sounds


# ─────────────────────────────────────────
# STEP 8: Extract S1/S2 Midpoints
# ─────────────────────────────────────────

def extract_midpoints(s1_sounds, s2_sounds):
    """
    Extract the midpoint time of each S1 and S2 sound.
    Midpoints are used for accuracy evaluation
    (as described in paper Section IV.C).

    Parameters:
        s1_sounds : list - identified S1 lobes
        s2_sounds : list - identified S2 lobes

    Returns:
        s1_midpoints : np.array - midpoint times of S1 (seconds)
        s2_midpoints : np.array - midpoint times of S2 (seconds)
    """
    s1_midpoints = np.array([
        (lobe['start_time'] + lobe['end_time']) / 2
        for lobe in s1_sounds
    ])
    s2_midpoints = np.array([
        (lobe['start_time'] + lobe['end_time']) / 2
        for lobe in s2_sounds
    ])

    return s1_midpoints, s2_midpoints


# ─────────────────────────────────────────
# STEP 9: Visualization
# ─────────────────────────────────────────

def plot_s1_s2_identification(x_norm, sr,
                               s1_sounds, s2_sounds):
    """
    Visualize the final S1/S2 identification results.
    """
    time_signal  = np.arange(len(x_norm)) / sr
    s1_midpoints, s2_midpoints = extract_midpoints(
        s1_sounds, s2_sounds
    )

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    fig.suptitle("S1/S2 Identification Results",
                 fontsize=14, fontweight='bold')

    # ── Plot 1: Signal with S1/S2 regions ──
    axes[0].plot(time_signal, x_norm,
                 color='steelblue', linewidth=0.6,
                 label='Signal', alpha=0.8)

    for i, lobe in enumerate(s1_sounds):
        axes[0].axvspan(
            lobe['start_time'], lobe['end_time'],
            alpha=0.35, color='red',
            label='S1' if i == 0 else ""
        )

    for i, lobe in enumerate(s2_sounds):
        axes[0].axvspan(
            lobe['start_time'], lobe['end_time'],
            alpha=0.35, color='green',
            label='S2' if i == 0 else ""
        )

    axes[0].set_title(
        "Heart Sound Segmentation  "
        "(Red = S1 | Green = S2)"
    )
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc='upper right')

    # ── Plot 2: Midpoint markers ──
    axes[1].plot(time_signal, x_norm,
                 color='lightgray', linewidth=0.6,
                 label='Signal')

    axes[1].vlines(s1_midpoints,
                   ymin=-1, ymax=1,
                   color='red', linewidth=1.5,
                   linestyle='-', label='S1 Midpoints')

    axes[1].vlines(s2_midpoints,
                   ymin=-1, ymax=1,
                   color='green', linewidth=1.5,
                   linestyle='-', label='S2 Midpoints')

    # Label each S1 and S2
    for i, t in enumerate(s1_midpoints):
        axes[1].annotate(
            f'S1',
            xy=(t, 0.85),
            ha='center', fontsize=7,
            color='red', fontweight='bold'
        )

    for i, t in enumerate(s2_midpoints):
        axes[1].annotate(
            f'S2',
            xy=(t, 0.70),
            ha='center', fontsize=7,
            color='green', fontweight='bold'
        )

    axes[1].set_title("S1/S2 Midpoint Locations")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    # Print summary table
    print(f"\n{'─'*55}")
    print(f"{'#':<5} {'Type':<6} {'Start(s)':<12} "
          f"{'End(s)':<12} {'Midpoint(s)':<12}")
    print(f"{'─'*55}")

    all_sounds = (
        [(l, 'S1') for l in s1_sounds] +
        [(l, 'S2') for l in s2_sounds]
    )
    all_sounds.sort(key=lambda x: x[0]['start_time'])

    for i, (lobe, label) in enumerate(all_sounds):
        midpoint = (lobe['start_time'] + lobe['end_time']) / 2
        print(f"{i+1:<5} {label:<6} "
              f"{lobe['start_time']:<12.3f} "
              f"{lobe['end_time']:<12.3f} "
              f"{midpoint:<12.3f}")


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────

if __name__ == "__main__":

    from preprocess_01     import preprocess
    from shannon_energy_02 import compute_shannon_envelope
    from noisy_lobe_03     import detect_and_remove_noise
    from lobe_validation_04 import validate_lobes

    FILE_PATH = "heart_sound.wav"

    # Pipeline
    x_norm, sr, valid = preprocess(FILE_PATH)

    if valid:
        ase, nase, time_ase, lobes = \
            compute_shannon_envelope(x_norm, sr)

        clean_lobes, noisy_lobes, clean_intervals = \
            detect_and_remove_noise(
                x_norm, lobes, nase, time_ase, sr
            )

        validated_lobes, intervals, estimated_hr, \
            cardiac_cycle_s = validate_lobes(
                clean_lobes, x_norm, sr
            )

        s1_sounds, s2_sounds = identify_s1_s2(
            validated_lobes, intervals,
            cardiac_cycle_s, x_norm, sr
        )

        plot_s1_s2_identification(x_norm, sr,
                                   s1_sounds, s2_sounds)

