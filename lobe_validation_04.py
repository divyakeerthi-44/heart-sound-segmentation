#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 04_lobe_validation.py

import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# STEP 1: Filter Lobes by Duration
# ─────────────────────────────────────────

def filter_by_duration(lobes, max_duration_ms=250):
    """
    Filter out lobes that are longer than the maximum
    allowed duration for S1 or S2 heart sounds.

    Rule (from paper):
        Both S1 and S2 are less than 250 ms in duration.
        Lobes longer than 250ms are likely noise or murmurs.

    Parameters:
        lobes          : list  - sound lobes from Shannon energy
        max_duration_ms: float - maximum allowed duration in ms
                                 (default 250ms as per paper)

    Returns:
        valid_lobes    : list - lobes within duration limit
        rejected_lobes : list - lobes exceeding duration limit
    """
    print(f"\n--- Duration Filter (max = {max_duration_ms}ms) ---")

    max_duration_s = max_duration_ms / 1000.0
    valid_lobes    = []
    rejected_lobes = []

    for lobe in lobes:
        if lobe['duration'] <= max_duration_s:
            valid_lobes.append(lobe)
        else:
            rejected_lobes.append(lobe)

    print(f"Input lobes         : {len(lobes)}")
    print(f"Valid lobes         : {len(valid_lobes)}")
    print(f"Rejected lobes      : {len(rejected_lobes)}"
          f" (duration > {max_duration_ms}ms)")

    return valid_lobes, rejected_lobes


# ─────────────────────────────────────────
# STEP 2: Handle Split Heart Sounds
# ─────────────────────────────────────────

def handle_split_sounds(lobes, x_norm, sr=4000,
                         max_split_ms=50, rms_ratio=0.4):
    """
    Handle split heart sounds where S1 or S2 is split
    into two closely spaced lobes.

    Rules (from paper):
        1. Maximum split interval between two split lobes
           is generally no greater than 50ms.
        2. Split sound lobes have lower intensity.
        3. RMS energy of one lobe is less than 40% of the other.
        4. If split detected → keep the higher energy lobe.
        5. If both lobes have similar energies → keep both as
           candidates (could be murmur or noise).

    Parameters:
        lobes        : list     - duration-filtered lobes
        x_norm       : np.array - normalized signal
        sr           : int      - sampling rate
        max_split_ms : float    - max gap between split lobes (ms)
        rms_ratio    : float    - RMS ratio threshold (default 0.4)

    Returns:
        processed_lobes : list - lobes after split handling
    """
    print(f"\n--- Split Sound Handler "
          f"(max gap={max_split_ms}ms, rms_ratio={rms_ratio}) ---")

    if len(lobes) < 2:
        print("Not enough lobes to check for splits.")
        return lobes

    max_split_s     = max_split_ms / 1000.0
    processed_lobes = []
    skip_next       = False

    for i in range(len(lobes) - 1):

        if skip_next:
            skip_next = False
            continue

        current_lobe = lobes[i]
        next_lobe    = lobes[i + 1]

        # Calculate gap between current and next lobe
        gap = next_lobe['start_time'] - current_lobe['end_time']

        # Check if gap is within split threshold
        if gap <= max_split_s and gap >= 0:

            # Calculate RMS energy of both lobes
            rms_current = compute_rms(
                x_norm,
                current_lobe['start_time'],
                current_lobe['end_time'],
                sr
            )
            rms_next = compute_rms(
                x_norm,
                next_lobe['start_time'],
                next_lobe['end_time'],
                sr
            )

            # Check if one lobe has less than 40% energy of the other
            if rms_next < rms_ratio * rms_current:
                # Next lobe is lower energy → split detected
                # Keep only the higher energy (current) lobe
                current_lobe['split_detected'] = True
                current_lobe['split_kept']     = 'current'
                processed_lobes.append(current_lobe)
                skip_next = True
                print(f"  Split detected at {current_lobe['start_time']:.3f}s"
                      f" → kept higher energy lobe")

            elif rms_current < rms_ratio * rms_next:
                # Current lobe is lower energy → split detected
                # Keep only the higher energy (next) lobe
                next_lobe['split_detected'] = True
                next_lobe['split_kept']     = 'next'
                processed_lobes.append(next_lobe)
                skip_next = True
                print(f"  Split detected at {current_lobe['start_time']:.3f}s"
                      f" → kept higher energy lobe")

            else:
                # Similar energies → keep both as candidates
                # Could be murmur or noise
                current_lobe['split_detected'] = False
                next_lobe['split_detected']    = False
                processed_lobes.append(current_lobe)
                print(f"  Similar energy lobes at "
                      f"{current_lobe['start_time']:.3f}s"
                      f" → kept both as candidates")
        else:
            # No split → keep current lobe
            current_lobe['split_detected'] = False
            processed_lobes.append(current_lobe)

    # Always add the last lobe if not skipped
    if not skip_next and len(lobes) > 0:
        last_lobe = lobes[-1]
        last_lobe['split_detected'] = False
        processed_lobes.append(last_lobe)

    print(f"Lobes after split handling: {len(processed_lobes)}")
    return processed_lobes


# ─────────────────────────────────────────
# STEP 3: Compute RMS Energy of a Lobe
# ─────────────────────────────────────────

def compute_rms(x_norm, start_time, end_time, sr=4000):
    """
    Compute the Root Mean Square (RMS) energy of the signal
    within a given time interval.

    Parameters:
        x_norm     : np.array - normalized signal
        start_time : float    - start time in seconds
        end_time   : float    - end time in seconds
        sr         : int      - sampling rate

    Returns:
        rms : float - RMS energy value
    """
    start_sample = int(start_time * sr)
    end_sample   = int(end_time   * sr)

    # Clamp to signal boundaries
    start_sample = max(0, start_sample)
    end_sample   = min(len(x_norm), end_sample)

    segment = x_norm[start_sample:end_sample]

    if len(segment) == 0:
        return 0.0

    rms = np.sqrt(np.mean(segment ** 2))
    return rms


# ─────────────────────────────────────────
# STEP 4: Compute Inter-Lobe Intervals
# ─────────────────────────────────────────

def compute_inter_lobe_intervals(lobes):
    """
    Compute the time intervals between consecutive lobes.
    This is used to distinguish systolic intervals (S1→S2)
    from diastolic intervals (S2→S1).

    Key physiological fact (from paper):
        The diastolic interval (S2→S1) is generally greater
        than the systolic interval (S1→S2).

    Parameters:
        lobes : list - validated candidate lobes

    Returns:
        lobes : list - updated with 'interval_to_next' key
        intervals : list - all inter-lobe intervals in seconds
    """
    print(f"\n--- Computing Inter-Lobe Intervals ---")

    intervals = []

    for i in range(len(lobes) - 1):
        current_lobe = lobes[i]
        next_lobe    = lobes[i + 1]

        # Interval = gap between end of current and start of next
        interval = next_lobe['start_time'] - current_lobe['end_time']

        lobes[i]['interval_to_next'] = interval
        intervals.append(interval)

    # Last lobe has no next lobe
    if lobes:
        lobes[-1]['interval_to_next'] = None

    if intervals:
        print(f"Number of intervals : {len(intervals)}")
        print(f"Min interval        : {min(intervals)*1000:.1f} ms")
        print(f"Max interval        : {max(intervals)*1000:.1f} ms")
        print(f"Mean interval       : {np.mean(intervals)*1000:.1f} ms")

    return lobes, intervals


# ─────────────────────────────────────────
# STEP 5: Estimate Heart Rate
# ─────────────────────────────────────────

def estimate_heart_rate(lobes, intervals):
    """
    Estimate the approximate heart rate from the inter-lobe
    intervals. This is used as a sanity check and to
    assist S1/S2 identification in the next step.

    Normal heart rate ranges:
        Adults   : 60–100 bpm
        Children : 70–120 bpm
        Infants  : 100–160 bpm

    Parameters:
        lobes     : list - validated lobes
        intervals : list - inter-lobe intervals in seconds

    Returns:
        estimated_hr    : float - estimated heart rate in bpm
        cardiac_cycle_s : float - estimated cardiac cycle in seconds
    """
    print(f"\n--- Estimating Heart Rate ---")

    if len(intervals) < 2:
        print("Not enough intervals to estimate heart rate.")
        return None, None

    # Use median interval to avoid outlier influence
    median_interval = np.median(intervals)

    # Cardiac cycle ≈ 2 × median inter-lobe interval
    # (since each cycle has S1 and S2)
    cardiac_cycle_s = 2 * median_interval

    # Heart rate in bpm
    if cardiac_cycle_s > 0:
        estimated_hr = 60.0 / cardiac_cycle_s
    else:
        estimated_hr = 0.0

    print(f"Median inter-lobe interval : {median_interval*1000:.1f} ms")
    print(f"Estimated cardiac cycle    : {cardiac_cycle_s*1000:.1f} ms")
    print(f"Estimated heart rate       : {estimated_hr:.1f} bpm")

    # Physiological warning
    if estimated_hr < 40 or estimated_hr > 220:
        print(f"Warning: Estimated HR {estimated_hr:.1f} bpm is outside"
              f" physiological range!")

    return estimated_hr, cardiac_cycle_s


# ─────────────────────────────────────────
# STEP 6: Full Lobe Validation Pipeline
# ─────────────────────────────────────────

def validate_lobes(clean_lobes, x_norm, sr=4000,
                   max_duration_ms=250,
                   max_split_ms=50,
                   rms_ratio=0.4):
    """
    Full lobe validation pipeline:
        Duration Filter → Split Handler
        → Inter-lobe Intervals → Heart Rate Estimate

    Parameters:
        clean_lobes    : list     - clean lobes from noise detection
        x_norm         : np.array - normalized signal
        sr             : int      - sampling rate
        max_duration_ms: float    - max lobe duration in ms
        max_split_ms   : float    - max split gap in ms
        rms_ratio      : float    - RMS split threshold

    Returns:
        validated_lobes  : list  - final candidate S1/S2 lobes
        intervals        : list  - inter-lobe intervals
        estimated_hr     : float - estimated heart rate
        cardiac_cycle_s  : float - estimated cardiac cycle length
    """
    print("\n====== Lobe Validation Pipeline ======")

    # Step 1: Filter by duration
    valid_lobes, rejected = filter_by_duration(
        clean_lobes, max_duration_ms
    )

    # Step 2: Handle split sounds
    processed_lobes = handle_split_sounds(
        valid_lobes, x_norm, sr, max_split_ms, rms_ratio
    )

    # Step 3: Compute inter-lobe intervals
    validated_lobes, intervals = compute_inter_lobe_intervals(
        processed_lobes
    )

    # Step 4: Estimate heart rate
    estimated_hr, cardiac_cycle_s = estimate_heart_rate(
        validated_lobes, intervals
    )

    print(f"\nValidation complete.")
    print(f"Final candidate lobes : {len(validated_lobes)}")

    return validated_lobes, intervals, estimated_hr, cardiac_cycle_s


# ─────────────────────────────────────────
# STEP 7: Visualization
# ─────────────────────────────────────────

def plot_lobe_validation(x_norm, sr,
                          clean_lobes,
                          validated_lobes,
                          intervals):
    """
    Visualize the lobe validation results showing:
        - Signal with validated candidate lobes
        - Inter-lobe interval distribution
        - RMS energy per lobe
    """
    time_signal = np.arange(len(x_norm)) / sr

    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    fig.suptitle("Lobe Validation Results",
                 fontsize=14, fontweight='bold')

    # ── Plot 1: Signal with validated lobes ──
    axes[0].plot(time_signal, x_norm,
                 color='steelblue', linewidth=0.6,
                 label='Signal')

    for i, lobe in enumerate(validated_lobes):
        axes[0].axvspan(
            lobe['start_time'],
            lobe['end_time'],
            alpha=0.3, color='green',
            label='Validated Lobe' if i == 0 else ""
        )

    # Mark rejected lobes if any
    for i, lobe in enumerate(clean_lobes):
        if lobe not in validated_lobes:
            axes[0].axvspan(
                lobe['start_time'],
                lobe['end_time'],
                alpha=0.3, color='orange',
                label='Rejected Lobe' if i == 0 else ""
            )

    axes[0].set_title("Signal with Validated Candidate Lobes "
                      "(Green=Valid, Orange=Rejected)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlabel("Time (s)")
    axes[0].legend(loc='upper right')

    # ── Plot 2: Inter-lobe interval distribution ──
    if intervals:
        intervals_ms = [iv * 1000 for iv in intervals]
        axes[1].bar(range(len(intervals_ms)),
                    intervals_ms,
                    color='steelblue',
                    edgecolor='black',
                    linewidth=0.5)
        axes[1].axhline(
            y=np.mean(intervals_ms),
            color='red', linestyle='--',
            linewidth=1.5,
            label=f'Mean = {np.mean(intervals_ms):.1f}ms'
        )
        axes[1].set_title("Inter-Lobe Intervals")
        axes[1].set_xlabel("Interval Index")
        axes[1].set_ylabel("Interval (ms)")
        axes[1].legend()

    # ── Plot 3: RMS energy per validated lobe ──
    rms_values = [
        compute_rms(x_norm,
                    lobe['start_time'],
                    lobe['end_time'], sr)
        for lobe in validated_lobes
    ]

    lobe_centers = [
        (lobe['start_time'] + lobe['end_time']) / 2
        for lobe in validated_lobes
    ]

    axes[2].bar(range(len(rms_values)),
                rms_values,
                color='darkorange',
                edgecolor='black',
                linewidth=0.5)
    axes[2].set_title("RMS Energy per Validated Lobe")
    axes[2].set_xlabel("Lobe Index")
    axes[2].set_ylabel("RMS Energy")

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────

if __name__ == "__main__":

    from preprocess_01        import preprocess
    from shannon_energy_02    import compute_shannon_envelope
    from noisy_lobe_03        import detect_and_remove_noise

    FILE_PATH = "heart_sound.wav"

    # Step 1: Preprocess
    x_norm, sr, valid = preprocess(FILE_PATH)

    if valid:
        # Step 2: Shannon energy
        ase, nase, time_ase, lobes = compute_shannon_envelope(
            x_norm, sr
        )

        # Step 3: Remove noise
        clean_lobes, noisy_lobes, clean_intervals = \
            detect_and_remove_noise(
                x_norm, lobes, nase, time_ase, sr
            )

        # Step 4: Validate lobes
        validated_lobes, intervals, estimated_hr, cardiac_cycle_s = \
            validate_lobes(clean_lobes, x_norm, sr)

        # Step 5: Visualize
        plot_lobe_validation(
            x_norm, sr,
            clean_lobes,
            validated_lobes,
            intervals
        )

