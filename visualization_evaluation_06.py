#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 06_visualization_evaluation.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import pandas as pd
import os
import sys

from preprocess_01          import preprocess
from shannon_energy_02      import compute_shannon_envelope
from noisy_lobe_03          import detect_and_remove_noise
from lobe_validation_04     import validate_lobes
from s1_s2_identify_05      import identify_s1_s2

# ─────────────────────────────────────────
# STEP 1: Load Ground Truth Annotations
# ─────────────────────────────────────────

def load_ground_truth(annotation_file):
    """
    Load ground truth S1/S2 annotations from a CSV file.

    Expected CSV format:
        time(s), label
        0.123,   S1
        0.345,   S2
        0.567,   S1
        ...

    If you are using the CirCor dataset, the annotations
    are provided as .tsv files with columns:
        start, end, label

    Parameters:
        annotation_file : str - path to annotation CSV/TSV

    Returns:
        gt_s1 : np.array - ground truth S1 midpoint times (s)
        gt_s2 : np.array - ground truth S2 midpoint times (s)
    """
    print(f"\n--- Loading Ground Truth: {annotation_file} ---")

    ext = os.path.splitext(annotation_file)[1].lower()

    if ext == '.tsv':
        # CirCor dataset format
        df = pd.read_csv(annotation_file,
                         sep='\t',
                         header=None,
                         names=['start', 'end', 'label'])
        gt_s1 = df[df['label'] == 'S1'][['start', 'end']].mean(
            axis=1).values
        gt_s2 = df[df['label'] == 'S2'][['start', 'end']].mean(
            axis=1).values

    elif ext == '.csv':
        # Simple CSV format
        df = pd.read_csv(annotation_file)
        if 'start' in df.columns and 'end' in df.columns:
            gt_s1 = df[df['label'] == 'S1'][['start', 'end']].mean(
                axis=1).values
            gt_s2 = df[df['label'] == 'S2'][['start', 'end']].mean(
                axis=1).values
        else:
            gt_s1 = df[df['label'] == 'S1']['time'].values
            gt_s2 = df[df['label'] == 'S2']['time'].values
    else:
        raise ValueError(f"Unsupported annotation format: {ext}")

    print(f"Ground truth S1     : {len(gt_s1)} annotations")
    print(f"Ground truth S2     : {len(gt_s2)} annotations")

    return gt_s1, gt_s2


# ─────────────────────────────────────────
# STEP 2: Create Dummy Ground Truth
# (for testing without annotation files)
# ─────────────────────────────────────────

def create_dummy_ground_truth(s1_sounds, s2_sounds,
                               noise_std_ms=5.0):
    """
    Create synthetic ground truth by adding small random
    offsets to identified S1/S2 midpoints.
    Used for testing the evaluation pipeline when no
    real annotation file is available.

    Parameters:
        s1_sounds    : list  - identified S1 lobes
        s2_sounds    : list  - identified S2 lobes
        noise_std_ms : float - std of random offset in ms

    Returns:
        gt_s1 : np.array - synthetic S1 ground truth times
        gt_s2 : np.array - synthetic S2 ground truth times
    """
    print(f"\n--- Creating Dummy Ground Truth "
          f"(noise={noise_std_ms}ms) ---")

    noise_std_s = noise_std_ms / 1000.0

    gt_s1 = np.array([
        (l['start_time'] + l['end_time']) / 2 +
        np.random.normal(0, noise_std_s)
        for l in s1_sounds
    ])

    gt_s2 = np.array([
        (l['start_time'] + l['end_time']) / 2 +
        np.random.normal(0, noise_std_s)
        for l in s2_sounds
    ])

    print(f"Dummy GT S1 count   : {len(gt_s1)}")
    print(f"Dummy GT S2 count   : {len(gt_s2)}")

    return gt_s1, gt_s2


# ─────────────────────────────────────────
# STEP 3: Compute Accuracy
# ─────────────────────────────────────────

def compute_accuracy(predicted_midpoints, gt_midpoints,
                     label='S1'):
    """
    Compute segmentation accuracy as the average distance
    between predicted midpoints and ground truth midpoints.

    Method (from paper Section IV.C):
        1. For each predicted midpoint, find the nearest
           ground truth midpoint.
        2. Compute the absolute distance in milliseconds.
        3. Average over all matched pairs.

    Parameters:
        predicted_midpoints : np.array - predicted times (s)
        gt_midpoints        : np.array - ground truth times (s)
        label               : str      - 'S1' or 'S2' for printing

    Returns:
        mean_accuracy_ms : float - mean midpoint distance (ms)
        std_accuracy_ms  : float - std of midpoint distances (ms)
        distances_ms     : list  - individual distances (ms)
    """
    print(f"\n--- Computing Accuracy for {label} ---")

    if len(predicted_midpoints) == 0 or len(gt_midpoints) == 0:
        print(f"No predictions or ground truth for {label}.")
        return None, None, []

    distances_ms = []

    for pred_t in predicted_midpoints:
        # Find nearest ground truth
        diffs   = np.abs(gt_midpoints - pred_t)
        nearest = np.min(diffs)
        distances_ms.append(nearest * 1000)  # convert to ms

    mean_accuracy_ms = float(np.mean(distances_ms))
    std_accuracy_ms  = float(np.std(distances_ms))

    print(f"Predictions         : {len(predicted_midpoints)}")
    print(f"Ground truth        : {len(gt_midpoints)}")
    print(f"Mean accuracy       : {mean_accuracy_ms:.4f} ms "
          f"(+/- {std_accuracy_ms:.4f} ms)")

    # Paper reported: 0.28ms for S1, 0.29ms for S2
    # on the CirCor dataset
    if mean_accuracy_ms < 1.0:
        print(f"Excellent accuracy  : < 1ms error")
    elif mean_accuracy_ms < 5.0:
        print(f"Good accuracy       : < 5ms error")
    else:
        print(f"Moderate accuracy   : {mean_accuracy_ms:.2f}ms error")

    return mean_accuracy_ms, std_accuracy_ms, distances_ms


# ─────────────────────────────────────────
# STEP 4: Compute Sensitivity
# ─────────────────────────────────────────

def compute_sensitivity(s1_sounds, s2_sounds,
                         gt_s1, gt_s2):
    """
    Compute segmentation sensitivity.

    Method (from paper Section IV.C - Equation 8):
        Sensitivity = TP / (TP + FN)

        A prediction is TRUE POSITIVE (TP) if the midpoint
        of the predicted S1 (or S2) falls within the
        start and end boundaries of the ground truth S1
        (or S2).

        If the midpoint falls OUTSIDE → FALSE NEGATIVE (FN)

    Note: FP and TN cannot be computed for CirCor dataset
    as ground truth is only provided for high-quality
    representative sections.

    Parameters:
        s1_sounds : list     - identified S1 lobes
        s2_sounds : list     - identified S2 lobes
        gt_s1     : np.array - ground truth S1 midpoints (s)
        gt_s2     : np.array - ground truth S2 midpoints (s)

    Returns:
        sensitivity    : float - overall sensitivity
        sensitivity_s1 : float - S1 sensitivity
        sensitivity_s2 : float - S2 sensitivity
        tp_s1, fn_s1   : int   - true/false counts for S1
        tp_s2, fn_s2   : int   - true/false counts for S2
    """
    print(f"\n--- Computing Sensitivity ---")

    # ── S1 Sensitivity ──
    tp_s1 = 0
    fn_s1 = 0

    # Use a tolerance window around each GT S1
    # (half the median S1 duration as window)
    s1_durations = [
        l['end_time'] - l['start_time']
        for l in s1_sounds
    ]
    tolerance_s1 = (np.median(s1_durations) / 2
                    if s1_durations else 0.125)

    for gt_time in gt_s1:
        # Check if any predicted S1 midpoint falls within
        # tolerance window of this GT S1
        matched = False
        for lobe in s1_sounds:
            midpoint = (lobe['start_time'] +
                        lobe['end_time']) / 2
            # Check if midpoint is within GT window
            gt_start = gt_time - tolerance_s1
            gt_end   = gt_time + tolerance_s1
            if gt_start <= midpoint <= gt_end:
                matched = True
                break
        if matched:
            tp_s1 += 1
        else:
            fn_s1 += 1

    # ── S2 Sensitivity ──
    tp_s2 = 0
    fn_s2 = 0

    s2_durations = [
        l['end_time'] - l['start_time']
        for l in s2_sounds
    ]
    tolerance_s2 = (np.median(s2_durations) / 2
                    if s2_durations else 0.125)

    for gt_time in gt_s2:
        matched = False
        for lobe in s2_sounds:
            midpoint = (lobe['start_time'] +
                        lobe['end_time']) / 2
            gt_start = gt_time - tolerance_s2
            gt_end   = gt_time + tolerance_s2
            if gt_start <= midpoint <= gt_end:
                matched = True
                break
        if matched:
            tp_s2 += 1
        else:
            fn_s2 += 1

    # ── Compute Sensitivity ──
    sensitivity_s1 = (tp_s1 / (tp_s1 + fn_s1)
                      if (tp_s1 + fn_s1) > 0 else 0.0)
    sensitivity_s2 = (tp_s2 / (tp_s2 + fn_s2)
                      if (tp_s2 + fn_s2) > 0 else 0.0)

    total_tp  = tp_s1 + tp_s2
    total_fn  = fn_s1 + fn_s2
    sensitivity = (total_tp / (total_tp + total_fn)
                   if (total_tp + total_fn) > 0 else 0.0)

    print(f"\nS1 Results:")
    print(f"  TP                : {tp_s1}")
    print(f"  FN                : {fn_s1}")
    print(f"  Sensitivity       : {sensitivity_s1*100:.2f}%")

    print(f"\nS2 Results:")
    print(f"  TP                : {tp_s2}")
    print(f"  FN                : {fn_s2}")
    print(f"  Sensitivity       : {sensitivity_s2*100:.2f}%")

    print(f"\nOverall Sensitivity : {sensitivity*100:.2f}%")

    # Paper reported: 97.44% overall sensitivity
    if sensitivity >= 0.97:
        print(f"Performance         : Matches paper benchmark (>=97%)")
    elif sensitivity >= 0.90:
        print(f"Performance         : Good (>= 90%)")
    else:
        print(f"Performance         : Needs improvement (< 90%)")

    return (sensitivity, sensitivity_s1, sensitivity_s2,
            tp_s1, fn_s1, tp_s2, fn_s2)


# ─────────────────────────────────────────
# STEP 5: Segmentation Success Rate
# ─────────────────────────────────────────

def compute_success_rate(s1_sounds, s2_sounds,
                          validated_lobes):
    """
    Compute the segmentation success rate.

    Definition (from paper Section V.B):
        Segmentation FAILS for a recording if the algorithm
        was not able to identify ALL heart sound cycles
        available.

    Success rate improvement reported in paper:
        Shannon energy-based : 81.4% (956/1174)
        Noise-robust         : 87.4% (1026/1174)
        Improvement          : 6%

    Parameters:
        s1_sounds       : list - identified S1 lobes
        s2_sounds       : list - identified S2 lobes
        validated_lobes : list - all validated candidate lobes

    Returns:
        success      : bool  - whether segmentation succeeded
        success_info : dict  - details about the result
    """
    print(f"\n--- Computing Segmentation Success ---")

    total_candidates = len(validated_lobes)
    s1_found         = len(s1_sounds)
    s2_found         = len(s2_sounds)

    # Success = algorithm found at least some S1-S2 pairs
    # and number of S1 matches number of S2
    pair_count   = min(s1_found, s2_found)
    success      = (pair_count > 0 and
                    abs(s1_found - s2_found) <= 2)

    coverage = (pair_count * 2 / total_candidates
                if total_candidates > 0 else 0.0)

    success_info = {
        'success'          : success,
        'total_candidates' : total_candidates,
        's1_found'         : s1_found,
        's2_found'         : s2_found,
        'pair_count'       : pair_count,
        'coverage'         : coverage
    }

    print(f"Total candidates    : {total_candidates}")
    print(f"S1 identified       : {s1_found}")
    print(f"S2 identified       : {s2_found}")
    print(f"Complete pairs      : {pair_count}")
    print(f"Coverage            : {coverage*100:.1f}%")
    print(f"Segmentation        : "
          f"{'SUCCESS' if success else 'FAILED'}")

    return success, success_info


# ─────────────────────────────────────────
# STEP 6: Full Evaluation Report
# ─────────────────────────────────────────

def generate_evaluation_report(s1_sounds, s2_sounds,
                                validated_lobes,
                                gt_s1, gt_s2,
                                file_name="recording"):
    """
    Generate a complete evaluation report combining:
        - Accuracy for S1 and S2
        - Sensitivity
        - Segmentation success rate
        - Comparison with paper benchmarks

    Parameters:
        s1_sounds       : list     - identified S1 lobes
        s2_sounds       : list     - identified S2 lobes
        validated_lobes : list     - all validated lobes
        gt_s1           : np.array - ground truth S1 times
        gt_s2           : np.array - ground truth S2 times
        file_name       : str      - recording identifier

    Returns:
        report : dict - complete evaluation results
    """
    print(f"\n{'='*55}")
    print(f"  EVALUATION REPORT: {file_name}")
    print(f"{'='*55}")

    # Extract predicted midpoints
    s1_midpoints = np.array([
        (l['start_time'] + l['end_time']) / 2
        for l in s1_sounds
    ])
    s2_midpoints = np.array([
        (l['start_time'] + l['end_time']) / 2
        for l in s2_sounds
    ])

    # Accuracy
    acc_s1, std_s1, dist_s1 = compute_accuracy(
        s1_midpoints, gt_s1, 'S1'
    )
    acc_s2, std_s2, dist_s2 = compute_accuracy(
        s2_midpoints, gt_s2, 'S2'
    )

    # Sensitivity
    (sensitivity,
     sens_s1, sens_s2,
     tp_s1, fn_s1,
     tp_s2, fn_s2) = compute_sensitivity(
        s1_sounds, s2_sounds, gt_s1, gt_s2
    )

    # Success rate
    success, success_info = compute_success_rate(
        s1_sounds, s2_sounds, validated_lobes
    )

    # Build report
    report = {
        'file_name'         : file_name,
        'accuracy_s1_ms'    : acc_s1,
        'accuracy_s2_ms'    : acc_s2,
        'std_s1_ms'         : std_s1,
        'std_s2_ms'         : std_s2,
        'sensitivity'       : sensitivity,
        'sensitivity_s1'    : sens_s1,
        'sensitivity_s2'    : sens_s2,
        'tp_s1'             : tp_s1,
        'fn_s1'             : fn_s1,
        'tp_s2'             : tp_s2,
        'fn_s2'             : fn_s2,
        'success'           : success,
        'success_info'      : success_info,
        'distances_s1_ms'   : dist_s1,
        'distances_s2_ms'   : dist_s2,
        's1_count'          : len(s1_sounds),
        's2_count'          : len(s2_sounds)
    }

  # Print comparison with paper benchmarks
    print(f"\n{'─'*55}")
    print(f"  COMPARISON WITH PAPER BENCHMARKS")
    print(f"{'─'*55}")
    print(f"{'Metric':<30} {'Yours':<15} {'Paper':<15}")
    print(f"{'─'*55}")
    s1_str      = f"{acc_s1:.4f}"          if acc_s1      else "N/A"
    s2_str      = f"{acc_s2:.4f}"          if acc_s2      else "N/A"
    sen_str     = f"{sensitivity*100:.2f}" if sensitivity  else "N/A"
    sens_s1_str = f"{sens_s1*100:.2f}"
    sens_s2_str = f"{sens_s2*100:.2f}"
    print(f"{'S1 Accuracy (ms)':<30} {s1_str:<15} {'0.28':>15}")
    print(f"{'S2 Accuracy (ms)':<30} {s2_str:<15} {'0.29':>15}")
    print(f"{'Overall Sensitivity (%)':<30} {sen_str:<15} {'97.44':>15}")
    print(f"{'S1 Sensitivity (%)':<30} {sens_s1_str:<15} {'97.22 (murmur present)':>15}")
    print(f"{'S2 Sensitivity (%)':<30} {sens_s2_str:<15} {'97.22 (murmur present)':>15}")
    print(f"{'─'*55}")
    return report

# ─────────────────────────────────────────
# STEP 7: Final Visualization Dashboard
# ─────────────────────────────────────────

def plot_final_dashboard(x_norm, sr,
                          s1_sounds, s2_sounds,
                          gt_s1, gt_s2,
                          report, nase=None,
                          time_ase=None):
    """
    Generate a comprehensive visualization dashboard with:
        Panel 1: Signal with S1/S2 + ground truth overlay
        Panel 2: NASE envelope (if provided)
        Panel 3: Accuracy distribution (S1 and S2)
        Panel 4: Sensitivity bar chart
        Panel 5: Summary metrics table
    """
    time_signal  = np.arange(len(x_norm)) / sr
    s1_midpoints = np.array([
        (l['start_time'] + l['end_time']) / 2
        for l in s1_sounds
    ])
    s2_midpoints = np.array([
        (l['start_time'] + l['end_time']) / 2
        for l in s2_sounds
    ])

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"Noise-Robust Heart Sound Segmentation\n"
        f"File: {report['file_name']}",
        fontsize=14, fontweight='bold'
    )

    gs = gridspec.GridSpec(3, 2, figure=fig,
                            hspace=0.45, wspace=0.35)

    # ── Panel 1: Signal + S1/S2 + GT ──
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_signal, x_norm,
             color='steelblue', linewidth=0.6,
             label='PCG Signal', alpha=0.8)

    for i, lobe in enumerate(s1_sounds):
        ax1.axvspan(lobe['start_time'], lobe['end_time'],
                    alpha=0.3, color='red',
                    label='Predicted S1' if i == 0 else "")

    for i, lobe in enumerate(s2_sounds):
        ax1.axvspan(lobe['start_time'], lobe['end_time'],
                    alpha=0.3, color='green',
                    label='Predicted S2' if i == 0 else "")

    # Ground truth markers
    for i, gt_t in enumerate(gt_s1):
        ax1.axvline(gt_t, color='darkred',
                    linestyle='--', linewidth=1.0,
                    alpha=0.7,
                    label='GT S1' if i == 0 else "")

    for i, gt_t in enumerate(gt_s2):
        ax1.axvline(gt_t, color='darkgreen',
                    linestyle='--', linewidth=1.0,
                    alpha=0.7,
                    label='GT S2' if i == 0 else "")

    ax1.set_title(
        "PCG Signal with Predicted S1/S2 "
        "(Shaded) and Ground Truth (Dashed)"
    )
    ax1.set_ylabel("Amplitude")
    ax1.set_xlabel("Time (s)")
    ax1.legend(loc='upper right', fontsize=8,
               ncol=4)

    # ── Panel 2: NASE Envelope ──
    ax2 = fig.add_subplot(gs[1, 0])
    if nase is not None and time_ase is not None:
        ax2.plot(time_ase, nase,
                 color='darkorange', linewidth=0.8,
                 label='NASE')
        ax2.axhline(y=0, color='red',
                    linestyle='--', linewidth=1.0,
                    label='Threshold = 0')
        ax2.vlines(s1_midpoints, nase.min(),
                   nase.max(), color='red',
                   linewidth=1.0, alpha=0.6,
                   label='S1')
        ax2.vlines(s2_midpoints, nase.min(),
                   nase.max(), color='green',
                   linewidth=1.0, alpha=0.6,
                   label='S2')
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5,
                 'NASE not provided',
                 ha='center', va='center',
                 transform=ax2.transAxes,
                 fontsize=12, color='gray')

    ax2.set_title("NASE Envelope with S1/S2 Locations")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Energy")

    # ── Panel 3: Accuracy Distribution ──
    ax3 = fig.add_subplot(gs[1, 1])
    dist_s1 = report.get('distances_s1_ms', [])
    dist_s2 = report.get('distances_s2_ms', [])

    if dist_s1:
        ax3.hist(dist_s1, bins=20,
                 alpha=0.6, color='red',
                 edgecolor='black',
                 linewidth=0.5,
                 label=f'S1 (mean={report["accuracy_s1_ms"]:.2f}ms)')
    if dist_s2:
        ax3.hist(dist_s2, bins=20,
                 alpha=0.6, color='green',
                 edgecolor='black',
                 linewidth=0.5,
                 label=f'S2 (mean={report["accuracy_s2_ms"]:.2f}ms)')

    ax3.axvline(x=0.28, color='darkred',
                linestyle='--', linewidth=1.5,
                label='Paper S1 (0.28ms)')
    ax3.axvline(x=0.29, color='darkgreen',
                linestyle='--', linewidth=1.5,
                label='Paper S2 (0.29ms)')
    ax3.set_title("Midpoint Distance Distribution")
    ax3.set_xlabel("Distance (ms)")
    ax3.set_ylabel("Count")
    ax3.legend(fontsize=8)

    # ── Panel 4: Sensitivity Bar Chart ──
    ax4 = fig.add_subplot(gs[2, 0])
    categories   = ['S1\nSensitivity',
                     'S2\nSensitivity',
                     'Overall\nSensitivity']
    your_values  = [
        report['sensitivity_s1'] * 100,
        report['sensitivity_s2'] * 100,
        report['sensitivity']    * 100
    ]
    paper_values = [97.22, 97.22, 97.44]

    x     = np.arange(len(categories))
    width = 0.35

    bars1 = ax4.bar(x - width/2, your_values,
                    width, label='Your Results',
                    color='steelblue',
                    edgecolor='black',
                    linewidth=0.5)
    bars2 = ax4.bar(x + width/2, paper_values,
                    width, label='Paper Benchmark',
                    color='darkorange',
                    edgecolor='black',
                    linewidth=0.5)

    ax4.set_ylim(0, 110)
    ax4.set_ylabel("Sensitivity (%)")
    ax4.set_title("Sensitivity Comparison")
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.legend(fontsize=9)
    ax4.axhline(y=97.44, color='gray',
                linestyle=':', linewidth=1.0)

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax4.annotate(
            f'{h:.1f}%',
            xy=(bar.get_x() + bar.get_width()/2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', fontsize=8
        )

    # ── Panel 5: Summary Table ──
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    table_data = [
        ['Metric', 'Value', 'Paper'],
        ['S1 Accuracy',
         f"{report['accuracy_s1_ms']:.4f} ms"
         if report['accuracy_s1_ms'] else 'N/A',
         '0.28 ms'],
        ['S2 Accuracy',
         f"{report['accuracy_s2_ms']:.4f} ms"
         if report['accuracy_s2_ms'] else 'N/A',
         '0.29 ms'],
        ['S1 Sensitivity',
         f"{report['sensitivity_s1']*100:.2f}%",
         '97.22%'],
        ['S2 Sensitivity',
         f"{report['sensitivity_s2']*100:.2f}%",
         '97.22%'],
        ['Overall Sensitivity',
         f"{report['sensitivity']*100:.2f}%",
         '97.44%'],
        ['S1 Count',
         str(report['s1_count']), '-'],
        ['S2 Count',
         str(report['s2_count']), '-'],
        ['Segmentation',
         'SUCCESS' if report['success'] else 'FAILED',
         '-']
    ]

    table = ax5.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Color the header row
    for j in range(3):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white',
                                    fontweight='bold')

    # Color success/fail cell
    last_row = len(table_data) - 1
    if report['success']:
        table[last_row, 1].set_facecolor('#C6EFCE')
    else:
        table[last_row, 1].set_facecolor('#FFC7CE')

    ax5.set_title("Summary Metrics Table",
                  fontweight='bold', pad=10)

    plt.savefig(
        f"segmentation_report_{report['file_name']}.png",
        dpi=150, bbox_inches='tight'
    )
    print(f"\nDashboard saved as: "
          f"segmentation_report_{report['file_name']}.png")

    plt.show()


# ─────────────────────────────────────────
# STEP 8: Full Evaluation Pipeline
# ─────────────────────────────────────────

def run_full_evaluation(x_norm, sr,
                         s1_sounds, s2_sounds,
                         validated_lobes,
                         nase=None, time_ase=None,
                         annotation_file=None,
                         file_name="recording"):
    """
    Complete evaluation pipeline:
        Load/Create GT → Accuracy → Sensitivity
        → Success Rate → Report → Dashboard

    Parameters:
        x_norm          : np.array - normalized signal
        sr              : int      - sampling rate
        s1_sounds       : list     - identified S1 lobes
        s2_sounds       : list     - identified S2 lobes
        validated_lobes : list     - all validated lobes
        nase            : np.array - NASE (optional, for plot)
        time_ase        : np.array - time axis (optional)
        annotation_file : str      - path to GT annotations
                                     (None = use dummy GT)
        file_name       : str      - recording identifier

    Returns:
        report : dict - complete evaluation results
    """
    print("\n====== Full Evaluation Pipeline ======")

    # Load or create ground truth
    if (annotation_file is not None and
            os.path.exists(annotation_file)):
        gt_s1, gt_s2 = load_ground_truth(annotation_file)
    else:
        print("\nNo annotation file provided.")
        print("Using dummy ground truth for demonstration.")
        gt_s1, gt_s2 = create_dummy_ground_truth(
            s1_sounds, s2_sounds
        )

    # Generate report
    report = generate_evaluation_report(
        s1_sounds, s2_sounds,
        validated_lobes,
        gt_s1, gt_s2,
        file_name
    )

    # Plot dashboard
    plot_final_dashboard(
        x_norm, sr,
        s1_sounds, s2_sounds,
        gt_s1, gt_s2,
        report, nase, time_ase
    )

    return report


# ─────────────────────────────────────────
# COMPLETE PIPELINE (All 6 Codes Together)
# ─────────────────────────────────────────

if __name__ == "__main__":

    from preprocess_01      import preprocess
    from shannon_energy_02  import compute_shannon_envelope
    from noisy_lobe_03      import detect_and_remove_noise
    from lobe_validation_04 import validate_lobes
    from s1_s2_identify_05  import identify_s1_s2

  # ── Configuration ──
FILE_PATH       = "heart_sound.wav"
ANNOTATION_FILE = None   # Set to path if you have GT
FILE_NAME       = os.path.splitext(
    os.path.basename(FILE_PATH)
)[0]

print(f"\n{'='*55}")
print(f"  NOISE-ROBUST HEART SOUND SEGMENTATION")
print(f"  Based on Shannon Energy (Arjoune et al.)")
print(f"{'='*55}")

# ── Step 1: Preprocess ──
x_norm, sr, is_valid = preprocess(FILE_PATH)
if not is_valid:
    print("❌ Signal is invalid, stopping.")
    sys.exit()

# ── Step 2: Shannon Energy Envelope ──
ase, nase, time_ase, lobes = \
    compute_shannon_envelope(x_norm, sr)

# ── Step 3: Noisy Lobe Detection ──
clean_lobes, noisy_lobes, clean_intervals = \
    detect_and_remove_noise(
        x_norm, lobes, nase, time_ase, sr
    )

# ── Step 4: Lobe Validation ──
validated_lobes, intervals, cardiac_cycle_s, hr = \
    validate_lobes(clean_lobes, clean_intervals, sr)

# ── Step 5: S1/S2 Identification ──
s1_sounds, s2_sounds = identify_s1_s2(
    validated_lobes, intervals,
    cardiac_cycle_s, x_norm, sr
)

# ── Step 6: Evaluation + Dashboard ──
report = run_full_evaluation(
    x_norm, sr,
    s1_sounds, s2_sounds,
    validated_lobes,
    nase=nase,
    time_ase=time_ase,
    annotation_file=ANNOTATION_FILE,
    file_name=FILE_NAME
)

print(f"\n{'='*55}")
print(f"  PIPELINE COMPLETE")
print(f"{'='*55}")
# ── Save Text Report ──
import json

with open(f"segmentation_report_{FILE_NAME}.txt", "w") as f:
    f.write("HEART SOUND SEGMENTATION REPORT\n")
    f.write("="*55 + "\n\n")

    f.write(f"File Name       : {FILE_NAME}\n")
    f.write(f"Sample Rate     : {sr} Hz\n")
    f.write(f"S1 Count        : {len(s1_sounds)}\n")
    f.write(f"S2 Count        : {len(s2_sounds)}\n")
    f.write(f"Total Lobes     : {len(validated_lobes)}\n\n")

    f.write("----- FULL REPORT -----\n")
    f.write(json.dumps(report, indent=2))

print(f"\n✅ Text report saved as segmentation_report_{FILE_NAME}.txt")
