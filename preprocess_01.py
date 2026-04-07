#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 01_load_and_preprocess.py

import numpy as np
import librosa
import scipy.signal as signal
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# STEP 1: Load Audio File
# ─────────────────────────────────────────

def load_audio(file_path, target_sr=4000):
    """
    Load a heart sound recording and resample to target sampling rate.

    Parameters:
        file_path  : str   - path to .wav file
        target_sr  : int   - target sampling rate (default 4000 Hz as per paper)

    Returns:
        audio      : np.array - resampled audio signal
        sr         : int      - sampling rate (4000 Hz)
    """
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    print(f"Loaded: {file_path}")
    print(f"Sampling Rate : {sr} Hz")
    print(f"Duration      : {len(audio)/sr:.2f} seconds")
    print(f"Total Samples : {len(audio)}")
    return audio, sr


# ─────────────────────────────────────────
# STEP 2: Bandpass Filter (40 Hz - 500 Hz)
# ─────────────────────────────────────────

def bandpass_filter(audio, lowcut=40, highcut=500, fs=4000, order=4):
    """
    Apply a Butterworth bandpass filter to remove noise outside
    the heart sound frequency range (40–500 Hz).

    Parameters:
        audio   : np.array - input audio signal
        lowcut  : float    - lower cutoff frequency (Hz)
        highcut : float    - upper cutoff frequency (Hz)
        fs      : int      - sampling frequency (Hz)
        order   : int      - filter order

    Returns:
        filtered_audio : np.array - bandpass filtered signal
    """
    nyquist = fs / 2.0
    low  = lowcut  / nyquist
    high = highcut / nyquist

    # Butterworth bandpass filter
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_audio = signal.filtfilt(b, a, audio)

    print(f"Bandpass filter applied: {lowcut} Hz - {highcut} Hz")
    return filtered_audio


# ─────────────────────────────────────────
# STEP 3: Normalize Signal to [-1, 1]
# ─────────────────────────────────────────

def normalize_signal(audio):
    """
    Normalize the filtered audio signal to the range [-1, 1].

    Parameters:
        audio : np.array - input signal

    Returns:
        normalized : np.array - normalized signal
    """
    max_val = np.max(np.abs(audio))

    if max_val == 0:
        print("Warning: Signal is all zeros. Returning as-is.")
        return audio

    normalized = audio / max_val
    print(f"Signal normalized to [-1, 1]. Max value was: {max_val:.4f}")
    return normalized


# ─────────────────────────────────────────
# STEP 4: Validity Check (min 3 seconds)
# ─────────────────────────────────────────

def check_validity(audio, sr=4000, min_duration=3):
    """
    Check if the audio signal is long enough for segmentation.
    The paper requires at least 3 seconds of valid data.

    Parameters:
        audio        : np.array - audio signal
        sr           : int      - sampling rate
        min_duration : float    - minimum required duration in seconds

    Returns:
        is_valid : bool
    """
    duration = len(audio) / sr
    if duration < min_duration:
        print(f"Invalid: Duration {duration:.2f}s is less than {min_duration}s minimum.")
        return False
    print(f"Valid signal: Duration = {duration:.2f} seconds")
    return True


# ─────────────────────────────────────────
# STEP 5: Full Preprocessing Pipeline
# ─────────────────────────────────────────

def preprocess(file_path, target_sr=4000, lowcut=40, highcut=500, order=4):
    """
    Full preprocessing pipeline:
        Load → Resample → Bandpass Filter → Normalize → Validity Check

    Parameters:
        file_path : str - path to .wav audio file
        target_sr : int - target sampling rate
        lowcut    : float - lower cutoff frequency
        highcut   : float - upper cutoff frequency
        order     : int   - filter order

    Returns:
        x_norm : np.array - preprocessed normalized signal
        sr     : int      - sampling rate
        valid  : bool     - whether signal passes validity check
    """
    # Load
    audio, sr = load_audio(file_path, target_sr)

    # Validity check
    valid = check_validity(audio, sr)
    if not valid:
        return None, sr, False

    # Bandpass filter
    filtered = bandpass_filter(audio, lowcut, highcut, sr, order)

    # Normalize
    x_norm = normalize_signal(filtered)

    return x_norm, sr, True


# ─────────────────────────────────────────
# STEP 6: Visualization
# ─────────────────────────────────────────

def plot_preprocessing(file_path, target_sr=4000):
    """
    Visualize the original vs filtered vs normalized signal
    for inspection purposes.
    """
    # Load raw
    raw, sr = librosa.load(file_path, sr=target_sr, mono=True)
    time = np.arange(len(raw)) / sr

    # Process
    filtered  = bandpass_filter(raw)
    normalized = normalize_signal(filtered)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Preprocessing Pipeline", fontsize=14, fontweight='bold')

    axes[0].plot(time, raw, color='steelblue', linewidth=0.7)
    axes[0].set_title("Original Signal (Resampled to 4000 Hz)")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(time, filtered, color='darkorange', linewidth=0.7)
    axes[1].set_title("After Bandpass Filter (40–500 Hz)")
    axes[1].set_ylabel("Amplitude")

    axes[2].plot(time, normalized, color='green', linewidth=0.7)
    axes[2].set_title("After Normalization ([-1, 1])")
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────
# QUICK TEST (run this cell directly)
# ─────────────────────────────────────────

if __name__ == "__main__":
    # Replace with your actual .wav file path
    FILE_PATH = "heart_sound.wav"

    x_norm, sr, valid = preprocess(FILE_PATH)

    if valid:
        print(f"\nPreprocessing complete.")
        print(f"Signal shape : {x_norm.shape}")
        print(f"Sample rate  : {sr} Hz")
        plot_preprocessing(FILE_PATH)


# In[ ]:




