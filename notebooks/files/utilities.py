import numpy as np
from scipy.signal import savgol_filter


# --- Function: Pure sine wave generator ---
def generate_tone(freq, duration, sr=16000, amplitude=0.5):
    """
    Create a pure sine wave at a given frequency.

    Math: y(t) = A * sin(2π*f*t)
    - A = 0.5 (Amplitude, controls loudness)
    - f = Frequency in Hz (e.g., 150 Hz = 150 cycles per second)
    - t = time array from 0 to duration

    Returns:
      t: time array (used for plotting)
      signal: the actual waveform (numpy array of audio samples)
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * freq * t)
    return t, signal


# --- Function: Phase Spectrum Computation ---
def compute_phase_spectrum(audio, sr, unwrap=True, mask_threshold=None):
    """
    Compute the phase spectrum of an audio signal using FFT.

    Parameters:
    -----------
    audio : numpy array
        Audio waveform (time-domain samples)
    sr : int
        Sample rate in Hz
    unwrap : bool
        If True, unwrap phase to remove 2π discontinuities
        (useful for studying delay or system behavior)
    mask_threshold : float or None
        If provided, mask phase values where magnitude is below this threshold
        (useful to focus on strong frequency components)

    Returns:
    --------
    freqs : numpy array
        Frequency bins in Hz
    phase : numpy array
        Phase spectrum in radians (wrapped or unwrapped)
    magnitude : numpy array
        Magnitude spectrum (for reference)

    Notes:
    ------
    - np.angle() gives phase in [-π, π] (wrapped)
    - For noise-dominated bins, phase becomes uniformly random in [-π, π]
    - Unwrapping: If phase jumps by more than π, assume it's a wrap and add/subtract 2π
      Formula: φ_unwrap(k) = φ(k) + 2π*n_k
    """
    N = len(audio)
    X = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(N, d=1 / sr)

    magnitude = np.abs(X)

    if unwrap:
        phase = np.unwrap(np.angle(X))
    else:
        phase = np.angle(X)

    if mask_threshold is not None:
        phase_masked = phase.copy()
        phase_masked[magnitude < mask_threshold] = np.nan
        phase = phase_masked

    return freqs, phase, magnitude


# --- Function: Group Delay Computation ---
def compute_group_delay(audio, sr, smooth_phase=True, window_length=101, polyorder=3):
    """
    Compute group delay of an audio signal from its phase spectrum.

    Parameters:
    -----------
    audio : numpy array
        Audio waveform (time-domain samples)
    sr : int
        Sample rate in Hz
    smooth_phase : bool
        If True, smooth phase using Savitzky-Golay filter before differentiation
        (reduces noise artifacts in group delay)
    window_length : int
        Window length for Savitzky-Golay filter (must be odd)
    polyorder : int
        Polynomial order for Savitzky-Golay filter

    Returns:
    --------
    freqs : numpy array
        Frequency bins in Hz
    group_delay : numpy array
        Group delay in seconds

    Notes:
    ------
    - Group delay measures frequency-dependent timing distortion
    - Formula: τ_g(ω) = -dφ/dω / (2π)
      where φ is phase and ω is angular frequency
    - Useful for debugging: Does processing introduce frequency-dependent timing distortion?
    - Best used for comparing two audio files (clean vs processed)
    - For signals (not systems), group delay shows timing estimate reliability at each frequency
    """
    N = len(audio)
    X = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(N, d=1 / sr)

    phase = np.unwrap(np.angle(X))

    if smooth_phase:
        phase = savgol_filter(phase, window_length, polyorder)

    # Numerical derivative of phase w.r.t frequency
    dphi_df = np.gradient(phase, freqs)

    # Group delay (seconds)
    group_delay = -dphi_df / (2 * np.pi)

    return freqs, group_delay


def interpret_signal(value, metric_type):
    # ============================================================================
    # INTERPRETATION GUIDE
    # ============================================================================
    # Use these guidelines to understand what the metrics mean:
    #
    # PROMINENCE:
    #   - < 2.0: Weak periodicity (noise, unvoiced sounds)
    #   - 2.0-5.0: Moderate periodicity (could be speech or music)
    #   - > 5.0: Strong periodicity (voiced speech, tonal music)
    #
    # ENTROPY:
    #   - < 3.0: Very structured (pure tones, synthetic)
    #   - 3.0-6.0: Moderate structure (some speech/music)
    #   - 6.0-9.5: Balanced (natural speech, complex music)
    #   - > 9.5: High disorder (noise, unvoiced)
    #
    # FLATNESS:
    #   - Close to 0: Energy concentrated (tonal)
    #   - 0.2-0.4: Moderate distribution (typical speech)
    #   - > 0.5: Very uniform (noise-like)
    #
    # HARMONICITY_MEAN:
    #   - < 0.3: Weak harmonics (likely not speech)
    #   - 0.3-0.6: Moderate harmonics (could be speech/music)
    #   - > 0.6: Strong harmonics (voiced speech/music)
    #
    # VOICED_RATIO:
    #   - < 0.3: Mostly unvoiced (noise, whispers)
    #   - 0.3-0.7: Mixed (typical natural speech)
    #   - > 0.7: Mostly voiced (sustained tones, singing)
    #
    # PITCH_STD:
    #   - ≈ 0: Constant pitch (synthetic, monotone)
    #   - 5-30 Hz: Moderate variation (natural speech)
    #   - > 30 Hz: High variation (expressive speech, music)
    # ============================================================================
    output_text = ''
    match metric_type:
        case "prominence":
            if value < 2.0:
                output_text = "Weak periodicity (noise, unvoiced sounds)"
            elif 2.0 <= value < 5:
                output_text = "Moderate periodicity (could be speech or music)"
            else:
                output_text = "Strong periodicity (voiced speech, tonal music)"
        case "entropy":
            if value < 3.0:
                output_text = "Very structured (pure tones, synthetic)"
            elif 3.0 <= value < 6:
                output_text = "Moderate structure (some speech/music)"
            elif 6.0 <= value < 9.5:
                output_text = "Balanced (natural speech, complex music)"
            else:
                output_text = "High disorder (noise, unvoiced)"
        case "flatness":
            if value < 0.2:
                output_text = "Energy concentrated (tonal)"
            elif 0.2 <= value < 0.4:
                output_text = "Moderate distribution (typical speech)"
            else:
                output_text = "Very uniform (noise-like)"
        case "harmonicity_mean":
            if value < 0.3:
                output_text = "Weak harmonics (likely not speech)"
            elif 0.3 <= value < 0.6:
                output_text = "Moderate harmonics (could be speech/music)"
            else:
                output_text = "Mostly voiced (sustained tones, singing)"
        case "voiced_ratio":
            if value < 0.3:
                output_text = "Mostly unvoiced (noise, whispers)"
            elif 0.3 <= value < 0.7:
                output_text = "Mixed (typical natural speech)"
            else:
                output_text = "Mostly voiced (sustained tones, singing)"
        case "pitch_std":
            if value < 5:
                output_text = "Constant pitch (synthetic, monotone)"
            elif 5 <= value < 30:
                output_text = "Moderate variation (natural speech)"
            else:
                output_text = "High variation (expressive speech, music)"

    return output_text
