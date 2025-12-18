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
