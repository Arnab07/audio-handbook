# AI Agent Instructions for audio-handbook

## Project Overview
**audio-handbook** is an educational Jupyter notebook project that teaches audio signal processing fundamentals. It explores the physics of sound, digital signal processing (DSP), and machine learning-ready features through hands-on Python code with visualizations.

**Key Focus**: Bridging theoretical DSP concepts with practical implementation patterns used in audio/ML pipelines.

## Architecture & Data Flow

### Notebook-Centric Design
- All content lives in **`notebooks/`** as self-contained Jupyter notebooks
- `basic.ipynb` is the foundational tutorial covering:
  1. **Signal Generation**: Creating synthetic sine waves as mathematical models (frequency domain physics)
  2. **Time Domain Analysis**: RMS calculations for amplitude/loudness metrics
  3. **Frequency Domain Conversion**: FFT to decompose signals into frequency components
  4. **Feature Extraction**: Mel spectrograms (logarithmic frequency scale humans/ML models perceive)
  5. **Signal Processing**: Butterworth low-pass filters for noise/frequency removal

### Standard Libraries & Their Roles
- **numpy**: Numerical operations, signal array manipulation, FFT computations
- **matplotlib**: Time domain and frequency domain visualization
- **librosa**: Audio-specific feature extraction (Mel spectrograms, frequency scale conversions)
- **scipy.signal**: Filter design (Butterworth), convolution-based filtering with `lfilter`

## Critical Patterns

### DSP Workflow Pattern
Every audio processing task follows this sequence:
1. **Generate/Load Signal** → time-domain array (sampling rate aware)
2. **Time Domain Analysis** → extract metrics (RMS, amplitude, envelope)
3. **Convert to Frequency Domain** → FFT or Mel Spectrogram
4. **Apply Transformations** → filters, feature extraction, modifications
5. **Validate & Visualize** → plot original vs. processed

**Example from `basic.ipynb`**: The low-pass filter (lines ~97-105) demonstrates this: noisy signal → butter filter design → lfilter application → RMS comparison before/after.

### Sampling Rate as First-Class Concept
- **Always declare `sr` (sampling rate) upfront** (e.g., `sr = 16000`)
- Pass `sr` explicitly to librosa functions: `librosa.feature.melspectrogram(y=signal, sr=sr, ...)`
- Nyquist frequency: `nyq = 0.5 * sr` (critical for filter design cutoffs)
- When adding features, ensure sr-dependent parameters scale correctly

### Visualization for Understanding
- **3-subplot pattern** is standard: Time Domain (raw waveform) → Frequency Domain (FFT) → Feature Space (Mel Spectrogram)
- Annotate frequency spikes directly on plots (e.g., lines 67-69 in `basic.ipynb`)
- Always show both "raw" and "processed" versions to demonstrate filter effects

### Mathematical Notation in Comments
Code comments include formulas (e.g., `A * sin(2 * pi * f * t)` for sine waves, line 12). This is intentional—preserve these when modifying, as they're teaching aids.

## Development Conventions

### Notebook Structure
- **Lead with imports** and parameter definitions (sr, duration, frequencies)
- **Function definitions before usage** (e.g., `generate_tone()` before calling it)
- **Long notebooks are acceptable** if logically chunked (each section = 1 major concept)
- **Cell 2+ should be empty or for extended exercises** (current: Cell 2 is blank for user work)

### Testing & Validation
- No test suite yet; validation is **visual inspection of plots + print statements**
- When adding features, include `print()` statements showing before/after metrics (e.g., line 102: comparing original vs. filtered RMS)
- Notebooks must be **runnable end-to-end** without errors

### Dependency Management
- No `requirements.txt` or `pyproject.toml` yet; dependencies are implicit in notebook imports
- If adding new packages, note them in this file and README.md for clarity
- Stick to **standard audio/ML stack**: librosa, scipy, numpy, matplotlib (no niche DSP libraries)

## Common Tasks & Patterns

### Adding a New Audio Processing Technique
1. Create a function that takes `(signal, sr, **params)` and returns processed signal
2. Add to appropriate section with math comment explaining the operation
3. Demonstrate with before/after RMS/spectrogram comparison
4. Example: `butter_lowpass_filter()` (lines 89-96) shows this pattern perfectly

### Extending the Mel Spectrogram Feature
- Librosa's `melspectrogram()` is the de facto standard for ML preprocessing
- Always convert to dB scale: `librosa.power_to_db(mel_spec, ref=np.max)`
- The `fmax=8000` cutoff is common for speech; adjust if analyzing other domains
- Never remove the spectrogram visualization—it's the bridge between theory and ML practice

## Key Files & Their Responsibilities
- **`notebooks/basic.ipynb`**: Foundational DSP theory + implementation + visualization
- **`README.md`**: Project overview (currently minimal; expand when feature-complete)
- **`LICENSE`**: Licensing info (preserve as-is)

## Integration & Extension Points
- **Future audio loading**: When adding file I/O, use `librosa.load(path)` (already imported, standard pattern)
- **Future ML models**: Mel Spectrograms (already generated) feed directly into audio classification/speech models
- **Future real-time processing**: Adapt signal generation → processing → visualization pipeline to streaming architecture

## Checklist for New Content
- [ ] Function has docstrings explaining mathematical or conceptual purpose
- [ ] All time-domain visualizations include axis labels and legends
- [ ] Frequency/Mel plots annotate key features with arrows/text
- [ ] New filters/transforms include before/after metric comparison (RMS, magnitude ranges)
- [ ] Sampling rate handling is explicit (no magic numbers)
- [ ] Code runs without errors in a fresh kernel
