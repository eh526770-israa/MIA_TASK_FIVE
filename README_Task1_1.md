# Task 1.1 – Recovering a Spoken Word from Noisy Audio

## Problem
The provided audio file (`task5_1.wav`) contains a spoken word buried under a
noisy buzz/hum. The goal is to recover a clean, intelligible version of the
speech using signal processing and filtering techniques — no deep learning.

## My Approach

I treated this as a classic **noise removal pipeline**, moving from a broad,
general cleanup down to a more targeted one. The idea was: first understand
what the noise actually *is* (in the frequency domain), remove it
surgically, then do a final general-purpose cleanup pass.

### 1. Load and inspect the audio
I read the `.wav` file with `scipy.io.wavfile`, converted it to `float64`,
and normalized it to the `[-1, 1]` range. Normalizing early makes every
filter downstream behave consistently and avoids clipping issues.

### 2. Visualize in the time domain
Plotting the raw waveform gave me a first look at the signal, but a buzz/hum
is much easier to diagnose in the **frequency domain** than in the time
domain — a steady hum looks like noise in time, but shows up as sharp,
isolated spikes in a frequency spectrum.

### 3. FFT to find the noise frequencies
I ran a full FFT (`np.fft.rfft`) on the signal and plotted the magnitude
spectrum. This is the key diagnostic step: I could visually spot several
sharp peaks that didn't belong to normal speech — these are the buzz
frequencies. I read off their approximate values from the plot:

```
1989 Hz, 4298 Hz, 6607 Hz, 8276 Hz
```

Notice they're roughly evenly spaced — that's typical of an electrical hum
and its harmonics, which confirmed I was looking at a real periodic
interference, not just random noise.

### 4. Notch filtering — remove exactly those frequencies
For each identified buzz frequency, I designed a narrow **IIR notch filter**
(`scipy.signal.iirnotch`) with `Q=50` (a high Q means a very narrow notch —
I only want to kill that specific frequency and its immediate neighborhood,
not eat into nearby speech content). I applied each notch with
`filtfilt` (zero-phase filtering) so the filtering doesn't shift/smear the
signal in time, which matters for speech intelligibility.

### 5. Band-pass filtering — keep only the speech-relevant range
Human speech mostly lives between roughly 80 Hz and a few kHz. I applied a
6th-order Butterworth band-pass filter (80 Hz–6000 Hz) using second-order
sections (`sos`) for numerical stability, again with `sosfiltfilt` for
zero-phase filtering. This throws away rumble/DC drift below 80 Hz and
high-frequency hiss above 6 kHz that isn't contributing to the word anyway.

At this point (notch + band-pass) the buzz was already much less prominent,
but there was still some residual broadband noise/hiss left over.

### 6. Wiener filter — general adaptive denoising
I applied `scipy.signal.wiener` as a general-purpose adaptive filter. Unlike
the notch/band-pass filters (which target specific frequencies I already
knew about), the Wiener filter estimates local signal-vs-noise statistics
and smooths out noise adaptively — a good general cleanup pass after the
targeted filtering.

### 7. STFT + spectral subtraction — final polish
As a last step, I moved to a time-frequency representation using
**Short-Time Fourier Transform** (`scipy.signal.stft`). I estimated the
noise floor per frequency bin as the **median magnitude across time** — the
median works well here because the noise is present continuously throughout
the clip, while speech energy comes and goes, so the median is robust to the
speech "outliers" and gives a good stationary-noise estimate.

I then subtracted a scaled version of that noise estimate from the
magnitude spectrogram (`alpha = 1.5`, i.e. slightly over-subtracting to be
aggressive about noise removal), clipped negative values to zero (magnitude
can't be negative), reconstructed the complex spectrogram using the
**original phase**, and inverted back to a time-domain signal with
`scipy.signal.istft`.

## Why this order of techniques
- **Notch first**: I know exactly which frequencies are the hum, so remove
  them surgically before doing anything more general — this avoids the
  general filters having to "fight" a strong periodic interferer.
- **Band-pass second**: cheap, safe way to discard frequency ranges that
  can't contain speech.
- **Wiener third**: adaptive statistical denoising for whatever noise is
  left that isn't tied to a specific frequency.
- **Spectral subtraction last**: the most aggressive/targeted step, applied
  only after the signal is already mostly clean, so the noise estimate
  (median across time) is more accurate.

## Tools & Libraries
- `numpy` – FFT, array operations
- `scipy.signal` – `iirnotch`, `butter`, `sosfiltfilt`, `filtfilt`, `wiener`,
  `stft`/`istft`
- `scipy.io.wavfile` – reading the `.wav` file
- `matplotlib` – waveform and spectrum visualization
- `IPython.display.Audio` – listening to intermediate results directly in
  the notebook, which was very useful for judging whether each step was
  actually helping before moving to the next one

## Result
A progressively cleaner signal at each stage: raw audio → notch-filtered →
band-pass filtered → Wiener filtered → spectral-subtraction cleaned, with
the buzz removed and the spoken word audible.
