# Methods and design decisions

This page records why the pipeline does what it does. The code docstrings carry the details. This page carries the
reasoning that connects them.

## 1. One anchor, one epoch grid

Every time is counted from the **recording start**, never from a round clock time. Sleep scorers cut 30 s epochs from
the recording start, so their bout boundaries fall on, for example, :07 and :37 s. An epoch grid anchored at :00 / :30
would move every sleep flag by up to one epoch. `signals.recording_window` is the single source of the anchor.

## 2. Timing is repaired, and the repairs are counted

The Apple export runs at ~50.24 Hz with ~2.6 % interval jitter. At batch boundaries its timestamps step backwards or
repeat. `load_recording` stable-sorts the rows and drops exact duplicates, keeping the first. It returns the counts in
`Timing`. These counts appear in the manifest and in the quality findings. Nothing else is altered.

## 3. Calibration before anything else

A still wrist measures exactly 1 g. Ten-second windows where every axis has SD < 13 mg are treated as still. Gain *g*
and offset *d* per axis are fitted by iteratively reweighted least squares until the calibrated window means lie on the
unit sphere (`x' = g·x + d`). This is a port of the MATLAB `calibrate_acc.m` and keeps its exact update rule.

There is one deliberate difference. MATLAB stops when the residual changes by less than 1e-10, which on quiet nights
halts 3–5 mg short of the fixed point. This port iterates until the *parameters* converge. The early-stopped fit is
still computed and reported beside the converged one, so the effect of that choice can be read off.

There are two guards. A fit whose still windows cover less than half the orientation sphere is refused, because it
diverges. A fit below full coverage is flagged `low_orientation_coverage`.

## 4. Lag from movement, frame from gravity

**Lag.** Signed waveforms from two wrists barely correlate, because the placement and the on-device filtering differ.
Their **movement envelope** does correlate: the mean of |band-passed VM| in 1 s bins, 0.5–10 Hz. Vector magnitude is
rotation-invariant, so the lag can be estimated before the frames are aligned. The search covers ±5 s, and
`at_boundary` is flagged if the peak sits at the edge of that range.

**Rotation.** In jointly still windows, each device measures only gravity. A Kabsch fit over those paired gravity
vectors gives the rotation that maps the Apple frame into the AX6 frame. The pipeline reports the RMS residual angle
and a block-bootstrap uncertainty. A large residual means that no single rotation relates the two frames, for example
when a watch strap shifts during the night. In that case the axis-level comparison is disabled by the quality gates
instead of being drawn wrong.

## 5. The common grid

Samples are interpolated onto a uniform axis at the device's native rate, which removes the jitter. They are then
decimated by `resample_poly`, which is zero-phase and so adds no delay. The grid is 30 Hz for agreement and 50 Hz for
spectral work, because 30 Hz has a Nyquist of 15 Hz and would discard the 15–25 Hz band. Gaps longer than 1 s become
NaN rather than a straight line. A grid that would need padding at either end is refused.

**ENMO is computed at native rate, then decimated.** ENMO = max(‖a‖ − 1, 0) is rectified, so anti-alias filtering
the axes first changes its value. Filtering the ENMO stream instead preserves every epoch mean. The cost is ringing
below zero at rectified edges. That ringing is documented, and it does not affect epoch means.

## 6. Agreement statistics that respect dependence

* **Moving-block bootstrap** for every interval. Consecutive epochs are not independent, so a plain bootstrap
  understates the interval. The interval depends on the block length (under AR(0.97) dependence longer than the block,
  a nominal 95 % interval covers about 60 %). The pipeline therefore reports r at 5, 15, 30 and 60 min blocks, never
  one alone.
* **Block sums for the 30 Hz stream.** Over 1 M samples × 2000 resamples, Pearson r is rebuilt from per-block sufficient
  statistics instead of resampling raw samples.
* **Phase-randomised null.** One stream's Fourier phases are randomised within 5 min blocks. This keeps its spectrum and
  autocorrelation and breaks only its pairing with the other device. The gap between r and this null is the quantity
  to read.
* **Effective n.** The number of independent samples implied by the bootstrap half-width,
  `(1.96 (1 − r²) / half_width)²`, shown so that "1 M samples" is not mistaken for 1 M observations.
* **Within-epoch peak r** is biased upward by its own ±5 s search. It is never shown without the 95th percentile of the
  same statistic computed on phase-randomised data.

## 7. Noise floors are per device and measured, not assumed

On still windows, the VM of each device is that device's own sensor noise. If that noise is Gaussian, its mean ENMO
is σ<sub>VM</sub>/√(2π). That value is the device's floor. An epoch below either device's floor is flagged
`noise_floor_flag`. Such epochs stay in the correlation statistics, are left out of ratio statistics (`log2_ratio` =
NaN), and are shown in the viewer. In sleep, most epochs are flagged, and the viewer states that at the top of every
page.

## 8. Spectral comparison

* Welch PSD per device. The **noise PSD** is measured on that device's still windows and subtracted (floored at zero).
* The bands are named by what they capture: posture 0.1–0.5 Hz, gross movement 0.5–3, jerk/twitch 3–8, tremor 8–15,
  diagnostic 15–20 Hz. The 20–25 Hz margin is reported per device and never compared, because the Apple's response
  near its own Nyquist frequency is unknown. AX6 content above 25 Hz has no Apple counterpart at all and is reported
  as a single night-level number.
* Magnitude-squared coherence is shown with its bias floor: an independent pair averages about 1/K over K segments, not 0.
* Per-epoch SEF95, spectral centroid, dominant frequency, normalised spectral entropy, and MODWT (sym8, 6 levels)
  energies. MODWT is undecimated, so its coefficients align in time between devices whose sample counts differ.
* The AX6 has a flat sensor noise floor that sits above the Apple's from roughly 12 Hz up. Without noise correction,
  that difference looks like extra high-frequency movement on the AX6.

## 9. Quality gates decide what may be compared

`preprocess/` judges completeness, gaps, rate error, jitter, reordering, duplicates, gravity presence, axis SD,
calibration coverage and RMS, lag correlation, and rotation residual and uncertainty. Each check gives an info / warn /
fail finding with its value and threshold, written to `findings.csv`. The findings feed a `Capabilities` record:
ENMO agreement, axis comparison, spectral comparison and per-epoch lag are each enabled separately. A night whose
rotation does not fit can still support ENMO and spectral comparison.

## 10. Artifact contract

The browser reads plain JSON and little-endian float32 chunk files (channels interleaved, 30 min per file):

| Tier | Rate | Use |
|---|---|---|
| `native` | 100 / 50 Hz | raw device samples, uniform axis |
| `grid30` | 30 Hz | common agreement grid |
| `grid50` | 50 Hz | common spectral grid |
| `ctx4` | 4 Hz | min/max envelope for windows ≥ 30 min (a mean would hide the transients) |
| `spec` | 0.25 Hz | dB spectrogram, frequency bins in place of channels |

Chunks hold **raw** device-frame values. The browser applies calibration, then rotation, then lag from the manifest,
so each correction can be switched off to show what it does. The build writes to `<out>.partial` and renames it only
on success.
