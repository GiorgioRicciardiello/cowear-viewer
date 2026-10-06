// Reviewed scientific prose, moved out of page render() template literals so page logic and
// copy can change independently. Wording here is unchanged from where it used to live, except the one explicit
// hue-free rewording called out by D2 (enmoBanner's former "blue tint" phrase, since device colour is no longer
// fixed-blue-for-AX6).
import { fmt } from "./common.js";

export const enmoBanner = (hasContext) =>
  `ENMO per 30 s epoch, mg. Difference = Apple &minus; AX6. ` +
  `${hasContext ? "Showing 5 min of context; the shaded band marks the current page window. " : ""}` +
  `Flagged epochs (below a device's own noise floor; about three quarters of the night) are marked by a dot in the lower strip and are excluded from ratio statistics.`;

export const enmoRkeyBanner =
  `<b>Lower strip:</b> bars = within-epoch Pearson r at lag 0 (the agreement measure). &#9679; = peak r over a &plusmn;5 s lag search, &#9675; = its per-epoch 95th-percentile null. ` +
  `The peak is inflated by the search itself (independent AR(0.9) noise alone averages 0.22), so only the excess over &#9675; means anything, and the lag at the peak is not a measured clock offset. Blank = r undefined (a device's ENMO is all zero in that epoch).`;

export const bandsBanner = () =>
  `Band-limited RMS per 30 s epoch (mg), noise-corrected, on the 50 Hz common grid ` +
  `(grid50, Nyquist 25 Hz). Calibrated variant only.`;

export const bandsMarginWarn = (ax6Only) =>
  `* margin (20-24.9 Hz) is reported per device, never compared: Apple's response near its own 25 Hz Nyquist is unknown. ` +
  `${ax6Only ? `AX6 alone also carries ${ax6Only.band_hz[0]}-${ax6Only.band_hz[1]} Hz content (RMS ${fmt(ax6Only.rms_mg, 2)} mg) with no Apple counterpart at all. ` : ""}` +
  `None of this band content reached the published RBD features: that pipeline resamples both devices to 25 Hz (Nyquist 12.5 Hz) before computing anything.`;

export const spectrumBanner = (specInfo, dbRange) =>
  `Spectrograms: dB(g&sup2;/Hz) of vector-magnitude acceleration on the 50 Hz common grid (grid50), ` +
  `${specInfo.seg_s} s Hann segments, ${specInfo.hop_s} s hop, colour range ${dbRange[0]} to ${dbRange[1]} dB. ` +
  `Vector magnitude is rotation-invariant: the raw/aligned frame toggle has no effect on this page.`;

export const spectrumNoiseWarn =
  `<b>Read beside the Bands page, not alone:</b> in quiet epochs AX6's 10-25 Hz floor sits ` +
  `7-11 dB above Apple's -- a flat sensor noise floor, not movement. Above about 10 Hz an uncorrected colour match ` +
  `mostly compares noise floors, not signal.`;

export const spectrumWelchNote =
  `<small>Welch PSD and coherence below are night-level (Python, once): they do not change as you page through time. Solid = observed; dashed = the device's own noise floor (still windows).</small>`;

export const agreementBanner =
  `Night-level agreement, n = 1 subject, 1 night. Intervals are moving-block bootstrap CIs and depend on the block length (see the block-length panel); per-epoch peak r is never shown without its null. No verdict is drawn here.`;
