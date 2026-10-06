// Page 5: per-30 s-epoch band RMS, a near-copy of pages/enmo.js's shape (bars, difference, a lower strip), with a
// band selector. Runs on the Python-computed `spectral` block (grid50, noise-corrected, calibrated-only; see
// spectral_epoch_rows' docstring), never recomputed here -- this file only draws.
import { COL, EPOCH_S, legendHTML, mkPlot, yAxis, windowShadePlugin, epochRange, alpha } from "../common.js";
import { bandsBanner, bandsMarginWarn } from "../copy.js";

const MIN_SPAN_S = 300;
const BANDS = [
  ["posture", "posture 0.1-0.5 Hz"], ["gross_movement", "gross movement 0.5-3 Hz"], ["jerk_twitch", "jerk/twitch 3-8 Hz"],
  ["tremor", "tremor 8-15 Hz"], ["diagnostic", "diagnostic 15-20 Hz"], ["margin", "margin 20-24.9 Hz*"],
];
let band = "jerk_twitch";   // survives page steps

export async function render(el, ctx) {
  const { epochs, t0, t1, spectra } = ctx;
  const spanEnd = Math.max(t1, t0 + MIN_SPAN_S);
  const [e0, e1] = epochRange(t0, spanEnd, epochs.length);
  const rows = epochs.slice(e0, e1).filter((r) => r.spectral);
  const start = (r) => r.epoch * EPOCH_S;
  const shade = spanEnd > t1 ? [windowShadePlugin(t0, t1)] : [];
  const opts = BANDS.map(([k, l]) => `<option value="${k}" ${k === band ? "selected" : ""}>${l}</option>`).join("");
  const wavelet = spectra?.wavelet, ax6Only = spectra?.ax6_only_band;

  el.innerHTML = `${legendHTML("device")}<div class="banner">${bandsBanner()} ` +
    `<label>Band <select class="bandsel">${opts}</select></label></div>` +
    `<div class="banner warn">${bandsMarginWarn(ax6Only)}</div>` +
    `<div class="stack bands"><div class="plotbox" data-p="bars"></div><div class="plotbox" data-p="diff"></div><div class="plotbox" data-p="centroid"></div></div>`;

  el.querySelector(".bandsel").addEventListener("change", (e) => { band = e.target.value; render(el, ctx); });
  if (!rows.length) return;

  const xRange = [t0, spanEnd], barPaths = uPlot.paths.bars({ size: [0.9, 80], gap: 1 });
  const opt = (extra) => ({ xRange, plugins: shade, sync: "bands", ...extra });
  const val = (r, dev, key) => r.spectral[dev][key];

  const xs = rows.flatMap((r) => [start(r) + 8, start(r) + 22]);
  const alt = (dev) => rows.flatMap((r) => (dev === 0 ? [val(r, "ax6", "band_rms_mg")[band], null] : [null, val(r, "apple", "band_rms_mg")[band]]));
  const top = Math.max(1, ...rows.map((r) => Math.max(val(r, "ax6", "band_rms_mg")[band], val(r, "apple", "band_rms_mg")[band]))) * 1.1;
  mkPlot(el.querySelector('[data-p="bars"]'), opt({
    series: [{ stroke: COL.ax6, fill: alpha(COL.ax6, 0.8), paths: barPaths, points: { show: false } }, { stroke: COL.apple, fill: alpha(COL.apple, 0.8), paths: barPaths, points: { show: false } }],
    scales: { y: { range: [0, top] } }, yAxes: [yAxis("band RMS (mg)")], data: [xs, alt(0), alt(1)] }));

  const mid = rows.map((r) => start(r) + EPOCH_S / 2);
  const diff = rows.map((r) => val(r, "apple", "band_rms_mg")[band] - val(r, "ax6", "band_rms_mg")[band]);
  const dmax = Math.max(0.5, ...diff.map(Math.abs)) * 1.1;
  mkPlot(el.querySelector('[data-p="diff"]'), opt({
    series: [{ stroke: COL.ink2, fill: alpha(COL.ink2, 0.67), paths: uPlot.paths.bars({ size: [0.7, 80], gap: 1 }), points: { show: false } }],
    scales: { y: { range: [-dmax, dmax] } }, yAxes: [yAxis("Apple − AX6 (mg)")], data: [mid, diff] }));

  // Spectral centroid (0.1-20 Hz, noise-corrected): where each device's power actually sits in that band -- context for
  // why a band RMS moved, not a repeat of it.
  mkPlot(el.querySelector('[data-p="centroid"]'), opt({
    series: [{ stroke: COL.ax6, width: 1.2, points: { show: true, size: 4, fill: COL.ax6, stroke: COL.ax6 } },
      { stroke: COL.apple, width: 1.2, dash: [6, 4], points: { show: true, size: 4, fill: COL.apple, stroke: COL.apple } }],
    scales: { y: { range: [0, 20] } }, yAxes: [yAxis("centroid Hz", { size: 58 })],
    data: [mid, rows.map((r) => val(r, "ax6", "centroid_hz")), rows.map((r) => val(r, "apple", "centroid_hz"))] }));

  if (wavelet) el.insertAdjacentHTML("beforeend",
    `<div class="banner" style="flex:none"><small>MODWT (${wavelet.name}, ${wavelet.level} levels) per-epoch energies are in the epoch JSON (\`spectral.{dev}.modwt_energy\`) but not yet plotted here.</small></div>`);
}
