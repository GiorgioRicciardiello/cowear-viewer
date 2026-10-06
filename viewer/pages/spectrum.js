// Page 6: per-device spectrograms of the visible window, plus the whole night's Welch PSD (with each device's own
// noise floor) and coherence. Python computes every number here (spectral.py -> viewer_export.spectra_summary /
// spectrogram_chunks); this file only draws. VM is rotation-invariant, so unlike
// pages 1-2 the raw|aligned frame toggle has no effect on anything on this page.
import { COL, legendHTML, mkPlot, alpha, canvasFont } from "../common.js";
import { segment } from "../data.js";
import { spectrumBanner, spectrumNoiseWarn, spectrumWelchNote } from "../copy.js";

const DB_RANGE = [-110, -40];
// Cividis (D5): CVD-safe by construction and perceptually uniform, so this sequential ramp no longer shares a hue
// with the salmon AX6 device colour (the former magma-like ramp's midtones were orange/red, the same hue family).
const RAMP = [[0, [0, 32, 76]], [0.25, [63, 72, 95]], [0.5, [112, 112, 110]], [0.75, [165, 149, 91]], [0.9, [211, 192, 79]], [1, [255, 234, 70]]];
const RANGES = [[0, 20, "0–20 Hz (comparable)"], [0, 25, "0–25 Hz (Apple Nyquist)"], [0, 50, "0–50 Hz (AX6 full)"]];
let freqMode = 0;   // index into RANGES; survives page steps

function colorFor(v, lo, hi) {
  if (v == null || !Number.isFinite(v)) return null;
  const t = Math.min(1, Math.max(0, (v - lo) / (hi - lo)));
  let a = RAMP[0], b = RAMP[RAMP.length - 1];
  for (let i = 0; i < RAMP.length - 1; i++) if (t >= RAMP[i][0] && t <= RAMP[i + 1][0]) { a = RAMP[i]; b = RAMP[i + 1]; break; }
  const f = (t - a[0]) / (b[0] - a[0] || 1);
  return a[1].map((c, i) => Math.round(c + f * (b[1][i] - c)));
}

// seg: { t, ch } from data.js's segment() on the "spec" tier -- ch[j] is frequency bin j's dB time series.
function heatmapPlugin(seg, nFreq, freqMax) {
  const nT = seg.t.length;
  const off = document.createElement("canvas");
  off.width = Math.max(nT, 1); off.height = Math.max(nFreq, 1);
  if (nT && nFreq) {
    const octx = off.getContext("2d"), img = octx.createImageData(nT, nFreq);
    for (let i = 0; i < nT; i++) for (let j = 0; j < nFreq; j++) {
      const c = colorFor(seg.ch[j][i], DB_RANGE[0], DB_RANGE[1]);
      const idx = ((nFreq - 1 - j) * nT + i) * 4;   // row 0 of the image = the highest frequency (drawn at the top)
      if (c) { img.data[idx] = c[0]; img.data[idx + 1] = c[1]; img.data[idx + 2] = c[2]; img.data[idx + 3] = 255; }
    }
    octx.putImageData(img, 0, 0);
  }
  return { hooks: { drawClear: [(u) => {
    if (!nT) return;
    const { ctx } = u, b = u.bbox;
    ctx.save(); ctx.imageSmoothingEnabled = false;
    ctx.beginPath(); ctx.rect(b.left, b.top, b.width, b.height); ctx.clip();
    const x0 = u.valToPos(seg.t[0], "x", true), x1 = u.valToPos(seg.t[nT - 1], "x", true);
    const y0 = u.valToPos(freqMax, "y", true), y1 = u.valToPos(0, "y", true);
    ctx.drawImage(off, x0, y0, Math.max(x1 - x0, 1), Math.max(y1 - y0, 1));
    ctx.restore();
  }] } };
}

// Absence, not silence: Apple has no samples above its own Nyquist at all. Only drawn when the selected range
// actually reaches above 25 Hz (§3.2): in 0-20/0-25 mode there is nothing above the visible top to grey out.
function noDataAbovePlugin(fromHz, toHz) {
  return { hooks: { draw: [(u) => {
    const { ctx } = u, b = u.bbox, dpr = devicePixelRatio || 1;
    const y0 = u.valToPos(toHz, "y", true), y1 = u.valToPos(fromHz, "y", true);
    ctx.save(); ctx.fillStyle = alpha(COL.grid, 0.92); ctx.fillRect(b.left, y0, b.width, y1 - y0);
    ctx.fillStyle = COL.ink2; ctx.font = canvasFont(11, dpr); ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(`NO DATA above ${fromHz} Hz (Apple's own Nyquist)`, b.left + b.width / 2, (y0 + y1) / 2);
    ctx.restore();
  }] } };
}

function hlinePlugin(v, label) {
  return { hooks: { draw: [(u) => {
    const { ctx } = u, b = u.bbox, dpr = devicePixelRatio || 1, y = u.valToPos(v, "y", true);
    if (y < b.top || y > b.top + b.height) return;
    ctx.save(); ctx.strokeStyle = COL.ink2; ctx.setLineDash([4 * dpr, 3 * dpr]); ctx.lineWidth = dpr;
    ctx.beginPath(); ctx.moveTo(b.left, y); ctx.lineTo(b.left + b.width, y); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = COL.ink2; ctx.font = canvasFont(10, dpr); ctx.textAlign = "left"; ctx.fillText(label, b.left + 4 * dpr, y - 4 * dpr);
    ctx.restore();
  }] } };
}

function freqAxis(label) {
  return { show: true, stroke: COL.ink2, font: "11px system-ui", size: 42, label, labelFont: "11px system-ui", labelSize: 14,
    grid: { stroke: COL.grid, width: 1 }, ticks: { stroke: COL.grid, width: 1 } };
}

export async function render(el, ctx) {
  const { man, t0, t1, spectra } = ctx;
  if (!spectra) { el.innerHTML = `<div class="banner warn">spectra.json did not load: rebuild the viewer artifacts.</div>`; return; }
  const specInfo = man.chunks.ax6.spec, nFreq = specInfo.n_freq;
  const [ax6Seg, appleSeg] = await Promise.all([segment(man, "ax6", "spec", t0, t1), segment(man, "apple", "spec", t0, t1)]);
  if (ctx.stale()) return;

  const [rLo, rHi] = RANGES[freqMode];
  const rangeOpts = RANGES.map(([, , l], i) => `<option value="${i}" ${i === freqMode ? "selected" : ""}>${l}</option>`).join("");

  el.innerHTML = `${legendHTML("device")}` +
    `<div class="banner">${spectrumBanner(specInfo, DB_RANGE)} <label>Frequency range <select class="rangesel">${rangeOpts}</select></label></div>` +
    `<div class="banner warn">${spectrumNoiseWarn}</div>` +
    `<div class="stack spectrum">` +
      `<div class="plotbox" data-p="ax6"></div><div class="plotbox" data-p="apple"></div>` +
      `<div class="plotbox" data-p="welch"></div><div class="plotbox" data-p="coh"></div>` +
    `</div>` +
    `<div class="banner">${spectrumWelchNote}</div>`;

  el.querySelector(".rangesel").addEventListener("change", (e) => { freqMode = Number(e.target.value); render(el, ctx); });

  const xRange = [t0, t1];
  mkPlot(el.querySelector('[data-p="ax6"]'), { xRange, sync: "spectrum", series: [{ show: false }],
    scales: { y: { range: [rLo, rHi] } }, yAxes: [freqAxis("AX6 Hz")], plugins: [heatmapPlugin(ax6Seg, nFreq, 50)], data: [ax6Seg.t] });
  mkPlot(el.querySelector('[data-p="apple"]'), { xRange, sync: "spectrum", series: [{ show: false }],
    scales: { y: { range: [rLo, rHi] } }, yAxes: [freqAxis("Apple Hz")],
    plugins: [heatmapPlugin(appleSeg, nFreq, 25), ...(rHi > 25 ? [noDataAbovePlugin(25, rHi)] : [])], data: [appleSeg.t] });

  const psd = spectra.psd, freqRange = [0.1, 25];
  mkPlot(el.querySelector('[data-p="welch"]'), { xRange: freqRange, xAxis: freqAxis("dB (g²/Hz)"), noData: false,
    scales: { y: { range: [-110, -20] } },
    series: [{ stroke: COL.ax6, width: 1.3, points: { show: false } }, { stroke: COL.ax6, width: 1, dash: [3, 3], points: { show: false } },
      { stroke: COL.apple, width: 1.3, points: { show: false } }, { stroke: COL.apple, width: 1, dash: [3, 3], points: { show: false } }],
    yAxes: [freqAxis("dB (g²/Hz)")], data: [psd.ax6.f, psd.ax6.raw_db, psd.ax6.noise_db, psd.apple.raw_db, psd.apple.noise_db] });

  const coh = spectra.coherence;
  mkPlot(el.querySelector('[data-p="coh"]'), { xRange: freqRange, xAxis: freqAxis("Hz"), noData: false,
    scales: { y: { range: [0, 1] } }, yAxes: [freqAxis("coherence")],
    plugins: [hlinePlugin(coh.bias_floor, `bias floor (independent noise averages here, not 0)`)],
    series: [{ stroke: COL.ink, width: 1.3, points: { show: false } }], data: [coh.f, coh.value] });
}
