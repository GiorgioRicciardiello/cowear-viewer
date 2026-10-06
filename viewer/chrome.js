// Persistent spine: header (toggles, night-level scalars, standing caveats) and footer (page width, stepping).
import { clock, fmt, geo, rotationFits, Q, ciHTML, noteHTML } from "./common.js";

const TOGGLES = [
  ["cal", "signal", [["raw", "raw"], ["calibrated", "calibrated"]]],
  ["lag", "clock lag", [["zero", "lag 0 s"], ["estimated", "lag estimated"]]],
  ["rate", "rate", [["native", "native"], ["grid30", "30 Hz"]]],
  ["frame", "frame", [["raw", "raw frame"], ["aligned", "aligned frame"]]],
];
export const WIDTHS = [[30, "30 s"], [120, "2 m"], [300, "5 m"], [1800, "30 m"]];

const seg = (key, items, cur) => `<span class="seg" data-key="${key}">${items.map(([v, l]) => `<button data-v="${v}" class="${String(v) === String(cur) ? "on" : ""}">${l}</button>`).join("")}</span>`;

function floorLine(nf, n, cal) {
  const d = (k) => `${nf[k].derivation} (${nf[k].n_windows} windows; sigma_VM mean ${fmt(nf[k].sigma_vm_mg_mean, 2)} / median ${fmt(nf[k].sigma_vm_mg_median, 2)} mg)`.replace(/"/g, "&quot;");
  const j = nf.joint;
  return `<div class="floor"><b>Noise floor (per device):</b> <span title="${d("ax6")}">AX6 ${Q.mg(nf.ax6.value_mg)}</span> &middot; <span title="${d("apple")}">Apple ${Q.mg(nf.apple.value_mg)}</span>` +
    ` &nbsp;|&nbsp; <b>Epochs below their own floor:</b> AX6 ${Q.pct(100 * nf.ax6.fraction_flagged)} (${Q.count(nf.ax6.n_flagged)}) &middot; Apple ${Q.pct(100 * nf.apple.fraction_flagged)} (${Q.count(nf.apple.n_flagged)}) &middot; ` +
    `<span class="headline">${Q.pct(100 * j.fraction_flagged)} (${Q.count(j.n_flagged)} of ${Q.count(n)}) noise-dominated on at least one device</span>` +
    `${cal === "raw" ? ` <small>(raw signal, offsets uncorrected: these fractions describe the raw ENMO, not the calibrated night)</small>` : ""}</div>`;
}

function blockLine(night) {
  const b = night.block_sensitivity, first = b[0], last = b[b.length - 1];
  return `<div class="blocks"><b>r interval depends on block length:</b> ${b.map((x) => `${x.block_min} min [${fmt(x.lo)}, ${fmt(x.hi)}]`).join(" &middot; ")}` +
    ` <small>(header intervals use the ${night.block_min}-min block; the ${last.block_min}-min interval is ${fmt(last.hi - last.lo, 2)} wide vs ${fmt(first.hi - first.lo, 2)})</small></div>`;
}

// The rotation misfit is a standing property of this night, not news on every tab: it lives once
// in the header, collapsed, rather than costing ~45px of plot height on all six pages (frameBanner still draws its
// own per-page consequence of a bad fit; this is the one summary statement of the fact itself).
function rotationNote(man) {
  const rot = man.rotation;
  if (rotationFits(rot)) return "";
  const res = `${fmt(rot.residual_deg, 1)}&deg; residual, uncertainty &plusmn;${fmt(rot.uncertainty_deg, 1)}&deg;${rot.n_windows == null ? "" : `, ${Q.count(rot.n_windows)} windows`}`;
  return `<details class="rotnote"><summary><b>The rotation does not fit</b> (${res}) -- axis-to-axis comparison is not available for this pair.</summary>` +
    noteHTML("warn", [`No single rotation relates the AX6 and Apple Watch frames (${res}). The gravity directions of the two devices are essentially unrelated (different wrists, or the watch moving on the arm).`]) +
    `</details>`;
}

function calibrationWarnings(man) {
  const out = [];
  for (const [k, d] of Object.entries(man.devices)) {
    const c = d.calibration;
    if (c.converged === false) out.push(`${d.label} calibration did NOT converge (rms ${Q.mg(c.rms_g * 1000)})`);
    if (c.low_orientation_coverage === true) out.push(`${d.label} calibration has low orientation coverage (${fmt(c.orientation_coverage, 2)}): offsets are poorly constrained`);
  }
  return out.length ? `<div class="hdrwarn">${noteHTML("warn", out)}</div>` : "";
}

// The night-level block follows the raw|calibrated toggle: night[cal] is what the toggle selects.
export function updateMetrics(el, man, night, cal) {
  const s = night.scalars;
  el.querySelector(".metrics").innerHTML =
    `<div class="row2"><b>Night level (${cal}), 30 s ENMO, 95% block-bootstrap CI:</b> Pearson r ${ciHTML(s.r)} &middot; Spearman &rho; ${ciHTML(s.rho)} &middot; CCC ${ciHTML(s.ccc)} &middot; ICC(2,1) ${ciHTML(s.icc)} &middot; ` +
    `bias ${ciHTML(s.bias, Q.mg)} &middot; 95% LoA ${Q.mg(s.loa_lo.est)} to ${Q.mg(s.loa_hi.est)} &middot; sleep-mask &kappa; ${ciHTML(s.kappa)}</div>` +
    `${blockLine(night)}${floorLine(night.noise_floor, night.n_epochs, cal)}${calibrationWarnings(man)}${rotationNote(man)}`;
}

export function buildHeader(el, man, night, st, onChange) {
  const w = man.window;
  const hm = (iso) => iso.split("T")[1].slice(0, 8);
  el.innerHTML = `<div class="row1"><h1>Co-wear viewer</h1><span class="sess">${man.session_id}</span>` +
    `<span class="span">recorded ${hm(w.available_start)}–${hm(w.available_end)} (requested ${hm(w.requested_start).slice(0, 5)}–${hm(w.requested_end).slice(0, 5)}; intersection shown, the rest is no data)</span>` +
    `<span class="toggles">${TOGGLES.map(([k, , items]) => seg(k, items, st[k])).join("")}</span></div><div class="metrics"></div>` +
    `<div class="row3">${man.synthetic ? `<span class="synth">SYNTHETIC DEMO DATA</span> ` : ""}<b>n = 1 subject, 1 night.</b> ` +
    `This viewer calibrates with gain + offset (calibrate_acc.m); the ABK feature pipeline used ICP with rotation, so ABK features and sleep flags come from a differently calibrated signal. ` +
    `Estimates and intervals only; no verdicts.</div>`;
  updateMetrics(el, man, night, st.cal);
  el.querySelector(".toggles").addEventListener("click", (e) => {
    const key = e.target.closest(".seg")?.dataset.key, v = e.target.dataset.v;
    if (!key || v == null) return;
    el.querySelector(`[data-key="${key}"]`).querySelectorAll("button").forEach((b) => b.classList.toggle("on", b === e.target));
    onChange({ [key]: v });
  });
}

export function buildFooter(el, st, onChange, step) {
  el.innerHTML = `<span class="lab">Page width</span>${seg("width", WIDTHS, st.width)}` +
    `<button class="step" data-d="-1" title="previous page (left arrow)">&#9664;</button><button class="step" data-d="1" title="next page (right arrow)">&#9654;</button>` +
    `<span class="pos"></span><span class="hint">Left / Right arrows: step one page. Shift + arrow: step one 30 s epoch. Overview: click or drag to move the page.</span>`;
  el.addEventListener("click", (e) => {
    if (e.target.dataset.d) return step(Number(e.target.dataset.d), false);
    const v = e.target.closest('[data-key="width"]') && e.target.dataset.v;
    if (v) onChange({ width: Number(v) });
  });
}

export function updateFooter(el, st) {
  el.querySelectorAll('[data-key="width"] button').forEach((b) => b.classList.toggle("on", Number(b.dataset.v) === st.width));
  el.querySelector(".pos").textContent = `${clock(st.pos)} – ${clock(st.pos + st.width)}  (t = ${fmt(st.pos, 0)} s from anchor)`;
}

export { geo };
