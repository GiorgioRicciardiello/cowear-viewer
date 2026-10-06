// Shared constants and uPlot helpers. Colours: single source is viewer/style.css `:root`; the axis
// triple was chosen by search for the best worst-case CIEDE2000 across normal vision + deuteran/protan/tritan simulation
// (min 29.6; figures.py's own bar is CVD dE 24.7), contrast >= 3:1 on the #fcfcfb surface, and distinct from the device pair.
const css = getComputedStyle(document.documentElement);
const tok = (n) => css.getPropertyValue(n).trim();

export const FONT_STACK = "system-ui";
// Canvas text is not scaled by uPlot itself (only its own axis/label fonts are); every plugin that writes to a
// canvas with ctx.font must multiply by devicePixelRatio by hand -- this is the one place that does it.
export const canvasFont = (px, dpr = devicePixelRatio || 1) => `${px * dpr}px ${FONT_STACK}`;

export const COL = Object.freeze({
  ax6: tok("--ax6"), apple: tok("--apple"),
  ax6Strong: tok("--ax6-strong"), appleStrong: tok("--apple-strong"),
  ink: tok("--ink"), ink2: tok("--ink2"),
  grid: tok("--grid"), surface: tok("--surface"),
  lineWeak: tok("--line-weak"), accent: tok("--accent"),
});

/** `a` in [0,1]. Accepts #rgb/#rrggbb; returns rgba() so callers stop concatenating strings onto hex tokens (D4) --
 * that concatenation silently breaks (paints transparent, no error) the moment a token stops being 6-digit hex. */
export function alpha(color, a) {
  const c = color.trim();
  if (c[0] === "#") {
    let hex = c.slice(1);
    if (hex.length === 3) hex = hex.split("").map((h) => h + h).join("");
    const r = parseInt(hex.slice(0, 2), 16), g = parseInt(hex.slice(2, 4), 16), b = parseInt(hex.slice(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${a})`;
  }
  return c;
}

export const AXIS_COL = ["#aa0929", "#248f8f", "#0909aa"];
export const AXES = ["x", "y", "z"];
export const DEVS = ["ax6", "apple"];
export const DEV_LABEL = { ax6: "AX6", apple: "Apple Watch" };
export const EPOCH_S = 30;
export const APPLE_DASH = [6, 4];

// D2: page 1 remaps device onto colour (so it reads consistently with every other page) and axis onto dash pattern.
export const DEV_DASH = Object.freeze({ ax6: undefined, apple: undefined });
export const AXIS_DASH = Object.freeze({ x: undefined, y: [6, 3], z: [2, 3] });

// geo: time geometry, all times in seconds relative to the epoch-grid anchor (recording start).
export const geo = { sod: 0, nightS: 0, reqStart: 0, reqEnd: 0, reqStartLabel: "", availEnd: 0 };

const secOfDay = (iso) => { const [h, m, s] = iso.split("T")[1].split(":").map(Number); return h * 3600 + m * 60 + s; };

export function initGeo(man) {
  geo.sod = secOfDay(man.anchor);
  geo.nightS = man.n_epochs * man.epoch_s;
  const rel = (iso) => (new Date(iso) - new Date(man.anchor)) / 1000;
  geo.reqStart = Math.round(rel(man.window.requested_start));
  geo.reqEnd = Math.round(rel(man.window.requested_end));
  geo.reqStartLabel = man.window.requested_start.split("T")[1].slice(0, 8);
}

const pad = (n) => String(n).padStart(2, "0");
export function clock(t, sec = true) {
  const total = Math.floor(((geo.sod + t) % 86400 + 86400) % 86400);
  const h = Math.floor(total / 3600), m = Math.floor(total / 60) % 60, s = total % 60;
  return sec ? `${pad(h)}:${pad(m)}:${pad(s)}` : `${pad(h)}:${pad(m)}`;
}

const STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200];
function timeSplits(u, ai, min, max) {
  const span = max - min;
  const step = STEPS.find((s) => span / s <= 9) || 7200;
  const first = Math.ceil((min + geo.sod) / step) * step - geo.sod;
  const out = [];
  for (let v = first; v <= max; v += step) out.push(v);
  return out;
}

export function timeAxis(show = true) {
  return { show, stroke: COL.ink2, font: "11px system-ui", grid: { stroke: COL.grid, width: 1 }, ticks: { stroke: COL.grid, width: 1 }, splits: timeSplits,
    values: (u, v) => { const st = v.length > 1 ? v[1] - v[0] : 60; return v.map((x) => clock(x, st < 60)); } };
}

export function yAxis(label, extra = {}) {
  return { stroke: COL.ink2, font: "11px system-ui", size: 58, label, labelFont: "11px system-ui", labelSize: 14,
    grid: { stroke: COL.grid, width: 1 }, ticks: { stroke: COL.grid, width: 1 }, ...extra };
}

let hatch = null;
function hatchPattern(ctx) {
  if (hatch) return hatch;
  const c = document.createElement("canvas");
  c.width = c.height = 10;
  const g = c.getContext("2d");
  g.fillStyle = "#dedcd6"; g.fillRect(0, 0, 10, 10);
  g.strokeStyle = "#bdbbb3"; g.lineWidth = 1.5;
  g.beginPath(); g.moveTo(0, 10); g.lineTo(10, 0); g.stroke();
  hatch = ctx.createPattern(c, "repeat");
  return hatch;
}

// Absence must not look like stillness: grey hatched + labelled, drawn under the series.
export function noDataPlugin() {
  return { hooks: { drawClear: [(u) => {
    const { ctx } = u, b = u.bbox, x = u.scales.x, dpr = devicePixelRatio || 1;
    const regions = [[-Infinity, 0, `NO DATA  ${geo.reqStartLabel}–${clock(0)}`], [geo.nightS, Infinity, "NO DATA  (after recording end)"]];
    ctx.save();
    ctx.beginPath(); ctx.rect(b.left, b.top, b.width, b.height); ctx.clip();
    for (const [a, z, label] of regions) {
      const lo = Math.max(a, x.min), hi = Math.min(z, x.max);
      if (hi <= lo) continue;
      const p0 = u.valToPos(lo, "x", true), p1 = u.valToPos(hi, "x", true);
      ctx.fillStyle = hatchPattern(ctx);
      ctx.fillRect(p0, b.top, p1 - p0, b.height);
      if ((p1 - p0) / dpr > 90) {
        ctx.fillStyle = COL.ink2; ctx.font = `bold ${canvasFont(12, dpr)}`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
        const txt = label;
        const w = ctx.measureText(txt).width;
        ctx.save(); ctx.fillStyle = alpha(COL.surface, 0.85); ctx.fillRect((p0 + p1) / 2 - w / 2 - 6 * dpr, b.top + b.height / 2 - 10 * dpr, w + 12 * dpr, 20 * dpr); ctx.restore();
        ctx.fillText(txt, (p0 + p1) / 2, b.top + b.height / 2);
      }
    }
    ctx.restore();
  }] } };
}

// Shade the actual page window when a plot shows a wider context (ENMO page at short page widths).
export function windowShadePlugin(t0, t1) {
  return { hooks: { drawClear: [(u) => {
    const { ctx } = u, b = u.bbox;
    const p0 = u.valToPos(t0, "x", true), p1 = u.valToPos(t1, "x", true);
    ctx.save(); ctx.fillStyle = alpha(COL.accent, 0.08); ctx.fillRect(p0, b.top, Math.max(p1 - p0, 2), b.height); ctx.restore();
  }] } };
}

// Live uPlot instances, plus any extra teardown a plugin registered against them (e.g. a ResizeObserver, see mkPlot).
const plots = [];
const teardowns = new Map();
export const register = (u, onDestroy) => { plots.push(u); if (onDestroy) teardowns.set(u, onDestroy); return u; };
export function destroyPlots() {
  while (plots.length) {
    const u = plots.pop();
    teardowns.get(u)?.();
    teardowns.delete(u);
    u.destroy();
  }
}

// Fixed x-range, canvas plot sized to its container, and kept in sync with the container (not only the window) via
// a ResizeObserver: a banner rewrapping or a section collapsing reaches every page this way,
// with no per-page edit. `xAxis`/`noData` default to the clock-time x-axis and the recording-absence hatch every
// page so far has used; a frequency-domain page (pages/spectrum.js) overrides both, since geo.reqStart/geo.nightS
// are real seconds and would hatch out a Hz axis as if it were almost all missing data.
export function mkPlot(el, { xRange, series, yAxes, scales = {}, plugins = [], data, sync, bands, legendShow = false, persist = false, xAxis = timeAxis(), noData = true }) {
  const opts = {
    width: Math.max(el.clientWidth, 100), height: Math.max(el.clientHeight, 60), legend: { show: legendShow }, padding: [6, 10, 0, 0],
    cursor: { drag: { x: false, y: false }, sync: sync ? { key: sync } : undefined },
    scales: { x: { time: false, range: () => xRange }, ...scales },
    axes: [xAxis, ...yAxes], series: [{}, ...series], plugins: [...(noData ? [noDataPlugin()] : []), ...plugins], bands,
  };
  const u = new uPlot(opts, data, el);
  const ro = new ResizeObserver(([e]) => {
    const { width, height } = e.contentRect;
    if (Math.abs(width - u.width) > 1 || Math.abs(height - u.height) > 1)
      u.setSize({ width: Math.max(width, 100), height: Math.max(height, 60) });
  });
  ro.observe(el);
  if (!persist) register(u, () => ro.disconnect());
  return u;
}

export function legendHTML(kind) {
  const sw = (c, dash) => `<span class="sw" style="--sw:${c}${dash ? ";border-top-style:dashed" : ""}"></span>`;
  if (kind === "axis")
    return `<div class="legend"><b>Colour = DEVICE</b> ${sw(COL.ax6)}AX6 &nbsp; ${sw(COL.apple)}Apple Watch &nbsp;|&nbsp; ` +
      `<b>Axis = line pattern</b> x solid &nbsp; y &ndash; &ndash; dashed &nbsp; z &middot;&middot;&middot; dotted</div>`;
  return `<div class="legend"><b>Colour = DEVICE</b> ${sw(COL.ax6)}AX6 &nbsp; ${sw(COL.apple, true)}Apple Watch (dashed on trace pages) &nbsp;|&nbsp; ` +
    `${sw(COL.ink2)}grey = derived quantity (comparison of the two devices)</div>`;
}

// A fitted rotation whose residual is at or above this does not relate the two frames (the fit's own uncertainty on the real night is ~15 deg).
export const ROT_MISFIT_DEG = 15;
export const rotationFits = (rot) => rot.residual_deg < ROT_MISFIT_DEG;

export function frameBanner(ctx, tier) {
  const { man, st } = ctx;
  const rot = man.rotation, res = `${fmt(rot.residual_deg, 1)}&deg; RMS residual, uncertainty &plusmn;${fmt(rot.uncertainty_deg, 1)}&deg;${rot.n_windows == null ? "" : `, ${rot.n_windows.toLocaleString()} windows`}`;
  if (st.frame === "raw")
    return `<div class="banner warn"><b>Raw frame.</b> The two sensors' frames differ (wrist and mounting orientation unknown), so axis-to-axis traces of AX6 and Apple Watch are NOT physically comparable here.` +
      `${rotationFits(rot) ? " Use the aligned frame to compare axes." : ` The fitted rotation does not fix this (${res}); see the aligned frame.`}</div>`;
  if (tier === "ctx4")
    return `<div class="banner warn"><b>30 m view = 4 Hz min/max envelope, rotation not applied.</b> An envelope cannot be rotated, so the Apple axes are in their own sensor frame here. Choose a page width of 5 m or less for the aligned frame.</div>`;
  if (!rotationFits(rot))
    return `<div class="banner warn big"><b>The rotation does not fit: ${fmt(rot.residual_deg, 1)}&deg; residual.</b> No single rotation relates the AX6 and Apple Watch frames (${res}). ` +
      `The gravity directions of the two devices are essentially unrelated (different wrists, or the watch moving on the arm), so the aligned axes below are NOT a corrected comparison and axis-to-axis comparison is not available for this pair. This view shows that, it does not enable it.</div>`;
  return `<div class="banner">Aligned frame: Apple rotated into the AX6 frame (${res}; roll about gravity is weakly identified).</div>`;
}

export const fmt = (v, d = 2) => (v == null || !Number.isFinite(v) ? "—" : v.toFixed(d));

/** Precision is a property of the QUANTITY, not of the surface that shows it. */
const unit = (s, u) => (s === "—" ? s : `${s} <span class="u">${u}</span>`);
export const Q = Object.freeze({
  mg: (v) => unit(fmt(v, 2), "mg"),
  g: (v) => unit(fmt(v, 4), "g"),
  db: (v) => unit(fmt(v, 1), "dB"),
  hz: (v) => unit(fmt(v, 2), "Hz"),
  deg: (v) => unit(fmt(v, 1), "°"),
  s: (v) => unit(fmt(v, 2), "s"),
  corr: (v) => fmt(v, 2),
  ratio: (v) => fmt(v, 2),
  count: (v) => (v == null ? "—" : v.toLocaleString()),
  pct: (v) => unit(fmt(v, 1), "%"),
});

/** est [lo, hi] -- the interval is always secondary to the estimate, never shown alone. */
export const ciHTML = (s, q = Q.corr) =>
  `<span class="est">${q(s.est)}</span><span class="ci">[${q(s.lo)}, ${q(s.hi)}]</span>`;

/** level: "info" | "caveat" | "warn". `items` are INDEPENDENT facts -- one <li> each, never joined. */
export const noteHTML = (level, items) => `<ul class="note note-${level}">` +
  items.map((t) => `<li>${t}</li>`).join("") + `</ul>`;

export const epochRange = (t0, t1, n) => { const e0 = Math.min(n, Math.max(0, Math.floor(t0 / EPOCH_S))); return [e0, Math.max(e0, Math.min(n, Math.ceil(t1 / EPOCH_S)))]; };
