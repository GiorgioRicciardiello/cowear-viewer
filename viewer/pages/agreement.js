// Page 4: Bland-Altman, ENMO scatter, correlation-vs-timescale (null floor shaded), and the epoch table.
// Routed through mkPlot: that is what gives this page the shared ResizeObserver and the
// canvasFont() DPR helper, same as every other page, instead of re-specifying uPlot's legend/padding/cursor by hand.
import { COL, mkPlot, legendHTML, yAxis, fmt, alpha, canvasFont, Q } from "../common.js";
import { mountTable } from "./agreement_table.js";
import { agreementBanner } from "../copy.js";

const NULL_FILL = alpha(COL.ink2, 0.3);
let metric = "enmo";

const sorted = (x, y) => { const o = x.map((_, i) => i).sort((a, b) => x[a] - x[b]); return [o.map((i) => x[i]), o.map((i) => y[i])]; };
const xAxis = (label, extra = {}) => ({ stroke: COL.ink2, font: "11px system-ui", label, labelFont: "11px system-ui", labelSize: 14, size: 40,
  grid: { stroke: COL.grid, width: 1 }, ticks: { stroke: COL.grid, width: 1 }, ...extra });
const panel = (el, cls) => el.querySelector(cls);
const dots = (fill) => ({ paths: () => null, points: { show: true, size: 4, fill, stroke: fill } });
const drawPlugin = (fn) => ({ hooks: { draw: [fn] } });

function hLine(u, y, dash, label) {
  const { ctx } = u, b = u.bbox, py = u.valToPos(y, "y", true), dpr = devicePixelRatio || 1;
  ctx.save(); ctx.strokeStyle = COL.ink; ctx.lineWidth = dpr; ctx.setLineDash(dash.map((d) => d * dpr));
  ctx.beginPath(); ctx.moveTo(b.left, py); ctx.lineTo(b.left + b.width, py); ctx.stroke();
  ctx.setLineDash([]); ctx.fillStyle = COL.ink; ctx.font = canvasFont(10, dpr); ctx.textAlign = "right"; ctx.fillText(label, b.left + b.width - 4 * dpr, py - 3 * dpr); ctx.restore();
}

function blandAltman(host, night) {
  const s = night.scalars, [x, y] = sorted(night.ba.mean, night.ba.diff);
  mkPlot(host, { xRange: undefined, noData: false, xAxis: xAxis("mean of both devices (mg)"), yAxes: [yAxis("Apple − AX6 (mg)")],
    scales: { x: { time: false } }, series: [dots(alpha(COL.ink2, 0.53))], data: [x, y],
    plugins: [drawPlugin((u) => { hLine(u, s.bias.est, [], `bias ${fmt(s.bias.est, 2)}`); hLine(u, s.loa_lo.est, [5, 4], `95% LoA ${fmt(s.loa_lo.est, 2)}`); hLine(u, s.loa_hi.est, [5, 4], `95% LoA ${fmt(s.loa_hi.est, 2)}`); })] });
}

function scatter(host, night) {
  const [x, y] = sorted(night.scatter.ax6, night.scatter.apple), all = [...x, ...y];
  const range = [Math.min(...all) * 0.8, Math.max(...all) * 1.25], log = { time: false, distr: 3, log: 10, range: () => range };
  mkPlot(host, { xRange: undefined, noData: false, xAxis: xAxis("ENMO AX6 (mg, log)"), yAxes: [yAxis("ENMO Apple (mg, log)")],
    scales: { x: log, y: log }, series: [dots(alpha(COL.ink2, 0.53))], data: [x, y],
    plugins: [drawPlugin((u) => { const { ctx } = u, a = u.valToPos(range[0], "x", true), z = u.valToPos(range[1], "x", true), ay = u.valToPos(range[0], "y", true), zy = u.valToPos(range[1], "y", true);
      ctx.save(); ctx.strokeStyle = COL.ink; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(a, ay); ctx.lineTo(z, zy); ctx.stroke(); ctx.restore(); })] });
}

const BASIS = { bootstrap_half_width: "implied by the bootstrap CI half-width (5-min block), not an independence count" };
const longest = (c) => c.block_sensitivity.map((b) => b[b.length - 1]);   // per w: the longest-block interval

function sweepData(night, m) {
  const c = night.sweep[m], idx = c.w_s.map((_, i) => i), L = longest(c);
  return { c, data: [idx, c.r, c.r_lo, c.r_hi, c.null_lo, c.null_hi, c.within_median, c.within_q25, c.within_q75, c.optimal_lag_s, L.map((b) => b.lo), L.map((b) => b.hi)] };
}

// Interval for the epoch-level ENMO r as a function of bootstrap block length: the interval is conditional on the block.
function blockPlot(host, night) {
  const b = night.block_sensitivity, idx = b.map((_, i) => i), est = night.scalars.r.est;
  const edge = { stroke: COL.ink2, width: 1.5, points: { show: true, size: 6, fill: COL.ink2 } };
  mkPlot(host, { xRange: undefined, noData: false,
    xAxis: xAxis("bootstrap block length", { splits: () => idx, values: (u, v) => v.map((i) => `${b[i].block_min} min`) }), yAxes: [yAxis("Pearson r, 95% CI")],
    scales: { x: { time: false, range: () => [-0.4, b.length - 0.4] }, y: { range: [-0.2, 1] } },
    series: [edge, edge, { stroke: COL.ink, dash: [4, 3], width: 1, points: { show: false } }],
    bands: [{ series: [2, 1], fill: alpha(COL.ink, 0.13) }],
    data: [idx, b.map((x) => x.lo), b.map((x) => x.hi), idx.map(() => est)] });
}

function timescale(host, night, info) {
  const { c, data } = sweepData(night, metric);
  const lag = { auto: true };
  const line = (stroke, extra = {}) => ({ stroke, width: 1.5, points: { show: false }, ...extra });
  const u = mkPlot(host, { xRange: undefined, noData: false,
    scales: { x: { time: false }, y: { range: [-0.1, 1.02] }, lag },
    xAxis: xAxis("moving-average window w", { splits: () => c.w_s.map((_, i) => i), values: (u, v) => v.map((i) => `${c.w_s[i]} s`) }),
    yAxes: [yAxis("r (night level)"), { scale: "lag", side: 1, stroke: COL.ink2, font: "11px system-ui", label: "lag maximising r (s)", labelFont: "11px system-ui", labelSize: 14, size: 44, grid: { show: false } }],
    series: [line(COL.ink, { width: 2.5, points: { show: true, size: 6, fill: COL.ink } }), line("transparent"), line("transparent"), line("transparent"), line("transparent"),
      line(COL.ink2, { dash: [2, 3] }), line("transparent"), line("transparent"), line(COL.ink2, { scale: "lag", dash: [8, 4], points: { show: true, size: 5, fill: COL.surface, stroke: COL.ink2 } }),
      line(COL.ink, { width: 1, dash: [1, 3] }), line(COL.ink, { width: 1, dash: [1, 3] })],
    bands: [{ series: [3, 2], fill: alpha(COL.ink, 0.13) }, { series: [5, 4], fill: NULL_FILL }, { series: [8, 7], fill: alpha(COL.ink2, 0.1) }],
    data });
  const write = () => {
    const d = sweepData(night, metric).c, bl = (i, k) => d.block_sensitivity[i][k === 0 ? 0 : d.block_sensitivity[i].length - 1];
    const ciAt = (k) => d.r.map((_, i) => { const b = bl(i, k); return `<td>[${fmt(b.lo, 2)}, ${fmt(b.hi, 2)}]</td>`; }).join("");
    const first = bl(0, 0).block_min, last = bl(0, 1).block_min, basis = [...new Set(d.n_eff_basis)].map((b) => BASIS[b] || b).join("; ");
    info.innerHTML = `<b>Read the gap between the curve and the shaded null floor, not the raw r.</b> ` +
      `<table class="mini"><tr><th>w</th>${d.w_s.map((w) => `<td>${w} s</td>`).join("")}</tr>` +
      `<tr><th>gap: r &minus; null upper</th>${d.r.map((r, i) => `<td>${fmt(r - d.null_hi[i], 2)}</td>`).join("")}</tr>` +
      `<tr><th>CI, ${first}-min block</th>${ciAt(0)}</tr><tr><th>CI, ${last}-min block</th>${ciAt(1)}</tr>` +
      `<tr><th>effective n*</th>${d.n_eff.map((n) => `<td>${Math.round(n).toLocaleString()}</td>`).join("")}</tr></table>` +
      `<div class="note11">* effective n is ${basis}. It is not a count of independent observations. Every CI is conditional on its block length; the lag maximising r is not a measured clock offset.</div>`;
  };
  write();
  return { u, write, data: (m) => sweepData(night, m).data };
}

export async function render(el, ctx) {
  const { night, epochs } = ctx;
  el.innerHTML = `${legendHTML("device")}<div class="banner">${agreementBanner}</div>` +
    `<div class="agr-top"><section><h3>Bland-Altman</h3><div class="plotbox ba"></div></section><section><h3>ENMO scatter, identity dashed</h3><div class="plotbox sc"></div></section>` +
    `<section><h3>Correlation vs timescale <span class="seg" data-seg="metric"><button data-m="enmo">ENMO</button><button data-m="vm">VM</button></span></h3><div class="plotbox ts"></div>` +
    `<div class="key"><span class="sw ci-shade"></span>bootstrap CI &nbsp;<span class="sw" style="--sw:${NULL_FILL}"></span><b>phase-randomised null floor</b> &nbsp;<span class="sw" style="height:0;border-top:2px dotted var(--ink)"></span>CI at the longest block &nbsp;<span class="sw dashed"></span>lag maximising r (right axis) &nbsp;dotted = within-epoch median (band IQR)</div>` +
    `</section>` +
    `<section><h3>Block-length sensitivity of r (30 s ENMO)</h3><div class="plotbox bl"></div>` +
    `<div class="note11">Solid points: 95% CI limits at each moving-block length; dashed: point estimate ${fmt(night.scalars.r.est, 2)}. ${night.block_sensitivity_note}</div></section><div class="tsinfo"></div></div><div class="agr-bottom"></div>`;
  blandAltman(panel(el, ".ba"), night);
  scatter(panel(el, ".sc"), night);
  blockPlot(panel(el, ".bl"), night);
  const ts = timescale(panel(el, ".ts"), night, panel(el, ".tsinfo"));
  const seg = el.querySelector('[data-seg="metric"]');
  const mark = () => seg.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.m === metric));
  seg.addEventListener("click", (e) => { if (!e.target.dataset.m) return; metric = e.target.dataset.m; mark(); ts.u.setData(ts.data(metric)); ts.write(); });
  mark();
  mountTable(panel(el, ".agr-bottom"), epochs, ctx.jumpToEpoch);
}
