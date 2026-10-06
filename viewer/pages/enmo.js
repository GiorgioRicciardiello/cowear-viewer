// Page 3: 30 s ENMO bars for both devices, their difference, and the within-epoch r strip. Colour = DEVICE.
import { COL, EPOCH_S, legendHTML, mkPlot, yAxis, windowShadePlugin, epochRange, alpha } from "../common.js";
import { enmoBanner, enmoRkeyBanner } from "../copy.js";

const MIN_SPAN_S = 300;   // a 30 s page would show a single bar: draw >= 5 min of context and shade the real page window

export async function render(el, ctx) {
  const { epochs, t0, t1 } = ctx;
  const spanEnd = Math.max(t1, t0 + MIN_SPAN_S);
  const [e0, e1] = epochRange(t0, spanEnd, epochs.length);
  const rows = epochs.slice(e0, e1);
  const start = (r) => r.epoch * EPOCH_S;
  const shade = spanEnd > t1 ? [windowShadePlugin(t0, t1)] : [];
  el.innerHTML = `${legendHTML("device")}<div class="banner">${enmoBanner(spanEnd > t1)}</div>` +
    `<div class="banner" data-key="rkey">${enmoRkeyBanner}</div>` +
    `<div class="stack enmo"><div class="plotbox" data-p="bars"></div><div class="plotbox" data-p="diff"></div><div class="plotbox" data-p="r"></div></div>`;
  const xRange = [t0, spanEnd], barPaths = uPlot.paths.bars({ size: [0.9, 80], gap: 1 });
  const opt = (extra) => ({ xRange, plugins: shade, sync: "enmo", ...extra });

  const xs = rows.flatMap((r) => [start(r) + 8, start(r) + 22]);
  const alt = (dev) => rows.flatMap((r, i) => (dev === 0 ? [r.enmo_ax6, null] : [null, r.enmo_apple]));
  const top = Math.max(1, ...rows.map((r) => Math.max(r.enmo_ax6, r.enmo_apple))) * 1.1;
  mkPlot(el.querySelector('[data-p="bars"]'), opt({
    series: [{ stroke: COL.ax6, fill: alpha(COL.ax6, 0.8), paths: barPaths, points: { show: false } }, { stroke: COL.apple, fill: alpha(COL.apple, 0.8), paths: barPaths, points: { show: false } }],
    scales: { y: { range: [0, top] } }, yAxes: [yAxis("ENMO (mg)")], data: [xs, alt(0), alt(1)] }));

  const mid = rows.map((r) => start(r) + EPOCH_S / 2);
  const dmax = Math.max(0.5, ...rows.map((r) => Math.abs(r.diff))) * 1.1;
  mkPlot(el.querySelector('[data-p="diff"]'), opt({
    series: [{ stroke: COL.ink2, fill: alpha(COL.ink2, 0.67), paths: uPlot.paths.bars({ size: [0.7, 80], gap: 1 }), points: { show: false } }],
    scales: { y: { range: [-dmax, dmax] } }, yAxes: [yAxis("Apple − AX6 (mg)")], data: [mid, rows.map((r) => r.diff)] }));

  // Primary: lag-0 r (bars). Peak r (filled) is drawn only with its 95th-percentile null (hollow) and a connector, so the gap is what is read.
  const pk = rows.map((r) => (r.within_epoch_r == null || r.within_epoch_r_null == null ? null : r.within_epoch_r));
  const nl = rows.map((r, i) => (pk[i] == null ? null : r.within_epoch_r_null));
  const dot = (fill, stroke) => ({ paths: () => null, points: { show: true, size: 6, fill, stroke, width: 1.5 } });
  const link = { hooks: { draw: [(u) => {
    const { ctx } = u, dpr = devicePixelRatio || 1;
    ctx.save(); ctx.strokeStyle = COL.ink2; ctx.lineWidth = dpr;
    mid.forEach((x, i) => { if (pk[i] == null) return; const px = u.valToPos(x, "x", true); ctx.beginPath(); ctx.moveTo(px, u.valToPos(nl[i], "y", true)); ctx.lineTo(px, u.valToPos(pk[i], "y", true)); ctx.stroke(); });
    ctx.restore();
  }] } };
  mkPlot(el.querySelector('[data-p="r"]'), opt({
    plugins: [...shade, link],
    series: [{ stroke: COL.ink2, fill: alpha(COL.ink2, 0.53), paths: uPlot.paths.bars({ size: [0.7, 80], gap: 1 }), points: { show: false } },
      dot(COL.ink, COL.ink), dot(COL.surface, COL.ink2),
      { stroke: COL.ink, paths: () => null, points: { show: true, size: 7, fill: COL.ink, stroke: COL.ink } }],
    scales: { y: { range: [-1, 1] } }, yAxes: [yAxis("lag-0 r", { size: 58 })],
    data: [mid, rows.map((r) => r.within_epoch_r_lag0), pk, nl, rows.map((r) => (r.noise_floor_flag ? -0.9 : null))] }));
}
