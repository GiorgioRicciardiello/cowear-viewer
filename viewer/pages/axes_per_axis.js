// Page 2: one panel per axis (x, y, z), both devices in each. Colour = DEVICE.
import { AXES, COL, APPLE_DASH, legendHTML, frameBanner, mkPlot, yAxis } from "../common.js";
import { traces, joinTraces } from "../data.js";

export async function render(el, ctx) {
  const { man, st, t0, t1 } = ctx;
  const tr = await traces(man, st, t0, t1);
  if (ctx.stale()) return;
  const C = tr.ax6.ch.length, data = joinTraces(tr);
  el.innerHTML = `${legendHTML("device")}${frameBanner(ctx, tr.tier)}<div class="stack">${AXES.map((a) => `<div class="plotbox" data-axis="${a}"></div>`).join("")}</div>`;
  AXES.forEach((a, ai) => {
    // columns of this axis: 1 (min or only) [+ 3 (max)] per device
    const per = C === 6 ? [ai, ai + 3] : [ai];
    const cols = [...per.map((c) => 1 + c), ...per.map((c) => 1 + C + c)];
    const series = cols.map((_, i) => ({ stroke: i < per.length ? COL.ax6 : COL.apple, width: 1.2, dash: i < per.length ? undefined : APPLE_DASH, points: { show: false } }));
    mkPlot(el.querySelector(`[data-axis="${a}"]`), { xRange: [t0, t1], series, yAxes: [yAxis(`${a} axis (g)`)], data: [data[0], ...cols.map((c) => data[c])], sync: "axes-per-axis" });
  });
}
