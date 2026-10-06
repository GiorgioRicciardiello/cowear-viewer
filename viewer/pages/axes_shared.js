// Page 1: six traces in one panel. D2: colour = DEVICE (consistent with every other page), axis = dash pattern
// (x solid, y dashed, z dotted) -- 2 colours x 3 patterns = 6 separable series in one panel.
import { AXES, COL, AXIS_DASH, DEV_LABEL, DEVS, legendHTML, frameBanner, mkPlot, yAxis } from "../common.js";
import { traces, joinTraces } from "../data.js";

const vis = { x: true, y: true, z: true, ax6: true, apple: true };   // survives page steps

export async function render(el, ctx) {
  const { man, st, t0, t1 } = ctx;
  const tr = await traces(man, st, t0, t1);
  if (ctx.stale()) return;
  const C = tr.ax6.ch.length;
  const boxes = [...AXES.map((a) => `<label><input type="checkbox" data-k="${a}" ${vis[a] ? "checked" : ""}> ${a}</label>`),
    ...DEVS.map((d) => `<label><input type="checkbox" data-k="${d}" ${vis[d] ? "checked" : ""}> ${DEV_LABEL[d]}</label>`)].join(" ");
  el.innerHTML = `${legendHTML("axis")}${frameBanner(ctx, tr.tier)}<div class="toolbar">Show: ${boxes}` +
    `<span class="note">${tr.tier === "ctx4" ? "4 Hz min/max envelope (each trace drawn as its min and max)" : tr.tier === "native" ? "native rates (AX6 100 Hz, Apple 50 Hz; not sample-paired)" : "common 30 Hz grid"}</span></div><div class="plotbox"></div>`;
  const series = Array.from({ length: 2 * C }, (_, i) => {
    const dev = DEVS[Math.floor(i / C)], axis = (i % C) % 3;
    return { stroke: COL[dev], width: 1.2, dash: AXIS_DASH[AXES[axis]], points: { show: false },
      show: vis[AXES[axis]] && vis[dev] };
  });
  const u = mkPlot(el.querySelector(".plotbox"), { xRange: [t0, t1], series, yAxes: [yAxis(st.cal === "calibrated" ? "acceleration, calibrated (g)" : "acceleration, raw (g)")],
    data: joinTraces(tr), sync: "axes-shared" });
  el.querySelector(".toolbar").addEventListener("change", (e) => {
    vis[e.target.dataset.k] = e.target.checked;
    series.forEach((_, i) => { const dev = DEVS[Math.floor(i / C)], axis = (i % C) % 3; u.setSeries(i + 1, { show: vis[AXES[axis]] && vis[dev] }); });
  });
}
