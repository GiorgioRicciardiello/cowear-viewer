// Always-visible overview strip: whole requested window, 30 s ENMO of both devices, abk sleep bands per device,
// greyed no-data region, page marker, drag/click brush to navigate.
//
// ENMO is drawn as two stacked per-device sparkline lanes, not one overlaid pair: a single lane at full
// opacity lets whichever device happens to sit higher (AX6's noise floor of 1.18 mg against Apple's 0.65 mg
// means that is usually AX6) paint over the other for most of the night, so the lower trace is only visible
// during its own spikes. Separate lanes make both always visible regardless of their relative level.
import { COL, DEV_LABEL, EPOCH_S, geo, noDataPlugin, mkPlot, alpha, canvasFont, fmt } from "./common.js";

const LANE = 9;        // css px per sleep lane
const ENMO_LANE = 26;  // css px per device ENMO sparkline lane
const ENMO_GAP = 3;    // css px between the two ENMO lanes

function sleepRuns(epochs, key) {
  const runs = [];
  epochs.forEach((r, i) => {
    if (!r[key]) return;
    const last = runs[runs.length - 1];
    if (last && last[1] === i) last[1] = i + 1; else runs.push([i, i + 1]);
  });
  return runs;
}

// A filled+stroked sparkline of `ys` against `xs`, inside [y0, y0 + laneH] with 0 at the lane's bottom.
function drawLane(ctx, u, xs, ys, cap, y0, laneH, color, dpr) {
  const pts = xs.map((t, i) => [u.valToPos(t, "x", true), y0 + laneH - Math.min(Math.max(ys[i], 0), cap) / cap * laneH]);
  if (!pts.length) return;
  ctx.beginPath(); ctx.moveTo(pts[0][0], y0 + laneH);
  for (const [x, y] of pts) ctx.lineTo(x, y);
  ctx.lineTo(pts[pts.length - 1][0], y0 + laneH); ctx.closePath();
  ctx.fillStyle = alpha(color, 0.18); ctx.fill();   // ~18% alpha fill under the line
  ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
  for (const [x, y] of pts) ctx.lineTo(x, y);
  ctx.strokeStyle = color; ctx.lineWidth = dpr; ctx.stroke();
}

export function mountOverview(el, epochs, getState, setPos) {
  const runs = { ax6: sleepRuns(epochs, "sleep_ax6"), apple: sleepRuns(epochs, "sleep_apple") };
  const mid = epochs.map((r) => r.epoch * EPOCH_S + EPOCH_S / 2);
  const capOf = (rows) => [...rows.map((r) => Math.max(r.enmo_ax6, r.enmo_apple))].sort((a, b) => a - b)[Math.floor(rows.length * 0.98)] * 1.2;
  let cap = capOf(epochs), rows = epochs;
  const bands = { hooks: { draw: [(u) => {
    const { ctx } = u, b = u.bbox, dpr = devicePixelRatio || 1, st = getState();
    ctx.save();
    ctx.beginPath(); ctx.rect(b.left, b.top, b.width, b.height); ctx.clip();

    ["ax6", "apple"].forEach((dev, li) => {
      const y0 = b.top + li * (ENMO_LANE + ENMO_GAP) * dpr, h = ENMO_LANE * dpr;
      ctx.fillStyle = COL.grid; ctx.fillRect(b.left, y0, b.width, h);
      drawLane(ctx, u, mid, rows.map((r) => r[dev === "ax6" ? "enmo_ax6" : "enmo_apple"]), cap, y0, h, COL[dev], dpr);
      ctx.fillStyle = COL.ink2; ctx.font = canvasFont(8, dpr); ctx.textAlign = "left"; ctx.textBaseline = "top";
      ctx.fillText(`${DEV_LABEL[dev]} ENMO`, b.left + 3 * dpr, y0 + 1 * dpr);
    });
    ctx.fillStyle = COL.ink2; ctx.font = canvasFont(8, dpr); ctx.textAlign = "right"; ctx.textBaseline = "top";
    ctx.fillText(`0–${fmt(cap, 1)} mg`, b.left + b.width - 3 * dpr, b.top + 1 * dpr);

    ["ax6", "apple"].forEach((dev, li) => {
      const y = b.top + b.height - (2 - li) * LANE * dpr - dpr;
      ctx.fillStyle = COL.grid; ctx.fillRect(u.valToPos(0, "x", true), y, u.valToPos(geo.nightS, "x", true) - u.valToPos(0, "x", true), (LANE - 1) * dpr);
      ctx.fillStyle = COL[dev];
      for (const [a, z] of runs[dev]) {
        const p0 = u.valToPos(a * EPOCH_S, "x", true);
        ctx.fillRect(p0, y, Math.max(u.valToPos(z * EPOCH_S, "x", true) - p0, 1), (LANE - 1) * dpr);
      }
      ctx.fillStyle = COL.ink2; ctx.font = canvasFont(8, dpr); ctx.textAlign = "right";
      ctx.fillText(dev === "ax6" ? "AX6 sleep" : "Apple sleep", b.left - 4 * dpr, y + (LANE - 1) * dpr);
    });
    const p0 = u.valToPos(st.pos, "x", true), p1 = u.valToPos(st.pos + st.width, "x", true);
    ctx.fillStyle = alpha(COL.ink, 0.12); ctx.strokeStyle = COL.ink; ctx.lineWidth = 1.5 * dpr;
    const w = Math.max(p1 - p0, 4 * dpr);
    ctx.fillRect(p0, b.top, w, b.height); ctx.strokeRect(p0, b.top, w, b.height);
    ctx.restore();
  }] } };
  const u = mkPlot(el, { xRange: [geo.reqStart, geo.reqEnd], series: [], yAxes: [{ show: true, size: 62, stroke: COL.ink2, grid: { show: false }, ticks: { show: false }, values: (u, v) => v.map(() => "") }],
    plugins: [bands], persist: true, data: [mid] });
  const at = (e) => u.posToVal(e.clientX - u.over.getBoundingClientRect().left, "x") - getState().width / 2;
  let down = false;
  u.over.addEventListener("mousedown", (e) => { down = true; setPos(at(e)); });
  window.addEventListener("mousemove", (e) => { if (down) setPos(at(e)); });
  window.addEventListener("mouseup", () => { down = false; });
  return { setEpochs: (r) => { rows = r; cap = capOf(r); u.setData([mid]); }, redraw: () => u.redraw(false), resize: () => u.setSize({ width: el.clientWidth, height: el.clientHeight }) };
}
