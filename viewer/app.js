// Co-wear viewer: state, routing, and the persistent spine. Python writes the artifacts; this file only draws.
import { EPOCH_S, geo, initGeo, destroyPlots } from "./common.js";
import { loadJSON } from "./data.js";
import { buildHeader, updateMetrics, buildFooter, updateFooter } from "./chrome.js";
import { mountOverview } from "./overview.js";
import { renderReadout } from "./readout.js";
import * as shared from "./pages/axes_shared.js";
import * as perAxis from "./pages/axes_per_axis.js";
import * as enmo from "./pages/enmo.js";
import * as agreement from "./pages/agreement.js";
import * as bands from "./pages/bands.js";
import * as spectrum from "./pages/spectrum.js";

// live: the page depends on the position / toggles and must redraw on every change.
const PAGES = [
  { id: "shared", label: "1 Axes shared", mod: shared, live: true },
  { id: "peraxis", label: "2 Axes per-axis", mod: perAxis, live: true },
  { id: "enmo", label: "3 ENMO", mod: enmo, live: true },
  { id: "agreement", label: "4 Agreement", mod: agreement, live: false },
  { id: "bands", label: "5 Bands", mod: bands, live: true },
  { id: "spectrum", label: "6 Spectrum", mod: spectrum, live: true },
];

let man, epochsAll, nightAll, spectra, epochs, night, overview, viewCal = null;
let st = { page: "shared", width: 300, pos: 0, cal: "calibrated", lag: "estimated", rate: "grid30", frame: "aligned" };
let token = 0, shownPage = null, shownCal = null, queued = false;
const $ = (id) => document.getElementById(id);

// The raw|calibrated toggle selects the variant of every metric, not only of the traces: epochs[i][cal], night[cal].
const EPOCH_KEYS = ["sleep_ax6", "sleep_apple"];
function selectVariant() {
  if (viewCal === st.cal) return;
  viewCal = st.cal;
  // spectral is calibrated-only (spectral_epoch_rows' own docstring): it does not vary with the raw|calibrated toggle.
  epochs = epochsAll.map((r) => ({ epoch: r.epoch, spectral: r.spectral, ...Object.fromEntries(EPOCH_KEYS.map((k) => [k, r[k]])), ...r[st.cal] }));
  night = nightAll[st.cal];
  if (overview) { overview.setEpochs(epochs); updateMetrics($("hdr"), man, night, st.cal); }   // overview exists once the header is built
}

const snap = (t) => Math.round(t / EPOCH_S) * EPOCH_S;
const clampPos = (pos, width) => Math.min(Math.max(snap(pos), Math.ceil(geo.reqStart / EPOCH_S) * EPOCH_S), Math.max(snap(geo.reqEnd - width), 0));

function setState(patch) {
  const next = { ...st, ...patch };
  st = { ...next, pos: clampPos(next.pos, next.width) };
  if (!queued) { queued = true; requestAnimationFrame(() => { queued = false; render(); }); }
}

function step(dir, epoch) { setState({ pos: st.pos + dir * (epoch ? EPOCH_S : st.width) }); }
function jumpToEpoch(e) { setState({ pos: e * EPOCH_S + EPOCH_S / 2 - st.width / 2 }); }

async function renderPage() {
  const page = PAGES.find((p) => p.id === st.page), el = $("page"), mine = ++token;
  destroyPlots();
  el.innerHTML = `<div class="loading">Loading…</div>`;
  const ctx = { man, epochs, night, spectra, st, t0: st.pos, t1: st.pos + st.width, stale: () => mine !== token, jumpToEpoch };
  try {
    await page.mod.render(el, ctx);
  } catch (err) {
    if (mine !== token) return;
    console.error(err);
    el.innerHTML = `<div class="banner warn"><b>Could not draw this page:</b> ${String(err.message || err)}</div>`;
  }
}

function render() {
  selectVariant();
  const page = PAGES.find((p) => p.id === st.page);
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.id === st.page));
  updateFooter($("foot"), st);
  overview.redraw();
  renderReadout($("readout"), epochs, st.pos, st.pos + st.width);
  if (page.live || shownPage !== st.page || shownCal !== st.cal) { shownPage = st.page; shownCal = st.cal; renderPage(); }
}

function keys(e) {
  const t = e.target;
  if (t.tagName === "SELECT" || t.tagName === "TEXTAREA" || (t.tagName === "INPUT" && t.type !== "checkbox")) return;
  if (e.key === "ArrowRight" || e.key === "ArrowLeft") { e.preventDefault(); step(e.key === "ArrowRight" ? 1 : -1, e.shiftKey); }
}

// Collapsible chrome (D6): a per-viewer convenience, so a failed localStorage read/write must never block render.
function wireCollapse(id) {
  const wrap = $(`${id}-wrap`), btn = wrap.querySelector(".collapse");
  const key = `cowear.collapsed.${id}`;
  let collapsed = false;
  try { collapsed = localStorage.getItem(key) === "true"; } catch { /* per-viewer convenience only */ }
  const apply = () => {
    wrap.dataset.collapsed = String(collapsed);
    btn.setAttribute("aria-expanded", String(!collapsed));
    btn.innerHTML = collapsed ? "&#9656;" : "&#9662;";
  };
  apply();
  btn.addEventListener("click", () => {
    collapsed = !collapsed;
    apply();
    try { localStorage.setItem(key, String(collapsed)); } catch { /* per-viewer convenience only */ }
    if (id === "overview") overview.resize();
  });
}

async function main() {
  try {
    [man, epochsAll, nightAll, spectra] = await Promise.all(
      ["manifest.json", "epochs.json", "night.json", "spectra.json"].map(loadJSON));
  } catch (err) {
    $("page").innerHTML = `<div class="banner warn"><b>Could not load the artifacts from data/:</b> ${err.message}. Run scripts/make_demo_data.py and scripts/build_viewer.py, then scripts/run_viewer.py.</div>`;
    throw err;
  }
  initGeo(man);
  selectVariant();
  buildHeader($("hdr"), man, night, st, setState);
  buildFooter($("foot"), st, setState, step);
  $("tabs").innerHTML = PAGES.map((p) => `<button data-id="${p.id}">${p.label}</button>`).join("");
  $("tabs").addEventListener("click", (e) => e.target.dataset.id && setState({ page: e.target.dataset.id }));
  overview = mountOverview($("overview"), epochs, () => st, (t) => setState({ pos: t }));
  wireCollapse("overview");
  wireCollapse("readout");
  document.addEventListener("keydown", keys);
  // Geometry (plot sizing) is now mkPlot's own ResizeObserver; a window resize only needs to
  // move the overview's persistent plot, which mkPlot's observer does not own (`persist: true`).
  let rz;
  window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(() => overview.resize(), 150); });
  window.viewer = { get state() { return st; }, setState };   // handle for scripted checks
  setState({});
}

main();
