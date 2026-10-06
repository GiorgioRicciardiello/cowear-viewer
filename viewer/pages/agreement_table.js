// Agreement page: "worst epochs" triage list and the sortable 1200-row epoch table. Clicking either jumps pages 1-3 to that epoch.
import { EPOCH_S, clock, Q } from "../common.js";

const COLS = [
  ["epoch", "epoch", (v) => v], ["t", "time", (v) => v], ["enmo_ax6", "ENMO AX6", (v) => Q.mg(v)], ["enmo_apple", "ENMO Apple", (v) => Q.mg(v)],
  ["diff", "Apple−AX6", (v) => Q.mg(v)], ["log2_ratio", "log2 ratio", (v) => Q.ratio(v)], ["within_epoch_r_lag0", "lag-0 r", (v) => Q.corr(v)],
  ["within_epoch_r", "peak r (±5 s lag search)", (v) => Q.corr(v)], ["within_epoch_r_null", "null p95 of peak r", (v) => Q.corr(v)],
  ["within_epoch_lag_s", "peak lag (s), not a clock offset", (v) => Q.s(v)], ["sleep_ax6", "AX6 sleep", (v) => (v ? "sleep" : "wake")], ["sleep_apple", "Apple sleep", (v) => (v ? "sleep" : "wake")],
  ["agreement_class", "class", (v) => v], ["noise_floor_flag", "noise floor", (v) => (v ? `<span class="chip chip-floor">below floor</span>` : "")],
];
const TRIAGE = {
  diff: ["largest |Apple − AX6|", (r) => -Math.abs(r.diff), (r) => Q.mg(r.diff)],
  r: ["lowest lag-0 r", (r) => (r.within_epoch_r_lag0 == null ? Infinity : r.within_epoch_r_lag0), (r) => `lag-0 r ${Q.corr(r.within_epoch_r_lag0)}`],
  ratio: ["largest |log2 ratio|", (r) => (r.log2_ratio == null ? Infinity : -Math.abs(r.log2_ratio)), (r) => `log2 ${Q.ratio(r.log2_ratio)}`],
};
const sortState = { key: "epoch", dir: 1 };
let selected = null;

const valueOf = (r, k) => (k === "t" ? r.epoch : r[k]);

function tableRows(epochs) {
  const { key, dir } = sortState;
  const sorted = [...epochs].sort((a, b) => {
    const x = valueOf(a, key), y = valueOf(b, key);
    if (x == null || y == null) return x == null && y == null ? 0 : x == null ? 1 : -1;   // nulls last
    return (x < y ? -1 : x > y ? 1 : 0) * dir;
  });
  return sorted.map((r) => `<tr data-epoch="${r.epoch}" class="${r.epoch === selected ? "sel" : ""}${r.noise_floor_flag ? " flagged" : ""}">` +
    COLS.map(([k, , f]) => `<td>${k === "t" ? clock(r.epoch * EPOCH_S) : f(r[k])}</td>`).join("") + "</tr>").join("");
}

export function mountTable(el, epochs, jump) {
  el.innerHTML = `<div class="triage"><h3>Worst epochs <select data-sel="triage">${Object.entries(TRIAGE).map(([k, v]) => `<option value="${k}">${v[0]}</option>`).join("")}</select></h3><ol></ol></div>` +
    `<div class="tablewrap"><table class="epochs"><thead><tr>${COLS.map(([k, l]) => `<th data-k="${k}">${l}</th>`).join("")}</tr></thead><tbody></tbody></table></div>`;
  const tbody = el.querySelector("tbody"), ol = el.querySelector("ol");
  const drawTable = () => {
    tbody.innerHTML = tableRows(epochs);
    el.querySelectorAll("th").forEach((th) => { th.className = th.dataset.k === sortState.key ? (sortState.dir > 0 ? "asc" : "desc") : ""; });
  };
  const drawTriage = (which) => {
    const [, score, show] = TRIAGE[which];
    const worst = epochs.filter((r) => !r.noise_floor_flag).sort((a, b) => score(a) - score(b)).slice(0, 12);
    ol.innerHTML = worst.map((r) => `<li><a href="#" data-epoch="${r.epoch}">#${r.epoch} ${clock(r.epoch * EPOCH_S)}</a> ${show(r)}</li>`).join("");
  };
  const go = (e) => {
    const n = e.target.closest("[data-epoch]");
    if (!n) return;
    e.preventDefault();
    selected = Number(n.dataset.epoch);
    tbody.querySelector("tr.sel")?.classList.remove("sel");
    tbody.querySelector(`tr[data-epoch="${selected}"]`)?.classList.add("sel");
    jump(selected);
  };
  tbody.addEventListener("click", go);
  ol.addEventListener("click", go);
  el.querySelector("thead").addEventListener("click", (e) => {
    const k = e.target.dataset.k;
    if (!k) return;
    sortState.dir = sortState.key === k ? -sortState.dir : 1;
    sortState.key = k;
    drawTable();
  });
  el.querySelector('[data-sel="triage"]').addEventListener("change", (e) => drawTriage(e.target.value));
  drawTable();
  drawTriage("diff");
}
