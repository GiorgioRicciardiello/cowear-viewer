// Always-visible epoch readout: per-epoch metrics for the epochs overlapping the current page.
import { EPOCH_S, clock, Q, epochRange } from "./common.js";

export function renderReadout(el, epochs, t0, t1) {
  const [e0, e1] = epochRange(t0, t1, epochs.length);
  const rows = epochs.slice(e0, e1);
  if (!rows.length) {
    el.innerHTML = `<div class="readout-head">Epoch readout</div><div class="nodata">No epochs on screen: this page is entirely inside the no-data region.</div>`;
    return;
  }
  const s = (v) => (v ? "sleep" : "wake");
  const floorChip = (v) => (v ? `<span class="chip chip-floor">below floor</span>` : "");
  el.innerHTML = `<div class="readout-head">Epoch readout: ${rows.length} epoch${rows.length > 1 ? "s" : ""} on screen (#${e0}${rows.length > 1 ? `–#${e1 - 1}` : ""}) <span class="note11">* peak r is the maximum over a &plusmn;5 s lag search and is inflated by that search; it is shown only next to its 95th-percentile null. Peak lag is not a measured clock offset.</span></div>` +
    `<div class="tablewrap"><table class="epochs"><thead>` +
    `<tr><th class="grp" colspan="2"></th><th class="grp" colspan="4">ENMO</th><th class="grp" colspan="4">agreement</th><th class="grp" colspan="4">classification</th></tr>` +
    `<tr><th>epoch</th><th>time</th><th>AX6</th><th>Apple</th><th>Apple&minus;AX6</th><th>log2 ratio</th>` +
    `<th title="Pearson r at zero lag, n = 900 on the 30 Hz grid">lag-0 r</th><th title="maximum r over the +/-5 s lag search: inflated by the maximisation, read only beside its null">peak r</th><th title="the per-epoch 95th-percentile null for peak r">null p95</th><th title="lag at the peak r: NOT a measured clock offset">peak lag s*</th>` +
    `<th>AX6</th><th>Apple</th><th>class</th><th>noise floor</th></tr>` +
    `</thead><tbody>` +
    rows.map((r) => `<tr><td>${r.epoch}</td><td>${clock(r.epoch * EPOCH_S)}</td><td>${Q.mg(r.enmo_ax6)}</td><td>${Q.mg(r.enmo_apple)}</td>` +
      `<td>${Q.mg(r.diff)}</td><td>${Q.ratio(r.log2_ratio)}</td><td>${Q.corr(r.within_epoch_r_lag0)}</td><td>${Q.corr(r.within_epoch_r)}</td><td>${Q.corr(r.within_epoch_r_null)}</td><td>${Q.s(r.within_epoch_lag_s)}</td><td>${s(r.sleep_ax6)}</td><td>${s(r.sleep_apple)}</td>` +
      `<td>${r.agreement_class}</td><td>${floorChip(r.noise_floor_flag)}</td></tr>`).join("") + "</tbody></table></div>";
}
