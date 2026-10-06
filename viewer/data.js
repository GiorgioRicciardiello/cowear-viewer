// Artifact loading and per-window signal preparation. Chunks hold RAW device-frame g; calibration, rotation and lag are applied here.
const BASE = "data/";
const MAX_CACHED = 10;
const cache = new Map();   // "dev/file" -> Promise<Float32Array>, insertion order = LRU

export async function loadJSON(name) {
  const r = await fetch(BASE + name);
  if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
  return r.json();
}

function chunk(dev, file) {
  const key = dev + "/" + file;
  if (cache.has(key)) { const p = cache.get(key); cache.delete(key); cache.set(key, p); return p; }
  const p = fetch(BASE + "chunks/" + file).then((r) => {
    if (!r.ok) throw new Error(`chunk ${file}: HTTP ${r.status}`);
    return r.arrayBuffer();
  }).then((b) => new Float32Array(b));   // little-endian host assumed (x86/ARM)
  cache.set(key, p);
  p.catch(() => cache.delete(key));
  while (cache.size > MAX_CACHED) cache.delete(cache.keys().next().value);
  return p;
}

// Samples of one device/tier inside [t0, t1) on that device's own clock. ch = 3 arrays (x,y,z), 6 for ctx4 (min xyz,
// max xyz), or `info.n_freq` for "spec" (one array per frequency bin, dB; `hz` there is time bins per second, not a
// sample rate, so a time axis built from it is already in real seconds).
export async function segment(man, dev, tier, t0, t1) {
  const info = man.chunks[dev][tier], hz = info.hz, C = tier === "ctx4" ? 6 : tier === "spec" ? info.n_freq : 3, perChunk = info.chunk_s * hz;
  const i0 = Math.max(0, Math.floor(t0 * hz)), i1 = Math.min(info.n_samples, Math.ceil(t1 * hz));
  const n = Math.max(0, i1 - i0);
  const t = new Float64Array(n), ch = Array.from({ length: C }, () => new Float32Array(n));
  if (!n) return { t, ch };
  const ks = [];
  for (let k = Math.floor(i0 / perChunk); k <= Math.floor((i1 - 1) / perChunk); k++) ks.push(k);
  const bufs = await Promise.all(ks.map((k) => chunk(dev, info.files[k])));
  ks.forEach((k, bi) => {
    const base = k * perChunk, buf = bufs[bi];
    for (let i = Math.max(i0, base); i < Math.min(i1, base + buf.length / C); i++) {
      const j = i - i0, o = (i - base) * C;
      t[j] = i / hz;
      for (let c = 0; c < C; c++) ch[c][j] = buf[o + c];
    }
  });
  return { t, ch };
}

const calibrate = (ch, cal) => ch.map((a, c) => { const g = cal.gain[c % 3], d = cal.offset_mg[c % 3] / 1000; return a.map((v) => g * v + d); });

function rotate(ch, R) {
  const [x, y, z] = ch, n = x.length;
  const out = [new Float32Array(n), new Float32Array(n), new Float32Array(n)];
  for (let i = 0; i < n; i++)
    for (let r = 0; r < 3; r++) out[r][i] = R[r][0] * x[i] + R[r][1] * y[i] + R[r][2] * z[i];
  return out;
}

// NaN marks a gap longer than max_gap_s. uPlot breaks a line at null but not at NaN, and 0 would be a false reading.
export const nanToNull = (a) => Array.from(a, (v) => (Number.isNaN(v) ? null : v));

export function tierFor(st) { return st.width >= 1800 ? "ctx4" : st.rate === "native" ? "native" : "grid30"; }

// Both devices for the visible window on the AX6 clock, with the header toggles applied.
export async function traces(man, st, t0, t1) {
  const tier = tierFor(st), L = st.lag === "estimated" ? man.lag_s : 0, m = 1;
  const [a, p] = await Promise.all([segment(man, "ax6", tier, t0 - m, t1 + m), segment(man, "apple", tier, t0 + L - m, t1 + L + m)]);
  const fix = (seg, dev, shift) => {
    let ch = st.cal === "calibrated" ? calibrate(seg.ch, man.devices[dev].calibration) : seg.ch;
    if (dev === "apple" && st.frame === "aligned" && tier !== "ctx4") ch = rotate(ch, man.rotation.matrix);
    return { t: seg.t.map((v) => v - shift), ch: ch.map(nanToNull) };
  };
  return { tier, ax6: fix(a, "ax6", 0), apple: fix(p, "apple", L) };
}

// uPlot.join over both devices: columns = [t, ax6 channels..., apple channels...] (nulls where a device has no sample).
export function joinTraces(tr) {
  return uPlot.join([[tr.ax6.t, ...tr.ax6.ch], [tr.apple.t, ...tr.apple.ch]]);
}
