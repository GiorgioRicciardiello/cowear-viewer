"""Build the static artifacts the web viewer reads from the signal chain.

Conventions the viewer relies on, mirrored from scripts/make_viewer_fixture.py:
- chunks hold RAW device-frame acceleration in g, as stored in the parquet. Calibration, rotation and lag are NOT applied:
  the viewer applies them from the manifest (calibrate -> rotate -> shift by lag). The only processing is putting the
  samples on a uniform time axis at the nominal rate (the Apple export runs at 50.24 Hz with jittered timestamps).
- `lag_s > 0`: Apple's clock is ahead, apple_time = ax6_time + lag_s; the signal chain corrects it with `build_grid(lag_s=)`.
- `rotation.matrix` maps the Apple frame into the AX6 frame and is applied after calibration.
- ENMO is in mg, computed at native rate and then decimated (signals.py docstring); `diff = apple - ax6`; NaN is null.
- epochs.json and night.json carry both variants, `calibrated` and `raw`. Top-level fields mirror `calibrated`, which is
  all the current viewer reads (it does not key by variant yet), so the two layouts coexist.
- `within_epoch_r` is the MAXIMUM r over the lag search (biased upwards); `within_epoch_r_lag0` and `within_epoch_r_null`
  ship beside it (align.epoch_lags). Every interval is conditional on its bootstrap block: see `block_sensitivity`.
- Every artifact parameter (chunk length, context rate, requested window hours) is read from config ax6apple.viewer.
"""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.signal import welch

from ax6apple import spectral
from ax6apple.agreement import (agreement_class, block_length, kappa_ci, log2_ratio, night_stats_ci, noise_floor_flag,
                                pearson_block_sensitivity)
from ax6apple.align import Lag, band_vm, epoch_lags, estimate_lag, fit_rotation, gravity_pairs
from ax6apple.calibration import MATLAB_TOL, apply, calibrate
from ax6apple.preprocess import BuildLog, Limits, alignment_quality, capabilities, device_quality
from ax6apple.preprocess.completeness import gap_stats
from ax6apple.preprocess.report import QualityReport, status_of
from ax6apple.preprocess.report import write as write_quality_report
from ax6apple.rbd_sleep import EPOCH, epoch_vector, load_bouts
from ax6apple.signals import (build_grid, epoch_matrix, gap_mask, load_recording, ma_sweep, recording_window,
                              sensor_noise_floor, vm)
from ax6apple.config import AX6APPLE, DATA_ROOT, PREPROCESSED_ROOT, RBD_ROOT

VIEWER = AX6APPLE["viewer"]
OUT_DIR = DATA_ROOT / "ax6apple" / "viewer"
DEVICES = ("ax6", "apple")
LABELS = {"ax6": "AX6", "apple": "Apple Watch"}
VARIANTS = ("calibrated", "raw")
SCALAR_KEYS = {"pearson": "r", "spearman": "rho", "ccc": "ccc", "icc21": "icc", "ba_bias": "bias",
               "ba_loa_low": "loa_lo", "ba_loa_high": "loa_hi"}
BLOCK_NOTE = ("every interval is conditional on its bootstrap block length and understates the uncertainty when the "
              "dependence is longer than the block; read them across block_min, not one alone")


def _iso(ts) -> str:
    return pd.Timestamp(ts).isoformat(timespec="milliseconds")


def clean(o, digits: int | None = 6):
    """JSON-ready copy: numpy to Python, NaN/inf to None, floats rounded to `digits` (None: full precision)."""
    if isinstance(o, dict):
        return {k: clean(v, digits) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v, digits) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist(), digits)
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else (float(o) if digits is None else round(float(o), digits))
    if isinstance(o, (np.integer, np.bool_)):
        return o.item()
    return o


def _sidecar(parquet: Path) -> dict:
    return json.loads(parquet.with_name(parquet.name + ".json").read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def uniform_native(t_s: np.ndarray, xyz: np.ndarray, fs: int, n_sec: int, max_gap_s: float = VIEWER["max_gap_s"]) -> np.ndarray:
    """Raw samples on the nominal-rate axis k / fs (the first step of `build_grid`), float32. No calibration, rotation or lag.

    Time inside a gap longer than `max_gap_s` is NaN, as on the grid, so a dropout is not drawn as a ramp.
    """
    tu = np.arange(n_sec * fs) / fs
    out = np.column_stack([np.interp(tu, t_s, xyz[:, k]) for k in range(3)]).astype(np.float32)
    out[gap_mask(t_s, tu, max_gap_s)] = np.nan
    return out


def envelope(arr: np.ndarray, fs: int, n_sec: int) -> np.ndarray:
    """4 Hz min/max envelope, n x 6: min_x, min_y, min_z, max_x, max_y, max_z. A mean would hide the transients."""
    idx = np.round(np.arange(n_sec * VIEWER["ctx_hz"]) * fs / VIEWER["ctx_hz"]).astype(int)
    return np.concatenate([np.minimum.reduceat(arr, idx, axis=0), np.maximum.reduceat(arr, idx, axis=0)], axis=1)


def write_chunks(out: Path, dev: str, tier: str, arr: np.ndarray, hz: float, chunk_s: int) -> dict:
    """Little-endian float32, channels interleaved, `chunk_s` seconds per file; returns the manifest's chunk-index entry.

    `hz` is normally an int (a sample rate); `round()` also allows the spectrogram tier's sub-1 Hz time-bin rate
    (1 / hop_s) through the same path without changing behaviour for any integer hz (round(int) == int).
    """
    per = round(chunk_s * hz)
    files = []
    for k in range(-(-len(arr) // per)):
        name = f"{dev}_{tier}_{k:02d}.f32"
        np.ascontiguousarray(arr[k * per:(k + 1) * per], dtype="<f4").tofile(out / "chunks" / name)
        files.append(name)
    return {"hz": hz, "chunk_s": chunk_s, "start_s": 0, "n_samples": len(arr), "files": files}


def _load_device(dev: str, start: pd.Timestamp, end: pd.Timestamp) -> SimpleNamespace:
    """The recording with a lag margin either side; calibration and noise floor use the window itself, not the margin."""
    rel = AX6APPLE["sources"][dev]["parquet"]
    path = PREPROCESSED_ROOT / rel
    rec = load_recording(path, start, end, margin_s=VIEWER["lag_search_s"])
    inside = (rec.t_s >= 0) & (rec.t_s <= (end - start).total_seconds())
    guard = {"min_coverage": VIEWER["calibration_min_coverage"], "low_coverage": VIEWER["calibration_low_coverage"]}
    xyz = rec.xyz[inside]
    args = (xyz, rec.fs, VIEWER["calibration_std_g"], VIEWER["still_window_s"])
    return SimpleNamespace(dev=dev, rel=rel, sidecar=_sidecar(path), rec=rec, xyz=xyz, cal=calibrate(*args, **guard),
                           matlab=calibrate(*args, tol=MATLAB_TOL, **guard),
                           noise=sensor_noise_floor(xyz, rec.fs, VIEWER["still_window_s"], VIEWER["calibration_std_g"]))


def epoch_table(en_ax6: np.ndarray, en_apple: np.ndarray, floors_mg: tuple[float, float]) -> dict:
    """Per-epoch metrics from two 30 Hz ENMO streams in g (Apple already lag corrected). ENMO in mg.

    `floors_mg` is each device's own noise floor (AX6, Apple). `within_epoch_r` is the maximum r over the lag search
    and is biased upwards; `within_epoch_r_lag0` and `within_epoch_r_null` (the 95th percentile of that same maximum
    against a phase-randomised Apple epoch) must be shown beside it, and its lag is not a measured offset.
    """
    boot = VIEWER["bootstrap"]
    n = VIEWER["epoch_s"] * VIEWER["grid_hz"]
    ea, ep = epoch_matrix(en_ax6, n), epoch_matrix(en_apple, n)
    a, p = ea.mean(axis=1) * 1000, ep.mean(axis=1) * 1000
    flag = noise_floor_flag(a, p, *floors_mg)
    lags = epoch_lags(ea, ep, VIEWER["grid_hz"], VIEWER["lag_search_s"], boot["epoch_null_draws"], np.random.default_rng(boot["seed"]))
    return {"enmo_ax6": a, "enmo_apple": p, "diff": p - a, "log2_ratio": log2_ratio(a, p, flag), "within_epoch_r": lags.r_peak,
            "within_epoch_r_lag0": lags.r_lag0, "within_epoch_r_null": lags.r_null, "within_epoch_lag_s": lags.lag_s,
            "noise_floor_flag": flag}


def _noise_floor(tab: dict, noise: dict) -> dict:
    """Per device: its floor (mg), how it was derived, and the fraction of epochs below it; and the fraction flagged jointly."""
    enmo = {"ax6": tab["enmo_ax6"], "apple": tab["enmo_apple"]}
    out = {}
    for d in DEVICES:
        below = enmo[d] < noise[d]["value_mg"]
        out[d] = {**noise[d], "n_flagged": int(below.sum()), "fraction_flagged": float(below.mean())}
    flag = tab["noise_floor_flag"]
    out["joint"] = {"rule": "either device below its own floor", "n_flagged": int(flag.sum()), "fraction_flagged": float(flag.mean())}
    return out


def night_summary(tab: dict, sleep_ax6: np.ndarray, sleep_apple: np.ndarray, noise: dict) -> dict:
    """Night scalars {est, lo, hi} over all epochs; Bland-Altman and scatter over the epochs above the noise floor.

    `block_sensitivity` is the interval of r at each bootstrap block length (the headline intervals use `block_min`).
    """
    boot = VIEWER["bootstrap"]
    rng = np.random.default_rng(boot["seed"])
    block = block_length(boot["block_min"], 1.0 / VIEWER["epoch_s"])
    stats = night_stats_ci(tab["enmo_ax6"], tab["enmo_apple"], block, boot["resamples"], rng)
    scalars = {name: {"est": stats[k]["estimate"], "lo": stats[k]["lo"], "hi": stats[k]["hi"]} for k, name in SCALAR_KEYS.items()}
    est, lo, hi = kappa_ci(sleep_ax6, sleep_apple, block, boot["resamples"], rng)
    scalars["kappa"] = {"est": est, "lo": lo, "hi": hi}
    blocks = sorted({*boot["block_sensitivity_min"], boot["block_min"]})
    sens = pearson_block_sensitivity(tab["enmo_ax6"], tab["enmo_apple"], blocks, 1.0 / VIEWER["epoch_s"], boot["resamples"], boot["seed"])
    ok = ~tab["noise_floor_flag"]
    a, p = tab["enmo_ax6"][ok], tab["enmo_apple"][ok]
    return {"n_epochs": len(ok), "n_flagged_epochs": int((~ok).sum()), "units": {"enmo": "mg", "diff": "mg"},
            "scalars": scalars, "block_min": boot["block_min"], "block_sensitivity": sens, "block_sensitivity_note": BLOCK_NOTE,
            "noise_floor": _noise_floor(tab, noise),
            "ba": {"mean": (a + p) / 2, "diff": p - a}, "scatter": {"ax6": a, "apple": p}}


def _sweep(ax6: np.ndarray, apple: np.ndarray, lag_s: float) -> dict:
    """The MA sweep in the contract's column layout. `optimal_lag_s` is absolute: the estimated lag plus the residual peak."""
    rows = ma_sweep(ax6, apple, VIEWER["grid_hz"], VIEWER["epoch_s"] * VIEWER["grid_hz"], VIEWER)
    out = {k: [r[k] for r in rows] for k in ("w_s", "r", "r_lo", "r_hi", "null_lo", "null_hi", "n_eff", "n_eff_basis",
                                             "block_sensitivity")}
    out["optimal_lag_s"] = [lag_s + r["lag_s"] for r in rows]
    for k, q in (("within_median", "q50"), ("within_q25", "q25"), ("within_q75", "q75")):
        out[k] = [r["within_epoch_r"][q] for r in rows]
    return out


def _cached_grid(cache: dict, dev: str, variant: str, devs: dict, xyz: np.ndarray, lag_s: float, n_out: int):
    """F2: `build_grid` is deterministic in `(dev, variant, grid_hz, lag_s)`; AX6's lag is always 0, so its grid for
    `raw` / `calibrated` is byte-identical to the one `build()` already built at step 8 -- reuse it instead of a second
    full `resample_poly` pass over the night.
    """
    key = (dev, variant, VIEWER["grid_hz"], lag_s)
    if key not in cache:
        cache[key] = build_grid(devs[dev].rec.t_s, xyz, devs[dev].rec.fs, VIEWER["grid_hz"], n_out, lag_s)
    return cache[key]


def _variant(xyz: dict, devs: dict, lag_s: float, sleep: dict, n_out: int, cache: dict, variant: str) -> tuple[dict, dict]:
    """(epoch table, night) for one of raw / calibrated: grids built (or reused from `cache`, F2) here, Apple lag corrected."""
    grids = {d: _cached_grid(cache, d, variant, devs, xyz[d], lag_s if d == "apple" else 0.0, n_out) for d in DEVICES}
    tab = epoch_table(grids["ax6"].enmo, grids["apple"].enmo, tuple(devs[d].noise["value_mg"] for d in DEVICES))
    night = night_summary(tab, sleep["ax6"], sleep["apple"], {d: devs[d].noise for d in DEVICES})
    night["sweep"] = {"enmo": _sweep(grids["ax6"].enmo, grids["apple"].enmo, lag_s),
                      "vm": _sweep(grids["ax6"].vm, grids["apple"].vm, lag_s)}
    return tab, night


def _sleep_flags(start: pd.Timestamp, end: pd.Timestamp) -> tuple[dict, dict]:
    """abk sleep epochs per device (the `device` comparison's two variants) and the bout files they came from."""
    comp = next(c for c in AX6APPLE["comparisons"] if c["name"] == "device")
    stem = next(v for v in AX6APPLE["variants"] if v["name"] == comp["ref"])["stem"]
    names = dict(zip(DEVICES, (comp["ref"], comp["test"])))
    files = {d: RBD_ROOT / "fit" / n / "auto_sleep" / "abk" / f"{stem}.csv" for d, n in names.items()}
    flags = {d: epoch_vector(load_bouts(p), start, end) for d, p in files.items()}
    return flags, {d: {"path": str(p.relative_to(RBD_ROOT)), "sha256": _sha256(p)} for d, p in files.items()}


def epoch_rows(tabs: dict, sleep: dict) -> list[dict]:
    """One row per epoch: no timestamp (time is epoch * 30 s). Top-level metrics are `calibrated`; both variants are keyed."""
    cls = agreement_class(sleep["ax6"], sleep["apple"])
    tabs = {v: {**t, "agreement_class": cls} for v, t in tabs.items()}
    rows = []
    for i in range(len(cls)):
        one = {v: {k: (x[i].item() if hasattr(x[i], "item") else x[i]) for k, x in t.items()} for v, t in tabs.items()}
        rows.append({"epoch": i, "sleep_ax6": bool(sleep["ax6"][i]), "sleep_apple": bool(sleep["apple"][i]),
                     **one["calibrated"], **one})
    return rows


def _device_entry(d: SimpleNamespace) -> dict:
    c = d.cal
    return {"label": LABELS[d.dev], "source_path": d.rel, "source_sha256": d.sidecar["source_sha256"],
            "source_relative_path": d.sidecar.get("source_relative_path"), "native_hz": d.rec.fs,
            "calibration": {"gain": c.gain, "offset_mg": c.offset * 1000, "n_windows": c.n_windows, "n_windows_total": c.n_retained,
                            "orientation_coverage": c.coverage, "residual": c.residual, "rms_g": c.rms_g,
                            "iterations": c.iterations, "converged": c.converged,
                            "low_orientation_coverage": c.low_orientation_coverage},
            "timing": d.rec.timing._asdict() | {"nominal_hz": d.rec.fs}}


def _caveats(devs: dict, rot) -> dict:
    return {
        "rotation": {"residual_deg": rot.residual_deg, "uncertainty_deg": rot.uncertainty_deg,
                     "note": "RMS angle between rotated Apple gravity and AX6 gravity over jointly still windows; a large "
                             "value means one fixed rotation does not relate the two frames. Reported as fitted, not corrected."},
        "calibration_vs_abk": {
            "viewer": "gain+offset (calibrate_acc.m) iterated to convergence",
            "abk_features": "MATLAB accelerometer_calibration.m (ICP with rotation), early stopped",
            "offset_diff_mg_converged_minus_matlab_early_stop": {d: (devs[d].cal.offset - devs[d].matlab.offset) * 1000 for d in DEVICES},
            "note": "measured against this port's MATLAB-tolerance early stop of gain+offset, a proxy for what the ABK features saw",
            "matlab_early_stop_fit": {d: {"gain": devs[d].matlab.gain, "coverage": devs[d].matlab.coverage,
                                          "rms_g": devs[d].matlab.rms_g, "iterations": devs[d].matlab.iterations,
                                          "converged": devs[d].matlab.converged} for d in DEVICES}},
        "timing": {"ax6": devs["ax6"].rec.timing._asdict(), "apple": devs["apple"].rec.timing._asdict(),
                   "note": "rows stable-sorted into time order and exact repeated timestamps dropped (first kept); nothing else repaired"}}


def _manifest(devs: dict, rot, lag, chunks: dict, sleep_src: dict, start: pd.Timestamp, n_epochs: int) -> dict:
    day = start.normalize()
    return {
        "synthetic": any(d.sidecar.get("synthetic", False) for d in devs.values()), "session_id": PureWindowsPath(devs["ax6"].rel).parts[1], "anchor": _iso(start),
        "epoch_s": VIEWER["epoch_s"], "n_epochs": n_epochs, "channels": ["x", "y", "z"], "unit": "g",
        "window": {"requested_start": _iso(day.replace(hour=VIEWER["requested_start_h"])),
                   "requested_end": _iso((day + pd.Timedelta(days=1)).replace(hour=VIEWER["requested_end_h"])),
                   "available_start": _iso(start), "available_end": _iso(start + n_epochs * EPOCH)},
        "devices": {d: _device_entry(devs[d]) for d in DEVICES},
        "rotation": {"matrix": rot.matrix, "maps": "apple_to_ax6", "residual_deg": rot.residual_deg,
                     "uncertainty_deg": rot.uncertainty_deg, "n_windows": rot.n_windows},
        "lag_s": lag.lag_s, "lag_convention": "apple_time = ax6_time + lag_s",
        "lag_diagnostics": {"peak_r": lag.r, "at_boundary": lag.at_boundary},
        "caveats": _caveats(devs, rot), "sleep_source": sleep_src,
        "config": {**VIEWER, "calibration": "gain_offset"},
        "built": datetime.now().isoformat(timespec="seconds"), "chunks": chunks}


def _write_all_chunks(out: Path, devs: dict, raw_grids: dict, n_sec: int) -> dict:
    """`grid50` is raw (uncalibrated, no lag), like `grid30`: a second common grid for spectral work whose Nyquist
    (25 Hz) reaches the 15-25 Hz band `grid30` cannot represent.
    """
    spectral_hz = VIEWER["spectral_grid_hz"]
    n_out50 = n_sec * spectral_hz
    chunks = {}
    for d in DEVICES:
        rec = devs[d].rec
        native = uniform_native(rec.t_s, rec.xyz, rec.fs, n_sec)
        ctx = envelope(native, rec.fs, n_sec)
        name = f"{d}_ctx4_00.f32"
        ctx.astype("<f4").tofile(out / "chunks" / name)
        grid50 = build_grid(rec.t_s, rec.xyz, rec.fs, spectral_hz, n_out50)
        chunks[d] = {"native": write_chunks(out, d, "native", native, rec.fs, VIEWER["chunk_s"]),
                     "grid30": write_chunks(out, d, "grid30", raw_grids[d].xyz, VIEWER["grid_hz"], VIEWER["chunk_s"]),
                     "grid50": write_chunks(out, d, "grid50", grid50.xyz, spectral_hz, VIEWER["chunk_s"]),
                     "ctx4": {"hz": VIEWER["ctx_hz"], "chunk_s": n_sec, "start_s": 0, "n_samples": len(ctx), "files": [name],
                              "layout": "min_xyz,max_xyz"}}
    return chunks


# --- spectral artifacts. Calibrated variant only: gain (~1.001) and a few mg of
# offset barely move an AC spectrum, so a second raw copy of every spectral artifact would not earn its size. ---

WAVELET, WAVELET_LEVEL = "sym8", 6


def _spectral_noise(devs: dict) -> dict:
    """Per device: (f, N) from `spectral.noise_psd` on the device's own still windows, raw native-rate xyz.

    F5: `devs[d].xyz` (margin stripped) is the same sample set `signals.sensor_noise_floor` measures (`_load_device`),
    not `devs[d].rec.xyz` (which still carries the +/- lag_search_s margin) -- the two noise floors used to be measured
    on different data.
    """
    return {d: spectral.noise_psd(devs[d].xyz, devs[d].rec.fs, VIEWER["still_window_s"], VIEWER["calibration_std_g"])
            for d in DEVICES}


def _spectral_vm(devs: dict, cal_xyz: dict, lag_s: float, n_out: int) -> dict:
    """Calibrated VM on `grid50`, Apple lag corrected: the stream every spectral artifact below is computed from."""
    hz = VIEWER["spectral_grid_hz"]
    return {d: build_grid(devs[d].rec.t_s, cal_xyz[d], devs[d].rec.fs, hz, n_out, lag_s if d == "apple" else 0.0).vm
            for d in DEVICES}


def spectral_epoch_rows(vm50: dict, fs: int, epoch_n: int, noise: dict) -> list[dict]:
    """One {ax6: {...}, apple: {...}} row per epoch: comparable-band RMS (mg), noise-corrected spectral scalars over
    0.1-20 Hz, and MODWT level energies. Merged into the epoch rows `build()` already writes.
    """
    per_device = {}
    for d in DEVICES:
        x = vm50[d]
        f, P = spectral.epoch_psd(x, fs, epoch_n)
        f_n, n = noise[d]
        scalars = spectral.spectral_scalars(f, spectral.noise_correct(f, P, f_n, n))
        bands = {name: spectral.band_epoch_rms(x, fs, spectral.BANDS[name], epoch_n) * 1000
                 for name in (*spectral.COMPARABLE_BANDS, "margin")}
        modwt = spectral.modwt_epoch_energy(x, epoch_n, WAVELET, WAVELET_LEVEL)
        per_device[d] = {"band_rms_mg": bands, "modwt_energy": modwt, **scalars}
    n_epochs = len(next(iter(vm50.values()))) // epoch_n
    return [{d: {"band_rms_mg": {k: v[i] for k, v in per_device[d]["band_rms_mg"].items()},
                "sef95_hz": per_device[d]["sef95_hz"][i], "centroid_hz": per_device[d]["centroid_hz"][i],
                "dominant_hz": per_device[d]["dominant_hz"][i], "spectral_entropy": per_device[d]["spectral_entropy"][i],
                "modwt_energy": per_device[d]["modwt_energy"][i]} for d in DEVICES} for i in range(n_epochs)]


def _ax6_only_band(devs: dict) -> dict:
    """25-49 Hz RMS on AX6's own native 100 Hz grid: Apple has no samples there at all, so there is nothing to pair it
    with. A single night-level number, not per epoch -- it is a noise diagnostic.
    """
    rec = devs["ax6"].rec
    hf = band_vm(vm(devs["ax6"].xyz), rec.fs, spectral.AX6_ONLY_BAND)   # F5: margin stripped, consistent with noise_psd
    return {"band_hz": list(spectral.AX6_ONLY_BAND), "rms_mg": float(np.sqrt((hf ** 2).mean()) * 1000),
            "note": "AX6 only: above Apple's 25 Hz Nyquist, it has no samples to compare against"}


def spectra_summary(vm50: dict, devs: dict, noise: dict) -> dict:
    """Night-level PSD (observed, noise floor, noise-corrected, all dB) per device, their coherence, and the band /
    wavelet definitions every spectral epoch field above is computed from: what the viewer's Spectrum page reads.
    """
    hz = VIEWER["spectral_grid_hz"]
    psd = {}
    for d in DEVICES:
        x = vm50[d]
        f, P = welch(x - x.mean(), fs=hz, nperseg=int(hz * VIEWER["epoch_s"]), detrend="constant")
        f_n, n = noise[d]
        psd[d] = {"f": f, "raw_db": 10 * np.log10(P + 1e-30), "noise_db": 10 * np.log10(np.interp(f, f_n, n) + 1e-30),
                  "corrected_db": 10 * np.log10(spectral.noise_correct(f, P, f_n, n) + 1e-30)}
    f_c, coh, floor = spectral.coherence(vm50["ax6"], vm50["apple"], hz, seg_s=VIEWER["epoch_s"])
    return {"grid_hz": hz, "bands": spectral.BANDS, "comparable_bands": list(spectral.COMPARABLE_BANDS), "psd": psd,
            "coherence": {"f": f_c, "value": coh, "bias_floor": floor},
            "wavelet": {"name": WAVELET, "level": WAVELET_LEVEL, "dyadic_bands_hz": spectral.dyadic_bands(hz, WAVELET_LEVEL)},
            "ax6_only_band": _ax6_only_band(devs)}


def spectrogram_chunks(out: Path, vm50: dict) -> dict:
    """dB spectrogram per device, written with the existing `write_chunks` (time bins in place of seconds, frequency
    bins in place of x/y/z -- the format does not care which). The viewer's Spectrum page reads `freq_hz` from here
    to label the rows; `write_chunks`' own `hz` is time bins per second (1 / hop_s), not a sample rate.
    """
    seg_s, hop_s, hz = VIEWER["spec_seg_s"], VIEWER["spec_hop_s"], VIEWER["spectral_grid_hz"]
    chunks = {}
    for d in DEVICES:
        f, _, S = spectral.spectrogram_db(vm50[d], hz, seg_s, hop_s)
        chunks[d] = write_chunks(out, d, "spec", S.T, 1 / hop_s, VIEWER["chunk_s"]) | \
            {"freq_hz": f.tolist(), "seg_s": seg_s, "hop_s": hop_s, "n_freq": len(f)}
    return chunks


def _estimate_lag_or_explain(ref_vm: np.ndarray, test_vm: np.ndarray, grid_hz: int, devs: dict, n_out: int) -> Lag:
    """F8: `estimate_lag`'s NaN raise never mentioned a gap, though a gap is the only way to reach it in this pipeline.
    Re-raises citing each device's `gap_stats` when that is the cause.
    """
    try:
        return estimate_lag(ref_vm, test_vm, grid_hz, tuple(VIEWER["lag_band_hz"]), VIEWER["lag_bin_s"],
                            VIEWER["lag_search_s"])
    except ValueError as exc:
        if "NaN or infinite" not in str(exc):
            raise
        gaps = {d: gap_stats(devs[d].rec.t_s, grid_hz, n_out, VIEWER["max_gap_s"]) for d in DEVICES}
        detail = "; ".join(f"{d}: {g.n_gap_runs} gap run(s) totalling {g.total_gap_s:.1f}s (longest {g.longest_gap_s:.1f}s)"
                          for d, g in gaps.items())
        raise ValueError(f"{exc} -- gap_stats: {detail}") from exc


def _quality_report(devs: dict, lag, rot, n_examined: int, grid_hz: int, n_out: int, log: BuildLog) -> QualityReport:
    """Assembles the preprocessing quality report from values this build
    already has in hand. Independent of the artifact files: a `fail` here does not stop them being written (D4).
    """
    lim = Limits.from_config()
    quality = {d: device_quality(d, devs[d].rec, devs[d].xyz, devs[d].cal, devs[d].noise, grid_hz, n_out,
                                 VIEWER["max_gap_s"], lim) for d in DEVICES}
    al = alignment_quality(lag, rot, n_examined, lim)
    caps = capabilities(quality, al, lim)
    findings = tuple(f for dq in quality.values() for f in dq.findings) + al.findings
    start, end = recording_window()
    return QualityReport(built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         window=(start.isoformat(), end.isoformat()), window_variant=VIEWER["window_variant"],
                         devices=quality, alignment=al, capabilities=caps, limits=lim, steps=log.rows(),
                         status=status_of(findings))


def build(out: Path = OUT_DIR) -> dict:
    """Run the signal chain on the configured night and write manifest.json, epochs.json, night.json, quality.json,
    findings.csv and chunks/.

    Built into `out`'s `.partial` sibling and swapped into place only on success (F1, D6): a load or fit failure no
    longer leaves a half-deleted artifact set beside a stale manifest.
    """
    log = BuildLog()
    start, end = recording_window()
    n_epochs = int((end - start) / EPOCH)
    n_sec, grid_hz = n_epochs * VIEWER["epoch_s"], VIEWER["grid_hz"]
    n_out = n_sec * grid_hz

    work = out.parent / (out.name + ".partial")
    if work.exists():
        shutil.rmtree(work)
    (work / "chunks").mkdir(parents=True)

    with log.step("load:ax6"):
        devs = {"ax6": _load_device("ax6", start, end)}
    with log.step("load:apple"):
        devs["apple"] = _load_device("apple", start, end)
    with log.step("sleep_flags"):
        sleep, sleep_src = _sleep_flags(start, end)
        if any(len(v) != n_epochs for v in sleep.values()):
            raise ValueError(f"sleep flags cover {[len(v) for v in sleep.values()]} epochs, expected {n_epochs}")

    grid = lambda d, xyz: build_grid(devs[d].rec.t_s, xyz, devs[d].rec.fs, grid_hz, n_out)
    raw_xyz = {d: devs[d].rec.xyz for d in DEVICES}
    cal_xyz = {d: apply(raw_xyz[d], devs[d].cal) for d in DEVICES}
    with log.step("grid:raw"):
        raw_grids = {d: grid(d, raw_xyz[d]) for d in DEVICES}
    with log.step("grid:calibrated"):
        cal_grids = {d: grid(d, cal_xyz[d]) for d in DEVICES}
    # F2: these two grids (lag 0, same xyz the "raw"/"calibrated" variants below use for AX6) seed the cache so
    # `_variant` never rebuilds AX6's grid, only Apple's lag-corrected one.
    grid_cache = {("ax6", "raw", grid_hz, 0.0): raw_grids["ax6"], ("ax6", "calibrated", grid_hz, 0.0): cal_grids["ax6"]}

    with log.step("lag"):
        lag = _estimate_lag_or_explain(cal_grids["ax6"].vm, cal_grids["apple"].vm, grid_hz, devs, n_out)
    with log.step("rotation"):
        boot = VIEWER["bootstrap"]
        rot = fit_rotation(cal_grids["ax6"].xyz, cal_grids["apple"].xyz, grid_hz, VIEWER["still_window_s"],
                           VIEWER["calibration_std_g"], int(boot["block_min"] * 60 / VIEWER["still_window_s"]),
                           boot["resamples"], np.random.default_rng(boot["seed"]))
        _, _, n_examined = gravity_pairs(cal_grids["ax6"].xyz, cal_grids["apple"].xyz, grid_hz, VIEWER["still_window_s"],
                                         VIEWER["calibration_std_g"])

    tabs, nights = {}, {}
    for v, xyz in (("calibrated", cal_xyz), ("raw", raw_xyz)):
        with log.step(f"variant:{v}"):
            tabs[v], nights[v] = _variant(xyz, devs, lag.lag_s, sleep, n_out, grid_cache, v)

    spectral_hz = VIEWER["spectral_grid_hz"]
    with log.step("spectral:vm"):
        vm50 = _spectral_vm(devs, cal_xyz, lag.lag_s, n_sec * spectral_hz)
    with log.step("spectral:noise"):
        noise = _spectral_noise(devs)
    with log.step("spectral:epochs"):
        spectral_rows = spectral_epoch_rows(vm50, spectral_hz, VIEWER["epoch_s"] * spectral_hz, noise)
    with log.step("spectral:summary"):
        spectra = spectra_summary(vm50, devs, noise)

    with log.step("chunks:all"):
        chunks = _write_all_chunks(work, devs, raw_grids, n_sec)
    with log.step("chunks:spectrogram"):
        spec_chunks = spectrogram_chunks(work, vm50)
        for d in DEVICES:
            chunks[d]["spec"] = spec_chunks[d]
    with log.step("manifest"):
        manifest = _manifest(devs, rot, lag, chunks, sleep_src, start, n_epochs)
    with log.step("epoch_rows"):
        rows = epoch_rows(tabs, sleep)
        for row, srow in zip(rows, spectral_rows, strict=True):
            row["spectral"] = srow
    with log.step("quality_report"):
        report = _quality_report(devs, lag, rot, n_examined, grid_hz, n_out, log)
    with log.step("write_json"):
        files = {"manifest": clean(manifest, None), "epochs": clean(rows), "night": clean({**nights["calibrated"], **nights}),
                 "spectra": clean(spectra)}
        for name, obj in files.items():
            (work / f"{name}.json").write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")
        write_quality_report(report, work)

    if out.exists():
        shutil.rmtree(out)
    work.rename(out)
    return manifest
