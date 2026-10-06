"""Viewer artifacts: the frozen JSON + float32 chunk contract the viewer reads.

The fast tests use synthetic arrays. The slow ones build the real pilot-night artifacts once into a temp folder and check them
against the source parquet and its regression anchors.
"""

import json
import sys
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ax6apple.calibration import apply, calibrate
from ax6apple.config import AX6APPLE
from ax6apple.signals import load_recording, recording_window
from ax6apple.viewer_export import (VARIANTS, VIEWER, WAVELET_LEVEL, _ax6_only_band, build, clean, envelope, epoch_rows,
                                    epoch_table, spectra_summary, spectral_epoch_rows, spectrogram_chunks, uniform_native,
                                    write_chunks)
from conftest import PARQUET

ANCHOR = pd.Timestamp("2026-07-13 21:53:37.540")
EPOCH_KEYS = {"epoch", "enmo_ax6", "enmo_apple", "diff", "log2_ratio", "within_epoch_r", "within_epoch_r_lag0",
              "within_epoch_r_null", "within_epoch_lag_s", "sleep_ax6", "sleep_apple", "agreement_class", "noise_floor_flag"}
SWEEP_KEYS = {"w_s", "r", "r_lo", "r_hi", "null_lo", "null_hi", "n_eff", "n_eff_basis", "block_sensitivity", "optimal_lag_s",
              "within_median", "within_q25", "within_q75"}
FLOORS = (0.5, 0.5)                    # mg


# --- chunks: raw, little-endian, interleaved ---

def test_native_samples_are_the_stored_samples_on_a_uniform_axis():
    t = np.arange(600) / 50.0
    xyz = np.random.default_rng(0).normal(size=(600, 3))
    got = uniform_native(t, xyz, 50, 12)
    assert got.dtype == np.float32 and got.shape == (600, 3)
    assert np.array_equal(got, xyz.astype(np.float32))          # nothing calibrated, rotated or shifted


def test_native_samples_are_nan_inside_a_gap_longer_than_the_threshold():
    t = np.arange(50 * 60) / 50.0
    keep = (t < 20) | (t >= 40)                                  # a 20 s dropout
    got = uniform_native(t[keep], np.ones((keep.sum(), 3)), 50, 60, max_gap_s=1.0)
    tu = np.arange(60 * 50) / 50
    assert np.isnan(got[(tu > 20) & (tu < 40)]).all() and np.isfinite(got[(tu < 20) | (tu >= 40)]).all()


def test_native_axis_removes_timestamp_jitter_without_changing_the_level():
    t = np.arange(600) / 50.24                                  # the Apple export runs slightly fast
    xyz = np.tile([0.1, -0.2, 0.97], (600, 1))
    got = uniform_native(t, xyz, 50, 10)
    assert np.allclose(got, xyz[:500].astype(np.float32))


def test_ctx4_is_a_min_max_envelope_not_a_mean():
    x = np.zeros((100 * 3, 3))
    x[50, 0], x[51, 1] = 2.0, -3.0                              # a one-sample transient inside the second 0.25 s bin
    env = envelope(x.astype(np.float32), 100, 3)
    assert env.shape == (3 * VIEWER["ctx_hz"], 6)
    assert env[2, 3] == 2.0 and env[2, 1] == -3.0                # max_x = 2, min_y = -3 survive; a mean would be ~0.02
    assert env[:, :3].max() <= 0 and env[:, 3:].min() >= 0


def test_ctx_rate_comes_from_config(monkeypatch):
    monkeypatch.setitem(VIEWER, "ctx_hz", 2)
    assert envelope(np.zeros((100 * 3, 3), dtype=np.float32), 100, 3).shape == (6, 6)


def test_ctx4_handles_a_rate_that_is_not_a_multiple_of_four():
    env = envelope(np.random.default_rng(0).normal(size=(50 * 8, 3)).astype(np.float32), 50, 8)
    assert env.shape == (32, 6)
    assert (env[:, 3:] >= env[:, :3]).all()


def test_chunks_are_little_endian_float32_interleaved_and_indexed(tmp_path):
    (tmp_path / "chunks").mkdir()
    arr = np.arange(30 * 3, dtype=np.float32).reshape(30, 3)
    info = write_chunks(tmp_path, "ax6", "native", arr, 2, chunk_s=8)   # 16 samples per chunk
    assert info == {"hz": 2, "chunk_s": 8, "start_s": 0, "n_samples": 30, "files": ["ax6_native_00.f32", "ax6_native_01.f32"]}
    back = np.concatenate([np.fromfile(tmp_path / "chunks" / f, dtype="<f4") for f in info["files"]]).reshape(-1, 3)
    assert np.array_equal(back, arr)
    assert (tmp_path / "chunks" / info["files"][0]).stat().st_size == 16 * 3 * 4
    assert (tmp_path / "chunks" / info["files"][1]).read_bytes()[:4] == np.float32(48.0).astype("<f4").tobytes()


# --- epoch table and serialisation ---

def _ninety_epochs_enmo(scale_ax6, scale_apple):
    n = VIEWER["epoch_s"] * VIEWER["grid_hz"] * 4
    rng = np.random.default_rng(1)
    slow = np.abs(rng.normal(size=n))
    return slow * scale_ax6, slow * scale_apple + rng.normal(scale=1e-5, size=n) ** 2


def test_epoch_table_is_in_mg_diff_is_apple_minus_ax6_and_flags_when_either_is_low():
    ax6, apple = _ninety_epochs_enmo(0.01, 0.005)
    ax6[:900] = 1e-5                                             # first epoch: AX6 below the 0.5 mg floor
    tab = epoch_table(ax6, apple, FLOORS)
    assert tab["enmo_ax6"][1] == pytest.approx(ax6[900:1800].mean() * 1000)
    assert tab["diff"] == pytest.approx(tab["enmo_apple"] - tab["enmo_ax6"])
    assert tab["noise_floor_flag"][0] and not tab["noise_floor_flag"][1]
    assert np.isnan(tab["log2_ratio"][0]) and np.isfinite(tab["log2_ratio"][1])


def test_epoch_table_floors_are_per_device_and_it_carries_the_peak_r_with_its_lag0_and_null():
    ax6, apple = _ninety_epochs_enmo(0.01, 0.005)
    ax6_mg, apple_mg = (epoch_table(ax6, apple, (0.0, 0.0))[k] for k in ("enmo_ax6", "enmo_apple"))
    floor = float(np.sort(apple_mg)[1]) + 1e-9                   # flags the two quietest Apple epochs only
    tab = epoch_table(ax6, apple, (0.0, floor))
    assert tab["noise_floor_flag"].tolist() == (apple_mg < floor).tolist() and tab["noise_floor_flag"].sum() == 2
    ax6_only = epoch_table(ax6, apple, (float(np.sort(ax6_mg)[0]) + 1e-9, 0.0))              # the AX6 floor alone flags its own quietest
    assert ax6_only["noise_floor_flag"].tolist() == (ax6_mg < np.sort(ax6_mg)[0] + 1e-9).tolist()
    for k in ("within_epoch_r", "within_epoch_r_lag0", "within_epoch_r_null"):
        assert len(tab[k]) == 4
    ok = ~np.isnan(tab["within_epoch_r"])
    assert (tab["within_epoch_r"][ok] >= tab["within_epoch_r_lag0"][ok] - 1e-12).all()
    assert np.isfinite(tab["within_epoch_r_null"][ok]).all()


def test_epoch_rows_carry_both_variants_and_no_timestamp():
    ax6, apple = _ninety_epochs_enmo(0.01, 0.005)
    cal, raw = epoch_table(ax6, apple, FLOORS), epoch_table(ax6 * 2, apple, FLOORS)
    sleep = {"ax6": np.array([True, True, False, False]), "apple": np.array([True, False, True, False])}
    rows = epoch_rows({"calibrated": cal, "raw": raw}, sleep)
    assert len(rows) == 4 and EPOCH_KEYS <= set(rows[0]) and not {"time", "timestamp"} & set(rows[0])
    assert [r["agreement_class"] for r in rows] == ["both_sleep", "ax6_only", "apple_only", "both_wake"]
    assert rows[0]["calibrated"]["enmo_ax6"] == rows[0]["enmo_ax6"] != rows[0]["raw"]["enmo_ax6"]


def test_clean_turns_nan_into_null_and_numpy_into_json():
    obj = clean({"a": np.array([1.0, np.nan]), "b": np.float32(2.5), "c": np.bool_(True), "d": float("inf")})
    assert json.loads(json.dumps(obj)) == {"a": [1.0, None], "b": 2.5, "c": True, "d": None}


# --- spectral artifacts ---

def _fake_dev(rng, fs, scale=0.003, n=100 * 600):
    xyz = rng.normal(scale=scale, size=(n, 3)) + [0, 0, 1.0]
    rec = SimpleNamespace(xyz=xyz, fs=fs)
    return SimpleNamespace(rec=rec, xyz=xyz)


def test_write_chunks_accepts_the_spectrogram_tiers_sub_1hz_rate(tmp_path):
    (tmp_path / "chunks").mkdir()
    arr = np.zeros((899, 5), dtype=np.float32)   # 899 time bins, 5 freq bins: a stand-in for a spec tier
    info = write_chunks(tmp_path, "ax6", "spec", arr, 1 / 4, 1800)   # hop_s = 4 s -> 450 time bins per 1800 s chunk
    assert info["files"] == ["ax6_spec_00.f32", "ax6_spec_01.f32"]
    assert (tmp_path / "chunks" / info["files"][0]).stat().st_size == 450 * 5 * 4


def test_spectral_epoch_rows_carries_both_devices_and_matches_epoch_count():
    rng = np.random.default_rng(2)
    fs, epoch_n, n_epochs = 50, 30 * 50, 3
    vm50 = {d: rng.normal(scale=0.01, size=epoch_n * n_epochs) + 1.0 for d in ("ax6", "apple")}
    f = np.linspace(0, 25, epoch_n // 2 + 1)
    noise = {d: (f, np.full_like(f, 1e-8)) for d in ("ax6", "apple")}
    rows = spectral_epoch_rows(vm50, fs, epoch_n, noise)
    assert len(rows) == n_epochs
    for row in rows:
        assert set(row) == {"ax6", "apple"}
        for d in ("ax6", "apple"):
            assert set(row[d]["band_rms_mg"]) == {"posture", "gross_movement", "jerk_twitch", "tremor", "diagnostic", "margin"}
            assert row[d]["modwt_energy"].shape == (WAVELET_LEVEL,)
            assert (row[d]["modwt_energy"] >= 0).all()
            assert all(np.isfinite(row[d][k]) for k in ("sef95_hz", "centroid_hz", "dominant_hz", "spectral_entropy"))


def test_ax6_only_band_has_no_apple_counterpart_and_is_a_single_scalar():
    rng = np.random.default_rng(3)
    devs = {"ax6": _fake_dev(rng, 100)}
    out = _ax6_only_band(devs)
    assert out["band_hz"] == [25.0, 49.0] and out["rms_mg"] >= 0 and "Apple" in out["note"]


def test_spectra_summary_ships_psd_coherence_and_the_ax6_only_diagnostic():
    rng = np.random.default_rng(4)
    n = 50 * 1800
    vm50 = {d: rng.normal(scale=0.01, size=n) + 1.0 for d in ("ax6", "apple")}
    f = np.linspace(0, 25, 30 * 50 // 2 + 1)
    noise = {d: (f, np.full_like(f, 1e-8)) for d in ("ax6", "apple")}
    devs = {d: _fake_dev(rng, 100) for d in ("ax6", "apple")}
    out = spectra_summary(vm50, devs, noise)
    assert out["grid_hz"] == 50 and set(out["comparable_bands"]) <= set(out["bands"])
    for d in ("ax6", "apple"):
        assert len(out["psd"][d]["f"]) == len(out["psd"][d]["raw_db"]) == len(out["psd"][d]["corrected_db"])
        assert (out["psd"][d]["corrected_db"] <= out["psd"][d]["raw_db"] + 1e-6).all()   # noise-corrected never exceeds raw
    assert 0 < out["coherence"]["bias_floor"] < 1
    assert out["wavelet"]["level"] == WAVELET_LEVEL and len(out["wavelet"]["dyadic_bands_hz"]) == WAVELET_LEVEL
    assert "ax6_only_band" in out


def test_spectrogram_chunks_write_one_device_pair_with_matching_freq_axes(tmp_path):
    (tmp_path / "chunks").mkdir()
    rng = np.random.default_rng(5)
    vm50 = {d: rng.normal(scale=0.01, size=50 * 1800) + 1.0 for d in ("ax6", "apple")}
    out = spectrogram_chunks(tmp_path, vm50)
    assert set(out) == {"ax6", "apple"}
    assert out["ax6"]["freq_hz"] == out["apple"]["freq_hz"]
    assert out["ax6"]["hz"] == 0.25 and out["ax6"]["n_freq"] == len(out["ax6"]["freq_hz"])
    for f in out["ax6"]["files"]:
        assert (tmp_path / "chunks" / f).exists()


# --- the real night ---

@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    if not all(p.exists() for p in PARQUET.values()):
        pytest.skip("source drive unreachable")
    out = tmp_path_factory.mktemp("viewer")
    build(out)
    load = lambda name: json.loads((out / f"{name}.json").read_text(encoding="utf-8"))
    return out, load("manifest"), load("epochs"), load("night")


def _chunk_sample(out, man, dev, tier, k, c):
    info = man["chunks"][dev][tier]
    per, ch = info["chunk_s"] * info["hz"], 6 if tier == "ctx4" else 3
    a = np.fromfile(out / "chunks" / info["files"][k // per], dtype="<f4").reshape(-1, ch)
    return a[k % per, :c] if c else a[k % per]


@pytest.mark.slow
def test_real_native_chunks_hold_raw_parquet_samples_not_calibrated_ones(artifacts):
    out, man, *_ = artifacts
    start, end = recording_window()
    for dev in ("ax6", "apple"):
        rec = load_recording(PARQUET[dev], start, end)
        cal = apply(rec.xyz, calibrate(rec.xyz, rec.fs, VIEWER["calibration_std_g"], VIEWER["still_window_s"]))
        for k in (0, 123456, 1_000_000, 1_700_000):
            i = np.searchsorted(rec.t_s, k / rec.fs)              # the stored sample nearest that time
            got = _chunk_sample(out, man, dev, "native", k, 3)
            assert np.abs(got - rec.xyz[i]).max() < 3e-3          # jitter and interpolation only
            if dev == "ax6":
                assert np.abs(got - cal[i]).max() > 0.03          # the -37 / +52 mg offsets: not calibrated


@pytest.mark.slow
def test_real_chunk_index_and_layout(artifacts):
    out, man, *_ = artifacts
    assert man["chunks"]["ax6"]["native"]["hz"] == 100 and man["chunks"]["apple"]["native"]["hz"] == 50
    for dev in ("ax6", "apple"):
        assert len(man["chunks"][dev]["native"]["files"]) == 20 and len(man["chunks"][dev]["grid30"]["files"]) == 20
        ctx = man["chunks"][dev]["ctx4"]
        assert ctx["hz"] == 4 and len(ctx["files"]) == 1 and ctx["n_samples"] == 36000 * 4
        assert (out / "chunks" / ctx["files"][0]).stat().st_size == ctx["n_samples"] * 6 * 4
        env = np.fromfile(out / "chunks" / ctx["files"][0], dtype="<f4").reshape(-1, 6)
        assert (env[:, 3:] >= env[:, :3]).all()
        assert (out / "chunks" / man["chunks"][dev]["grid30"]["files"][0]).stat().st_size == 1800 * 30 * 3 * 4
        g50 = man["chunks"][dev]["grid50"]
        assert g50["hz"] == 50 == VIEWER["spectral_grid_hz"] and len(g50["files"]) == 20
        assert g50["n_samples"] == man["n_epochs"] * VIEWER["epoch_s"] * 50
        assert (out / "chunks" / g50["files"][0]).stat().st_size == 1800 * 50 * 3 * 4
        arr50 = np.fromfile(out / "chunks" / g50["files"][0], dtype="<f4").reshape(-1, 3)
        assert np.isfinite(arr50).all()   # mid-file: no gap-NaN, no resample_poly edge transient here


@pytest.mark.slow
def test_real_manifest_has_provenance_and_surfaces_the_caveats(artifacts):
    _, man, *_ = artifacts
    assert man["anchor"] == "2026-07-13T21:53:37.540" and pd.Timestamp(man["anchor"]) == ANCHOR
    assert man["window"]["requested_start"] == "2026-07-13T18:00:00.000" and man["window"]["available_start"] == man["anchor"]
    assert man["session_id"] == PureWindowsPath(AX6APPLE["sources"]["ax6"]["parquet"]).parts[1] and man["synthetic"] is False and man["built"]
    for dev in ("ax6", "apple"):
        assert len(man["devices"][dev]["source_sha256"]) == 64 and man["devices"][dev]["source_path"].endswith(".parquet")
    assert all(len(v["sha256"]) == 64 for v in man["sleep_source"].values())
    assert man["config"]["bootstrap"]["seed"] == VIEWER["bootstrap"]["seed"]
    rot = man["rotation"]
    assert rot["maps"] == "apple_to_ax6" and rot["residual_deg"] == pytest.approx(56.8, abs=0.5)
    assert rot["uncertainty_deg"] == pytest.approx(14.9, abs=0.5)
    assert man["lag_s"] == pytest.approx(3.0, abs=0.5) and "lag_s" in man["lag_convention"]
    t = man["caveats"]["timing"]
    assert (t["apple"]["reordered"], t["apple"]["duplicates_dropped"]) == (97, 6)
    assert (t["ax6"]["reordered"], t["ax6"]["duplicates_dropped"]) == (0, 0)
    diff = np.abs(man["caveats"]["calibration_vs_abk"]["offset_diff_mg_converged_minus_matlab_early_stop"]["ax6"])
    assert 2 < diff.max() < 6                                      # 3 to 5 mg on some axes


@pytest.mark.slow
def test_real_epochs_and_night_carry_raw_and_calibrated_variants(artifacts):
    _, _, epochs, night = artifacts
    assert len(epochs) == 1200 and [r["epoch"] for r in epochs] == list(range(1200))
    assert all(EPOCH_KEYS <= set(r) for r in epochs)
    for v in VARIANTS:
        assert set(night[v]["sweep"]["enmo"]) == SWEEP_KEYS and set(night[v]["sweep"]["vm"]) == SWEEP_KEYS
        assert {"r", "rho", "ccc", "icc", "bias", "loa_lo", "loa_hi", "kappa"} == set(night[v]["scalars"])
        assert all(set(s) == {"est", "lo", "hi"} for s in night[v]["scalars"].values())
        ok = [r for r in epochs if not r[v]["noise_floor_flag"]]
        assert len(night[v]["ba"]["mean"]) == len(night[v]["scatter"]["ax6"]) == len(ok) == 1200 - night[v]["n_flagged_epochs"]
        assert all(r[v]["log2_ratio"] is None for r in epochs if r[v]["noise_floor_flag"])
    assert night["scalars"] == night["calibrated"]["scalars"]        # top level mirrors calibrated
    assert night["raw"]["scalars"]["r"]["est"] != night["calibrated"]["scalars"]["r"]["est"]
    assert epochs[5]["raw"]["enmo_ax6"] != epochs[5]["calibrated"]["enmo_ax6"]


@pytest.mark.slow
def test_real_section_2_anchors(artifacts):
    _, man, epochs, night = artifacts
    cal = man["devices"]
    assert cal["ax6"]["calibration"]["gain"] == pytest.approx([1.001, 0.989, 0.999], abs=0.001)
    assert cal["ax6"]["calibration"]["offset_mg"] == pytest.approx([-37.5, 52.4, -5.1], abs=0.2)
    assert cal["apple"]["calibration"]["offset_mg"] == pytest.approx([-0.5, 2.5, -4.0], abs=0.3)
    en = lambda v, d: np.mean([r[v][f"enmo_{d}"] for r in epochs])
    assert (en("calibrated", "ax6"), en("calibrated", "apple")) == pytest.approx((2.09, 1.29), abs=0.02)
    assert night["calibrated"]["scalars"]["r"]["est"] == pytest.approx(0.55, abs=0.02)
    assert night["raw"]["scalars"]["r"]["est"] == pytest.approx(0.18, abs=0.02)
    w = dict(zip(night["calibrated"]["sweep"]["enmo"]["w_s"], night["calibrated"]["sweep"]["enmo"]["r"]))
    assert w[30] == pytest.approx(0.55, abs=0.02)
    sleep_ax6, sleep_apple = (np.array([r[k] for r in epochs]) for k in ("sleep_ax6", "sleep_apple"))
    assert (sleep_ax6 == sleep_apple).mean() == pytest.approx(0.9333, abs=1e-4)
    assert (sleep_ax6.sum() / 2, sleep_apple.sum() / 2) == (544, 524)
    assert night["calibrated"]["scalars"]["kappa"]["est"] == pytest.approx(0.6605, abs=1e-3)


@pytest.mark.slow
def test_real_calibration_diagnostics_and_config_driven_artifact_parameters(artifacts):
    _, man, *_ = artifacts
    for dev, rms_mg in (("ax6", 2.90), ("apple", 0.19)):
        c = man["devices"][dev]["calibration"]
        assert c["converged"] is True and c["low_orientation_coverage"] is False
        assert c["rms_g"] * 1000 == pytest.approx(rms_mg, abs=0.02) and c["iterations"] < 1000
    assert man["devices"]["apple"]["calibration"]["offset_mg"][2] == pytest.approx(-3.96, abs=0.02)
    assert man["window"]["requested_start"] == "2026-07-13T18:00:00.000" and man["window"]["requested_end"] == "2026-07-14T08:00:00.000"
    assert man["config"]["chunk_s"] == VIEWER["chunk_s"] == 1800 and man["config"]["ctx_hz"] == VIEWER["ctx_hz"] == 4
    assert man["chunks"]["ax6"]["native"]["chunk_s"] == 1800 and man["chunks"]["apple"]["ctx4"]["hz"] == 4


@pytest.mark.slow
def test_real_epoch_peak_r_ships_with_its_lag0_r_and_null(artifacts):
    _, _, epochs, _ = artifacts
    for v in VARIANTS:
        r = {k: np.array([np.nan if row[v][k] is None else row[v][k] for row in epochs]) for k in
             ("within_epoch_r", "within_epoch_r_lag0", "within_epoch_r_null")}
        ok = ~np.isnan(r["within_epoch_r"])
        assert ok.sum() == (1200 if v == "calibrated" else 929)          # raw: 271 epochs where one device's ENMO is all zero
        assert np.isfinite(r["within_epoch_r_null"][ok]).all() and (np.isnan(r["within_epoch_r_lag0"]) == ~ok).all()
        assert (r["within_epoch_r"][ok] >= r["within_epoch_r_lag0"][ok] - 1e-9).all()      # a maximum over lags includes lag 0
        assert np.nanmedian(r["within_epoch_r"]) > np.nanmedian(r["within_epoch_r_lag0"]) + 0.1   # the selection bias, on real data
    assert epochs[0]["within_epoch_r_null"] == epochs[0]["calibrated"]["within_epoch_r_null"]


@pytest.mark.slow
def test_real_spectral_noise_floor_uses_the_same_sample_set_as_the_enmo_noise_floor():
    """F5: `spectral.noise_psd` and `signals.sensor_noise_floor` must agree on which still windows they measured --
    before this fix `noise_psd` saw `rec.xyz` (with the +/- lag_search_s margin) while `sensor_noise_floor` saw
    `rec.xyz[inside]` (margin stripped)."""
    from ax6apple import spectral
    from ax6apple.calibration import still_windows
    from ax6apple.viewer_export import _load_device

    start, end = recording_window()
    if not all(p.exists() for p in PARQUET.values()):
        pytest.skip("source drive unreachable")
    for dev in ("ax6", "apple"):
        d = _load_device(dev, start, end)
        _, N = spectral.noise_psd(d.xyz, d.rec.fs, VIEWER["still_window_s"], VIEWER["calibration_std_g"])
        assert np.isfinite(N).all()
        still, _ = still_windows(d.xyz, d.rec.fs, VIEWER["still_window_s"], VIEWER["calibration_std_g"])
        assert len(still) == d.noise["n_windows"]      # same still-window count as the ENMO noise floor


@pytest.mark.slow
def test_real_quality_report_matches_the_known_night(artifacts):
    """Phases 3 and 6 Done-when: the report reproduces the pilot night's known state, and the build is atomic (F1)."""
    out, man, *_ = artifacts
    assert (out / "quality.json").exists() and (out / "findings.csv").exists()
    assert not (out.parent / (out.name + ".partial")).exists()           # F1: no partial directory left behind
    quality = json.loads((out / "quality.json").read_text(encoding="utf-8"))
    assert quality["capabilities"] == {"enmo_agreement": True, "axis_comparison": False,
                                       "spectral_comparison": True, "per_epoch_lag": True}
    codes = {f["code"] for dq in quality["devices"].values() for f in dq["findings"]}
    assert "timing.reordered" in codes             # Apple's 97 reorderings land as a finding ...
    apple_reordered = next(f for f in quality["devices"]["apple"]["findings"] if f["code"] == "timing.reordered")
    assert apple_reordered["level"] == "info"       # ... at info, not warn (5.3e-5 of rows, section 3)
    assert not any(f["code"] == "calibration.coverage" for f in quality["devices"]["ax6"]["findings"])  # 1.786 passes 1.2
    align_codes = {f["code"] for f in quality["alignment"]["findings"]}
    assert {"rotation.residual", "rotation.uncertainty"} <= align_codes    # 56.8 / 14.9 both fail their limits
    assert "lag.weak_r" not in align_codes                                 # r = 0.45 passes narrowly (lag_r_warn 0.30)
    assert quality["status"] == "marginal"
    assert len(quality["steps"]) > 0 and any(s["status"] == "ok" for s in quality["steps"])


@pytest.mark.slow
def test_real_night_noise_floors_block_sensitivity_and_n_eff_fields(artifacts):
    _, _, _, night = artifacts
    nf = night["calibrated"]["noise_floor"]
    assert nf["ax6"]["value_mg"] == pytest.approx(1.18, abs=0.02) and nf["apple"]["value_mg"] == pytest.approx(0.65, abs=0.02)
    assert all("derivation" in nf[d] for d in ("ax6", "apple"))
    assert (nf["ax6"]["fraction_flagged"], nf["apple"]["fraction_flagged"], nf["joint"]["fraction_flagged"]) == pytest.approx(
        (0.55, 0.47, 0.76), abs=0.01)                                     # most of this night is noise-dominated
    assert nf["joint"]["n_flagged"] == night["calibrated"]["n_flagged_epochs"]
    assert night["raw"]["noise_floor"]["ax6"]["value_mg"] == nf["ax6"]["value_mg"]                 # a sensor property
    assert night["raw"]["noise_floor"]["joint"]["fraction_flagged"] > nf["joint"]["fraction_flagged"]
    assert night["noise_floor"] == nf and night["block_sensitivity"] == night["calibrated"]["block_sensitivity"]
    for v in VARIANTS:
        sens = night[v]["block_sensitivity"]
        assert [b["block_min"] for b in sens] == [5, 15, 30, 60] and "conditional" in night[v]["block_sensitivity_note"]
        assert sens[0]["lo"] == night[v]["scalars"]["r"]["lo"] and sens[0]["hi"] == night[v]["scalars"]["r"]["hi"]
        if v == "calibrated":
            assert sens[-1]["hi"] - sens[-1]["lo"] > 1.3 * (sens[0]["hi"] - sens[0]["lo"])     # block 5 understates it
        for kind in ("enmo", "vm"):
            sw = night[v]["sweep"][kind]
            assert set(sw["n_eff_basis"]) == {"bootstrap_half_width"} and len(sw["block_sensitivity"]) == len(sw["w_s"])
            assert [b["block_min"] for b in sw["block_sensitivity"][0]] == [5, 15, 30, 60]
            for r, lo, hi, n_eff in zip(sw["r"], sw["r_lo"], sw["r_hi"], sw["n_eff"], strict=True):
                assert n_eff == pytest.approx(min(max((1.96 * (1 - r ** 2) / ((hi - lo) / 2)) ** 2, 1), 1_080_000), rel=1e-3)
    e = night["calibrated"]["sweep"]["enmo"]
    assert e["n_eff"][e["w_s"].index(30)] == pytest.approx(84, abs=25) and e["n_eff"][0] < 78533 / 10
