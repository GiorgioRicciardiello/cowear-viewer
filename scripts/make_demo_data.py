"""Write a SYNTHETIC co-wear night: an AX6 and an Apple Watch on the same wearer, plus each device's sleep bouts.

    python scripts/make_demo_data.py

The output has the layout config/config.yaml points at, so `scripts/build_viewer.py` runs the real signal chain on it
unchanged. Nothing here reads or resembles a real participant recording. The generator plants every artefact the
pipeline exists to undo, each with a known answer:

  - per-axis gain and offset errors on both devices (recovered by the still-window calibration),
  - a different mounting frame on the Apple Watch (recovered by the gravity-vector rotation fit),
  - an Apple clock running LAG_S ahead of the AX6 (recovered by the movement-envelope lag search),
  - Apple timestamps at 50.24 Hz with 2.6 % jitter, rows out of order and exact duplicates (repaired and counted on load),
  - a flat AX6 noise floor above the Apple's (the per-device noise floor and noise-corrected spectra),
  - brief 6-9 Hz twitch bursts during sleep (visible in the jerk/twitch band and the spectrogram).
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ax6apple.config import AX6APPLE, PREPROCESSED_ROOT, RBD_ROOT

SEED = 20260110
EPOCH_S, MARGIN_S = 30, 60                    # the devices record a minute either side of the analysed window
TRUE_HZ, APPLE_HZ, APPLE_JITTER = 100, 50.24, 0.026
LAG_S = 3.4                                   # apple_time = ax6_time + LAG_S
NOISE_MG = {"ax6": 1.2, "apple": 0.65}        # per-axis white sensor noise
CAL = {"ax6": ([1.004, 0.991, 0.998], [-37.5, 52.4, -5.1]),    # gain, offset (mg): measured = (true - offset) / gain
       "apple": ([1.001, 1.002, 0.999], [-0.5, 2.5, -4.0])}
SHARED, OWN = 0.7, 0.6                        # Apple movement = SHARED * whole-body movement + OWN * its wrist's own
N_REORDERED, N_DUPLICATES = 40, 12


def rotation(axis=(0.3, 0.5, 0.8), deg=35.0) -> np.ndarray:
    """Rodrigues: the matrix that maps the Apple frame into the AX6 frame (x_ax6 = R @ x_apple)."""
    k = np.asarray(axis) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    a = np.deg2rad(deg)
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def sleep_epochs(rng, n: int, n_margin: int) -> np.ndarray:
    """Sleep after a 20 min latency, with brief awakenings; awake in the margins and the last 15 min."""
    sleep = np.zeros(n, dtype=bool)
    i = n_margin + 40
    while i < n - n_margin - 30:
        run = int(rng.exponential(70)) + 10
        sleep[i:min(i + run, n - n_margin - 30)] = True
        i += run + int(rng.integers(1, 10))
    return sleep


def truth(rng, sleep: np.ndarray, n_s: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(gravity, movement, the other wrist's own movement), n x 3 g at 100 Hz on the AX6 clock. Posture changes are
    15 s turns, more often awake."""
    n = n_s * TRUE_HZ
    amp = np.where(sleep, rng.lognormal(np.log(0.0008), 0.5, len(sleep)), rng.lognormal(np.log(0.03), 0.6, len(sleep)))
    amp[sleep & (rng.random(len(sleep)) < 0.08)] = 0.015                   # occasional movement in sleep
    env = lambda a: uniform_filter1d(np.repeat(a, EPOCH_S * TRUE_HZ)[:n], TRUE_HZ)[:, None]
    smooth = lambda a: uniform_filter1d(rng.standard_normal((n, 3)), 5, axis=0) * np.sqrt(5) * env(a) * [1.0, 0.6, 0.8]
    move, own = smooth(amp), smooth(amp * rng.lognormal(0, 0.8, len(amp)))   # same state, its own level per epoch

    t = np.arange(n) / TRUE_HZ
    for e in np.flatnonzero(sleep & (rng.random(len(sleep)) < 0.03)):     # twitches: 0.3 s at 6-9 Hz
        t0 = e * EPOCH_S + rng.uniform(0, EPOCH_S - 1)
        i = slice(int(t0 * TRUE_HZ), int((t0 + 0.3) * TRUE_HZ))
        move[i] += 0.05 * np.sin(2 * np.pi * rng.uniform(6, 9) * (t[i] - t0))[:, None] * rng.standard_normal(3)

    knots, k = [0.0], 0.0
    while k < n_s:
        k += rng.exponential(1500 if sleep[min(int(k // EPOCH_S), len(sleep) - 1)] else 300)
        knots.append(k)
    post = 0.7 * rng.standard_normal((len(knots), 3)) + [0, 0, 1.0]       # mostly lying on the back of the wrist
    kt = np.repeat(knots, 2)[1:] + np.tile([0, 15.0], len(knots))[1:]       # hold, then a 15 s turn to the next posture
    g = np.stack([np.interp(t, kt, np.repeat(post, 2, axis=0)[:-1, c]) for c in range(3)], axis=1)
    return g / np.linalg.norm(g, axis=1, keepdims=True), move, own


def measured(true: np.ndarray, dev: str, rng) -> np.ndarray:
    gain, off = (np.asarray(v) for v in CAL[dev])
    return ((true - off / 1000) / gain + rng.standard_normal(true.shape) * NOISE_MG[dev] / 1000).astype(np.float32)


def apple_clock(rng, n_s: int) -> np.ndarray:
    """Device timestamps (s on the Apple clock, from -MARGIN_S): 50.24 Hz with jitter, covering the whole span."""
    dt = np.clip(1 / APPLE_HZ * (1 + APPLE_JITTER * rng.standard_normal(int(n_s * APPLE_HZ))), 0.002, None)
    t = -MARGIN_S + LAG_S + np.concatenate([[0], np.cumsum(dt)[:-1]])
    return t[t < n_s - MARGIN_S - 1 + LAG_S]


def export_quirks(df: pd.DataFrame, rng) -> pd.DataFrame:
    """Rows swapped with their neighbour and exact duplicate rows, as the Apple export produces at batch boundaries."""
    order = np.arange(len(df))
    for i in rng.choice(len(df) - 1, N_REORDERED, replace=False):
        order[[i, i + 1]] = order[[i + 1, i]]
    out = df.iloc[order].reset_index(drop=True)
    dup = rng.choice(len(out), N_DUPLICATES, replace=False)
    return pd.concat([out, out.iloc[dup]]).sort_index(kind="stable").reset_index(drop=True)   # each copy after its original


def write_parquet(df: pd.DataFrame, rel: str) -> None:
    path = PREPROCESSED_ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    sidecar = {"synthetic": True, "source_relative_path": f"SYNTHETIC/{Path(rel).name}",
               "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path.with_name(path.name + ".json").write_text(json.dumps(sidecar, indent=1), encoding="utf-8")


def bouts(sleep: np.ndarray, anchor: pd.Timestamp) -> pd.DataFrame:
    """Runs of sleep epochs as bouts; `annotation_end` is the start of the last epoch (one-epoch bout: end == start)."""
    edges = np.diff(np.concatenate([[0], sleep.astype(int), [0]]))
    first, last = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    at = lambda i: (anchor + pd.Timedelta(seconds=float(i) * EPOCH_S)).isoformat(sep=" ", timespec="milliseconds")
    return pd.DataFrame({"annotation_start": [at(i) for i in first], "annotation_end": [at(i) for i in last]})


def apple_sleep(rng, sleep: np.ndarray) -> np.ndarray:
    """The Apple's scorer disagrees at bout edges: every transition moves by up to two epochs."""
    out = sleep.copy()
    for i in np.flatnonzero(np.diff(sleep.astype(int))) + 1:
        shift = int(rng.integers(-2, 3))
        lo, hi = sorted((i, i + shift))
        out[lo:hi] = sleep[i - 1] if shift > 0 else sleep[i]
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    window = next(v for v in AX6APPLE["variants"] if v["name"] == AX6APPLE["viewer"]["window_variant"])
    anchor, end = pd.Timestamp(window["start"]), pd.Timestamp(window["end"])
    night_s = int((end - anchor).total_seconds())
    span_s = night_s + 2 * MARGIN_S
    n_margin = MARGIN_S // EPOCH_S
    sleep = sleep_epochs(rng, span_s // EPOCH_S, n_margin)
    g, move, own = truth(rng, sleep, span_s)
    t_true = np.arange(span_s * TRUE_HZ) / TRUE_HZ - MARGIN_S             # AX6 clock, s after the anchor

    stamp = lambda t: anchor + pd.to_timedelta(np.round(t * 1e6).astype("int64"), unit="us")
    ax6 = measured(g + move, "ax6", rng)
    write_parquet(pd.DataFrame({"time": stamp(t_true), "x": ax6[:, 0], "y": ax6[:, 1], "z": ax6[:, 2]}),
                  AX6APPLE["sources"]["ax6"]["parquet"])

    t_apple = apple_clock(rng, span_s)
    body = g + SHARED * move + OWN * own
    at_true = np.stack([np.interp(t_apple - LAG_S, t_true, body[:, c]) for c in range(3)], axis=1)
    apple = measured(at_true @ rotation(), "apple", rng)                   # row form of R^T x: the Apple frame
    df = pd.DataFrame({"time": stamp(t_apple), "x": apple[:, 0], "y": apple[:, 1], "z": apple[:, 2]})
    write_parquet(export_quirks(df, rng), AX6APPLE["sources"]["apple"]["parquet"])

    in_window = sleep[n_margin:n_margin + night_s // EPOCH_S]
    comp = next(c for c in AX6APPLE["comparisons"] if c["name"] == "device")
    stem = next(v for v in AX6APPLE["variants"] if v["name"] == comp["ref"])["stem"]
    for name, flags in ((comp["ref"], in_window), (comp["test"], apple_sleep(rng, in_window))):
        path = RBD_ROOT / "fit" / name / "auto_sleep" / "abk" / f"{stem}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        bouts(flags, anchor).to_csv(path, index=False)
    print(f"synthetic night written: {night_s / 3600:.1f} h, {in_window.mean():.0%} asleep, Apple lag {LAG_S} s "
          f"-> {PREPROCESSED_ROOT} and {RBD_ROOT}")


if __name__ == "__main__":
    main()
