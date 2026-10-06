"""Build the viewer artifacts (manifest.json, epochs.json, night.json, chunks/*.f32) into data/ax6apple/viewer.

    python scripts/build_viewer.py [--out DIR]

Reads the device Parquet files and sleep bouts named in config/config.yaml (by default the synthetic night written by
scripts/make_demo_data.py). See docs/METHODS.md for what each step computes.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ax6apple.viewer_export import OUT_DIR, build


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    out = ap.parse_args().out
    t0 = time.time()
    manifest = build(out)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"viewer artifacts written to {out}: {manifest['n_epochs']} epochs, {size / 1e6:.0f} MB, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
