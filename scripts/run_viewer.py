"""Serve viewer/ (and the artifact directory, mounted at /data/) on localhost only, then open a browser.

    python scripts/run_viewer.py               # artifacts built by scripts/build_viewer.py: data/ax6apple/viewer
    python scripts/run_viewer.py --data DIR    # any other artifact directory

Real recordings are identifiable human-subject accelerometry: the server binds 127.0.0.1 only.
"""

import argparse
import functools
import http.server
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIEWER = ROOT / "viewer"
DEFAULT = ROOT / "data" / "ax6apple" / "viewer"


class Handler(http.server.SimpleHTTPRequestHandler):
    data_dir: Path

    def translate_path(self, path: str) -> str:
        clean = super().translate_path(path)                       # resolved under `directory` (= viewer/)
        rel = Path(clean).relative_to(VIEWER)
        if rel.parts[:1] == ("data",):
            return str(self.data_dir.joinpath(*rel.parts[1:]))
        return clean

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT, help="artifact directory")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    data = args.data.resolve()
    if not (data / "manifest.json").exists():
        raise SystemExit(f"no manifest.json in {data}; build the artifacts first: python scripts/make_demo_data.py && python scripts/build_viewer.py")
    Handler.data_dir = data
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), functools.partial(Handler, directory=str(VIEWER)))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"serving viewer/ + {data} at {url}  (localhost only; Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
