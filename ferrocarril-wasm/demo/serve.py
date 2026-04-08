#!/usr/bin/env python3
"""Serve the ferrocarril-wasm demo on http://localhost:8080.

Routes:
  /             -> the demo directory (index.html, main.js, pkg/)
  /weights/...  -> the ../../../ferrocarril_weights directory

The weights directory is expected to contain the output of
`weight_converter.py` (i.e. `config.json`, `model/metadata.json`, per-
tensor `.bin` files, and `voices/voices.json` + per-voice `.bin` files).
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import os
import sys

DEMO_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = DEMO_DIR.parent.parent  # ferrocarril/ (workspace root)
WEIGHTS_DIR = (WORKSPACE_ROOT.parent / "ferrocarril_weights").resolve()

PORT = int(os.environ.get("PORT", "8080"))
HOST = os.environ.get("HOST", "0.0.0.0")


def _is_inside(path: Path, root: Path) -> bool:
    """Return True if `path` is `root` itself or strictly inside it.

    Uses Path.relative_to inside a try/except, which correctly handles
    `..` traversal *and* the sibling-prefix attack (a sibling directory
    whose name shares the same string prefix as `root`). Both `path`
    and `root` must already be absolute / resolved.
    """
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class DemoHandler(SimpleHTTPRequestHandler):
    # Route /weights/ to the real weights directory; everything else
    # comes from the demo directory.
    def translate_path(self, path: str) -> str:
        # Strip query string + fragment before path-mapping.
        q = path.find("?")
        if q >= 0:
            path = path[:q]
        frag = path.find("#")
        if frag >= 0:
            path = path[:frag]

        if path.startswith("/weights/"):
            rel = path[len("/weights/"):]
            # Reject any leading slash on `rel` so the join doesn't
            # silently anchor at the filesystem root.
            rel = rel.lstrip("/")
            target = (WEIGHTS_DIR / rel).resolve()
            if not _is_inside(target, WEIGHTS_DIR):
                # Out-of-tree access attempt — fall back to the
                # weights root itself, which will produce a 404 from
                # the default handler if it's a directory listing
                # context, or be rejected by `do_GET`.
                return str(WEIGHTS_DIR)
            return str(target)
        # Default: serve relative to the demo directory.
        return super().translate_path(path)

    def end_headers(self) -> None:
        # Modern browsers require `application/wasm` for streaming
        # compile; SimpleHTTPServer guesses it correctly on newer
        # Pythons but we set it explicitly to be safe.
        if self.path.endswith(".wasm"):
            self.send_header("Content-Type", "application/wasm")
        # Permissive CORS for local dev.
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()


def main() -> None:
    if not WEIGHTS_DIR.exists():
        print(
            f"!! Weights directory not found at {WEIGHTS_DIR}.",
            file=sys.stderr,
        )
        print(
            "   Run `python3 weight_converter.py --huggingface hexgrad/Kokoro-82M "
            "--output ferrocarril_weights` from the repo root first.",
            file=sys.stderr,
        )
    else:
        print(f">> Serving weights from {WEIGHTS_DIR}")

    pkg_dir = DEMO_DIR / "pkg"
    if not pkg_dir.exists():
        print(
            f"!! Wasm package directory not found at {pkg_dir}.",
            file=sys.stderr,
        )
        print(
            "   Run ./build.sh from inside this directory first.",
            file=sys.stderr,
        )

    os.chdir(DEMO_DIR)
    print(f">> Serving demo root  from {DEMO_DIR}")
    print(f">> Listening on http://{HOST}:{PORT}")
    print(">> (Ctrl-C to stop)")
    server = HTTPServer((HOST, PORT), DemoHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n>> stopping")
        server.server_close()


if __name__ == "__main__":
    main()