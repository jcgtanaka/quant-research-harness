"""Read-only JSON server for the dashboard: GET only, bound to 127.0.0.1 only."""

from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from qrh.ui import bench_export, data

LOCAL_HOST = "127.0.0.1"
_TRIAL = re.compile(r"^/api/trials/([^/]+)$")
_WARNINGS = re.compile(r"^/api/trials/([^/]+)/warnings$")


STATIC_DIR = Path(__file__).resolve().parent / "static"
# Exact allow-list: the only files ever served from disk. Anything else is a 404.
_STATIC_FILES = {
    "index.html": "text/html; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "app.css": "text/css; charset=utf-8",
}


class _Handler(BaseHTTPRequestHandler):
    server_version = "qrh-ui"

    def _send_static(self, name: str | None) -> None:
        ctype = _STATIC_FILES.get(name) if name else None
        try:
            raw = (STATIC_DIR / name).read_bytes() if ctype else None
        except OSError:
            raw = None
        if raw is None:
            self._send(404, {"error": f"no such file {name!r}"})
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(raw)

    def _send(self, status: int, body: Any, extra: dict[str, str] | None = None) -> None:
        raw = json.dumps(body, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        home: Path = self.server.home  # type: ignore[attr-defined]
        path = self.path.split("?", 1)[0]
        try:
            if path == "/":
                self._send_static("index.html")
            elif path.startswith("/static/"):
                self._send_static(path[len("/static/"):])
            elif path == "/api/health":
                self._send(200, {"status": "ok", "bound_to": LOCAL_HOST})
            elif path == "/api/trials":
                self._send(200, {"trials": data.list_trials(home)})
            elif path == "/api/parity":
                self._send(200, {"items": data.list_parity(home)})
            elif path == "/api/data-checks":
                self._send(200, {"items": data.list_data_checks(home)})
            elif path == "/api/bench":
                self._send(200, bench_export.list_benches(home))
            elif m := _WARNINGS.match(path):
                self._send(200, data.trial_warnings(home, m.group(1)))
            elif m := _TRIAL.match(path):
                self._send(200, data.trial_detail(home, m.group(1)))
            else:
                self._send(404, {"error": f"no such endpoint {path!r}"})
        except data.DataError as exc:
            self._send(exc.status, {"error": exc.message})
        except Exception as exc:  # never a fabricated value
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def _refuse(self) -> None:
        self._send(405, {"error": f"method {self.command} not allowed; this server is read-only (GET only)"},
                   {"Allow": "GET"})

    do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = _refuse  # noqa: N815

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass


def make_server(home: str | Path, host: str = LOCAL_HOST, port: int = 8765) -> ThreadingHTTPServer:
    """A server on 127.0.0.1 only; any other host is refused."""
    if host != LOCAL_HOST:
        raise ValueError(f"refusing host {host!r}: the dashboard binds to {LOCAL_HOST} only")
    server = ThreadingHTTPServer((LOCAL_HOST, port), _Handler)
    server.daemon_threads = True
    server.home = Path(home)  # type: ignore[attr-defined]
    return server
