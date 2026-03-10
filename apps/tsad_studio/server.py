from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from studio_core import get_bootstrap_payload, preview_sample, randomize_config


STATIC_DIR = Path(__file__).resolve().parent / "static"


class StudioRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/bootstrap":
            self._write_json(HTTPStatus.OK, get_bootstrap_payload())
            return
        if parsed.path in {"/", "/index.html"}:
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/randomize":
            self._handle_randomize()
            return
        if parsed.path == "/api/preview":
            self._handle_preview()
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint: {parsed.path}"})

    def _handle_randomize(self) -> None:
        try:
            body = self._read_json_body(optional=True)
            seed = None
            if isinstance(body, dict) and body.get("seed") is not None:
                seed = int(body["seed"])
            self._write_json(HTTPStatus.OK, {"config": randomize_config(seed=seed)})
        except Exception as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _handle_preview(self) -> None:
        try:
            body = self._read_json_body(optional=False)
            if not isinstance(body, dict) or "config" not in body:
                raise ValueError("Expected JSON object with 'config'.")
            preview = preview_sample(body["config"])
            self._write_json(HTTPStatus.OK, {"preview": preview})
        except Exception as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _read_json_body(self, optional: bool) -> Any:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            if optional:
                return {}
            raise ValueError("Missing JSON request body.")
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _write_json(self, status: HTTPStatus, payload: Any) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def make_server(host: str, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), StudioRequestHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description="TSAD Studio interactive UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind")
    parser.add_argument("--open-browser", action="store_true", help="Open the browser after startup")
    args = parser.parse_args()

    server = make_server(args.host, int(args.port))
    url = f"http://{args.host}:{args.port}"
    print(f"TSAD Studio listening on {url}")

    if args.open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
