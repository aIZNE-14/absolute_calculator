from __future__ import annotations

import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

from calculator import SUPPORTED_BASES, convert_base, evaluate


ROOT = Path(__file__).resolve().parent
_calculator = None
_calculator_lock = Lock()


def get_calculator():
    global _calculator
    if _calculator is None:
        with _calculator_lock:
            if _calculator is None:
                from umbac.facade import UniversalAdvancedCalculator

                _calculator = UniversalAdvancedCalculator()
    return _calculator


class CalculatorHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 100_000:
                raise ValueError("Запрос слишком большой.")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if route == "/api/convert":
                result = convert_base(
                    str(payload.get("value", "")),
                    int(payload.get("source", 10)),
                    int(payload.get("target", 2)),
                )
            elif route == "/api/evaluate":
                result = evaluate(str(payload.get("expression", "")))
            elif route == "/api/compute":
                result = get_calculator().compute(
                    expression=str(payload.get("expression", "")),
                    operation=str(payload.get("operation", "analysis.evaluate")),
                    base_in=int(payload.get("base_in", 10)),
                    base_out=int(payload.get("base_out", 10)),
                    **payload.get("params", {}),
                ).to_dict()
                if not result["ok"]:
                    self._send_json(400, {"error": result["error"], "hint": result["hint"]})
                    return
            else:
                self.send_error(404, "Неизвестный маршрут")
                return
            self._send_json(200, {"result": result})
        except (ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as error:
            self._send_json(400, {"error": str(error)})
        except Exception as error:
            self._send_json(400, {"error": f"Ошибка вычисления: {error}"})

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")


def main() -> None:
    port = int(os.environ.get("CALCULATOR_PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), CalculatorHandler)
    print(f"Калькулятор открыт: http://127.0.0.1:{port}")
    print(f"Доступные основания: {min(SUPPORTED_BASES)}–{max(SUPPORTED_BASES)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nСервер остановлен.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
