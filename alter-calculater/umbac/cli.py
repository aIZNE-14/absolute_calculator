from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from typing import Any

from umbac import CalcConfig, UniversalAdvancedCalculator


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UMBAC: Universal Multi-Base Advanced Calculator")
    parser.add_argument("--op", default="base", help="Operation id; use --list-ops to show all")
    parser.add_argument("--expr", default="", help="Expression or number to calculate")
    parser.add_argument("--base-in", type=int, default=10)
    parser.add_argument("--base-out", type=int, default=10)
    parser.add_argument("--params", default="{}", help="Extra operation parameters as JSON object")
    parser.add_argument("--dps", type=int, default=50)
    parser.add_argument("--frac-digits", type=int, default=40)
    parser.add_argument("--angle-unit", choices=("rad", "deg"), default="rad")
    parser.add_argument("--units", choices=("atomic", "SI"), default="atomic")
    parser.add_argument("--list-ops", action="store_true")
    parser.add_argument("--repl", action="store_true", help="Start interactive REPL")
    return parser


def _params(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"--params must be valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("--params must decode to a JSON object.")
    return value


def _print_result(calculator: UniversalAdvancedCalculator, operation: str, expression: str,
                  base_in: int, base_out: int, params: dict[str, Any]) -> int:
    result = calculator.compute(expression, operation, base_in, base_out, **params)
    print(result.to_json())
    return 0 if result.ok else 1


def _repl(calculator: UniversalAdvancedCalculator, args: argparse.Namespace) -> int:
    print("UMBAC REPL. Команды: :base N, :dps N, :help, :quit")
    base_in, base_out = args.base_in, args.base_out
    while True:
        try:
            line = input("umbac> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in {":quit", ":q", ":exit"}:
            return 0
        if line == ":help":
            print("Команды: :base N задаёт оба основания; :dps N меняет точность; :quit завершает.")
            print("Операции: " + ", ".join(calculator.operations))
            continue
        if line.startswith(":base "):
            try:
                base_in = base_out = int(line.split(maxsplit=1)[1])
                calculator.base_converter.validate_base(base_in)
                print(f"Основание установлено: {base_in}")
            except (ValueError, TypeError) as error:
                print(f"Ошибка: {error}")
            continue
        if line.startswith(":dps "):
            try:
                calculator.config = replace(calculator.config, dps=int(line.split(maxsplit=1)[1]))
                calculator.base_converter.config = calculator.config
                calculator.analysis.config = calculator.config
                print(f"Точность установлена: {calculator.config.dps}")
            except (ValueError, TypeError) as error:
                print(f"Ошибка: {error}")
            continue
        print(calculator.compute(line, "analysis.evaluate", base_in, base_out).to_json())


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = CalcConfig(dps=args.dps, frac_digits_out=args.frac_digits, angle_unit=args.angle_unit)
    calculator = UniversalAdvancedCalculator(config, units=args.units)
    if args.list_ops:
        print("\n".join(calculator.operations))
        return 0
    if args.repl:
        return _repl(calculator, args)
    try:
        parameters = _params(args.params)
    except ValueError as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    return _print_result(calculator, args.op, args.expr, args.base_in, args.base_out, parameters)


if __name__ == "__main__":
    raise SystemExit(main())
