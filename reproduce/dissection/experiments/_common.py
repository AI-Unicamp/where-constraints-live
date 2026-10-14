from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmarks import Benchmark, fast_benchmarks, full_benchmarks


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--oracle-milp", action="store_true", help="reserved for external solver validation")
    p.add_argument("--timeout-ms", type=int, default=None)
    return p


def selected_benchmarks(args) -> list[Benchmark]:
    return full_benchmarks() if args.full else fast_benchmarks()


def solution_from_result(result):
    if result is None:
        return None
    if hasattr(result, "solution"):
        return result.solution
    return result
