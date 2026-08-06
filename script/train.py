#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.engine import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one E1-E9 or E8 scaling experiment.")
    parser.add_argument("experiment", help="E1-E9 or E8_p01/E8_p02/E8_p03/E8_p04")
    args = parser.parse_args()
    run_training(args.experiment)


if __name__ == "__main__":
    main()
