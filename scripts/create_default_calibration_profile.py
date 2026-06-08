"""Write the default ICC pitch calibration template used by the manual workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.pitch_calibration import default_icc_profile
from utils.helpers import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the default ICC pitch calibration profile template.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/calibration/default_icc_profile.json"),
        help="Output path for the default profile JSON",
    )
    args = parser.parse_args()
    path = save_json(default_icc_profile(), args.out)
    print(f"Wrote default ICC calibration profile to {path}")


if __name__ == "__main__":
    main()
