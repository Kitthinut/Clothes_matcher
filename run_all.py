"""One-command pipeline (run_all.py).

split -> train -> (open closet search GUI).

The closet (closet/closet.json) is built separately from photos of your own
clothes with closet/add_to_closet.py — see README step 5.

Usage:
  python run_all.py --garment top --backend pytorch
  python run_all.py --garment top --skip-train      # reuse trained model
  python run_all.py --no-gui                        # train only
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run(cmd):
    print("\n==>", " ".join(cmd))
    if subprocess.call(cmd, cwd=ROOT) != 0:
        raise SystemExit(f"Step failed: {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--garment", choices=("top", "bottom"), default="top")
    ap.add_argument("--backend", choices=("pytorch", "tensorflow"), default="pytorch")
    ap.add_argument("--skip-split", action="store_true", help="reuse existing splits")
    ap.add_argument("--skip-train", action="store_true", help="reuse existing model")
    ap.add_argument("--no-gui", action="store_true", help="no search window")
    a = ap.parse_args()

    if not a.skip_split:
        run([PY, str(ROOT / "shared" / "split.py"), "--garment", a.garment])

    if not a.skip_train:
        script = ROOT / "2_train" / ("pytorch" if a.backend == "pytorch" else "tensorflow") / "train.py"
        run([PY, str(script), "--garment", a.garment])

    if not a.no_gui:
        run([PY, str(ROOT / "3_match" / "search_app.py")])


if __name__ == "__main__":
    main()
