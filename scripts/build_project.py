"""
End-to-end pipeline: extract dataset -> preprocess videos -> train model.

Usage:
    python scripts/build_project.py
    python scripts/build_project.py --skip-extract
    python scripts/build_project.py --category Greetings --max-videos-per-class 10
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_step(label: str, cmd: list[str]) -> None:
    print(f"\n>>> {label}")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(f"Step failed: {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SilentTalk from dataset zips")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--category", help="Optional category filter for preprocessing")
    parser.add_argument("--max-videos-per-class", type=int, default=0)
    args = parser.parse_args()

    python = sys.executable

    if not args.skip_extract:
        run_step("List archives", [python, "scripts/extract_dataset.py", "--list"])
        run_step("Extract archives", [python, "scripts/extract_dataset.py"])

    if not args.skip_preprocess:
        cmd = [python, "scripts/preprocess_videos.py"]
        if args.category:
            cmd.extend(["--category", args.category])
        if args.max_videos_per_class:
            cmd.extend(["--max-videos-per-class", str(args.max_videos_per_class)])
        run_step("Preprocess videos", cmd)

    if not args.skip_train:
        run_step(
            "Train model",
            [python, "scripts/train_model.py", "--dataset-csv", "data/landmarks/isl_landmarks.csv"],
        )

    print("\nBuild complete. Run: streamlit run app.py")


if __name__ == "__main__":
    main()
