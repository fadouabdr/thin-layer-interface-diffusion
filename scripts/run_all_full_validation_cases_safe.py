
import subprocess
import sys

from src.validation_config import THICKNESSES


def main():
    for h in THICKNESSES:
        print("=" * 70)
        print(f"Running full validation case h={h}")
        cmd = [sys.executable, "scripts/run_full_validation_case.py", str(h)]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
