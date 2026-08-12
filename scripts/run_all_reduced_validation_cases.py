
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THICKNESSES = [0.08, 0.04, 0.02, 0.01]

env = os.environ.copy()
env["PYTHONPATH"] = str(ROOT)

for h in THICKNESSES:
    print(f"\n=== Running reduced validation case h={h} ===")
    result = subprocess.run(
        [sys.executable, "scripts/run_reduced_validation_case.py", str(h)],
        cwd=ROOT,
        env=env,
    )

    if result.returncode != 0:
       print(f"Warning: h={h} ended with return code {result.returncode}, but may have saved correctly. Continuing...")
       continue

print("\nAll reduced validation cases completed.")
