import csv
import numpy as np
from pathlib import Path


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_csv(path, header, rows):
    ensure_parent(path)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def save_npz(path, **arrays):
    ensure_parent(path)
    np.savez(path, **arrays)