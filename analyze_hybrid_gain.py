#!/usr/bin/env python3
"""Rank hybrid-vs-LDPC gains from a summary.csv file.

Usage:
  python analyze_hybrid_gain.py results/.../summary.csv
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path


def load_rows(path: Path):
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python analyze_hybrid_gain.py <summary.csv>")
        return 2
    path = Path(sys.argv[1])
    rows = load_rows(path)
    by_snr = defaultdict(dict)
    for row in rows:
        by_snr[float(row["snr_db"])][row["decoder"]] = row

    print(f"Summary: {path}")
    print()
    best = []
    for snr in sorted(by_snr):
        decs = by_snr[snr]
        for dec_name, row in sorted(decs.items()):
            if not dec_name.startswith("hybbgr"):
                continue
            it = dec_name.replace("hybbgr", "")
            base = decs.get(f"ldpc{it}")
            if base is None:
                continue
            fer_h = float(row["fer"])
            fer_b = float(base["fer"])
            rel = (fer_b - fer_h) / fer_b if fer_b > 0 else 0.0
            abs_gain = fer_b - fer_h
            best.append((rel, abs_gain, snr, it, fer_b, fer_h))
            print(
                f"snr={snr:4.1f}  it={it:>3s}  ldpc={fer_b:.6f}  hybbgr={fer_h:.6f}  "
                f"abs_gain={abs_gain:.6f}  rel_gain={100.0*rel:.2f}%"
            )

    print()
    best.sort(reverse=True)
    print("Top FER gains:")
    for rel, abs_gain, snr, it, fer_b, fer_h in best[:10]:
        print(
            f"snr={snr:4.1f}  it={it:>3s}  ldpc={fer_b:.6f}  hybbgr={fer_h:.6f}  "
            f"abs_gain={abs_gain:.6f}  rel_gain={100.0*rel:.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
