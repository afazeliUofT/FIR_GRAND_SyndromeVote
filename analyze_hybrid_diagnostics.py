#!/usr/bin/env python3
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def f(x):
    try:
        return float(x)
    except Exception:
        return math.nan


def main():
    if len(sys.argv) < 3:
        print("usage: analyze_hybrid_diagnostics.py SUMMARY.csv DIAGNOSTICS.csv")
        sys.exit(1)
    summary = load_csv(sys.argv[1])
    diag = load_csv(sys.argv[2])

    by_snr = defaultdict(dict)
    for row in summary:
        by_snr[row["snr_db"]][row["decoder"]] = row
    diag_by = {(r["snr_db"], r["decoder"]): r for r in diag}

    print("=== Relative FER gain vs matching LDPC ===")
    for snr, rows in sorted(by_snr.items(), key=lambda kv: float(kv[0])):
        print(f"\nSNR={snr} dB")
        for dec, row in sorted(rows.items()):
            if dec.startswith("ldpc"):
                continue
            suffix = "".join(ch for ch in dec if ch.isdigit())
            base = rows.get(f"ldpc{suffix}")
            if not base:
                continue
            fer0 = f(base.get("fer"))
            fer1 = f(row.get("fer"))
            if not math.isfinite(fer0) or fer0 <= 0 or not math.isfinite(fer1):
                continue
            rel = 100.0 * (fer0 - fer1) / fer0
            print(f"  {dec:10s} vs ldpc{suffix:<3s}: FER {fer1:.6f} vs {fer0:.6f}  gain={rel:6.2f}%")
            d = diag_by.get((snr, dec))
            if d:
                inv = f(d.get("stage2_invocation_rate"))
                fix = f(d.get("stage2_true_fix_rate_if_invoked"))
                span = f(d.get("avg_stage1_error_span_if_invoked"))
                conc = f(d.get("avg_stage1_block_concentration_if_invoked"))
                print(f"      invoked={inv:.3f}  fix_if_invoked={fix:.3f}  span={span:.1f}  block_conc={conc:.3f}")


if __name__ == "__main__":
    main()
