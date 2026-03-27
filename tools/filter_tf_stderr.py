#!/usr/bin/env python3
import os
import re
import sys

USE_GPU = str(os.getenv("USE_GPU", "0")).strip().lower() in {"1", "true", "yes", "on"}

# Only suppress known non-fatal CPU-node TensorFlow/Sionna GPU-probe noise.
BENIGN_PATTERNS = [
    re.compile(r"Unable to register cuDNN factory"),
    re.compile(r"Unable to register cuFFT factory"),
    re.compile(r"Unable to register cuBLAS factory"),
    re.compile(r"computation placer already registered"),
    re.compile(r"All log messages before absl::InitializeLog\(\) is called are written to STDERR"),
    re.compile(r"failed call to cuInit"),
    re.compile(r"Could not find cuda drivers"),
    re.compile(r"Could not find TensorRT"),
    re.compile(r"successful NUMA node read from SysFS had negative value"),
]

for line in sys.stdin:
    if not USE_GPU and any(p.search(line) for p in BENIGN_PATTERNS):
        continue
    sys.stderr.write(line)
