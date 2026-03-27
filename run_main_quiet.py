#!/usr/bin/env python3
import os
import runpy
import sys


def _is_true(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    use_gpu = _is_true(os.getenv("USE_GPU", "0"))
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ.setdefault("NVIDIA_VISIBLE_DEVICES", "void")
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.getenv("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("ABSL_LOG_LEVEL", os.getenv("ABSL_LOG_LEVEL", "3"))
        os.environ.setdefault("GLOG_minloglevel", os.getenv("GLOG_minloglevel", "3"))
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "0")

    target = os.path.join(os.path.dirname(__file__), "HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py")
    sys.argv = [target, *sys.argv[1:]]
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
