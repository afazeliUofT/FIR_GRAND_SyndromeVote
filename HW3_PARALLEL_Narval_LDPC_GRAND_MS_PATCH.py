"""
Receiver-4R launcher drop-in for FIR_GRAND_SyndromeVote.

This file replaces HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py.
It bootstraps the historical Receiver-4R wrapper from the local git history,
pins that wrapper to a historical non-wrapper base revision of the same file,
and hot-patches the wrapper's dynamic module loading so it works on Python 3.13.

Why this exists:
- The historical Receiver-4R wrapper expects a non-wrapper base revision of the
  same file to be reachable via git refs.
- After wrapper revisions were committed on top, HEAD/origin/main can all point
  to wrapper/launcher revisions instead of the original base.
- The previous launcher fixed that lookup problem, but the wrapper still creates
  a synthetic module for the base script without registering it in sys.modules.
  On Python 3.13 this breaks dataclass processing while importing the base.

This launcher therefore does three things:
1) Finds a wrapper revision from local git history.
2) Finds a non-wrapper base revision from local git history.
3) Patches the wrapper source in-memory to register the synthetic base module
   in sys.modules before executing it.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

LAUNCHER_SENTINEL = "Receiver-4R launcher drop-in for FIR_GRAND_SyndromeVote"
WRAPPER_SENTINEL = "Receiver-4R drop-in for FIR_GRAND_SyndromeVote"
REL_PATH = Path(__file__).name
REPO_ROOT = Path(__file__).resolve().parent


def _git(args: List[str]) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(REPO_ROOT),
        stderr=subprocess.DEVNULL,
        text=True,
    )


def _git_try_show(refspec: str) -> Optional[str]:
    try:
        return _git(["show", refspec])
    except Exception:
        return None


def _list_file_history() -> List[str]:
    try:
        out = _git(["log", "--format=%H", "--follow", "--", REL_PATH])
        shas = [x.strip() for x in out.splitlines() if x.strip()]
        if shas:
            return shas
    except Exception:
        pass
    return []


def _looks_like_base_source(src: Optional[str]) -> bool:
    if not src:
        return False
    if LAUNCHER_SENTINEL in src:
        return False
    if WRAPPER_SENTINEL in src:
        return False
    return ("def run_awgn_sweep_for_code" in src) and ("ClusterGrandConfig" in src)


def _looks_like_wrapper_source(src: Optional[str]) -> bool:
    if not src:
        return False
    if LAUNCHER_SENTINEL in src:
        return False
    return WRAPPER_SENTINEL in src


def _pick_wrapper_and_base() -> Tuple[str, str, str]:
    commits = _list_file_history()
    candidate_refs = [f"{sha}:{REL_PATH}" for sha in commits]

    # Known repository revisions observed in the public history.
    for sha in [
        "4e59ddb",  # historical Receiver-4R wrapper
        "37aad60",  # current launcher commit
        "98f6431",  # updated sbatch commit
        "37bf246",  # strong non-wrapper receiver2/3 base
        "6ae98a3",
        "a20d377",
        "74080c3",
    ]:
        refspec = f"{sha}:{REL_PATH}"
        if refspec not in candidate_refs:
            candidate_refs.append(refspec)

    wrapper_ref: Optional[str] = None
    wrapper_src: Optional[str] = None
    base_ref: Optional[str] = None

    for refspec in candidate_refs:
        src = _git_try_show(refspec)
        if wrapper_ref is None and _looks_like_wrapper_source(src):
            wrapper_ref = refspec
            wrapper_src = src
        if base_ref is None and _looks_like_base_source(src):
            base_ref = refspec
        if wrapper_ref is not None and wrapper_src is not None and base_ref is not None:
            break

    if wrapper_ref is None or wrapper_src is None or base_ref is None:
        checked = ", ".join(commits[:8]) if commits else "<no history>"
        raise RuntimeError(
            "Receiver-4R launcher could not resolve both a historical wrapper "
            "revision and a historical non-wrapper base revision of "
            f"{REL_PATH}. First few history SHAs seen: {checked}"
        )

    return wrapper_ref, wrapper_src, base_ref


def _resolve_base_git_ref(auto_base_ref: str) -> str:
    env_ref = (os.environ.get("BASE_GIT_REF", "") or "").strip()
    if env_ref and _looks_like_base_source(_git_try_show(env_ref)):
        return env_ref
    os.environ["BASE_GIT_REF"] = auto_base_ref
    return auto_base_ref


def _patch_wrapper_source(wrapper_src: str) -> Tuple[str, bool]:
    """
    Patch the historical wrapper so its dynamic base-module loader registers
    the synthetic module in sys.modules before exec(). This is required for
    Python 3.13 dataclass processing.
    """
    if 'sys.modules[mod.__name__] = mod' in wrapper_src:
        return wrapper_src, False

    old = '''def _load_base_module() -> Tuple[types.ModuleType, str]:
    src, refspec = _load_base_source()
    mod = types.ModuleType("fir_grand_base")
    mod.__file__ = str(_repo_root() / _this_relpath())
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod, refspec
'''

    new = '''def _load_base_module() -> Tuple[types.ModuleType, str]:
    src, refspec = _load_base_source()
    mod = types.ModuleType("fir_grand_base")
    mod.__file__ = str(_repo_root() / _this_relpath())
    sys.modules[mod.__name__] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod, refspec
'''

    if old in wrapper_src:
        return wrapper_src.replace(old, new, 1), True

    raise RuntimeError(
        "Could not patch the historical Receiver-4R wrapper in-memory. "
        "Its _load_base_module() body no longer matches the expected pattern."
    )


def _main() -> None:
    wrapper_ref, wrapper_src, auto_base_ref = _pick_wrapper_and_base()
    base_ref = _resolve_base_git_ref(auto_base_ref)
    wrapper_src, patched = _patch_wrapper_source(wrapper_src)

    print(f"[Receiver-4R launcher] repo_root={REPO_ROOT}")
    print(f"[Receiver-4R launcher] wrapper_ref={wrapper_ref}")
    print(f"[Receiver-4R launcher] BASE_GIT_REF={base_ref}")
    print(f"[Receiver-4R launcher] wrapper_patched_sys_modules={int(patched)}")

    glb = {
        "__file__": str(REPO_ROOT / REL_PATH),
        "__name__": "__main__",
        "__package__": None,
        "__cached__": None,
    }
    exec(compile(wrapper_src, str(REPO_ROOT / REL_PATH), "exec"), glb, glb)


if __name__ == "__main__":
    _main()
