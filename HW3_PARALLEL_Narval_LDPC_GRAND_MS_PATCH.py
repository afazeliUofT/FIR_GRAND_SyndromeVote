### CELL number 1 ###
import os

try:
    from numba import set_num_threads, get_num_threads
    _NUMBA_THREADING_AVAILABLE = True
except ImportError:
    set_num_threads = None
    get_num_threads = None
    _NUMBA_THREADING_AVAILABLE = False


def _detect_num_threads():
    """
    Detect a reasonable number of CPU threads to use on this node.

    Priority (from most to least authoritative):

      0) LDPC_GRAND_NUM_THREADS  (explicit override for this script)
      1) SLURM_CPUS_PER_TASK     (set by Slurm when you use --cpus-per-task)
      2) NUMBA_NUM_THREADS       (Numba's own override if you set it)
      3) multiprocessing.cpu_count()  (what the OS reports as available)
      4) OMP_NUM_THREADS         (used only as a last resort)

    Rationale:
      - On Narval your job stats show ~11× wall‑clock CPU usage over 3.25 h
        (≈17% of 64 cores), which strongly suggests that OMP_NUM_THREADS is
        limiting Numba to ~11 threads even though you requested 64 cores.
      - By making OMP_NUM_THREADS the *last* fallback, and preferring Slurm /
        Numba / cpu_count, we let the job actually use the full core allocation.
    """

    # 0) Explicit override for this script
    env_val = os.environ.get("LDPC_GRAND_NUM_THREADS")
    if env_val:
        try:
            n = int(env_val)
            if n > 0:
                return n
        except ValueError:
            pass  # fall through to the other heuristics

    # 1) Slurm hint: CPUs per task
    env_val = os.environ.get("SLURM_CPUS_PER_TASK")
    if env_val:
        try:
            n = int(env_val)
            if n > 0:
                return n
        except ValueError:
            pass

    # 2) Numba-specific override
    env_val = os.environ.get("NUMBA_NUM_THREADS")
    if env_val:
        try:
            n = int(env_val)
            if n > 0:
                return n
        except ValueError:
            pass

    # 3) What the OS / cgroup says is available to this process
    try:
        import multiprocessing
        n = multiprocessing.cpu_count()
        if n > 0:
            return n
    except Exception:
        pass

    # 4) As a *last resort*, honour OMP_NUM_THREADS
    env_val = os.environ.get("OMP_NUM_THREADS")
    if env_val:
        try:
            n = int(env_val)
            if n > 0:
                return n
        except ValueError:
            pass

    # Fallback if everything else fails
    return 1


# Global thread budget seen by the rest of the script
NUMBA_THREADS = _detect_num_threads()

if _NUMBA_THREADING_AVAILABLE:
    try:
        set_num_threads(NUMBA_THREADS)
    except Exception:
        pass

    try:
        current = get_num_threads()
    except Exception:
        current = NUMBA_THREADS

    print(f"Numba threads: {current}")
else:
    print("Numba not available; using default threading.")



# %%
### CELL number 2 ###
import sys
import platform
import datetime
import os
import time  # NEW: for latency measurements

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from functools import partial

import numpy as np

# Numba for JIT acceleration
try:
    import numba as nb
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    nb = None
    njit = None
    prange = range
    NUMBA_AVAILABLE = False

# Joblib for potential multiprocessing (kept for compatibility)
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    Parallel = None
    delayed = None
    JOBLIB_AVAILABLE = False

# TensorFlow / Sionna (optional; required for 5G NR LDPC + 3GPP channels)
# TensorFlow / Sionna (optional; required for 5G NR LDPC + 3GPP channels)
try:
    # Force CPU-only TF unless explicitly enabled (pre-import; avoids CUDA init noise on CPU nodes)
    if os.getenv("USE_GPU", "0").lower() not in ("1", "true", "yes"):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import tensorflow as tf
    # Sionna v1.x
    try:
        from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
    except Exception:
        LDPC5GEncoder = None
    try:
        from sionna.phy.channel.tr38901.tdl import TDL
    except Exception:
        TDL = None

    # Fallback for older Sionna (<1.0) where module paths differ
    if LDPC5GEncoder is None:
        try:
            from sionna.fec.ldpc.encoding import LDPC5GEncoder  # type: ignore
        except Exception:
            LDPC5GEncoder = None
    if TDL is None:
        try:
            from sionna.channel.tr38901.tdl import TDL  # type: ignore
        except Exception:
            TDL = None

    SIONNA_AVAILABLE = (LDPC5GEncoder is not None) and (TDL is not None)
    _SIONNA_IMPORT_ERROR = None

    # Avoid oversubscribing threads when joblib/numba is used for parallel sweeps
    try:
        tf.config.threading.set_intra_op_parallelism_threads(int(os.getenv("TF_INTRA_OP", "1")))
        tf.config.threading.set_inter_op_parallelism_threads(int(os.getenv("TF_INTER_OP", "1")))
    except Exception:
        pass

    # Default: CPU-only unless explicitly enabled
    if os.getenv("USE_GPU", "0").lower() not in ("1", "true", "yes"):
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

except Exception as _e:
    tf = None  # type: ignore
    LDPC5GEncoder = None  # type: ignore
    TDL = None  # type: ignore
    SIONNA_AVAILABLE = False
    _SIONNA_IMPORT_ERROR = _e


# %%
### CELL number 3 ###
@dataclass
class CodeConfig:
    """Static information about the code itself."""
    code_name: str
    N: int          # codeword length
    K: int          # information length
    rate: float
    H_path: Optional[str] = None  # where parity-check matrix will live

    # Tanner-graph neighbourhoods for checks and variables
    checks_to_vars: Optional[List[np.ndarray]] = field(default=None, repr=False)
    vars_to_checks: Optional[List[np.ndarray]] = field(default=None, repr=False)

    # NEW: for each variable, position of that variable inside each neighbouring check
    # var_to_checks_edge_pos[v][k] = local edge index e such that
    #   checks_to_vars[ vars_to_checks[v][k] ][ e ] == v
    var_to_checks_edge_pos: Optional[List[np.ndarray]] = field(default=None, repr=False)


@dataclass
class InterleaverConfig:
    """Bit-level interleaver/de-interleaver description."""
    name: str                   # e.g. "identity", "random"
    pattern: np.ndarray         # permutation of [0..N-1]
    inverse_pattern: np.ndarray # inverse permutation


@dataclass
class ChannelConfig:
    """Channel model configuration."""
    name: str     # "SIONNA_TDL"
    snr_db: float




# %%
### CELL number 4 ###
@dataclass
class SimulationConfig:
    """Global config for a simulation run."""
    code: CodeConfig
    channel: ChannelConfig
    interleaver: InterleaverConfig
    rng_seed_global: int
    # Minor suggestion applied: snapshot iterations prepared from day one
    snapshot_iters: List[int] = field(default_factory=lambda: [4, 8, 12, 40])


@dataclass
class FrameLog:
    """
    Per-frame log. 
    We define all fields we know we’ll need later so we don't have to change this structure.
    """
    frame_id: int
    rng_seed_frame: int

    # --- Encoder / input ---
    u_bits: np.ndarray                  # length K
    c_bits: np.ndarray                  # length N (pre-interleaver)
    interleaver_pattern: np.ndarray     # length N
    deinterleaver_pattern: np.ndarray   # length N

    # --- Channel ---
    s_symbols: np.ndarray               # BPSK symbols actually sent (interleaver order)
    y_channel: np.ndarray               # received samples (interleaver order)
    y_received: np.ndarray              # deinterleaved samples (decoder input)
    channel_realization: Dict[str, np.ndarray]

    # Precomputed channel LLRs aligned to the decoding graph (mother code length)
    llr_channel: Optional[np.ndarray] = None

    # --- Decoder outputs (to be filled later) ---

    # --- Decoder outputs (to be filled later) ---
    dec_success: Optional[bool] = None
    iter_used: Optional[int] = None
    hard_bits_final: Optional[np.ndarray] = None
    llr_final: Optional[np.ndarray] = None
    syndrome_final: Optional[np.ndarray] = None

    # Explicit error positions for analysis
    error_positions_final: Optional[np.ndarray] = None

    # NEW: per-iteration snapshots for GRAND experiments
    # snapshots["llr"][it]       -> LLRs at iteration it
    # snapshots["hard_bits"][it] -> hard bits at iteration it
    # snapshots["syndrome"][it]  -> syndrome at iteration it
    snapshots: Dict[str, Dict[int, np.ndarray]] = field(default_factory=dict)




# %%
### CELL number 5 ###
def create_identity_interleaver(N: int) -> InterleaverConfig:
    """Identity interleaver: useful as a default, and keeps the API general."""
    pattern = np.arange(N, dtype=np.int64)
    inverse_pattern = np.argsort(pattern)  # for identity this is the same array
    return InterleaverConfig(name="identity", pattern=pattern, inverse_pattern=inverse_pattern)


def create_interleaver_from_pattern(pattern: np.ndarray,
                                    name: str = "custom") -> InterleaverConfig:
    """
    Create a general interleaver from an arbitrary permutation of [0..N-1].

    pattern: 1D array of length N containing a permutation of 0..N-1.
    """
    pattern = np.asarray(pattern, dtype=np.int64)
    if pattern.ndim != 1:
        raise ValueError("Interleaver pattern must be a 1D array")

    N = pattern.size
    # Basic sanity: pattern must be a permutation of 0..N-1
    if np.unique(pattern).size != N or pattern.min() != 0 or pattern.max() != N - 1:
        raise ValueError("Interleaver pattern must be a permutation of 0..N-1")

    # Build inverse permutation explicitly
    inverse_pattern = np.empty_like(pattern)
    inverse_pattern[pattern] = np.arange(N, dtype=np.int64)

    return InterleaverConfig(name=name, pattern=pattern, inverse_pattern=inverse_pattern)


def interleave(bits: np.ndarray, ilv: InterleaverConfig) -> np.ndarray:
    return bits[ilv.pattern]


def deinterleave(bits: np.ndarray, ilv: InterleaverConfig) -> np.ndarray:
    return bits[ilv.inverse_pattern]










# -------------------- Sionna 5G NR helpers (rate-matching + 3GPP channels) --------------------
_TDL_CACHE: Dict[Tuple[Any, ...], Any] = {}


def _sionna5g_internal_tx_positions(code_cfg: CodeConfig) -> np.ndarray:
    """Return internal VN indices (mother graph) corresponding to transmitted bits (RV=0-style).

    This matches the simplified Sionna encoder behavior:
      - remove filler bits (shortening)
      - puncture first 2*Z bits
      - take next n_tx bits
      - apply output interleaver (if qm>1)

    We return indices in the *internal/mother* VN ordering (after re-inserting filler bits),
    aligned with the LLR vector fed to the LDPC decoder.
    """
    if not hasattr(code_cfg, "sionna"):
        raise ValueError("code_cfg has no .sionna metadata")
    s = code_cfg.sionna
    z = int(s.get("z", 0))
    n_tx = int(s.get("n_tx"))
    k_filler = int(s.get("k_filler", 0))
    k_info = int(code_cfg.K)

    N_int = int(code_cfg.N)
    L_pre = N_int - k_filler  # length before re-inserting filler bits

    start = 2 * z
    stop = start + n_tx
    if stop > L_pre:
        raise ValueError(
            f"Invalid 5G mapping: need stop={stop} <= L_pre={L_pre}. "
            f"(N_int={N_int}, k_filler={k_filler}, z={z}, n_tx={n_tx})"
        )

    pos = np.arange(start, stop, dtype=np.int32)
    if k_filler > 0:
        pos = pos + (pos >= k_info).astype(np.int32) * k_filler
    return pos


def _sionna5g_tx_llr_to_internal_llr(
    llr_tx: np.ndarray,
    code_cfg: CodeConfig,
    llr_max: float = 50.0,
) -> np.ndarray:
    """Rate-recover transmitted-bit LLRs into the mother LDPC graph LLR vector.

    llr_tx must be LLRs in the convention log p(x=0|y)/p(x=1|y).
    """
    if not hasattr(code_cfg, "sionna"):
        raise ValueError("code_cfg has no .sionna metadata")
    s = code_cfg.sionna
    z = int(s.get("z", 0))
    n_tx = int(s.get("n_tx"))
    k_filler = int(s.get("k_filler", 0))
    out_int_inv = s.get("out_int_inv", None)
    k_info = int(code_cfg.K)
    N_int = int(code_cfg.N)

    llr_tx = np.asarray(llr_tx, dtype=np.float32).reshape(-1)
    if llr_tx.size != n_tx:
        raise ValueError(f"llr_tx length mismatch: got {llr_tx.size}, expected {n_tx}")

    # Undo output interleaver (if any)
    if out_int_inv is not None:
        llr_tx = llr_tx[np.asarray(out_int_inv, dtype=np.int32)]

    # Build the pre-filler vector: [0...(2Z) | llr_tx | 0...(tail)]
    L_pre = N_int - k_filler
    tail_len = L_pre - (2 * z + n_tx)
    if tail_len < 0:
        raise ValueError(
            f"Invalid tail_len={tail_len}. (N_int={N_int}, k_filler={k_filler}, z={z}, n_tx={n_tx})"
        )
    llr_pre = np.concatenate(
        [
            np.zeros(2 * z, dtype=np.float32),
            llr_tx,
            np.zeros(tail_len, dtype=np.float32),
        ],
        axis=0,
    )

    # Re-insert filler bits with strong LLR towards 0
    if k_filler > 0:
        filler = np.full(k_filler, float(llr_max), dtype=np.float32)
        llr_int = np.concatenate([llr_pre[:k_info], filler, llr_pre[k_info:]], axis=0)
    else:
        llr_int = llr_pre

    if llr_int.size != N_int:
        raise RuntimeError(f"Internal LLR length mismatch: got {llr_int.size}, expected {N_int}")
    return llr_int


def _get_cached_tdl_model() -> Any:
    """Create/cache a Sionna TDL channel object (SISO) based on env vars."""
    if not SIONNA_AVAILABLE:
        raise RuntimeError(
            "Sionna/TensorFlow not available (needed for SIONNA_TDL channel). "
            f"Import error was: {_SIONNA_IMPORT_ERROR}"
        )

    model = os.getenv("SIONNA_TDL_MODEL", "A")
    delay_spread_s = float(os.getenv("SIONNA_TDL_DELAY_SPREAD_S", "3e-7"))  # 300 ns
    carrier_frequency_hz = float(os.getenv("SIONNA_TDL_CARRIER_FREQUENCY_HZ", "3.5e9"))
    min_speed = float(os.getenv("SIONNA_TDL_MIN_SPEED", "0.0"))
    max_speed = float(os.getenv("SIONNA_TDL_MAX_SPEED", "0.0"))

    key = ("TDL", model, delay_spread_s, carrier_frequency_hz, min_speed, max_speed)
    if key in _TDL_CACHE:
        return _TDL_CACHE[key]

    tdl = TDL(
        model=model,
        delay_spread=delay_spread_s,
        carrier_frequency=carrier_frequency_hz,
        min_speed=min_speed,
        max_speed=max_speed,
    )
    _TDL_CACHE[key] = tdl
    return tdl


def sionna_tdl_ofdm_siso_bpsk(
    n_bits: int,
    snr_db: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Generate SISO BPSK over a 3GPP TDL channel (frequency-domain single-tap per RE).

    Returns:
      y_vec: complex received samples for the transmitted REs (length n_bits)
      h_vec: complex channel frequency response on those REs (length n_bits)
      no: noise variance per complex sample (E|w|^2)
    """
    tdl = _get_cached_tdl_model()

    fft_size = int(os.getenv("SIONNA_OFDM_FFT_SIZE", "256"))
    scs_hz = float(os.getenv("SIONNA_OFDM_SUBCARRIER_SPACING_HZ", "15000"))
    if fft_size <= 0:
        raise ValueError("SIONNA_OFDM_FFT_SIZE must be > 0")
    n_ofdm = int(math.ceil(n_bits / fft_size))
    pad = n_ofdm * fft_size - n_bits

    # all-zero CW -> all +1 symbols
    x = np.ones(n_bits + pad, dtype=np.complex64).reshape(n_ofdm, fft_size)

    sampling_frequency = float(fft_size) * scs_hz

    # Seed Sionna RNG from numpy rng for deterministic per-frame channel draws
    try:
        from sionna.phy import config as sionna_config  # Sionna v1.x
        sionna_config.seed = int(rng.integers(0, 2**31 - 1))
    except Exception:
        pass

    a, tau = tdl(batch_size=1, num_time_steps=n_ofdm, sampling_frequency=sampling_frequency)
    a = a.numpy()
    tau = tau.numpy()

    # SISO extraction (per Sionna docs):
    # a: [batch, rx=1, rx_ant=1, tx=1, tx_ant=1, num_paths, num_time_steps]
    a_siso = a[0, 0, 0, 0, 0, :, :]          # [n_paths, n_ofdm]
    a_siso = np.transpose(a_siso, (1, 0))    # [n_ofdm, n_paths]
    tau_siso = tau[0, 0, 0, :]               # [n_paths]

    # Frequency response on subcarriers (index 0..fft_size-1)
    f = (np.arange(fft_size, dtype=np.float32) * scs_hz)[None, :]        # [1, fft_size]
    phase = np.exp(-1j * 2.0 * np.pi * tau_siso[:, None] * f)            # [n_paths, fft_size]
    h = (a_siso @ phase).astype(np.complex64)                             # [n_ofdm, fft_size]

    snr_lin = 10.0 ** (snr_db / 10.0)
    no = 1.0 / snr_lin  # noise variance per complex sample
    w = (
        rng.standard_normal((n_ofdm, fft_size)).astype(np.float32)
        + 1j * rng.standard_normal((n_ofdm, fft_size)).astype(np.float32)
    ) * np.sqrt(no / 2.0)
    y = h * x + w

    # Optional CFO knob (simple per-OFDM-symbol phase rotation)
    cfo_hz = float(os.getenv("SIONNA_CFO_HZ", "0.0"))
    if cfo_hz != 0.0:
        t_sym = 1.0 / scs_hz  # rough OFDM symbol duration (no CP)
        rot = np.exp(1j * 2.0 * np.pi * cfo_hz * t_sym * np.arange(n_ofdm, dtype=np.float32))
        y = (rot[:, None] * y).astype(np.complex64)

    y_vec = y.reshape(-1)[:n_bits]
    h_vec = h.reshape(-1)[:n_bits]
    return y_vec, h_vec, float(no)


def _llr_bpsk_known_h(y: np.ndarray, h: np.ndarray, no: float) -> np.ndarray:
    """LLR for BPSK (0->+1,1->-1) over complex channel y=h*x+w, w~CN(0,no)."""
    # LLR = log p(y|x=+1)/p(y|x=-1) = 4*Re(h^* y)/no
    return (4.0 / float(no)) * np.real(np.conj(h) * y).astype(np.float32)


def run_single_frame_sionna5g(sim_cfg: SimulationConfig, frame_id: int, global_rng: np.random.Generator) -> FrameLog:
    """Single-frame runner for Sionna 5G NR LDPC codes.

    We transmit only the rate-matched `n_tx` bits, then rate-recover into the mother graph
    (pcm.shape[1]) before LDPC decoding / GRAND.
    """
    code_cfg = sim_cfg.code
    if not hasattr(code_cfg, "sionna"):
        raise ValueError("run_single_frame_sionna5g called but code_cfg has no .sionna")

    s = code_cfg.sionna
    n_tx = int(s["n_tx"])
    tx_pos_int = np.asarray(s["tx_pos"], dtype=np.int32)

    rng_seed_frame = int(global_rng.integers(0, 2**32 - 1))
    rng = np.random.default_rng(rng_seed_frame)

    # All-zero CW (symmetry)
    u_bits = np.zeros(code_cfg.K, dtype=np.uint8)
    c_bits = np.zeros(code_cfg.N, dtype=np.uint8)

    # Transmitted BPSK symbols (only n_tx positions). For logging, embed into length-N arrays.
    x_tx = np.ones(n_tx, dtype=np.float32)  # all +1

    ch_name = str(sim_cfg.channel.name).upper()
    channel_realization: Dict[str, Any] = {}

    if ch_name in ("SIONNA_TDL", "TDL"):
        y_c, h_c, no = sionna_tdl_ofdm_siso_bpsk(n_tx, sim_cfg.channel.snr_db, rng)
        llr_tx = _llr_bpsk_known_h(y_c, h_c, no)
        y_tx = y_c  # complex
        channel_realization.update({
            "no": np.array([no], dtype=np.float64),
            "tdl_model": os.getenv("SIONNA_TDL_MODEL", "A"),
        })
    else:
        raise ValueError(
            f"Unsupported channel name '{sim_cfg.channel.name}'. "
            "This cleaned version supports only CHANNEL_NAME=SIONNA_TDL."
        )

    # Rate recovery into mother-graph LLRs
    llr_max = float(os.getenv("SIONNA_LLR_MAX", "50.0"))
    llr_int = _sionna5g_tx_llr_to_internal_llr(llr_tx, code_cfg, llr_max=llr_max)

    # Build logging arrays (length = N_internal)
    s_symbols = np.zeros(code_cfg.N, dtype=np.complex64)
    y_received = np.zeros(code_cfg.N, dtype=np.complex64)
    s_symbols[tx_pos_int] = x_tx.astype(np.complex64)
    y_received[tx_pos_int] = y_tx.astype(np.complex64)

    frame_log = FrameLog(
        frame_id=frame_id,
        rng_seed_frame=rng_seed_frame,
        u_bits=u_bits,
        c_bits=c_bits,
        interleaver_pattern=sim_cfg.interleaver.pattern,
        deinterleaver_pattern=sim_cfg.interleaver.inverse_pattern,
        s_symbols=s_symbols,
        y_channel=y_received,
        y_received=y_received,
        channel_realization=channel_realization,
    )

    # Decoder uses mother-graph channel LLRs
    frame_log.llr_channel = llr_int
    return frame_log


def run_single_frame(sim_cfg: SimulationConfig, frame_id: int, global_rng: np.random.Generator) -> FrameLog:
    """Dispatch per-frame simulation (cleaned: only Sionna 5G NR LDPC + SIONNA_TDL)."""
    if not hasattr(sim_cfg.code, "sionna"):
        raise ValueError("Only the Sionna 5G NR LDPC path ('sionna5g') is supported in this cleaned script.")
    return run_single_frame_sionna5g(sim_cfg, frame_id, global_rng)



### CELL number 8 ###
# (Removed) AWGN LLR helper.
# This cleaned script supports only CHANNEL_NAME=SIONNA_TDL and expects
# FrameLog.llr_channel to be set by run_single_frame_sionna5g().

# %%




# %%
### CELL number 9 ###
def prepare_flat_adjacency(code_cfg):
    """Convert checks_to_vars to flat CSR format for Numba."""
    M = code_cfg.M
    checks_to_vars = code_cfg.checks_to_vars

    total_edges = sum(len(cv) for cv in checks_to_vars)
    check_ptrs = np.zeros(M + 1, dtype=np.int64)
    check_indices = np.zeros(total_edges, dtype=np.int64)

    ptr = 0
    for j in range(M):
        check_ptrs[j] = ptr
        cv = checks_to_vars[j]
        check_indices[ptr:ptr + len(cv)] = cv
        ptr += len(cv)
    check_ptrs[M] = ptr

    return check_ptrs, check_indices


if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def _compute_syndrome_numba(bits, check_ptrs, check_indices, M):
        """
        Numba-accelerated syndrome computation, parallel over checks.

        For each check j:
            s[j] = XOR_{v in N(j)} bits[v]
        """
        s = np.zeros(M, dtype=np.uint8)
        for j in prange(M):
            start = check_ptrs[j]
            end = check_ptrs[j + 1]
            parity = 0
            for idx in range(start, end):
                parity ^= bits[check_indices[idx]]
            s[j] = parity
        return s


# %%
### CELL number 10 ###
def compute_syndrome_from_checks(bits: np.ndarray, code_cfg) -> np.ndarray:
    """
    Compute syndrome s = H * bits (mod 2).
    Uses Numba acceleration if available.
    """
    # Use fast path if Numba structures are prepared
    if NUMBA_AVAILABLE and hasattr(code_cfg, '_check_ptrs'):
        return _compute_syndrome_numba(
            bits,
            code_cfg._check_ptrs,
            code_cfg._check_indices,
            code_cfg.M
        )
    
    # Fallback to original
    M = code_cfg.M
    s = np.zeros(M, dtype=np.uint8)
    for j, var_indices in enumerate(code_cfg.checks_to_vars):
        if var_indices.size == 0:
            s[j] = 0
        else:
            s[j] = int(bits[var_indices].sum() % 2)
    return s

# %%
### CELL number 11 ###
import numpy as np
from typing import List

def build_systematic_ldpc_H(code_cfg: CodeConfig,
                             dv_info: int = 3,
                             dv_parity_extra: int = 1,
                             rng_seed: int = 2025) -> None:
    """
    Build a 'less easy' LDPC-style parity-check matrix for the current (N, K):

      - Let M = N - K.
      - H = [A | P], where:
          * A is M x K, sparse with dv_info ones per *information* column.
          * P is M x M, upper-triangular with:
              - 1s on the diagonal (always invertible over GF(2))
              - plus dv_parity_extra extra 1s ABOVE the diagonal in each column
                (so parity bits also have degree > 1).

    Properties:
      * rank(H) = M  => code dimension = K (with very high probability).
      * Systematic encoder: c = [u | p], with P p = A u (mod 2).
      * Variable node degrees:
          - info bits: ≈ dv_info
          - parity bits: ≥ 1 + dv_parity_extra (except maybe very first few cols)
      * Check node degrees: ≈ dv_info + (# of parity bits in each row).

    Also builds Tanner graph adjacency lists:
      - checks_to_vars[j] = array of variable indices connected to check j
      - vars_to_checks[v] = array of check indices connected to variable v
      - var_to_checks_edge_pos[v][k] = local edge index inside checks_to_vars[ vars_to_checks[v][k] ]

    NEW:
      - code_cfg.P_sys: the parity submatrix P
      - code_cfg.P_row_upper_indices: for each row i, the indices of columns k>i
        where P[i, k] = 1, used for fast back-substitution in encoding.
    """
    N, K = code_cfg.N, code_cfg.K
    M = N - K
    rng = np.random.default_rng(rng_seed)

    # ---- 1) Build sparse A for info bits ----
    A = np.zeros((M, K), dtype=np.uint8)
    for col in range(K):
        # dv_info distinct rows per info column
        rows = rng.choice(M, size=dv_info, replace=False)
        A[rows, col] = 1

    # ---- 2) Build upper-triangular P for parity bits ----
    # Start with identity (guarantees invertible, diag=1).
    P = np.eye(M, dtype=np.uint8)

    # Add extra 1s above the diagonal to increase parity column degrees.
    # For column 'col', we may add up to dv_parity_extra ones in rows < col.
    if dv_parity_extra > 0:
        for col in range(1, M):
            # how many extra ones we can place in this column (bounded by 'col')
            n_extra = min(col, dv_parity_extra)
            if n_extra == 0:
                continue
            rows = rng.choice(col, size=n_extra, replace=False)
            P[rows, col] ^= 1  # toggle bits (0->1 or 1->0, though diag never touched)

    # ---- 3) Assemble H = [A | P] ----
    H = np.concatenate([A, P], axis=1)  # shape (M, N)

    # ---- 4) Build Tanner-graph adjacency lists ----
    checks_to_vars: List[np.ndarray] = []
    vars_to_checks_lists: List[list] = [[] for _ in range(N)]
    edge_pos_lists: List[list] = [[] for _ in range(N)]

    for j in range(M):
        cols = np.flatnonzero(H[j])              # variable indices for check j
        cols_int = cols.astype(np.int32)
        checks_to_vars.append(cols_int)

        for local_e, v in enumerate(cols_int):
            vars_to_checks_lists[v].append(j)    # check index
            edge_pos_lists[v].append(local_e)    # position in checks_to_vars[j]

    vars_to_checks = [np.array(lst, dtype=np.int32) for lst in vars_to_checks_lists]
    var_to_checks_edge_pos = [np.array(lst, dtype=np.int32) for lst in edge_pos_lists]

    # ---- 5) Precompute row structure of P for back-substitution ----
    # For each row i, we want the list of columns k > i where P[i, k] = 1.
    P_row_upper_indices: List[np.ndarray] = []
    for i in range(M):
        nz = np.flatnonzero(P[i])
        nz_upper = nz[nz > i]  # strictly above diagonal
        P_row_upper_indices.append(nz_upper.astype(np.int32))

    # ---- 6) Attach to code_cfg ----
    code_cfg.M = M
    code_cfg.H = H
    code_cfg.A_sys = A                  # used by encoder
    code_cfg.P_sys = P                  # new: parity submatrix for encoder
    code_cfg.dv_info = dv_info
    code_cfg.checks_to_vars = checks_to_vars
    code_cfg.vars_to_checks = vars_to_checks
    code_cfg.var_to_checks_edge_pos = var_to_checks_edge_pos
    code_cfg.P_row_upper_indices = P_row_upper_indices

    # ---- 7) Quick stats ----
    var_degrees = np.array([len(v) for v in vars_to_checks])
    check_degrees = np.array([len(c) for c in checks_to_vars])
    parity_degrees = var_degrees[K:]  # last M columns are parity bits

    print(f"Built systematic LDPC-style H = [A | P] for code '{code_cfg.code_name}':")
    print(f" - N = {N}, K = {K}, M = {M}")
    print(f" - Info column degree target dv_info      = {dv_info}")
    print(f" - Variable degrees (all bits): min={var_degrees.min()}, "
          f"max={var_degrees.max()}, mean={var_degrees.mean():.2f}")
    print(f" - Parity column degrees:      min={parity_degrees.min()}, "
          f"max={parity_degrees.max()}, mean={parity_degrees.mean():.2f}")
    print(f" - Check degrees:              min={check_degrees.min()}, "
          f"max={check_degrees.max()}, mean={check_degrees.mean():.2f}")


### CELL number 12 ###
def prepare_code_for_fast_decoding(code_cfg) -> None:
    """
    Prepare flattened adjacency structures for Numba-accelerated decoding.

    MUST be called once after building code_cfg.checks_to_vars / vars_to_checks /
    var_to_checks_edge_pos (for any Tanner graph construction).
    """
    if not NUMBA_AVAILABLE:
        print("Numba not available - skipping fast preparation")
        return
    
    M = code_cfg.M
    N = code_cfg.N
    checks_to_vars = code_cfg.checks_to_vars
    vars_to_checks = code_cfg.vars_to_checks
    var_to_checks_edge_pos = code_cfg.var_to_checks_edge_pos
    
    # Syndrome computation structures
    check_ptrs, check_indices = prepare_flat_adjacency(code_cfg)
    code_cfg._check_ptrs = check_ptrs
    code_cfg._check_indices = check_indices
    
    # Min-sum decoder structures (CSR format)
    total_c2v = sum(len(cv) for cv in checks_to_vars)
    
    c2v_ptrs = np.zeros(M + 1, dtype=np.int32)
    c2v_indices = np.zeros(total_c2v, dtype=np.int32)
    ptr = 0
    for j in range(M):
        c2v_ptrs[j] = ptr
        cv = checks_to_vars[j]
        c2v_indices[ptr:ptr + len(cv)] = cv
        ptr += len(cv)
    c2v_ptrs[M] = ptr
    
    total_v2c = sum(len(vc) for vc in vars_to_checks)
    v2c_ptrs = np.zeros(N + 1, dtype=np.int32)
    v2c_checks = np.zeros(total_v2c, dtype=np.int32)
    v2c_edge_pos = np.zeros(total_v2c, dtype=np.int32)
    ptr = 0
    for v in range(N):
        v2c_ptrs[v] = ptr
        vc = vars_to_checks[v]
        ep = var_to_checks_edge_pos[v]
        v2c_checks[ptr:ptr + len(vc)] = vc
        v2c_edge_pos[ptr:ptr + len(ep)] = ep
        ptr += len(vc)
    v2c_ptrs[N] = ptr
    
    check_degrees = np.array([len(cv) for cv in checks_to_vars], dtype=np.int32)
    
    code_cfg._c2v_ptrs = c2v_ptrs
    code_cfg._c2v_indices = c2v_indices
    code_cfg._v2c_ptrs = v2c_ptrs
    code_cfg._v2c_checks = v2c_checks
    code_cfg._v2c_edge_pos = v2c_edge_pos
    code_cfg._check_degrees = check_degrees
    code_cfg._max_check_degree = int(check_degrees.max()) if M > 0 else 0
    
    print(f"Fast decoding structures prepared: {total_c2v} edges")


# NOTE: Legacy systematic-LDPC import-time build disabled/removed.
# It caused import-time side effects and a NameError after refactors (code_cfg is not module-global anymore).


# %%
### CELL number 13 ###
def encode_bits_ldpc_systematic(u_bits: np.ndarray, code_cfg: CodeConfig) -> np.ndarray:
    """
    Systematic LDPC encoder for H = [A | P]:

      Given info bits u (length K) in positions [0..K-1], produce codeword
          c = [u | p]
      where
          A u + P p = 0 (mod 2)   =>   P p = A u  (mod 2),

      with:
        - A = code_cfg.A_sys, shape (M, K)
        - P = code_cfg.P_sys, shape (M, M), upper triangular with 1s on the diagonal.

    We solve P p = t over GF(2) by back-substitution, where t = A u (mod 2).
    """
    A = code_cfg.A_sys          # shape (M, K)
    P = getattr(code_cfg, "P_sys", None)
    u = u_bits.astype(np.uint8)

    # Right-hand side: t = A * u (mod 2)
    # Use int32 for dot, then reduce mod 2.
    t = (A.dot(u.astype(np.int32)) & 1).astype(np.uint8)

    # If no P_sys is present (backward compatibility), treat P as identity
    if P is None:
        p = t
    else:
        M = P.shape[0]
        p = np.zeros(M, dtype=np.uint8)

        # Precomputed from build_systematic_ldpc_H: list of columns > i where P[i,k] = 1
        upper = code_cfg.P_row_upper_indices

        # Back substitution from bottom row up (since P is upper triangular with diag=1)
        for i in range(M - 1, -1, -1):
            acc = t[i]
            # subtract (XOR) contributions from already-solved p[k], k > i
            for k in upper[i]:
                acc ^= p[k]
            # P[i,i] = 1, so p[i] = acc (mod 2)
            p[i] = acc & 1

    # Assemble full codeword in [info | parity] layout
    c = np.zeros(code_cfg.N, dtype=np.uint8)
    c[:code_cfg.K] = u_bits
    c[code_cfg.K:] = p
    return c


def sanity_check_ldpc_encoder(code_cfg: CodeConfig,
                              num_tests: int = 3,
                              rng_seed: int = 999) -> None:
    """
    Verify that our encoder and H are consistent:
      For random u_bits, H @ c_bits % 2 == 0.
    """
    rng = np.random.default_rng(rng_seed)
    H = code_cfg.H
    M, N = H.shape

    print("=== LDPC encoder vs H consistency check ===")
    for t_idx in range(num_tests):
        u = rng.integers(0, 2, size=code_cfg.K, dtype=np.uint8)
        c = encode_bits_ldpc_systematic(u, code_cfg)

        if c.shape[0] != N:
            print(f"  [Test {t_idx}] ERROR: c_bits length {c.shape[0]} != N={N}")
            continue

        syn = (H.astype(np.int32) @ c.astype(np.int32)) & 1
        syn_weight = int(syn.sum())

        print(f"  Test {t_idx}: syndrome weight = {syn_weight}")
        if syn_weight == 0:
            print("    -> OK (valid codeword)")
        else:
            print("    -> ERROR (H c != 0), something is inconsistent!")

    print("Encoder/H consistency check done.")


### CELL number 14 ###
def encode_bits_simple(u_bits: np.ndarray, code_cfg: CodeConfig) -> np.ndarray:
    """
    Unified encoder used by the simulation pipeline.

    Supported modes:
      - encoder_mode == "all_zero": always return the all-zero codeword (valid for any linear code).
      - If code_cfg.A_sys exists (LDPC-style H = [A | I]), use systematic encoding: c=[u | A u].
      - Otherwise: fall back to the old placeholder encoder that appends zeros as parity bits.

    NOTE:
      For non-systematic codes (e.g., Gallager/5G Tanner graphs), use encoder_mode="all_zero"
      to avoid building a generator matrix.
    """
    enc_mode = str(getattr(code_cfg, "encoder_mode", "systematic")).lower()
    if enc_mode == "all_zero":
        return np.zeros(code_cfg.N, dtype=np.uint8)

    if hasattr(code_cfg, "A_sys"):
        # Use our proper LDPC-style encoder
        return encode_bits_ldpc_systematic(u_bits, code_cfg)

    # Fallback: old behaviour (u | zeros)
    N, K = code_cfg.N, code_cfg.K
    c_bits = np.zeros(N, dtype=np.uint8)
    c_bits[:K] = u_bits
    return c_bits




# %%
### CELL number 15 ###
@dataclass
class DecoderConfig:
    """Config for LDPC min-sum / normalized min-sum decoder."""
    max_iters: int = 40      # maximum number of iterations
    alpha: float = 0.8       # normalization factor (1.0 = pure min-sum)
    early_stop: bool = True  # stop when syndrome is 0




# %%
### CELL number 16 ###
dec_cfg_min_sum_4_early = DecoderConfig(
    max_iters=4,
    alpha=0.8,
    early_stop=True,   # baseline: allow early convergence
)

dec_cfg_min_sum_4_no_early = DecoderConfig(
    max_iters=4,
    alpha=0.8,
    early_stop=True,   # hybrid stage-1: early-stop; GRAND only if LDPC fails
)


# %%
### CELL number 17 ###
if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def _check_node_update_numba(msg_v2c_flat, msg_c2v_flat, c2v_ptrs, M, alpha):
        """
        Numba-accelerated check node update, parallel over checks.

        For each check j, we compute outgoing CN->VN messages on all its edges
        using the normalized min-sum rule with factor 'alpha'.
        """
        for j in prange(M):
            start = c2v_ptrs[j]
            end = c2v_ptrs[j + 1]
            d = end - start

            if d == 0:
                continue

            sign_all = 1.0
            min1 = 1e30
            min2 = 1e30
            idx_min1 = 0

            # Scan incoming messages on this check
            for e in range(d):
                msg = msg_v2c_flat[start + e]

                if msg < 0.0:
                    sign_all *= -1.0
                    abs_val = -msg
                else:
                    # Treat very small positives as zero to match original behaviour
                    abs_val = msg if msg > 0.0 else 0.0

                if abs_val < min1:
                    min2 = min1
                    min1 = abs_val
                    idx_min1 = e
                elif abs_val < min2:
                    min2 = abs_val

            if d == 1:
                # For degree-1 checks, fall back to min1 for all edges (same as original)
                min2 = min1

            # Produce outgoing messages on each edge of this check
            for e in range(d):
                msg = msg_v2c_flat[start + e]
                sign_e = -sign_all if msg < 0.0 else sign_all
                mag_e = min2 if e == idx_min1 else min1
                msg_c2v_flat[start + e] = alpha * sign_e * mag_e

    @njit(parallel=True, cache=True)
    def _variable_node_update_numba(llr_channel,
                                    msg_c2v_flat,
                                    v2c_ptrs,
                                    v2c_checks,
                                    v2c_edge_pos,
                                    c2v_ptrs,
                                    N):
        """
        Numba-accelerated variable node update, parallel over variables.

        For each variable v:
            L_post[v] = L_ch[v] + sum_{j in N(v)} m_{j->v}
        """
        llr_posterior = np.empty_like(llr_channel)

        for v in prange(N):
            start = v2c_ptrs[v]
            end = v2c_ptrs[v + 1]

            total = 0.0
            for idx in range(start, end):
                j = v2c_checks[idx]
                e = v2c_edge_pos[idx]
                total += msg_c2v_flat[c2v_ptrs[j] + e]

            llr_posterior[v] = llr_channel[v] + total

        return llr_posterior

    @njit(parallel=True, cache=True)
    def _vn_to_cn_update_numba(llr_posterior,
                               msg_c2v_flat,
                               msg_v2c_flat,
                               v2c_ptrs,
                               v2c_checks,
                               v2c_edge_pos,
                               c2v_ptrs,
                               N):
        """
        Numba-accelerated VN->CN message update, parallel over variables.

        For each variable v and each neighbouring check j:
            m_{v->j} = L_post[v] - m_{j->v}
        """
        for v in prange(N):
            start = v2c_ptrs[v]
            end = v2c_ptrs[v + 1]
            L_v = llr_posterior[v]

            for idx in range(start, end):
                j = v2c_checks[idx]
                e = v2c_edge_pos[idx]
                base = c2v_ptrs[j] + e
                msg_v2c_flat[base] = L_v - msg_c2v_flat[base]


# %%
### CELL number 18 ###
def ldpc_min_sum_decode(llr_channel: np.ndarray,
                        code_cfg,
                        dec_cfg,
                        snapshot_iters: Optional[List[int]] = None,
                        snapshots: Optional[Dict[str, Dict[int, np.ndarray]]] = None):
    """
    Normalized min-sum LDPC decoding.
    Uses Numba acceleration if available.
    """
    N = code_cfg.N
    M = code_cfg.M
    
    # Use fast path if Numba structures are prepared
    use_fast = NUMBA_AVAILABLE and hasattr(code_cfg, '_c2v_ptrs')
    
    if use_fast:
        return _ldpc_min_sum_decode_fast(llr_channel, code_cfg, dec_cfg,
                                          snapshot_iters, snapshots)
    else:
        return _ldpc_min_sum_decode_original(llr_channel, code_cfg, dec_cfg,
                                              snapshot_iters, snapshots)


def _ldpc_min_sum_decode_fast(llr_channel, code_cfg, dec_cfg,
                               snapshot_iters, snapshots):
    """Numba-accelerated implementation."""
    N = code_cfg.N
    M = code_cfg.M
    
    c2v_ptrs = code_cfg._c2v_ptrs
    c2v_indices = code_cfg._c2v_indices
    v2c_ptrs = code_cfg._v2c_ptrs
    v2c_checks = code_cfg._v2c_checks
    v2c_edge_pos = code_cfg._v2c_edge_pos
    check_ptrs = code_cfg._check_ptrs
    check_indices = code_cfg._check_indices
    
    total_edges = c2v_ptrs[M]
    msg_v2c_flat = np.zeros(total_edges, dtype=np.float64)
    msg_c2v_flat = np.zeros(total_edges, dtype=np.float64)
    
    # Initialize VN->CN messages
    for j in range(M):
        start = c2v_ptrs[j]
        end = c2v_ptrs[j + 1]
        for idx in range(start, end):
            v = c2v_indices[idx]
            msg_v2c_flat[idx] = llr_channel[v]
    
    iter_used = dec_cfg.max_iters
    alpha = dec_cfg.alpha
    snapshot_set = set(snapshot_iters) if snapshot_iters else set()
    
    if snapshots is not None:
        snapshots.setdefault("llr", {})
        snapshots.setdefault("hard_bits", {})
        snapshots.setdefault("syndrome", {})
        snapshots.setdefault("unsat_checks", {})
    
    llr_posterior = llr_channel.copy()
    
    for it in range(1, dec_cfg.max_iters + 1):
        _check_node_update_numba(msg_v2c_flat, msg_c2v_flat, c2v_ptrs, M, alpha)
        
        llr_posterior = _variable_node_update_numba(
            llr_channel, msg_c2v_flat, v2c_ptrs, v2c_checks,
            v2c_edge_pos, c2v_ptrs, N
        )
        
        hard_bits = (llr_posterior < 0.0).astype(np.uint8)
        syndrome = _compute_syndrome_numba(hard_bits, check_ptrs, check_indices, M)
        
        if snapshots is not None and it in snapshot_set:
            snapshots["llr"][it] = llr_posterior.copy()
            snapshots["hard_bits"][it] = hard_bits.copy()
            snapshots["syndrome"][it] = syndrome.copy()
            snapshots["unsat_checks"][it] = np.flatnonzero(syndrome)
        
        if dec_cfg.early_stop and syndrome.sum() == 0:
            iter_used = it
            break
        
        _vn_to_cn_update_numba(
            llr_posterior, msg_c2v_flat, msg_v2c_flat,
            v2c_ptrs, v2c_checks, v2c_edge_pos, c2v_ptrs, N
        )
        iter_used = it
    
    hard_bits = (llr_posterior < 0.0).astype(np.uint8)
    syndrome = _compute_syndrome_numba(hard_bits, check_ptrs, check_indices, M)
    
    return hard_bits, llr_posterior, syndrome, iter_used


def _ldpc_min_sum_decode_original(llr_channel, code_cfg, dec_cfg,
                                   snapshot_iters, snapshots):
    """Original implementation (fallback)."""
    # [Keep your original implementation here as fallback]
    # Copy lines 813-928 from your original CELL 24
    N = code_cfg.N
    M = code_cfg.M
    checks_to_vars = code_cfg.checks_to_vars
    vars_to_checks = code_cfg.vars_to_checks
    var_to_checks_edge_pos = code_cfg.var_to_checks_edge_pos

    msg_v2c = [np.zeros(len(checks_to_vars[j]), dtype=np.float64) for j in range(M)]
    msg_c2v = [np.zeros(len(checks_to_vars[j]), dtype=np.float64) for j in range(M)]

    for j in range(M):
        neigh_vars = checks_to_vars[j]
        if neigh_vars.size == 0:
            continue
        msg_v2c[j][:] = llr_channel[neigh_vars]

    llr_posterior = llr_channel.copy()
    iter_used = dec_cfg.max_iters

    if snapshot_iters is not None:
        snapshot_set = set(snapshot_iters)
    else:
        snapshot_set = set()

    if snapshots is not None:
        snapshots.setdefault("llr", {})
        snapshots.setdefault("hard_bits", {})
        snapshots.setdefault("syndrome", {})
        snapshots.setdefault("unsat_checks", {})

    for it in range(1, dec_cfg.max_iters + 1):
        alpha = dec_cfg.alpha
        for j in range(M):
            msgs = msg_v2c[j]
            d = msgs.size
            if d == 0:
                continue

            signs = np.sign(msgs)
            signs[signs == 0] = 1.0
            abs_vals = np.abs(msgs)

            idx_min1 = int(np.argmin(abs_vals))
            min1 = abs_vals[idx_min1]
            if d > 1:
                tmp = abs_vals.copy()
                tmp[idx_min1] = np.inf
                min2 = float(tmp.min())
            else:
                min2 = min1

            sign_all = float(np.prod(signs))
            out = msg_c2v[j]

            for e in range(d):
                sign_e = sign_all * signs[e]
                mag_e = min2 if e == idx_min1 else min1
                out[e] = alpha * sign_e * mag_e

        llr_posterior = llr_channel.copy()

        for v in range(N):
            checks = vars_to_checks[v]
            if checks.size == 0:
                continue
            edge_pos = var_to_checks_edge_pos[v]
            total = 0.0
            for k, j in enumerate(checks):
                e = int(edge_pos[k])
                total += msg_c2v[j][e]
            llr_posterior[v] += total

        hard_bits = (llr_posterior < 0.0).astype(np.uint8)
        syndrome = compute_syndrome_from_checks(hard_bits, code_cfg)

        if snapshots is not None and it in snapshot_set:
            snapshots["llr"][it] = llr_posterior.copy()
            snapshots["hard_bits"][it] = hard_bits.copy()
            snapshots["syndrome"][it] = syndrome.copy()
            unsat_idx = np.flatnonzero(syndrome)
            snapshots["unsat_checks"][it] = unsat_idx

        if dec_cfg.early_stop and int(syndrome.sum()) == 0:
            iter_used = it
            break

        for v in range(N):
            checks = vars_to_checks[v]
            if checks.size == 0:
                continue
            edge_pos = var_to_checks_edge_pos[v]
            L_v = llr_posterior[v]
            for k, j in enumerate(checks):
                e = int(edge_pos[k])
                msg_v2c[j][e] = L_v - msg_c2v[j][e]

        iter_used = it

    hard_bits = (llr_posterior < 0.0).astype(np.uint8)
    syndrome = compute_syndrome_from_checks(hard_bits, code_cfg)

    return hard_bits, llr_posterior, syndrome, iter_used

# %%
def ldpc_min_sum_decoder_frame(frame: FrameLog,
                               sim_cfg: SimulationConfig,
                               dec_cfg: DecoderConfig) -> None:
    """
    Run LDPC normalized min-sum decoding on a single frame and populate the FrameLog
    with decoder outputs + per-iteration snapshots.
    """
    # Channel LLRs must be provided by run_single_frame_sionna5g() in this cleaned script.
    llr_ch = frame.llr_channel
    if llr_ch is None:
        raise RuntimeError(
            "FrameLog.llr_channel is None. "
            "This cleaned script supports only CHANNEL_NAME=SIONNA_TDL and requires "
            "run_single_frame() to populate FrameLog.llr_channel."
        )

   

    # Prepare snapshots dict on the frame
    frame.snapshots = {
        "llr": {},
        "hard_bits": {},
        "syndrome": {},
        "unsat_checks": {},
    }

    # Core LDPC decoding with snapshot support
    hard_bits, llr_post, syndrome, iter_used = ldpc_min_sum_decode(
        llr_ch,
        sim_cfg.code,
        dec_cfg,
        snapshot_iters=sim_cfg.snapshot_iters,
        snapshots=frame.snapshots,
    )

    # Fill standard FrameLog fields
    frame.hard_bits_final = hard_bits
    frame.llr_final = llr_post
    frame.syndrome_final = syndrome
    frame.iter_used = iter_used

    # Compare against transmitted codeword
    diff = (hard_bits != frame.c_bits)
    frame.error_positions_final = np.flatnonzero(diff)
    frame.dec_success = bool(frame.error_positions_final.size == 0)




### CELL number 20 ###
# Optional sanity check was removed (legacy AWGN path).
# Use the provided bash probes instead.
_run_sanity = str(os.environ.get("RUN_SANITY_CHECKS", "0")).strip().lower()
if _run_sanity not in ("0", "", "false", "no", "off"):
    raise RuntimeError(
        "RUN_SANITY_CHECKS is not supported in this cleaned Sionna-only script. "
        "Run the probe script instead."
    )




# %%
### CELL number 21 ###
def find_variable_clusters_from_syndrome(syndrome: np.ndarray,
                                         code_cfg: CodeConfig) -> List[np.ndarray]:
    """
    Given a syndrome vector (0/1) and the Tanner graph in code_cfg, return
    clusters of variable nodes that are connected via *unsatisfied* checks.

    Each cluster is a connected component in the graph where:
      - Nodes are variable indices v with degree > 0 in the unsatisfied-check subgraph.
      - There is an (undirected) edge between v1 and v2 if they share an unsatisfied check.
    """
    checks_to_vars = code_cfg.checks_to_vars
    N = code_cfg.N
    M = code_cfg.M

    # 1) Indices of unsatisfied checks
    unsat_checks = np.flatnonzero(syndrome)
    if unsat_checks.size == 0:
        return []

    # 2) Collect all variables touched by unsatisfied checks
    active_vars_set = set()
    for j in unsat_checks:
        for v in checks_to_vars[j]:
            active_vars_set.add(int(v))

    if not active_vars_set:
        return []

    active_vars = sorted(active_vars_set)
    L = len(active_vars)

    # Map variable index -> local index in [0..L-1]
    var_to_local = {v: i for i, v in enumerate(active_vars)}

    # 3) Build adjacency list for variable-variable graph (only active vars)
    adj = [[] for _ in range(L)]
    for j in unsat_checks:
        vars_j = [var_to_local[int(v)] for v in checks_to_vars[j]]
        d = len(vars_j)
        # Fully connect all vars in this unsatisfied check
        for a in range(d):
            va = vars_j[a]
            for b in range(a + 1, d):
                vb = vars_j[b]
                adj[va].append(vb)
                adj[vb].append(va)

    # 4) Find connected components via BFS/DFS
    clusters: List[np.ndarray] = []
    visited = np.zeros(L, dtype=bool)

    for start in range(L):
        if visited[start]:
            continue
        # BFS from 'start'
        queue = [start]
        visited[start] = True
        comp_local = []

        while queue:
            u = queue.pop()
            comp_local.append(u)
            for w in adj[u]:
                if not visited[w]:
                    visited[w] = True
                    queue.append(w)

        # Map local indices back to global variable indices
        comp_vars = np.array([active_vars[idx] for idx in comp_local], dtype=np.int32)
        clusters.append(comp_vars)

    return clusters




# %%
### CELL number 22 ###
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class ClusterGrandConfig:
    '''
    Config for local GRAND search on clusters / unions of clusters.

    Ordering rule used in this code:
      - enumerate flip patterns of weight 1..max_weight over a set of
        candidate bits;
      - assign each pattern a cost = sum(|LLR|) of the flipped bits;
      - test patterns in ascending cost (ties broken by lower weight).

    Key knobs:
      max_bits_from_cluster:
        - If an integer: use at most this many candidate bits (after sorting).
        - If None: AUTO-pick the number of candidate bits so that the total
          number of generated patterns stays close to max_patterns.
          (controlled by pattern_overgen_ratio).

      llr_source:
        - "posterior": use the LDPC posterior LLR snapshot for ordering/cost.
        - "channel"  : use the channel LLRs for ordering/cost (often better
                       when the LDPC posterior becomes over-confident on wrong bits).

      max_syndrome_weight_for_grand:
        - Optional guardrail. If set and the snapshot syndrome weight exceeds
          this threshold, skip GRAND (likely too many errors for a low-weight search).

    Notes:
      - max_patterns is BOTH a cap on *tested* patterns and (when AUTO bit-picking
        is used) an implicit cap on *generated/sorted* patterns, which is important
        for keeping the GRAND front-end cost reasonable.
    '''
    # Search space
    max_weight: int = 2
    max_patterns: int = 2000
    max_bits_from_cluster: Optional[int] = None

    # Logging / debug
    verbose: bool = True

    # Reliability / fade-aware gating (used by union-of-clusters code)
    low_llr_fraction: Optional[float] = None
    num_worst_blocks: Optional[int] = None

    # Chunked batching: number of patterns per Numba batch (if enabled)
    batch_size: int = 256

    # --- New knobs (hybrid performance + robustness) ---
    llr_source: str = "posterior"          # "posterior", "channel", or "mixed"
    pattern_overgen_ratio: float = 1.02    # used only when max_bits_from_cluster is None
    max_syndrome_weight_for_grand: Optional[int] = None

    # Receiver front-end selection
    selection_mode: str = "llr"            # "llr" (Receiver 1) or "syndrome_vote" (Receiver 2)
    sv_epsilon: float = 1e-3               # epsilon in eta_v = u_v / (rho_v + epsilon)
    sv_check_cover_k: int = 0              # k_cc ; 0 disables check-cover seeding

    # Receiver 3 / stronger pre-solver knobs
    pre_solver_mode: str = "none"          # "none" or "peel_gf2"
    peel_candidate_ratio: float = 1.50     # L_peel ~= ratio * L_search
    peel_max_bits: Optional[int] = None    # hard cap on peel candidate size
    peel_dense_max_vars: int = 32          # exact weighted GF(2) solve only if residual vars <= this
    peel_max_free_enum: int = 12           # enumerate weighted nullspace only if free dimension <= this
    peel_extra_llr_bits: int = 0           # add this many plain-LLR candidates to the peel set



### CELL number 23 ###
from dataclasses import dataclass
from typing import List, Tuple
import itertools
import numpy as np

@dataclass
class ClusterGrandResult:
    success: bool
    pattern_weight: int
    flipped_vars: np.ndarray
    patterns_tested: int
    initial_syndrome_weight: int
    final_syndrome_weight: int
    initial_bit_errors: int
    final_bit_errors: int

    # -------- NEW: op / complexity counters for hardware-like modeling --------
    # Total # of variable-to-check edges visited while testing patterns
    total_v2c_edge_visits: int = 0
    # Total # of UNIQUE checks encountered while testing patterns
    total_unique_checks_visited: int = 0
    # Total # of UNIQUE checks toggled odd times (i.e., weight-update touches)
    total_unique_checks_toggled: int = 0
    # Total patterns generated (before applying max_patterns cap)
    patterns_generated: int = 0


def _syndrome_weight_and_counts_after_flips_from_base(
    base_syndrome: np.ndarray,
    base_weight: int,
    flipped_vars: List[int],
    code_cfg: CodeConfig,
) -> Tuple[int, int, int, int]:
    """
    Exact incremental syndrome-weight update + op counters.

    Candidate = base_bits with flips at flipped_vars.
    Syndrome update:
        syn(cand) = syn(base) XOR (XOR of columns for flipped vars)

    Returns:
        syn_weight,
        v2c_edge_visits          : sum_{v in flipped} deg(v)
        unique_checks_visited    : | union of neighbouring checks |
        unique_checks_toggled    : | checks toggled odd times |
    """
    if not flipped_vars:
        return int(base_weight), 0, 0, 0

    toggled_checks = set()   # odd-toggle set
    visited_checks = set()   # union of checks seen at least once
    v2c = code_cfg.vars_to_checks

    edge_visits = 0
    for v in flipped_vars:
        for j in v2c[v]:
            edge_visits += 1
            j_int = int(j)
            visited_checks.add(j_int)
            if j_int in toggled_checks:
                toggled_checks.remove(j_int)
            else:
                toggled_checks.add(j_int)

    w = int(base_weight)
    for j in toggled_checks:
        if base_syndrome[j] == 0:
            w += 1
        else:
            w -= 1

    return w, edge_visits, len(visited_checks), len(toggled_checks)


def _bit_errors_after_flips_from_base(
    base_bits: np.ndarray,
    true_bits: np.ndarray,
    base_bit_errors: int,
    flipped_vars: List[int],
) -> int:
    """
    Exact incremental bit-error-count update (vs. true_bits).
    Only depends on whether each flipped bit was correct/incorrect in base_bits.
    """
    err = int(base_bit_errors)
    for v in flipped_vars:
        if base_bits[v] == true_bits[v]:
            err += 1
        else:
            err -= 1
    return err


# Batched GRAND pattern evaluation (per-batch, parallel over patterns)
# Optimized: incremental syndrome update + op counters
_GRAND_MAX_TOGGLES = 256  # must exceed (max_weight * max_variable_degree)


if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def _grand_eval_batch_numba_incremental(
        base_syndrome,
        base_syndrome_weight,
        base_bit_errors,
        base_bits,
        true_c_bits,
        search_vars,
        pattern_starts,
        pattern_lengths,
        pattern_positions,
        v2c_ptrs,
        v2c_checks,
    ):
        num_patterns = pattern_lengths.shape[0]
        M = base_syndrome.shape[0]

        syn_weights = np.empty(num_patterns, dtype=np.int32)
        bit_errors = np.full(num_patterns, -1, dtype=np.int32)

        # ---- NEW: per-pattern op counters ----
        edge_visits = np.zeros(num_patterns, dtype=np.int32)
        uniq_checks_visited = np.zeros(num_patterns, dtype=np.int32)
        uniq_checks_toggled = np.zeros(num_patterns, dtype=np.int32)

        for p in prange(num_patterns):
            start_p = pattern_starts[p]
            len_p = pattern_lengths[p]

            overflow = False

            # Track unique checks encountered; check_tog holds odd/even parity.
            check_ids = np.empty(_GRAND_MAX_TOGGLES, dtype=np.int32)
            check_tog = np.empty(_GRAND_MAX_TOGGLES, dtype=np.uint8)
            num_tog = 0

            err = base_bit_errors
            edges = 0

            # Build toggle set + error count
            for k in range(len_p):
                local_pos = pattern_positions[start_p + k]
                v = search_vars[local_pos]

                # bit-error update
                if base_bits[v] == true_c_bits[v]:
                    err += 1
                else:
                    err -= 1

                # syndrome toggles induced by flipping bit v
                vs = v2c_ptrs[v]
                ve = v2c_ptrs[v + 1]
                for idx in range(vs, ve):
                    edges += 1
                    j = v2c_checks[idx]

                    # find j in current unique list
                    found_idx = -1
                    for t in range(num_tog):
                        if check_ids[t] == j:
                            found_idx = t
                            break

                    if found_idx >= 0:
                        check_tog[found_idx] ^= np.uint8(1)
                    else:
                        if num_tog < _GRAND_MAX_TOGGLES:
                            check_ids[num_tog] = j
                            check_tog[num_tog] = np.uint8(1)
                            num_tog += 1
                        else:
                            overflow = True
                            break

                if overflow:
                    break

            if overflow:
                # Very rare for this project, but keep correctness.
                cand_syn = base_syndrome.copy()
                err2 = base_bit_errors
                edges2 = 0

                for k in range(len_p):
                    local_pos = pattern_positions[start_p + k]
                    v = search_vars[local_pos]

                    if base_bits[v] == true_c_bits[v]:
                        err2 += 1
                    else:
                        err2 -= 1

                    vs = v2c_ptrs[v]
                    ve = v2c_ptrs[v + 1]
                    for idx in range(vs, ve):
                        edges2 += 1
                        j = v2c_checks[idx]
                        cand_syn[j] ^= np.uint8(1)

                w = 0
                toggled = 0
                for j in range(M):
                    if cand_syn[j] != 0:
                        w += 1
                    if cand_syn[j] != base_syndrome[j]:
                        toggled += 1

                syn_weights[p] = w
                bit_errors[p] = err2 if w == 0 else -1

                edge_visits[p] = edges2
                uniq_checks_visited[p] = -1   # unknown in overflow path
                uniq_checks_toggled[p] = toggled
                continue

            # Store op counts
            edge_visits[p] = edges
            uniq_checks_visited[p] = num_tog

            toggled_cnt = 0
            for t in range(num_tog):
                if check_tog[t] != 0:
                    toggled_cnt += 1
            uniq_checks_toggled[p] = toggled_cnt

            # Fast weight update from base_syndrome_weight
            w = base_syndrome_weight
            for t in range(num_tog):
                if check_tog[t] != 0:
                    j = check_ids[t]
                    if base_syndrome[j] == 0:
                        w += 1
                    else:
                        w -= 1

            syn_weights[p] = w
            if w == 0:
                bit_errors[p] = err

        return syn_weights, bit_errors, edge_visits, uniq_checks_visited, uniq_checks_toggled


def run_local_grand_on_cluster(frame: FrameLog,
                               sim_cfg: SimulationConfig,
                               snapshot_iter: int,
                               cluster_index: int,
                               cfg: ClusterGrandConfig) -> ClusterGrandResult:
    """
    Local GRAND-style search on a single variable cluster at a given
    decoder iteration snapshot, with chunked batched pattern testing.

    Membership test:
      - incremental syndrome updates (no full syndrome recomputation per pattern)

    NEW:
      - accumulates per-frame op counters for hardware-like modeling.
    """
    code_cfg = sim_cfg.code

    # ---- Extract snapshot data ----
    snaps = frame.snapshots
    syn_snaps = snaps.get("syndrome", {})
    hard_snaps = snaps.get("hard_bits", {})
    llr_snaps = snaps.get("llr", {})

    if (snapshot_iter not in syn_snaps or
        snapshot_iter not in hard_snaps or
        snapshot_iter not in llr_snaps):
        raise ValueError(
            f"Snapshot at iter {snapshot_iter} is not fully available "
            f"(keys: syndrome={list(syn_snaps.keys())}, "
            f"hard_bits={list(hard_snaps.keys())}, "
            f"llr={list(llr_snaps.keys())})"
        )

    syndrome = syn_snaps[snapshot_iter]
    hard_bits_snapshot = hard_snaps[snapshot_iter].copy()
    llr_snapshot = llr_snaps[snapshot_iter]

    initial_syndrome_weight = int(syndrome.sum())

    # If already a codeword, nothing to do
    if initial_syndrome_weight == 0:
        diff_init = (hard_bits_snapshot != frame.c_bits)
        initial_bit_errors = int(diff_init.sum())
        return ClusterGrandResult(
            success=True,
            pattern_weight=0,
            flipped_vars=np.array([], dtype=np.int32),
            patterns_tested=0,
            initial_syndrome_weight=initial_syndrome_weight,
            final_syndrome_weight=initial_syndrome_weight,
            initial_bit_errors=initial_bit_errors,
            final_bit_errors=initial_bit_errors,
            total_v2c_edge_visits=0,
            total_unique_checks_visited=0,
            total_unique_checks_toggled=0,
            patterns_generated=0,
        )

    # ---- Build clusters from current syndrome ----
    clusters = find_variable_clusters_from_syndrome(syndrome, code_cfg)
    if not clusters:
        diff_init = (hard_bits_snapshot != frame.c_bits)
        initial_bit_errors = int(diff_init.sum())
        return ClusterGrandResult(
            success=False,
            pattern_weight=-1,
            flipped_vars=np.array([], dtype=np.int32),
            patterns_tested=0,
            initial_syndrome_weight=initial_syndrome_weight,
            final_syndrome_weight=initial_syndrome_weight,
            initial_bit_errors=initial_bit_errors,
            final_bit_errors=initial_bit_errors,
            total_v2c_edge_visits=0,
            total_unique_checks_visited=0,
            total_unique_checks_toggled=0,
            patterns_generated=0,
        )

    if cluster_index < 0 or cluster_index >= len(clusters):
        raise IndexError(f"cluster_index {cluster_index} out of range [0, {len(clusters)-1}]")

    cluster_vars = clusters[cluster_index]
    cluster_size = cluster_vars.size

    # Initial bit errors at snapshot vs ground truth
    diff_init = (hard_bits_snapshot != frame.c_bits)
    initial_bit_errors = int(diff_init.sum())

    # ---- Order cluster bits by |LLR| (least reliable first) ----
    abs_llr_cluster = np.abs(llr_snapshot[cluster_vars])
    order = np.argsort(abs_llr_cluster)   # ascending |LLR|
    cluster_vars_sorted = cluster_vars[order]

    if cfg.max_bits_from_cluster is not None and cfg.max_bits_from_cluster < cluster_size:
        search_vars = cluster_vars_sorted[:cfg.max_bits_from_cluster]
    else:
        search_vars = cluster_vars_sorted
    L = search_vars.size

    patterns_tested = 0
    found = False
    found_weight = -1
    found_flipped = np.array([], dtype=np.int32)
    final_syn_weight = initial_syndrome_weight
    final_bit_errors = initial_bit_errors

    # ---- NEW totals ----
    total_edge_visits = 0
    total_uniq_checks_visited = 0
    total_uniq_checks_toggled = 0

    if L == 0 or cfg.max_weight <= 0:
        return ClusterGrandResult(
            success=False,
            pattern_weight=-1,
            flipped_vars=np.array([], dtype=np.int32),
            patterns_tested=0,
            initial_syndrome_weight=initial_syndrome_weight,
            final_syndrome_weight=initial_syndrome_weight,
            initial_bit_errors=initial_bit_errors,
            final_bit_errors=final_bit_errors,
            total_v2c_edge_visits=0,
            total_unique_checks_visited=0,
            total_unique_checks_toggled=0,
            patterns_generated=0,
        )

    max_w = min(cfg.max_weight, L)

    # ---- Build all patterns and order them by sum‑|LLR| ----
    pattern_items: List[tuple] = []
    abs_llr_local = np.abs(llr_snapshot[search_vars])

    for w in range(1, max_w + 1):
        for comb in itertools.combinations(range(L), w):
            cost = float(abs_llr_local[list(comb)].sum())
            pattern_items.append((cost, w, comb))

    pattern_items.sort(key=lambda t: (t[0], t[1]))
    patterns_generated = len(pattern_items)

    if not pattern_items:
        return ClusterGrandResult(
            success=False,
            pattern_weight=-1,
            flipped_vars=np.array([], dtype=np.int32),
            patterns_tested=0,
            initial_syndrome_weight=initial_syndrome_weight,
            final_syndrome_weight=initial_syndrome_weight,
            initial_bit_errors=initial_bit_errors,
            final_bit_errors=initial_bit_errors,
            total_v2c_edge_visits=0,
            total_unique_checks_visited=0,
            total_unique_checks_toggled=0,
            patterns_generated=0,
        )

    use_batch = (
        NUMBA_AVAILABLE
        and hasattr(code_cfg, "_v2c_ptrs")
        and hasattr(code_cfg, "_v2c_checks")
        and getattr(cfg, "batch_size", 0) > 0
    )

    if use_batch:
        base_bits = hard_bits_snapshot.astype(np.uint8)
        true_c_bits = frame.c_bits.astype(np.uint8)
        search_vars_int = search_vars.astype(np.int64)

        base_syn = syndrome.astype(np.uint8)
        base_syn_w = np.int32(initial_syndrome_weight)
        base_bit_err = np.int32(initial_bit_errors)

        total_patterns = len(pattern_items)
        max_patterns = int(cfg.max_patterns)
        limit = min(total_patterns, max_patterns)
        batch_size = int(getattr(cfg, "batch_size", 256))
        if batch_size <= 0:
            batch_size = limit

        for start_idx in range(0, limit, batch_size):
            end_idx = min(start_idx + batch_size, limit)
            num_batch = end_idx - start_idx

            # Pack this batch
            total_positions = 0
            for i in range(start_idx, end_idx):
                total_positions += pattern_items[i][1]

            pattern_starts = np.zeros(num_batch, dtype=np.int32)
            pattern_lengths = np.zeros(num_batch, dtype=np.int32)
            pattern_positions = np.zeros(total_positions, dtype=np.int32)

            pos_ptr = 0
            for b in range(num_batch):
                _, w, comb = pattern_items[start_idx + b]
                pattern_starts[b] = pos_ptr
                pattern_lengths[b] = w
                for lp in comb:
                    pattern_positions[pos_ptr] = int(lp)
                    pos_ptr += 1

            syn_w_arr, bit_err_arr, edge_arr, uniq_arr, tog_arr = _grand_eval_batch_numba_incremental(
                base_syn,
                base_syn_w,
                base_bit_err,
                base_bits,
                true_c_bits,
                search_vars_int,
                pattern_starts,
                pattern_lengths,
                pattern_positions,
                code_cfg._v2c_ptrs,
                code_cfg._v2c_checks,
            )

            # Find first success in this batch
            success_rel = -1
            for b in range(num_batch):
                if syn_w_arr[b] == 0:
                    success_rel = b
                    break

            if success_rel >= 0:
                # Accumulate counters up to and including success_rel
                total_edge_visits += int(edge_arr[:success_rel + 1].sum())
                total_uniq_checks_visited += int(np.maximum(uniq_arr[:success_rel + 1], 0).sum())
                total_uniq_checks_toggled += int(tog_arr[:success_rel + 1].sum())

                global_idx = start_idx + success_rel
                patterns_tested = global_idx + 1
                _, w, comb = pattern_items[global_idx]
                flipped = [int(search_vars[pos]) for pos in comb]
                found_flipped = np.array(flipped, dtype=np.int32)
                found_weight = w
                final_syn_weight = 0
                be = int(bit_err_arr[success_rel])
                final_bit_errors = be if be >= 0 else initial_bit_errors
                found = True
                break
            else:
                # Accumulate full batch counters
                total_edge_visits += int(edge_arr.sum())
                total_uniq_checks_visited += int(np.maximum(uniq_arr, 0).sum())
                total_uniq_checks_toggled += int(tog_arr.sum())
                patterns_tested = end_idx

    else:
        # Sequential one‑by‑one testing (incremental membership + counters)
        for _, w, comb in pattern_items:
            patterns_tested += 1
            if patterns_tested > cfg.max_patterns:
                break

            flipped = [int(search_vars[pos]) for pos in comb]

            syn_w, e_cnt, uq_cnt, tg_cnt = _syndrome_weight_and_counts_after_flips_from_base(
                base_syndrome=syndrome,
                base_weight=initial_syndrome_weight,
                flipped_vars=flipped,
                code_cfg=code_cfg,
            )

            total_edge_visits += e_cnt
            total_uniq_checks_visited += uq_cnt
            total_uniq_checks_toggled += tg_cnt

            if syn_w == 0:
                bit_err_cand = _bit_errors_after_flips_from_base(
                    base_bits=hard_bits_snapshot,
                    true_bits=frame.c_bits,
                    base_bit_errors=initial_bit_errors,
                    flipped_vars=flipped,
                )

                found = True
                found_weight = w
                found_flipped = np.array(flipped, dtype=np.int32)
                final_syn_weight = 0
                final_bit_errors = bit_err_cand
                break

    return ClusterGrandResult(
        success=found,
        pattern_weight=found_weight if found else -1,
        flipped_vars=found_flipped,
        patterns_tested=patterns_tested,
        initial_syndrome_weight=initial_syndrome_weight,
        final_syndrome_weight=final_syn_weight,
        initial_bit_errors=initial_bit_errors,
        final_bit_errors=final_bit_errors,
        total_v2c_edge_visits=int(total_edge_visits),
        total_unique_checks_visited=int(total_uniq_checks_visited),
        total_unique_checks_toggled=int(total_uniq_checks_toggled),
        patterns_generated=int(patterns_generated),
    )







# %%
### CELL number 26 ###
def build_allowed_mask_from_config(frame: FrameLog,
                                   sim_cfg: SimulationConfig,
                                   snapshot_iter: int,
                                   cfg: ClusterGrandConfig) -> np.ndarray:
    """
    Build a boolean mask allowed_mask[v] telling GRAND which variable
    nodes it is allowed to flip at the given snapshot.

    Combines:
      - global low-|LLR| gating via cfg.low_llr_fraction in (0, 1),
      - block-fading gating via cfg.num_worst_blocks (integer >= 1).

    If those attributes are missing or None, we fall back to
    'all bits allowed'.
    """
    N = sim_cfg.code.N

    # Start with everything allowed
    allowed = np.ones(N, dtype=bool)

    # Pull LLR snapshot
    llr_snaps = frame.snapshots.get("llr", {})
    if snapshot_iter not in llr_snaps:
        # No LLR at this iteration; don't gate anything.
        return allowed

    llr_snapshot = llr_snaps[snapshot_iter]
    if llr_snapshot.shape[0] != N:
        raise ValueError(f"LLR snapshot length {llr_snapshot.shape[0]} != N={N}")

    # ---- 1) low-|LLR| gating (reliability gating) ----
    low_frac = getattr(cfg, "low_llr_fraction", None)
    if isinstance(low_frac, (int, float)) and 0.0 < low_frac < 1.0:
        abs_llr = np.abs(llr_snapshot)
        K = max(1, int(np.round(low_frac * N)))
        if K < N:
            # Threshold tau: K smallest |LLR|
            tau = np.partition(abs_llr, K - 1)[K - 1]
            mask_low = abs_llr <= tau
        else:
            mask_low = np.ones_like(allowed)
        allowed &= mask_low

    # ---- 2) block-fading gating (worst B blocks) ----
    num_worst_blocks = getattr(cfg, "num_worst_blocks", None)
    if isinstance(num_worst_blocks, int) and num_worst_blocks > 0:
        ch = frame.channel_realization
        block_idx = ch.get("block_index_per_bit", None)
        num_blocks_arr = ch.get("num_blocks", None)

        if block_idx is not None and num_blocks_arr is not None:
            block_idx = block_idx.astype(np.int64)
            F = int(num_blocks_arr[0])
            abs_llr = np.abs(llr_snapshot)

            # Per-block mean |LLR|
            block_scores = np.full(F, np.inf, dtype=np.float64)
            for b in range(F):
                mask_b = (block_idx == b)
                if not mask_b.any():
                    continue
                # Score = avg |LLR| in the block
                block_scores[b] = abs_llr[mask_b].mean()

            B = min(num_worst_blocks, F)
            worst_ids = np.argsort(block_scores)[:B]
            in_worst = np.isin(block_idx, worst_ids)
            allowed &= in_worst

    return allowed




def _auto_pick_grand_search_size(L_full: int, cfg: ClusterGrandConfig) -> int:
    """Choose the GRAND search size L.

    If max_bits_from_cluster is None, pick the largest L whose generated pattern
    count stays close to cfg.max_patterns (using cfg.pattern_overgen_ratio).
    Otherwise, clamp to max_bits_from_cluster.
    """
    L_full = int(max(L_full, 0))
    if L_full <= 0:
        return 0

    if cfg.max_bits_from_cluster is None:
        over = float(getattr(cfg, "pattern_overgen_ratio", 1.02) or 1.02)
        target = max(1, int(round(over * int(cfg.max_patterns))))
        max_w = max(1, int(cfg.max_weight))

        L = 1
        while True:
            L_try = L + 1
            total = 0
            for w in range(1, max_w + 1):
                total += math.comb(L_try, w)
                if total > target:
                    break
            if total > target:
                break
            if L_try >= L_full:
                L = L_try
                break
            L = L_try
        return int(min(max(L, 1), L_full))

    return int(min(max(int(cfg.max_bits_from_cluster), 0), L_full))



def _select_search_vars_llr(union_vars: np.ndarray,
                            llr_for_sort: np.ndarray,
                            L: int) -> Tuple[np.ndarray, Dict[str, int]]:
    """Receiver 1 front-end: pick the L least reliable bits in the union."""
    L = int(max(L, 0))
    union_vars = np.asarray(union_vars, dtype=np.int32)
    if L <= 0 or union_vars.size == 0:
        return np.array([], dtype=np.int32), {
            "selection_mode_used": "llr",
            "sv_seeded_count": 0,
            "sv_neighbor_visits": 0,
            "sv_score_len": int(union_vars.size),
        }

    abs_llr_union = np.abs(llr_for_sort[union_vars])
    order = np.argsort(abs_llr_union, kind="stable")
    search_vars = union_vars[order[:L]].astype(np.int32, copy=False)
    return search_vars, {
        "selection_mode_used": "llr",
        "sv_seeded_count": 0,
        "sv_neighbor_visits": 0,
        "sv_score_len": int(union_vars.size),
    }



def _select_search_vars_syndrome_vote(union_vars: np.ndarray,
                                      unsat_checks: np.ndarray,
                                      code_cfg: CodeConfig,
                                      llr_for_sort: np.ndarray,
                                      L: int,
                                      cfg: ClusterGrandConfig) -> Tuple[np.ndarray, Dict[str, int]]:
    """Receiver 2 front-end: syndrome-vote ranking with optional check-cover seeding.

    Score:
        eta_v = u_v / (rho_v + epsilon)
    where u_v counts the number of unsatisfied checks touching v and
    rho_v = |LLR_v| from the selected LLR source.

    Check-cover seeding:
        For each unsatisfied check j, include up to k_cc of its least reliable
        neighbours before filling the remaining budget from the global score rank.
    """
    union_vars = np.asarray(union_vars, dtype=np.int32)
    unsat_checks = np.asarray(unsat_checks, dtype=np.int32)
    L = int(max(L, 0))

    if L <= 0 or union_vars.size == 0:
        return np.array([], dtype=np.int32), {
            "selection_mode_used": "syndrome_vote",
            "sv_seeded_count": 0,
            "sv_neighbor_visits": 0,
            "sv_score_len": int(union_vars.size),
        }

    abs_llr_union = np.abs(llr_for_sort[union_vars]).astype(np.float64, copy=False)
    eps = float(getattr(cfg, "sv_epsilon", 1e-3) or 1e-3)
    k_cc = max(0, int(getattr(cfg, "sv_check_cover_k", 0) or 0))

    vote_counts = np.zeros(union_vars.size, dtype=np.int32)
    var_to_local = {int(v): i for i, v in enumerate(union_vars.tolist())}

    seed_list: List[int] = []
    seeded = set()
    sv_neighbor_visits = 0

    for j in unsat_checks:
        local_neighbors: List[int] = []
        for v in code_cfg.checks_to_vars[int(j)]:
            v_int = int(v)
            loc = var_to_local.get(v_int, None)
            if loc is None:
                continue
            vote_counts[loc] += 1
            local_neighbors.append(loc)
            sv_neighbor_visits += 1

        if k_cc > 0 and local_neighbors:
            # Least reliable neighbours first; deterministic tie-break by vote then index.
            local_neighbors_sorted = sorted(
                local_neighbors,
                key=lambda loc: (float(abs_llr_union[loc]), -int(vote_counts[loc]), int(union_vars[loc]))
            )
            take = min(k_cc, len(local_neighbors_sorted))
            for loc in local_neighbors_sorted[:take]:
                v_int = int(union_vars[loc])
                if v_int not in seeded:
                    seed_list.append(v_int)
                    seeded.add(v_int)

    # Global score rank for Receiver 2
    scores = vote_counts.astype(np.float64) / (abs_llr_union + eps)
    global_order = sorted(
        range(union_vars.size),
        key=lambda loc: (-float(scores[loc]), float(abs_llr_union[loc]), -int(vote_counts[loc]), int(union_vars[loc]))
    )

    selected: List[int] = []
    seen = set()

    for v_int in seed_list:
        if len(selected) >= L:
            break
        if v_int not in seen:
            selected.append(int(v_int))
            seen.add(int(v_int))

    for loc in global_order:
        if len(selected) >= L:
            break
        v_int = int(union_vars[loc])
        if v_int not in seen:
            selected.append(v_int)
            seen.add(v_int)

    return np.asarray(selected[:L], dtype=np.int32), {
        "selection_mode_used": "syndrome_vote",
        "sv_seeded_count": int(min(len(seed_list), L)),
        "sv_neighbor_visits": int(sv_neighbor_visits),
        "sv_score_len": int(union_vars.size),
    }



def _resolve_sort_llr_vector(llr_snapshot: np.ndarray,
                             llr_channel: Optional[np.ndarray],
                             cfg: ClusterGrandConfig) -> Tuple[np.ndarray, str]:
    """Resolve the LLR vector used for ranking/costing.

    Supported sources:
      - "posterior": use the stage-1 snapshot posterior LLRs
      - "channel"  : use the channel LLRs when available
      - "mixed"    : conservative magnitude = min(|posterior|, |channel|)
                     with posterior sign carried for deterministic ordering
    """
    llr_snapshot = np.asarray(llr_snapshot, dtype=np.float32)
    llr_source = str(getattr(cfg, "llr_source", "posterior") or "posterior").strip().lower()

    if llr_source == "channel":
        if llr_channel is not None:
            return np.asarray(llr_channel, dtype=np.float32), "channel"
        return llr_snapshot, "posterior"

    if llr_source in ("mixed", "hybrid", "minabs"):
        if llr_channel is not None:
            llr_channel = np.asarray(llr_channel, dtype=np.float32)
            abs_mix = np.minimum(np.abs(llr_snapshot), np.abs(llr_channel)).astype(np.float32, copy=False)
            sign_ref = np.sign(llr_snapshot).astype(np.float32, copy=False)
            zero_mask = (sign_ref == 0)
            if np.any(zero_mask):
                sign_ref = sign_ref.copy()
                if llr_channel is not None:
                    sign_ref[zero_mask] = np.sign(llr_channel[zero_mask]).astype(np.float32, copy=False)
                    zero_mask = (sign_ref == 0)
                if np.any(zero_mask):
                    sign_ref[zero_mask] = 1.0
            return (sign_ref * abs_mix).astype(np.float32, copy=False), "mixed"
        return llr_snapshot, "posterior"

    return llr_snapshot, "posterior"


def _auto_pick_peel_candidate_size(L_full: int,
                                   L_search: int,
                                   cfg: ClusterGrandConfig) -> int:
    """Choose L_peel >= L_search, capped by peel_max_bits / L_full."""
    L_full = int(max(L_full, 0))
    L_search = int(max(L_search, 0))
    if L_full <= 0:
        return 0

    ratio = float(getattr(cfg, "peel_candidate_ratio", 1.0) or 1.0)
    ratio = max(1.0, ratio)
    L_peel = int(np.ceil(ratio * max(L_search, 1)))

    peel_max_bits = getattr(cfg, "peel_max_bits", None)
    if isinstance(peel_max_bits, int) and peel_max_bits > 0:
        L_peel = min(L_peel, int(peel_max_bits))

    L_peel = max(L_peel, max(L_search, 1))
    return int(min(L_peel, L_full))


def _select_presolver_vars(union_vars: np.ndarray,
                           unsat_checks: np.ndarray,
                           code_cfg: CodeConfig,
                           llr_for_sort: np.ndarray,
                           L_peel: int,
                           cfg: ClusterGrandConfig) -> Tuple[np.ndarray, Dict[str, int]]:
    """Receiver-3 candidate-set construction.

    Base list:
      - uses the configured front-end selection mode (LLR or syndrome-vote)

    Optional strengthening:
      - append a small number of extra plain-LLR candidates so the pre-solver
        is less brittle if the syndrome-vote list misses a true error bit.
    """
    union_vars = np.asarray(union_vars, dtype=np.int32)
    L_peel = int(max(L_peel, 0))
    if L_peel <= 0 or union_vars.size == 0:
        return np.array([], dtype=np.int32), {
            "selection_mode_used": str(getattr(cfg, "selection_mode", "llr")),
            "sv_seeded_count": 0,
            "sv_neighbor_visits": 0,
            "sv_score_len": int(union_vars.size),
            "peel_extra_llr_added": 0,
        }

    selection_mode = str(getattr(cfg, "selection_mode", "llr") or "llr").strip().lower()
    if selection_mode in ("syndrome_vote", "sv", "receiver2"):
        base_vars, meta = _select_search_vars_syndrome_vote(
            union_vars=union_vars,
            unsat_checks=unsat_checks,
            code_cfg=code_cfg,
            llr_for_sort=llr_for_sort,
            L=L_peel,
            cfg=cfg,
        )
    else:
        base_vars, meta = _select_search_vars_llr(
            union_vars=union_vars,
            llr_for_sort=llr_for_sort,
            L=L_peel,
        )

    selected = [int(v) for v in np.asarray(base_vars, dtype=np.int32).tolist()]
    seen = set(selected)

    extra_llr = max(0, int(getattr(cfg, "peel_extra_llr_bits", 0) or 0))
    extra_added = 0
    if extra_llr > 0 and len(selected) < L_peel:
        llr_vars, _ = _select_search_vars_llr(
            union_vars=union_vars,
            llr_for_sort=llr_for_sort,
            L=min(int(union_vars.size), int(L_peel + extra_llr)),
        )
        for v in np.asarray(llr_vars, dtype=np.int32).tolist():
            v_int = int(v)
            if v_int not in seen:
                selected.append(v_int)
                seen.add(v_int)
                extra_added += 1
            if len(selected) >= L_peel:
                break

    out = np.asarray(selected[:L_peel], dtype=np.int32)
    meta = dict(meta)
    meta["peel_extra_llr_added"] = int(extra_added)
    return out, meta


def _build_local_subsystem_for_candidate(candidate_vars: np.ndarray,
                                         unsat_checks: np.ndarray,
                                         code_cfg: CodeConfig,
                                         syndrome: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build A, b for H_sub e = syndrome on a localized row set.

    Row set = all checks touched by candidate_vars UNION all unsatisfied checks.
    If a selected candidate set cannot touch some unsatisfied check, that row
    becomes all-zero with RHS=1, and the solver correctly declares failure.
    """
    candidate_vars = np.asarray(candidate_vars, dtype=np.int32)
    unsat_checks = np.asarray(unsat_checks, dtype=np.int32)

    row_set = set(int(j) for j in unsat_checks.tolist())
    for v in candidate_vars.tolist():
        for j in code_cfg.vars_to_checks[int(v)]:
            row_set.add(int(j))

    rows = np.array(sorted(row_set), dtype=np.int32)
    m = int(rows.size)
    n = int(candidate_vars.size)
    if m == 0 or n == 0:
        return np.zeros((0, n), dtype=np.uint8), np.zeros((0,), dtype=np.uint8)

    row_to_idx = {int(j): i for i, j in enumerate(rows.tolist())}
    A = np.zeros((m, n), dtype=np.uint8)
    for c_idx, v in enumerate(candidate_vars.tolist()):
        for j in code_cfg.vars_to_checks[int(v)]:
            r_idx = row_to_idx[int(j)]
            A[r_idx, c_idx] ^= np.uint8(1)

    b = syndrome[rows].astype(np.uint8, copy=True)
    return A, b


def _peel_reduce_system(A: np.ndarray,
                        b: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray, int]:
    """Peel degree-1 equations in a binary linear system.

    Returns:
      ok,
      fixed_assignments (len n; -1 = unresolved, else 0/1),
      unresolved_col_idx,
      unresolved_row_idx,
      peel_edge_work
    """
    A = np.asarray(A, dtype=np.uint8).copy()
    b = np.asarray(b, dtype=np.uint8).copy()

    m, n = A.shape
    fixed = np.full(n, -1, dtype=np.int8)
    active_rows = np.ones(m, dtype=np.bool_)
    active_cols = np.ones(n, dtype=np.bool_)

    row_deg = A.sum(axis=1).astype(np.int32, copy=False)
    peel_edge_work = 0

    changed = True
    while changed:
        changed = False

        # Contradiction rows (0 = 1)
        bad_rows = np.flatnonzero(active_rows & (row_deg == 0) & (b != 0))
        if bad_rows.size > 0:
            return False, fixed, np.flatnonzero(active_cols), np.flatnonzero(active_rows), int(peel_edge_work)

        deg1_rows = np.flatnonzero(active_rows & (row_deg == 1))
        if deg1_rows.size == 0:
            break

        changed = True
        for r in deg1_rows.tolist():
            if not bool(active_rows[r]) or int(row_deg[r]) != 1:
                continue

            cols = np.flatnonzero(A[r] & active_cols)
            if cols.size != 1:
                continue
            c = int(cols[0])

            x = int(b[r] & 1)
            fixed[c] = np.int8(x)

            touched_rows = np.flatnonzero(A[:, c] & active_rows)
            peel_edge_work += int(touched_rows.size)

            if x != 0:
                b[touched_rows] ^= np.uint8(1)

            A[touched_rows, c] = np.uint8(0)
            row_deg[touched_rows] -= 1
            active_cols[c] = False
            active_rows[r] = False
            row_deg[r] = 0

    # Final contradiction check
    bad_rows = np.flatnonzero(active_rows & (row_deg == 0) & (b != 0))
    if bad_rows.size > 0:
        return False, fixed, np.flatnonzero(active_cols), np.flatnonzero(active_rows), int(peel_edge_work)

    unresolved_cols = np.flatnonzero(active_cols)
    unresolved_rows = np.flatnonzero(active_rows & (row_deg > 0))
    return True, fixed, unresolved_cols.astype(np.int32), unresolved_rows.astype(np.int32), int(peel_edge_work)


def _gf2_weighted_solve(A: np.ndarray,
                        b: np.ndarray,
                        weights: np.ndarray,
                        max_free_enum: int = 12) -> Tuple[bool, np.ndarray, int, int]:
    """Solve A x = b over GF(2) and choose a low-cost solution.

    If the solution space has free dimension <= max_free_enum, enumerate the
    affine nullspace and pick the minimum weighted cost solution.
    Otherwise return failure so the caller can fall back to GRAND.
    """
    A = np.asarray(A, dtype=np.uint8).copy()
    b = np.asarray(b, dtype=np.uint8).copy()
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)

    m, n = A.shape
    if n == 0:
        ok = bool(np.all((b & 1) == 0))
        return ok, np.zeros((0,), dtype=np.uint8), 0, 0

    xor_ops = 0
    row = 0
    pivot_cols: List[int] = []
    pivot_rows: List[int] = []

    for col in range(n):
        pivot = None
        for r in range(row, m):
            if int(A[r, col]) != 0:
                pivot = r
                break
        if pivot is None:
            continue

        if pivot != row:
            A[[row, pivot], :] = A[[pivot, row], :]
            b[[row, pivot]] = b[[pivot, row]]

        # Full elimination to RREF-style pivot columns
        for r in range(m):
            if r != row and int(A[r, col]) != 0:
                A[r, :] ^= A[row, :]
                b[r] ^= b[row]
                xor_ops += int(n + 1)

        pivot_cols.append(int(col))
        pivot_rows.append(int(row))
        row += 1
        if row >= m:
            break

    # Inconsistency: 0 = 1
    for r in range(m):
        if int(A[r].sum()) == 0 and int(b[r]) != 0:
            return False, np.zeros((n,), dtype=np.uint8), 0, int(xor_ops)

    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(n) if c not in pivot_set]
    free_dim = int(len(free_cols))

    # One particular solution (free vars = 0)
    x0 = np.zeros((n,), dtype=np.uint8)
    for prow, pcol in zip(pivot_rows, pivot_cols):
        x0[pcol] = np.uint8(b[prow] & 1)

    if free_dim == 0:
        return True, x0, 0, int(xor_ops)

    if free_dim > int(max_free_enum):
        return False, np.zeros((n,), dtype=np.uint8), free_dim, int(xor_ops)

    pivot_row_for_col = {int(c): int(r) for r, c in zip(pivot_rows, pivot_cols)}
    basis = []
    for fcol in free_cols:
        vec = np.zeros((n,), dtype=np.uint8)
        vec[int(fcol)] = np.uint8(1)
        for pcol in pivot_cols:
            prow = pivot_row_for_col[int(pcol)]
            if int(A[prow, int(fcol)]) != 0:
                vec[int(pcol)] = np.uint8(1)
        basis.append(vec)

    best_x = x0.copy()
    best_cost = float(np.dot(weights, best_x.astype(np.float64)))
    best_weight = int(best_x.sum())

    for mask in range(1, 1 << free_dim):
        x = x0.copy()
        for i in range(free_dim):
            if (mask >> i) & 1:
                x ^= basis[i]
        cost = float(np.dot(weights, x.astype(np.float64)))
        hwt = int(x.sum())
        if (cost < best_cost - 1e-12) or (abs(cost - best_cost) <= 1e-12 and hwt < best_weight):
            best_cost = cost
            best_weight = hwt
            best_x = x

    return True, best_x, free_dim, int(xor_ops)



def _run_presolver_peel_gf2(frame: FrameLog,
                            sim_cfg: SimulationConfig,
                            snapshot_iter: int,
                            cfg: ClusterGrandConfig) -> Optional[ClusterGrandResult]:
    """Receiver 3+ pre-solver: peel + weighted small GF(2) solve on a larger local set.

    Returns:
      - successful ClusterGrandResult if the pre-solver alone fixes the frame
      - failed ClusterGrandResult with pre-solver metadata if it *attempted* but
        could not certify a correction (so the caller can still charge its cost)
      - None if the pre-solver was not applicable / not meaningfully attempted
    """
    code_cfg = sim_cfg.code

    snaps = frame.snapshots
    syn_snaps = snaps.get("syndrome", {})
    hard_snaps = snaps.get("hard_bits", {})
    llr_snaps = snaps.get("llr", {})

    if (snapshot_iter not in syn_snaps or
        snapshot_iter not in hard_snaps or
        snapshot_iter not in llr_snaps):
        raise ValueError(f"Snapshot at iter {snapshot_iter} is not fully available for pre-solver.")

    syndrome = syn_snaps[snapshot_iter]
    hard_bits_snapshot = hard_snaps[snapshot_iter].copy()
    llr_snapshot = llr_snaps[snapshot_iter]

    initial_syndrome_weight = int(syndrome.sum())
    if initial_syndrome_weight == 0:
        return None

    diff_init = (hard_bits_snapshot != frame.c_bits)
    initial_bit_errors = int(diff_init.sum())
    unsat_checks = np.flatnonzero(syndrome).astype(np.int32)

    cluster_unsat_edges = 0
    cluster_pair_edges = 0
    if unsat_checks.size > 0:
        for j in unsat_checks:
            neigh = code_cfg.checks_to_vars[int(j)]
            d = int(neigh.size)
            cluster_unsat_edges += d
            if d >= 2:
                cluster_pair_edges += int(d * (d - 1) // 2)

    llr_for_sort, llr_source_used = _resolve_sort_llr_vector(
        llr_snapshot=llr_snapshot,
        llr_channel=getattr(frame, "llr_channel", None),
        cfg=cfg,
    )

    allowed_mask = build_allowed_mask_from_config(frame, sim_cfg, snapshot_iter, cfg)
    clusters = find_variable_clusters_from_syndrome(syndrome, code_cfg)
    if not clusters:
        return None

    union_vars = np.unique(np.concatenate(clusters)).astype(np.int32)
    union_vars = union_vars[allowed_mask[union_vars]]
    L_full = int(union_vars.size)
    if L_full == 0:
        return None

    L_search = _auto_pick_grand_search_size(L_full, cfg)
    L_peel = _auto_pick_peel_candidate_size(L_full, L_search, cfg)
    peel_vars, front_end_meta = _select_presolver_vars(
        union_vars=union_vars,
        unsat_checks=unsat_checks,
        code_cfg=code_cfg,
        llr_for_sort=llr_for_sort,
        L_peel=L_peel,
        cfg=cfg,
    )

    if peel_vars.size == 0:
        return None

    # Common metadata carried on both success and attempted-failure returns
    def _make_attempt_result(success: bool,
                             flipped_vars: Optional[np.ndarray] = None,
                             final_bit_errors: Optional[int] = None,
                             peel_edge_work: int = 0,
                             dense_xor_ops: int = 0,
                             free_dim: int = 0,
                             residual_vars: int = 0,
                             residual_rows: int = 0,
                             e_cnt: int = 0,
                             uq_cnt: int = 0,
                             tg_cnt: int = 0) -> ClusterGrandResult:
        if flipped_vars is None:
            flipped_vars = np.array([], dtype=np.int32)
        if final_bit_errors is None:
            final_bit_errors = int(initial_bit_errors)
        res_local = ClusterGrandResult(
            success=bool(success),
            pattern_weight=int(flipped_vars.size) if bool(success) else -1,
            flipped_vars=np.asarray(flipped_vars, dtype=np.int32),
            patterns_tested=0,
            initial_syndrome_weight=int(initial_syndrome_weight),
            final_syndrome_weight=0 if bool(success) else int(initial_syndrome_weight),
            initial_bit_errors=int(initial_bit_errors),
            final_bit_errors=int(final_bit_errors),
            total_v2c_edge_visits=int(e_cnt),
            total_unique_checks_visited=int(uq_cnt),
            total_unique_checks_toggled=int(tg_cnt),
            patterns_generated=0,
        )
        setattr(res_local, "patterns_evaluated", 0)
        setattr(res_local, "total_v2c_edge_visits_evaluated", 0)
        setattr(res_local, "total_unique_checks_visited_evaluated", 0)
        setattr(res_local, "total_unique_checks_toggled_evaluated", 0)
        setattr(res_local, "union_size", int(L_full))
        setattr(res_local, "search_size", int(L_search))
        setattr(res_local, "llr_sort_len", int(L_full))
        setattr(res_local, "sum_pattern_weights_generated", 0)
        setattr(res_local, "cluster_unsat_edges", int(cluster_unsat_edges))
        setattr(res_local, "cluster_pair_edges", int(cluster_pair_edges))
        setattr(res_local, "num_batches_evaluated", 0)
        setattr(res_local, "positions_packed_evaluated", 0)
        setattr(res_local, "batch_size_used", 0)
        setattr(res_local, "llr_source_used", str(llr_source_used))
        setattr(res_local, "selection_mode_used", str(front_end_meta.get("selection_mode_used", getattr(cfg, "selection_mode", "llr"))))
        setattr(res_local, "sv_seeded_count", int(front_end_meta.get("sv_seeded_count", 0)))
        setattr(res_local, "sv_neighbor_visits", int(front_end_meta.get("sv_neighbor_visits", 0)))
        setattr(res_local, "sv_score_len", int(front_end_meta.get("sv_score_len", L_full)))
        setattr(res_local, "pre_solver_mode_used", "peel_gf2")
        setattr(res_local, "pre_solver_attempted", 1)
        setattr(res_local, "pre_solver_success", 1 if bool(success) else 0)
        setattr(res_local, "peel_candidate_size", int(peel_vars.size))
        setattr(res_local, "peel_residual_vars", int(residual_vars))
        setattr(res_local, "peel_residual_rows", int(residual_rows))
        setattr(res_local, "peel_edge_work", int(peel_edge_work))
        setattr(res_local, "peel_dense_xor_ops", int(dense_xor_ops))
        setattr(res_local, "peel_free_dim", int(free_dim))
        setattr(res_local, "peel_extra_llr_added", int(front_end_meta.get("peel_extra_llr_added", 0)))
        return res_local

    A, b = _build_local_subsystem_for_candidate(
        candidate_vars=peel_vars,
        unsat_checks=unsat_checks,
        code_cfg=code_cfg,
        syndrome=syndrome,
    )

    ok_peel, fixed, unresolved_cols, unresolved_rows, peel_edge_work = _peel_reduce_system(A, b)
    if not ok_peel:
        return _make_attempt_result(
            success=False,
            peel_edge_work=int(peel_edge_work),
            residual_vars=int(unresolved_cols.size),
            residual_rows=int(unresolved_rows.size),
        )

    solution = np.zeros((peel_vars.size,), dtype=np.uint8)
    fixed_mask = (fixed >= 0)
    if np.any(fixed_mask):
        solution[fixed_mask] = fixed[fixed_mask].astype(np.uint8, copy=False)

    dense_xor_ops = 0
    free_dim = 0
    residual_vars = int(unresolved_cols.size)
    residual_rows = int(unresolved_rows.size)

    if residual_vars > 0:
        dense_limit = max(0, int(getattr(cfg, "peel_dense_max_vars", 0) or 0))
        if residual_vars > dense_limit:
            return _make_attempt_result(
                success=False,
                peel_edge_work=int(peel_edge_work),
                residual_vars=int(residual_vars),
                residual_rows=int(residual_rows),
            )

        A_red = A[np.asarray(unresolved_rows, dtype=np.int32)][:, np.asarray(unresolved_cols, dtype=np.int32)]
        b_red = b[np.asarray(unresolved_rows, dtype=np.int32)]
        w_red = np.abs(llr_for_sort[peel_vars[np.asarray(unresolved_cols, dtype=np.int32)]]).astype(np.float64, copy=False)

        ok_dense, x_red, free_dim, dense_xor_ops = _gf2_weighted_solve(
            A=A_red,
            b=b_red,
            weights=w_red,
            max_free_enum=int(getattr(cfg, "peel_max_free_enum", 12) or 12),
        )
        if not ok_dense:
            return _make_attempt_result(
                success=False,
                peel_edge_work=int(peel_edge_work),
                dense_xor_ops=int(dense_xor_ops),
                free_dim=int(free_dim),
                residual_vars=int(residual_vars),
                residual_rows=int(residual_rows),
            )

        solution[np.asarray(unresolved_cols, dtype=np.int32)] = x_red.astype(np.uint8, copy=False)

    flipped_vars = peel_vars[solution.astype(bool)]
    if flipped_vars.size == 0:
        return _make_attempt_result(
            success=False,
            peel_edge_work=int(peel_edge_work),
            dense_xor_ops=int(dense_xor_ops),
            free_dim=int(free_dim),
            residual_vars=int(residual_vars),
            residual_rows=int(residual_rows),
        )

    syn_w, e_cnt, uq_cnt, tg_cnt = _syndrome_weight_and_counts_after_flips_from_base(
        base_syndrome=syndrome,
        base_weight=int(initial_syndrome_weight),
        flipped_vars=[int(v) for v in flipped_vars.tolist()],
        code_cfg=code_cfg,
    )
    if int(syn_w) != 0:
        return _make_attempt_result(
            success=False,
            peel_edge_work=int(peel_edge_work),
            dense_xor_ops=int(dense_xor_ops),
            free_dim=int(free_dim),
            residual_vars=int(residual_vars),
            residual_rows=int(residual_rows),
        )

    final_bit_errors = _bit_errors_after_flips_from_base(
        base_bits=hard_bits_snapshot,
        true_bits=frame.c_bits,
        base_bit_errors=int(initial_bit_errors),
        flipped_vars=[int(v) for v in flipped_vars.tolist()],
    )

    return _make_attempt_result(
        success=True,
        flipped_vars=flipped_vars.astype(np.int32, copy=False),
        final_bit_errors=int(final_bit_errors),
        peel_edge_work=int(peel_edge_work),
        dense_xor_ops=int(dense_xor_ops),
        free_dim=int(free_dim),
        residual_vars=int(residual_vars),
        residual_rows=int(residual_rows),
        e_cnt=int(e_cnt),
        uq_cnt=int(uq_cnt),
        tg_cnt=int(tg_cnt),
    )



def run_local_rescue_with_optional_presolver(frame: FrameLog,
                                             sim_cfg: SimulationConfig,
                                             snapshot_iter: int,
                                             cfg: ClusterGrandConfig) -> ClusterGrandResult:
    """Stage-2 wrapper.

    If cfg.pre_solver_mode requests a stronger Receiver-3-style pre-solver,
    try that first; otherwise (or on failure) fall back to the existing
    enumerative GRAND engine unchanged.
    """
    mode = str(getattr(cfg, "pre_solver_mode", "none") or "none").strip().lower()
    res_pre: Optional[ClusterGrandResult] = None

    if mode in ("peel_gf2", "receiver3", "ptg"):
        res_pre = _run_presolver_peel_gf2(
            frame=frame,
            sim_cfg=sim_cfg,
            snapshot_iter=snapshot_iter,
            cfg=cfg,
        )
        if (res_pre is not None) and bool(res_pre.success):
            return res_pre

    res = run_local_grand_on_union_of_clusters(
        frame=frame,
        sim_cfg=sim_cfg,
        snapshot_iter=snapshot_iter,
        cfg=cfg,
    )

    # Attach "attempted but fell back" metadata when a pre-solver mode was enabled.
    if (mode in ("peel_gf2", "receiver3", "ptg")) and (res_pre is not None):
        for attr in [
            "pre_solver_mode_used",
            "pre_solver_attempted",
            "pre_solver_success",
            "peel_candidate_size",
            "peel_residual_vars",
            "peel_residual_rows",
            "peel_edge_work",
            "peel_dense_xor_ops",
            "peel_free_dim",
            "peel_extra_llr_added",
        ]:
            if hasattr(res_pre, attr):
                setattr(res, attr, getattr(res_pre, attr))
    elif mode in ("peel_gf2", "receiver3", "ptg"):
        setattr(res, "pre_solver_mode_used", mode)
        setattr(res, "pre_solver_attempted", 1)
        setattr(res, "pre_solver_success", 0)

    return res


def run_local_grand_on_union_of_clusters(frame: FrameLog,
                                         sim_cfg: SimulationConfig,
                                         snapshot_iter: int,
                                         cfg: ClusterGrandConfig) -> ClusterGrandResult:
    """
    Local GRAND-style search over the *union* of all variable clusters
    induced by the snapshot syndrome at iteration `snapshot_iter`.

    Membership test:
      - incremental syndrome updates (no full syndrome recomputation per pattern)

    IMPORTANT for hardware-time modeling:
      - We distinguish "tested" work (up to first success) from "evaluated" work
        (the actual chunk/batch work performed by the batch-parallel engine).
      - We also expose front-end GRAND work: union sorting length, pattern generation,
        and pattern ordering complexity proxies.

    Notes:
      - The returned ClusterGrandResult keeps the original fields for backward
        compatibility (tested counters).
      - Extra hardware-relevant fields are attached dynamically as attributes:
            patterns_evaluated
            total_v2c_edge_visits_evaluated
            total_unique_checks_visited_evaluated
            total_unique_checks_toggled_evaluated
            union_size, search_size, llr_sort_len
            sum_pattern_weights_generated
            cluster_unsat_edges, cluster_pair_edges
            num_batches_evaluated
            positions_packed_evaluated
            batch_size_used
            llr_source_used
    """
    code_cfg = sim_cfg.code

    # ---- Extract snapshot data ----
    snaps = frame.snapshots
    syn_snaps = snaps.get("syndrome", {})
    hard_snaps = snaps.get("hard_bits", {})
    llr_snaps = snaps.get("llr", {})

    if (snapshot_iter not in syn_snaps or
        snapshot_iter not in hard_snaps or
        snapshot_iter not in llr_snaps):
        raise ValueError(
            f"Snapshot at iter {snapshot_iter} is not fully available "
            f"(keys: syndrome={list(syn_snaps.keys())}, "
            f"hard_bits={list(hard_snaps.keys())}, "
            f"llr={list(llr_snaps.keys())})"
        )

    syndrome = syn_snaps[snapshot_iter]
    hard_bits_snapshot = hard_snaps[snapshot_iter].copy()
    llr_snapshot = llr_snaps[snapshot_iter]

    initial_syndrome_weight = int(syndrome.sum())
    diff_init = (hard_bits_snapshot != frame.c_bits)
    initial_bit_errors = int(diff_init.sum())

    # ---- LLR source selection (posterior vs channel vs mixed) ----
    llr_for_sort, llr_source_used = _resolve_sort_llr_vector(
        llr_snapshot=llr_snapshot,
        llr_channel=getattr(frame, "llr_channel", None),
        cfg=cfg,
    )

    # ---- Cluster-complexity proxy counters from the snapshot syndrome ----
    #   cluster_unsat_edges: total degree sum over unsatisfied checks
    #   cluster_pair_edges : sum over unsat checks of (deg choose 2)
    unsat_checks = np.flatnonzero(syndrome).astype(np.int32)

    cluster_unsat_edges = 0
    cluster_pair_edges = 0
    if unsat_checks.size > 0:
        for j in unsat_checks:
            # FIX: CodeConfig uses checks_to_vars (plural), not check_to_vars
            neigh = code_cfg.checks_to_vars[int(j)]
            d = int(neigh.size)
            cluster_unsat_edges += d
            if d >= 2:
                cluster_pair_edges += int(d * (d - 1) // 2)

    # If already a codeword, nothing to do
    if initial_syndrome_weight == 0:
        res = ClusterGrandResult(
            success=True,
            pattern_weight=0,
            flipped_vars=np.array([], dtype=np.int32),
            patterns_tested=0,
            initial_syndrome_weight=initial_syndrome_weight,
            final_syndrome_weight=initial_syndrome_weight,
            initial_bit_errors=initial_bit_errors,
            final_bit_errors=initial_bit_errors,
            total_v2c_edge_visits=0,
            total_unique_checks_visited=0,
            total_unique_checks_toggled=0,
            patterns_generated=0,
        )
        # Attach meta (mostly zeros)
        setattr(res, "patterns_evaluated", 0)
        setattr(res, "total_v2c_edge_visits_evaluated", 0)
        setattr(res, "total_unique_checks_visited_evaluated", 0)
        setattr(res, "total_unique_checks_toggled_evaluated", 0)
        setattr(res, "union_size", 0)
        setattr(res, "search_size", 0)
        setattr(res, "llr_sort_len", 0)
        setattr(res, "sum_pattern_weights_generated", 0)
        setattr(res, "cluster_unsat_edges", int(cluster_unsat_edges))
        setattr(res, "cluster_pair_edges", int(cluster_pair_edges))
        setattr(res, "num_batches_evaluated", 0)
        setattr(res, "positions_packed_evaluated", 0)
        setattr(res, "batch_size_used", 0)
        setattr(res, "llr_source_used", str(llr_source_used))
        return res

    # Optional guardrail: skip GRAND when syndrome is huge (likely too many errors)
    max_syn = getattr(cfg, "max_syndrome_weight_for_grand", None)
    if isinstance(max_syn, int) and max_syn > 0 and initial_syndrome_weight > int(max_syn):
        res = ClusterGrandResult(
            success=False,
            pattern_weight=-1,
            flipped_vars=np.array([], dtype=np.int32),
            patterns_tested=0,
            initial_syndrome_weight=initial_syndrome_weight,
            final_syndrome_weight=initial_syndrome_weight,
            initial_bit_errors=initial_bit_errors,
            final_bit_errors=initial_bit_errors,
            total_v2c_edge_visits=0,
            total_unique_checks_visited=0,
            total_unique_checks_toggled=0,
            patterns_generated=0,
        )
        setattr(res, "patterns_evaluated", 0)
        setattr(res, "total_v2c_edge_visits_evaluated", 0)
        setattr(res, "total_unique_checks_visited_evaluated", 0)
        setattr(res, "total_unique_checks_toggled_evaluated", 0)
        setattr(res, "union_size", 0)
        setattr(res, "search_size", 0)
        setattr(res, "llr_sort_len", 0)
        setattr(res, "sum_pattern_weights_generated", 0)
        setattr(res, "cluster_unsat_edges", int(cluster_unsat_edges))
        setattr(res, "cluster_pair_edges", int(cluster_pair_edges))
        setattr(res, "num_batches_evaluated", 0)
        setattr(res, "positions_packed_evaluated", 0)
        setattr(res, "batch_size_used", 0)
        setattr(res, "llr_source_used", "skipped")
        return res

    # Reliability + fade gating
    allowed_mask = build_allowed_mask_from_config(frame, sim_cfg, snapshot_iter, cfg)

    # Build unsatisfied-check clusters (structure only)
    clusters = find_variable_clusters_from_syndrome(syndrome, code_cfg)
    if not clusters:
        res = ClusterGrandResult(
            success=False,
            pattern_weight=-1,
            flipped_vars=np.array([], dtype=np.int32),
            patterns_tested=0,
            initial_syndrome_weight=initial_syndrome_weight,
            final_syndrome_weight=initial_syndrome_weight,
            initial_bit_errors=initial_bit_errors,
            final_bit_errors=initial_bit_errors,
            total_v2c_edge_visits=0,
            total_unique_checks_visited=0,
            total_unique_checks_toggled=0,
            patterns_generated=0,
        )
        setattr(res, "patterns_evaluated", 0)
        setattr(res, "total_v2c_edge_visits_evaluated", 0)
        setattr(res, "total_unique_checks_visited_evaluated", 0)
        setattr(res, "total_unique_checks_toggled_evaluated", 0)
        setattr(res, "union_size", 0)
        setattr(res, "search_size", 0)
        setattr(res, "llr_sort_len", 0)
        setattr(res, "sum_pattern_weights_generated", 0)
        setattr(res, "cluster_unsat_edges", int(cluster_unsat_edges))
        setattr(res, "cluster_pair_edges", int(cluster_pair_edges))
        setattr(res, "num_batches_evaluated", 0)
        setattr(res, "positions_packed_evaluated", 0)
        setattr(res, "batch_size_used", 0)
        setattr(res, "llr_source_used", str(llr_source_used))
        return res

    # Union of all cluster variables, then intersect with allowed_mask
    union_vars = np.unique(np.concatenate(clusters)).astype(np.int32)
    union_vars = union_vars[allowed_mask[union_vars]]
    L_full = int(union_vars.size)

    if L_full == 0:
        res = ClusterGrandResult(
            success=False,
            pattern_weight=-1,
            flipped_vars=np.array([], dtype=np.int32),
            patterns_tested=0,
            initial_syndrome_weight=initial_syndrome_weight,
            final_syndrome_weight=initial_syndrome_weight,
            initial_bit_errors=initial_bit_errors,
            final_bit_errors=initial_bit_errors,
            total_v2c_edge_visits=0,
            total_unique_checks_visited=0,
            total_unique_checks_toggled=0,
            patterns_generated=0,
        )
        setattr(res, "patterns_evaluated", 0)
        setattr(res, "total_v2c_edge_visits_evaluated", 0)
        setattr(res, "total_unique_checks_visited_evaluated", 0)
        setattr(res, "total_unique_checks_toggled_evaluated", 0)
        setattr(res, "union_size", int(L_full))
        setattr(res, "search_size", 0)
        setattr(res, "llr_sort_len", int(L_full))
        setattr(res, "sum_pattern_weights_generated", 0)
        setattr(res, "cluster_unsat_edges", int(cluster_unsat_edges))
        setattr(res, "cluster_pair_edges", int(cluster_pair_edges))
        setattr(res, "num_batches_evaluated", 0)
        setattr(res, "positions_packed_evaluated", 0)
        setattr(res, "batch_size_used", 0)
        setattr(res, "llr_source_used", str(llr_source_used))
        return res

    # Determine the search-space budget L, then choose the front-end ranking.
    L = _auto_pick_grand_search_size(L_full, cfg)
    selection_mode = str(getattr(cfg, "selection_mode", "llr") or "llr").strip().lower()

    if selection_mode in ("syndrome_vote", "sv", "receiver2"):
        search_vars, front_end_meta = _select_search_vars_syndrome_vote(
            union_vars=union_vars,
            unsat_checks=unsat_checks,
            code_cfg=code_cfg,
            llr_for_sort=llr_for_sort,
            L=L,
            cfg=cfg,
        )
    else:
        search_vars, front_end_meta = _select_search_vars_llr(
            union_vars=union_vars,
            llr_for_sort=llr_for_sort,
            L=L,
        )

    selection_mode_used = str(front_end_meta.get("selection_mode_used", "llr"))
    sv_seeded_count = int(front_end_meta.get("sv_seeded_count", 0))
    sv_neighbor_visits = int(front_end_meta.get("sv_neighbor_visits", 0))
    sv_score_len = int(front_end_meta.get("sv_score_len", L_full))

    L = int(search_vars.size)

    if L == 0:
        res = ClusterGrandResult(
            success=False,
            pattern_weight=-1,
            flipped_vars=np.array([], dtype=np.int32),
            patterns_tested=0,
            initial_syndrome_weight=initial_syndrome_weight,
            final_syndrome_weight=initial_syndrome_weight,
            initial_bit_errors=initial_bit_errors,
            final_bit_errors=initial_bit_errors,
            total_v2c_edge_visits=0,
            total_unique_checks_visited=0,
            total_unique_checks_toggled=0,
            patterns_generated=0,
        )
        setattr(res, "patterns_evaluated", 0)
        setattr(res, "total_v2c_edge_visits_evaluated", 0)
        setattr(res, "total_unique_checks_visited_evaluated", 0)
        setattr(res, "total_unique_checks_toggled_evaluated", 0)
        setattr(res, "union_size", int(L_full))
        setattr(res, "search_size", int(L))
        setattr(res, "llr_sort_len", int(L_full))
        setattr(res, "sum_pattern_weights_generated", 0)
        setattr(res, "cluster_unsat_edges", int(cluster_unsat_edges))
        setattr(res, "cluster_pair_edges", int(cluster_pair_edges))
        setattr(res, "num_batches_evaluated", 0)
        setattr(res, "positions_packed_evaluated", 0)
        setattr(res, "batch_size_used", 0)
        setattr(res, "llr_source_used", str(llr_source_used))
        setattr(res, "selection_mode_used", str(selection_mode_used))
        setattr(res, "sv_seeded_count", int(sv_seeded_count))
        setattr(res, "sv_neighbor_visits", int(sv_neighbor_visits))
        setattr(res, "sv_score_len", int(sv_score_len))
        return res

    patterns_tested = 0
    patterns_evaluated = 0
    found = False
    found_weight = -1
    found_flipped = np.array([], dtype=np.int32)
    final_syn_weight = initial_syndrome_weight
    final_bit_errors = initial_bit_errors

    # ---- Counters (tested vs evaluated) ----
    total_edge_visits_tested = 0
    total_uniq_checks_visited_tested = 0
    total_uniq_checks_toggled_tested = 0

    total_edge_visits_eval = 0
    total_uniq_checks_visited_eval = 0
    total_uniq_checks_toggled_eval = 0

    # Batch/packing overhead proxy: total pattern positions packed (sum of weights)
    positions_packed_eval = 0
    num_batches_evaluated = 0
    batch_size_used = 0

    if cfg.max_weight <= 0:
        res = ClusterGrandResult(
            success=False,
            pattern_weight=-1,
            flipped_vars=np.array([], dtype=np.int32),
            patterns_tested=0,
            initial_syndrome_weight=initial_syndrome_weight,
            final_syndrome_weight=initial_syndrome_weight,
            initial_bit_errors=initial_bit_errors,
            final_bit_errors=initial_bit_errors,
            total_v2c_edge_visits=0,
            total_unique_checks_visited=0,
            total_unique_checks_toggled=0,
            patterns_generated=0,
        )
        setattr(res, "patterns_evaluated", 0)
        setattr(res, "total_v2c_edge_visits_evaluated", 0)
        setattr(res, "total_unique_checks_visited_evaluated", 0)
        setattr(res, "total_unique_checks_toggled_evaluated", 0)
        setattr(res, "union_size", int(L_full))
        setattr(res, "search_size", int(L))
        setattr(res, "llr_sort_len", int(L_full))
        setattr(res, "sum_pattern_weights_generated", 0)
        setattr(res, "cluster_unsat_edges", int(cluster_unsat_edges))
        setattr(res, "cluster_pair_edges", int(cluster_pair_edges))
        setattr(res, "num_batches_evaluated", 0)
        setattr(res, "positions_packed_evaluated", 0)
        setattr(res, "batch_size_used", 0)
        setattr(res, "llr_source_used", str(llr_source_used))
        return res

    max_w = min(int(cfg.max_weight), int(L))

    # ---- Build all patterns and order them by sum‑|LLR| ----
    pattern_items: List[tuple] = []
    abs_llr_local = np.abs(llr_for_sort[search_vars])

    # This counts the total number of abs-LLR terms summed across all generated patterns.
    # It is a direct proxy for "pattern cost computation" complexity.
    sum_pattern_weights_generated = 0

    for w in range(1, max_w + 1):
        for comb in itertools.combinations(range(L), w):
            # cost = sum_{i in comb} |LLR_i|
            # Hardware ops proxy: w reads + (w-1) adds (we store w; model can convert)
            sum_pattern_weights_generated += int(w)
            cost = float(abs_llr_local[list(comb)].sum())
            pattern_items.append((cost, w, comb))

    pattern_items.sort(key=lambda t: (t[0], t[1]))
    patterns_generated = int(len(pattern_items))

    if not pattern_items:
        res = ClusterGrandResult(
            success=False,
            pattern_weight=-1,
            flipped_vars=np.array([], dtype=np.int32),
            patterns_tested=0,
            initial_syndrome_weight=initial_syndrome_weight,
            final_syndrome_weight=initial_syndrome_weight,
            initial_bit_errors=initial_bit_errors,
            final_bit_errors=initial_bit_errors,
            total_v2c_edge_visits=0,
            total_unique_checks_visited=0,
            total_unique_checks_toggled=0,
            patterns_generated=0,
        )
        setattr(res, "patterns_evaluated", 0)
        setattr(res, "total_v2c_edge_visits_evaluated", 0)
        setattr(res, "total_unique_checks_visited_evaluated", 0)
        setattr(res, "total_unique_checks_toggled_evaluated", 0)
        setattr(res, "union_size", int(L_full))
        setattr(res, "search_size", int(L))
        setattr(res, "llr_sort_len", int(L_full))
        setattr(res, "sum_pattern_weights_generated", int(sum_pattern_weights_generated))
        setattr(res, "cluster_unsat_edges", int(cluster_unsat_edges))
        setattr(res, "cluster_pair_edges", int(cluster_pair_edges))
        setattr(res, "num_batches_evaluated", 0)
        setattr(res, "positions_packed_evaluated", 0)
        setattr(res, "batch_size_used", 0)
        setattr(res, "llr_source_used", str(llr_source_used))
        return res

    # ---- Pattern testing: batch-parallel (evaluated counters) or sequential (tested counters) ----
    use_batch = (
        NUMBA_AVAILABLE
        and hasattr(code_cfg, "_v2c_ptrs")
        and hasattr(code_cfg, "_v2c_checks")
        and getattr(cfg, "batch_size", 0) > 0
    )

    if use_batch:
        base_bits = hard_bits_snapshot.astype(np.uint8)
        true_c_bits = frame.c_bits.astype(np.uint8)
        search_vars_int = search_vars.astype(np.int64)

        base_syn = syndrome.astype(np.uint8)
        base_syn_w = np.int32(initial_syndrome_weight)
        base_bit_err = np.int32(initial_bit_errors)

        total_patterns = int(len(pattern_items))
        max_patterns = int(cfg.max_patterns)
        limit = int(min(total_patterns, max_patterns))
        batch_size = int(getattr(cfg, "batch_size", 256))
        if batch_size <= 0:
            batch_size = limit
        batch_size_used = int(batch_size)

        for start_idx in range(0, limit, batch_size):
            end_idx = min(start_idx + batch_size, limit)
            num_batch = end_idx - start_idx
            if num_batch <= 0:
                continue

            num_batches_evaluated += 1
            patterns_evaluated += int(num_batch)

            # Pack this batch (proxy cost: positions_packed_eval += sum(weights))
            total_positions = 0
            for i in range(start_idx, end_idx):
                total_positions += int(pattern_items[i][1])
            positions_packed_eval += int(total_positions)

            pattern_starts = np.zeros(num_batch, dtype=np.int32)
            pattern_lengths = np.zeros(num_batch, dtype=np.int32)
            pattern_positions = np.zeros(total_positions, dtype=np.int32)

            pos_ptr = 0
            for b in range(num_batch):
                _, w, comb = pattern_items[start_idx + b]
                pattern_starts[b] = pos_ptr
                pattern_lengths[b] = int(w)
                for lp in comb:
                    pattern_positions[pos_ptr] = int(lp)
                    pos_ptr += 1

            syn_w_arr, bit_err_arr, edge_arr, uniq_arr, tog_arr = _grand_eval_batch_numba_incremental(
                base_syn,
                base_syn_w,
                base_bit_err,
                base_bits,
                true_c_bits,
                search_vars_int,
                pattern_starts,
                pattern_lengths,
                pattern_positions,
                code_cfg._v2c_ptrs,
                code_cfg._v2c_checks,
            )

            # Hardware-evaluated counters: full batch always evaluated
            total_edge_visits_eval += int(edge_arr.sum())
            total_uniq_checks_visited_eval += int(np.maximum(uniq_arr, 0).sum())
            total_uniq_checks_toggled_eval += int(tog_arr.sum())

            # Find first success in this batch
            success_rel = -1
            for b in range(num_batch):
                if syn_w_arr[b] == 0:
                    success_rel = b
                    break

            if success_rel >= 0:
                # Tested counters: only up to success
                total_edge_visits_tested += int(edge_arr[:success_rel + 1].sum())
                total_uniq_checks_visited_tested += int(np.maximum(uniq_arr[:success_rel + 1], 0).sum())
                total_uniq_checks_toggled_tested += int(tog_arr[:success_rel + 1].sum())

                global_idx = start_idx + success_rel
                patterns_tested = int(global_idx + 1)

                _, w, comb = pattern_items[global_idx]
                flipped = [int(search_vars[pos]) for pos in comb]

                found = True
                found_weight = int(w)
                found_flipped = np.array(flipped, dtype=np.int32)
                final_syn_weight = 0
                be = int(bit_err_arr[success_rel])
                final_bit_errors = be if be >= 0 else int(initial_bit_errors)
                break
            else:
                # Tested counters: entire batch tested
                total_edge_visits_tested += int(edge_arr.sum())
                total_uniq_checks_visited_tested += int(np.maximum(uniq_arr, 0).sum())
                total_uniq_checks_toggled_tested += int(tog_arr.sum())
                patterns_tested = int(end_idx)

    else:
        # Sequential one‑by‑one testing (incremental membership + counters)
        for _, w, comb in pattern_items:
            patterns_tested += 1
            if patterns_tested > int(cfg.max_patterns):
                break

            flipped = [int(search_vars[pos]) for pos in comb]

            syn_w, e_cnt, uq_cnt, tg_cnt = _syndrome_weight_and_counts_after_flips_from_base(
                base_syndrome=syndrome,
                base_weight=int(initial_syndrome_weight),
                flipped_vars=flipped,
                code_cfg=code_cfg,
            )

            total_edge_visits_tested += int(e_cnt)
            total_uniq_checks_visited_tested += int(uq_cnt)
            total_uniq_checks_toggled_tested += int(tg_cnt)

            # In sequential mode, evaluated == tested
            patterns_evaluated = int(patterns_tested)
            total_edge_visits_eval = int(total_edge_visits_tested)
            total_uniq_checks_visited_eval = int(total_uniq_checks_visited_tested)
            total_uniq_checks_toggled_eval = int(total_uniq_checks_toggled_tested)

            if syn_w == 0:
                bit_err_cand = _bit_errors_after_flips_from_base(
                    base_bits=hard_bits_snapshot,
                    true_bits=frame.c_bits,
                    base_bit_errors=int(initial_bit_errors),
                    flipped_vars=flipped,
                )

                found = True
                found_weight = int(w)
                found_flipped = np.array(flipped, dtype=np.int32)
                final_syn_weight = 0
                final_bit_errors = int(bit_err_cand)
                break

    # Build result (keep legacy semantics: these are TESTED counters)
    res = ClusterGrandResult(
        success=bool(found),
        pattern_weight=int(found_weight) if found else -1,
        flipped_vars=found_flipped,
        patterns_tested=int(patterns_tested),
        initial_syndrome_weight=int(initial_syndrome_weight),
        final_syndrome_weight=int(final_syn_weight),
        initial_bit_errors=int(initial_bit_errors),
        final_bit_errors=int(final_bit_errors),
        total_v2c_edge_visits=int(total_edge_visits_tested),
        total_unique_checks_visited=int(total_uniq_checks_visited_tested),
        total_unique_checks_toggled=int(total_uniq_checks_toggled_tested),
        patterns_generated=int(patterns_generated),
    )

    # Attach evaluated counters + front-end meta for hardware-time model
    setattr(res, "patterns_evaluated", int(patterns_evaluated))
    setattr(res, "total_v2c_edge_visits_evaluated", int(total_edge_visits_eval))
    setattr(res, "total_unique_checks_visited_evaluated", int(total_uniq_checks_visited_eval))
    setattr(res, "total_unique_checks_toggled_evaluated", int(total_uniq_checks_toggled_eval))

    setattr(res, "union_size", int(L_full))
    setattr(res, "search_size", int(L))
    setattr(res, "llr_sort_len", int(L_full))

    setattr(res, "sum_pattern_weights_generated", int(sum_pattern_weights_generated))

    setattr(res, "cluster_unsat_edges", int(cluster_unsat_edges))
    setattr(res, "cluster_pair_edges", int(cluster_pair_edges))

    setattr(res, "num_batches_evaluated", int(num_batches_evaluated))
    setattr(res, "positions_packed_evaluated", int(positions_packed_eval))
    setattr(res, "batch_size_used", int(batch_size_used))

    setattr(res, "llr_source_used", str(llr_source_used))
    setattr(res, "selection_mode_used", str(selection_mode_used))
    setattr(res, "sv_seeded_count", int(sv_seeded_count))
    setattr(res, "sv_neighbor_visits", int(sv_neighbor_visits))
    setattr(res, "sv_score_len", int(sv_score_len))

    return res







# ==================== CELL 28 (DROP-IN REPLACEMENT) ====================
import os

def _get_int_env(name: str, default: int) -> int:
    v = os.environ.get(name, None)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default

def _get_float_env(name: str, default: float) -> float:
    v = os.environ.get(name, None)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default

# ---- Baseline GRAND (keep your current behavior) ----
grand_cfg_awgn = ClusterGrandConfig(
    max_weight=_get_int_env("GRAND_MAX_WEIGHT", 5),
    max_patterns=_get_int_env("GRAND_MAX_PATTERNS", 5000),
    max_bits_from_cluster=None,     # keep auto-pick L
    verbose=False,
    # IMPORTANT: channel-LLR ordering is usually more robust than posterior-LLR ordering
    llr_source=os.environ.get("GRAND_LLR_SOURCE", "channel").strip().lower(),
    pattern_overgen_ratio=_get_float_env("GRAND_OVERGEN", 1.02),
    max_syndrome_weight_for_grand=None,
    batch_size=_get_int_env("GRAND_BATCH_SIZE", 256),
)

# ---- Boost GRAND (only for baseline-exhausted hard frames) ----
GRAND_USE_BOOST = bool(_get_int_env("GRAND_USE_BOOST", 1))

grand_cfg_awgn_boost = ClusterGrandConfig(
    max_weight=_get_int_env("GRAND_BOOST_MAX_WEIGHT", 5),
    max_patterns=_get_int_env("GRAND_BOOST_MAX_PATTERNS", 15000),
    max_bits_from_cluster=None,     # keep auto-pick L
    verbose=False,
    llr_source=os.environ.get("GRAND_LLR_SOURCE", "channel").strip().lower(),
    pattern_overgen_ratio=_get_float_env("GRAND_BOOST_OVERGEN", 1.02),
    # Optional: set to e.g. 0 or leave None; using None keeps boost always eligible
    max_syndrome_weight_for_grand=None,
    batch_size=_get_int_env("GRAND_BATCH_SIZE", 256),
)

# ---- Receiver 2: syndrome-vote + check-cover front-end ----
RUN_RECEIVER2 = bool(_get_int_env("RUN_RECEIVER2", 0))
GRAND_SV_USE_BOOST = bool(_get_int_env("GRAND_SV_USE_BOOST", _get_int_env("GRAND_USE_BOOST", 1)))

grand_cfg_awgn_sv = ClusterGrandConfig(
    max_weight=_get_int_env("GRAND_SV_MAX_WEIGHT", _get_int_env("GRAND_MAX_WEIGHT", 5)),
    max_patterns=_get_int_env("GRAND_SV_MAX_PATTERNS", _get_int_env("GRAND_MAX_PATTERNS", 5000)),
    max_bits_from_cluster=None,     # keep auto-pick L
    verbose=False,
    llr_source=os.environ.get(
        "GRAND_SV_LLR_SOURCE",
        os.environ.get("GRAND_LLR_SOURCE", "channel"),
    ).strip().lower(),
    pattern_overgen_ratio=_get_float_env(
        "GRAND_SV_OVERGEN",
        _get_float_env("GRAND_OVERGEN", 1.02),
    ),
    max_syndrome_weight_for_grand=None,
    batch_size=_get_int_env("GRAND_SV_BATCH_SIZE", _get_int_env("GRAND_BATCH_SIZE", 256)),
    selection_mode="syndrome_vote",
    sv_epsilon=_get_float_env("GRAND_SV_EPSILON", 1e-3),
    sv_check_cover_k=_get_int_env("GRAND_SV_CHECK_COVER_K", 1),
)

grand_cfg_awgn_sv_boost = ClusterGrandConfig(
    max_weight=_get_int_env("GRAND_SV_BOOST_MAX_WEIGHT", _get_int_env("GRAND_BOOST_MAX_WEIGHT", 5)),
    max_patterns=_get_int_env("GRAND_SV_BOOST_MAX_PATTERNS", _get_int_env("GRAND_BOOST_MAX_PATTERNS", 15000)),
    max_bits_from_cluster=None,     # keep auto-pick L
    verbose=False,
    llr_source=os.environ.get(
        "GRAND_SV_LLR_SOURCE",
        os.environ.get("GRAND_LLR_SOURCE", "channel"),
    ).strip().lower(),
    pattern_overgen_ratio=_get_float_env(
        "GRAND_SV_BOOST_OVERGEN",
        _get_float_env("GRAND_BOOST_OVERGEN", 1.02),
    ),
    max_syndrome_weight_for_grand=None,
    batch_size=_get_int_env("GRAND_SV_BATCH_SIZE", _get_int_env("GRAND_BATCH_SIZE", 256)),
    selection_mode="syndrome_vote",
    sv_epsilon=_get_float_env("GRAND_SV_EPSILON", 1e-3),
    sv_check_cover_k=_get_int_env("GRAND_SV_CHECK_COVER_K", 1),
)
# ---- Receiver 3+: syndrome-vote front-end + peel/weighted-GF(2) pre-solver ----
RUN_RECEIVER3 = bool(_get_int_env("RUN_RECEIVER3", 0))
GRAND_PTG_USE_BOOST = bool(_get_int_env("GRAND_PTG_USE_BOOST", _get_int_env("GRAND_SV_USE_BOOST", _get_int_env("GRAND_USE_BOOST", 1))))

grand_cfg_awgn_ptg = ClusterGrandConfig(
    max_weight=_get_int_env("GRAND_PTG_MAX_WEIGHT", _get_int_env("GRAND_SV_MAX_WEIGHT", _get_int_env("GRAND_MAX_WEIGHT", 5))),
    max_patterns=_get_int_env("GRAND_PTG_MAX_PATTERNS", _get_int_env("GRAND_SV_MAX_PATTERNS", _get_int_env("GRAND_MAX_PATTERNS", 5000))),
    max_bits_from_cluster=None,
    verbose=False,
    llr_source=os.environ.get(
        "GRAND_PTG_LLR_SOURCE",
        os.environ.get("GRAND_SV_LLR_SOURCE", os.environ.get("GRAND_LLR_SOURCE", "mixed")),
    ).strip().lower(),
    pattern_overgen_ratio=_get_float_env(
        "GRAND_PTG_OVERGEN",
        _get_float_env("GRAND_SV_OVERGEN", _get_float_env("GRAND_OVERGEN", 1.02)),
    ),
    max_syndrome_weight_for_grand=None,
    batch_size=_get_int_env("GRAND_PTG_BATCH_SIZE", _get_int_env("GRAND_SV_BATCH_SIZE", _get_int_env("GRAND_BATCH_SIZE", 256))),
    selection_mode="syndrome_vote",
    sv_epsilon=_get_float_env("GRAND_PTG_EPSILON", _get_float_env("GRAND_SV_EPSILON", 1e-3)),
    sv_check_cover_k=_get_int_env("GRAND_PTG_CHECK_COVER_K", _get_int_env("GRAND_SV_CHECK_COVER_K", 1)),
    pre_solver_mode="peel_gf2",
    peel_candidate_ratio=_get_float_env("GRAND_PTG_PEEL_RATIO", 1.75),
    peel_max_bits=_get_int_env("GRAND_PTG_PEEL_MAX_BITS", 48),
    peel_dense_max_vars=_get_int_env("GRAND_PTG_PEEL_DENSE_MAX_VARS", 28),
    peel_max_free_enum=_get_int_env("GRAND_PTG_PEEL_MAX_FREE_ENUM", 12),
    peel_extra_llr_bits=_get_int_env("GRAND_PTG_PEEL_EXTRA_LLR_BITS", 8),
)

grand_cfg_awgn_ptg_boost = ClusterGrandConfig(
    max_weight=_get_int_env("GRAND_PTG_BOOST_MAX_WEIGHT", _get_int_env("GRAND_SV_BOOST_MAX_WEIGHT", _get_int_env("GRAND_BOOST_MAX_WEIGHT", 5))),
    max_patterns=_get_int_env("GRAND_PTG_BOOST_MAX_PATTERNS", _get_int_env("GRAND_SV_BOOST_MAX_PATTERNS", _get_int_env("GRAND_BOOST_MAX_PATTERNS", 15000))),
    max_bits_from_cluster=None,
    verbose=False,
    llr_source=os.environ.get(
        "GRAND_PTG_LLR_SOURCE",
        os.environ.get("GRAND_SV_LLR_SOURCE", os.environ.get("GRAND_LLR_SOURCE", "mixed")),
    ).strip().lower(),
    pattern_overgen_ratio=_get_float_env(
        "GRAND_PTG_BOOST_OVERGEN",
        _get_float_env("GRAND_SV_BOOST_OVERGEN", _get_float_env("GRAND_BOOST_OVERGEN", 1.02)),
    ),
    max_syndrome_weight_for_grand=None,
    batch_size=_get_int_env("GRAND_PTG_BATCH_SIZE", _get_int_env("GRAND_SV_BATCH_SIZE", _get_int_env("GRAND_BATCH_SIZE", 256))),
    selection_mode="syndrome_vote",
    sv_epsilon=_get_float_env("GRAND_PTG_EPSILON", _get_float_env("GRAND_SV_EPSILON", 1e-3)),
    sv_check_cover_k=_get_int_env("GRAND_PTG_CHECK_COVER_K", _get_int_env("GRAND_SV_CHECK_COVER_K", 1)),
    pre_solver_mode="none",  # avoid repeating the same pre-solver on the boost path
    peel_candidate_ratio=_get_float_env("GRAND_PTG_PEEL_RATIO", 1.75),
    peel_max_bits=_get_int_env("GRAND_PTG_PEEL_MAX_BITS", 48),
    peel_dense_max_vars=_get_int_env("GRAND_PTG_PEEL_DENSE_MAX_VARS", 28),
    peel_max_free_enum=_get_int_env("GRAND_PTG_PEEL_MAX_FREE_ENUM", 12),
    peel_extra_llr_bits=_get_int_env("GRAND_PTG_PEEL_EXTRA_LLR_BITS", 8),
)
# ======================================================================






# ======================================================================





### CELL number 28-B ###
import os
import math
from dataclasses import dataclass, asdict

def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return int(default)
    try:
        return int(float(v))
    except Exception:
        return int(default)

@dataclass
class HardwareTimingModel:
    # Global clock (MHz)
    fclk_mhz: int = 800

    # LDPC throughputs
    ldpc_edges_per_cycle: int = 64
    ldpc_bits_per_cycle_hd: int = 64
    ldpc_iter_overhead_cycles: int = 20

    # GRAND throughputs (membership test)
    grand_edges_per_cycle: int = 64
    grand_checks_per_cycle: int = 64
    grand_batch_overhead_cycles: int = 200

    # GRAND front-end (LLR sorting + pattern formation)
    grand_abs_per_cycle: int = 64
    grand_sort_elem_log2_per_cycle: int = 64
    grand_cost_add_per_cycle: int = 64
    grand_patsort_patlog2_per_cycle: int = 64

    # Optional: cluster clique-building proxy
    cluster_pair_edges_per_cycle: int = 0

def load_hw_model_from_env() -> HardwareTimingModel:
    return HardwareTimingModel(
        fclk_mhz=_env_int("HW_FCLK_MHZ", 800),

        ldpc_edges_per_cycle=_env_int("HW_LDPC_EDGES_PER_CYCLE", 64),
        ldpc_bits_per_cycle_hd=_env_int("HW_LDPC_BITS_PER_CYCLE_HD", 64),
        ldpc_iter_overhead_cycles=_env_int("HW_LDPC_ITER_OVERHEAD_CYCLES", 20),

        grand_edges_per_cycle=_env_int("HW_GRAND_EDGES_PER_CYCLE", 64),
        grand_checks_per_cycle=_env_int("HW_GRAND_CHECKS_PER_CYCLE", 64),
        grand_batch_overhead_cycles=_env_int("HW_GRAND_BATCH_OVERHEAD_CYCLES", 200),

        grand_abs_per_cycle=_env_int("HW_GRAND_ABS_PER_CYCLE", 64),
        grand_sort_elem_log2_per_cycle=_env_int("HW_GRAND_SORT_ELEMLOG2_PER_CYCLE", 64),
        grand_cost_add_per_cycle=_env_int("HW_GRAND_COST_ADD_PER_CYCLE", 64),
        grand_patsort_patlog2_per_cycle=_env_int("HW_GRAND_PATSORT_PATLOG2_PER_CYCLE", 64),

        cluster_pair_edges_per_cycle=_env_int("HW_CLUSTER_PAIR_EDGES_PER_CYCLE", 0),
    )

def _ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError("Throughput parameter must be > 0")
    if a <= 0:
        return 0
    return int((a + b - 1) // b)

def _safe_log2_int(n: int) -> int:
    """Return ceil(log2(n)) for n>=2, else 0."""
    if n <= 1:
        return 0
    return int(math.ceil(math.log2(float(n))))

def cycles_to_us(cycles: int, hw: HardwareTimingModel) -> float:
    """Convert cycles to microseconds (fclk_mhz cycles per microsecond)."""
    return float(cycles) / float(hw.fclk_mhz)

def _ldpc_total_edges(code_cfg: CodeConfig) -> int:
    """Total Tanner-graph edges E = |{(check,var): H[check,var]=1}|."""
    if hasattr(code_cfg, "_c2v_ptrs"):
        return int(code_cfg._c2v_ptrs[int(code_cfg.M)])
    # Fallback (slower)
    return int(sum(len(cv) for cv in code_cfg.checks_to_vars))

def ldpc_hw_cycles_frame(
    iter_used: int,
    code_cfg: CodeConfig,
    hw: HardwareTimingModel,
    final_vn2cn_executed: bool = False,
) -> int:
    """
    Cycle model for ONE LDPC decoding invocation on ONE frame.

    IMPORTANT:
      - This is an ALGORITHMIC hardware-intent model,
        NOT a model of Python overhead.

    Counted operations:
      (A) Message initialization (VN->CN): write E messages.
      (B) For each executed iteration:
            - check node update        : process E edges
            - variable node update     : process E edges
            - hard decision            : process N bits
            - syndrome computation     : process E edges
            - iteration control overhead (constant)
      (C) VN->CN update pass:
            - for iterations 1..(iter_used-1): ALWAYS executed
            - on the LAST iteration: executed only if the software actually ran it,
              controlled by final_vn2cn_executed.

    This makes the HW-cycle accounting match the *actual control flow* of the
    current software implementation:
      - if early-stop triggers, the last VN->CN update is skipped
      - if early-stop does NOT trigger (e.g. decode fails at max_iters),
        the last VN->CN update *is* executed by the current code.
    """
    it = int(iter_used) if iter_used is not None else 0
    if it <= 0:
        return 0

    N = int(code_cfg.N)
    E = _ldpc_total_edges(code_cfg)

    c_edge = _ceil_div(E, int(hw.ldpc_edges_per_cycle))
    c_hd = _ceil_div(N, int(hw.ldpc_bits_per_cycle_hd))

    # Init: VN->CN message init across all edges
    c_init = c_edge

    # Iteration core (excluding VN->CN)
    c_iter_core = (
        c_edge               # CN update
        + c_edge             # VN update
        + c_hd               # hard decision
        + c_edge             # syndrome compute
        + int(hw.ldpc_iter_overhead_cycles)
    )

    # VN->CN:
    # - Always for the first (it-1) iterations
    # - Plus the last one iff the software executed it (final_vn2cn_executed)
    c_vn2cn = c_edge * max(0, it - 1)
    if bool(final_vn2cn_executed):
        c_vn2cn += c_edge

    return int(c_init + it * c_iter_core + c_vn2cn)

def grand_hw_cycles_from_result(
    result: ClusterGrandResult,
    sim_cfg: SimulationConfig,
    hw: HardwareTimingModel
) -> int:
    """
    Cycle model for ONE GRAND (union-of-clusters) decoding invocation.

    IMPORTANT:
      - Uses the "evaluated" counters, which represent the workload actually
        executed by the chunked/batch-parallel engine. This is the correct
        quantity for serial-parallel (chunked) hardware timing.
      - Front-end costs (LLR abs+sort, pattern cost computation, pattern sort)
        are ALSO charged using metadata recorded by run_local_grand_on_union_of_clusters().
    """
    code_cfg = sim_cfg.code
    M = int(code_cfg.M)

    # --- Evaluated counters (batch-true workload) ---
    edge_visits = int(getattr(result, "total_v2c_edge_visits_evaluated",
                              getattr(result, "total_v2c_edge_visits", 0)))
    checks_toggled = int(getattr(result, "total_unique_checks_toggled_evaluated",
                                 getattr(result, "total_unique_checks_toggled", 0)))
    num_batches = int(getattr(result, "num_batches_evaluated", 0))
    positions_packed = int(getattr(result, "positions_packed_evaluated", 0))

    # --- Front-end meta (always generated, even if early success) ---
    llr_sort_len = int(getattr(result, "llr_sort_len", 0))
    search_sz = int(getattr(result, "search_size", 0))
    patterns_gen = int(getattr(result, "patterns_generated", 0))
    sumw_gen = int(getattr(result, "sum_pattern_weights_generated", 0))
    selection_mode_used = str(getattr(result, "selection_mode_used", "llr") or "llr").strip().lower()
    sv_score_len = int(getattr(result, "sv_score_len", llr_sort_len))

    # --- Cluster extraction proxies ---
    cluster_unsat_edges = int(getattr(result, "cluster_unsat_edges", 0))
    cluster_pair_edges = int(getattr(result, "cluster_pair_edges", 0))

    # --- Throughputs ---
    epc = int(hw.grand_edges_per_cycle)
    cpc = int(hw.grand_checks_per_cycle)
    abs_pc = int(hw.grand_abs_per_cycle)
    sort_pc = int(hw.grand_sort_elem_log2_per_cycle)
    add_pc = int(hw.grand_cost_add_per_cycle)
    psort_pc = int(hw.grand_patsort_patlog2_per_cycle)

    cycles = 0

    # (0) Scan syndrome to identify unsatisfied checks (proxy = M checks)
    cycles += _ceil_div(M, cpc)

    # (1) Cluster extraction proxies
    cycles += _ceil_div(cluster_unsat_edges, epc)
    if int(hw.cluster_pair_edges_per_cycle) > 0:
        cycles += _ceil_div(cluster_pair_edges, int(hw.cluster_pair_edges_per_cycle))

    # (2) LLR abs computations:
    #     - abs for union bits (length llr_sort_len)
    #     - abs AGAIN for the truncated search set (length search_sz)
    cycles += _ceil_div(llr_sort_len, abs_pc)
    cycles += _ceil_div(search_sz, abs_pc)

    # (3) Sort union bits by the front-end key:
    #     - Receiver 1 : |LLR|
    #     - Receiver 2 : eta_v = u_v / (rho_v + epsilon)
    cycles += _ceil_div(llr_sort_len * _safe_log2_int(llr_sort_len), sort_pc)

    # Small extra arithmetic for Receiver 2 score formation. The unsatisfied-check
    # neighbour scan itself is already covered by cluster_unsat_edges above.
    if selection_mode_used in ("syndrome_vote", "sv", "receiver2"):
        cycles += _ceil_div(sv_score_len, add_pc)

    # (4) Receiver-3-style pre-solver cost (if enabled)
    peel_candidate_size = int(getattr(result, "peel_candidate_size", 0))
    peel_edge_work = int(getattr(result, "peel_edge_work", 0))
    peel_dense_xor_ops = int(getattr(result, "peel_dense_xor_ops", 0))
    if peel_candidate_size > 0:
        cycles += _ceil_div(peel_candidate_size, abs_pc)   # extra reliability work on V_peel
    if peel_edge_work > 0:
        cycles += _ceil_div(peel_edge_work, epc)
    if peel_dense_xor_ops > 0:
        cycles += _ceil_div(peel_dense_xor_ops, add_pc)

    # (5) Pattern cost computation: total number of |LLR| terms summed
    cycles += _ceil_div(sumw_gen, add_pc)

    # (6) Pattern ordering sort: O(P log2 P)
    cycles += _ceil_div(patterns_gen * _safe_log2_int(patterns_gen), psort_pc)

    # (6) Packing overhead proxy: sum of weights of evaluated patterns
    cycles += _ceil_div(positions_packed, add_pc)

    # (7) Membership test evaluation workload (evaluated counters)
    cycles += _ceil_div(edge_visits, epc)
    cycles += _ceil_div(checks_toggled, cpc)

    # (8) Chunk/batch overhead (serial-parallel engine granularity)
    cycles += int(num_batches) * int(hw.grand_batch_overhead_cycles)

    return int(cycles)

# A single global instance (each Loky process will have its own copy)
HW_MODEL = load_hw_model_from_env()
print("[HW model] ", asdict(HW_MODEL))




### CELL number 29 ###
from dataclasses import dataclass
from typing import Dict, Any, Optional

MAX_FRAMES_CAP = 1600

@dataclass
class AdaptiveMCConfig:
    """
    Configuration for adaptive Monte-Carlo simulations.

    target_frame_errors:
        Stop when this many frame errors have been observed
        (unless max_frames is hit first).

    min_frames:
        Reserved for future use (no minimum-frame requirement in current runs).

    max_frames:
        Hard cap on number of frames simulated.
    """
    target_frame_errors: int = 200
    min_frames: int = 0
    max_frames: int = MAX_FRAMES_CAP


def run_ldpc_min_sum_adaptive(
    sim_cfg: SimulationConfig,
    dec_cfg: DecoderConfig,
    mc_cfg: AdaptiveMCConfig,
    rng_seed: Optional[int] = None,
    label: Optional[str] = None,
    hw_model: Optional[HardwareTimingModel] = None,
) -> Dict[str, Any]:
    """
    Adaptive LDPC-only Monte-Carlo (channel = sim_cfg.channel.name).

    Hardware timing model:
      - NO CPU wall-time is used.
      - Per-frame LDPC cycles are computed from:
            * iter_used (early stopping / max-iters)
            * code Tanner-graph edge count E
            * throughput parameters in hw_model / environment variables
      - IMPORTANT detail:
            The last VN->CN update pass is charged iff the current *software*
            actually executed it. This occurs when early-stop did not trigger
            (e.g., failure at max_iters, or early_stop=False).
    """
    if hw_model is None:
        hw_model = HW_MODEL

    if rng_seed is None:
        rng_seed = sim_cfg.rng_seed_global + 1234

    global_rng = np.random.default_rng(rng_seed)

    N = sim_cfg.code.N
    max_frames = int(mc_cfg.max_frames)
    target_fe = int(mc_cfg.target_frame_errors)

    per_frame_errs = []
    per_frame_iters = []
    per_frame_hw_cycles = []
    per_frame_hw_time_us = []

    total_bits = 0
    total_bit_errs = 0
    frame_errors = 0
    total_unsat_checks = 0
    total_iters = 0

    total_hw_cycles = 0

    frame_id = 0
    while True:
        if frame_id >= max_frames:
            break

        frame = run_single_frame(sim_cfg, frame_id, global_rng)

        # ---- LDPC decode ----
        ldpc_min_sum_decoder_frame(frame, sim_cfg, dec_cfg)

        num_err = int(frame.error_positions_final.size)
        unsat = int(frame.syndrome_final.sum())
        it_used = int(frame.iter_used if frame.iter_used is not None else 0)

        # ---- Hardware cycles / time ----
        # If early-stop did NOT trigger, the software executed the last VN->CN pass.
        final_vn2cn_executed = bool((not dec_cfg.early_stop) or (unsat != 0))

        hw_cycles = ldpc_hw_cycles_frame(
            it_used,
            sim_cfg.code,
            hw_model,
            final_vn2cn_executed=final_vn2cn_executed,
        )
        hw_time_us = cycles_to_us(hw_cycles, hw_model)

        # Accumulate
        per_frame_errs.append(num_err)
        per_frame_iters.append(it_used)
        per_frame_hw_cycles.append(hw_cycles)
        per_frame_hw_time_us.append(hw_time_us)

        total_bits += int(N)
        total_bit_errs += num_err
        total_unsat_checks += unsat
        total_iters += it_used
        total_hw_cycles += hw_cycles

        if num_err > 0:
            frame_errors += 1

        frame_id += 1

        # Stopping condition
        if frame_errors >= target_fe:
            break

    num_frames = frame_id
    if num_frames == 0:
        return {
            "ber": 0.0,
            "fer": 0.0,
            "avg_unsat_checks": 0.0,
            "avg_iters": 0.0,
            "num_frames": 0,
            "frame_errors": 0,
            "per_frame_errs": np.array([], dtype=np.int32),
            "per_frame_iters": np.array([], dtype=np.int32),
            "per_frame_hw_cycles": np.array([], dtype=np.int64),
            "per_frame_hw_time_us": np.array([], dtype=np.float64),
            "avg_hw_cycles_per_frame": 0.0,
            "avg_hw_time_us_per_frame": 0.0,
        }

    ber = total_bit_errs / total_bits
    fer = frame_errors / num_frames
    avg_unsat_checks = total_unsat_checks / num_frames
    avg_iters = total_iters / num_frames

    avg_hw_cycles_per_frame = total_hw_cycles / num_frames
    avg_hw_time_us_per_frame = cycles_to_us(avg_hw_cycles_per_frame, hw_model)

    if label is None:
        label = (
            f"LDPC-only, max_iters={dec_cfg.max_iters}, "
            f"early_stop={dec_cfg.early_stop}"
        )

    print(f"\n=== Adaptive LDPC-only Monte-Carlo (channel={sim_cfg.channel.name}) ===")
    print(f"Decoder label                 : {label}")
    print(f"SNR (dB)                      : {sim_cfg.channel.snr_db:.2f}")
    print(f"Frames simulated              : {num_frames}")
    print(f"Frame errors                  : {frame_errors} (target={target_fe})")
    print(f"Bit error rate (BER)          : {ber:.3e}")
    print(f"Frame error rate (FER)        : {fer:.3e}")
    print(f"Avg LDPC iterations/frame     : {avg_iters:.2f}")
    print(f"Avg HW cycles/frame           : {avg_hw_cycles_per_frame:.2f}")
    print(f"Avg HW decode time/frame      : {avg_hw_time_us_per_frame:.2f} µs")
    print(f"HW model (fclk_mhz)            : {hw_model.fclk_mhz:.1f} MHz")

    return {
        "ber": float(ber),
        "fer": float(fer),
        "avg_unsat_checks": float(avg_unsat_checks),
        "avg_iters": float(avg_iters),
        "num_frames": int(num_frames),
        "frame_errors": int(frame_errors),
        "per_frame_errs": np.array(per_frame_errs, dtype=np.int32),
        "per_frame_iters": np.array(per_frame_iters, dtype=np.int32),
        "per_frame_hw_cycles": np.array(per_frame_hw_cycles, dtype=np.int64),
        "per_frame_hw_time_us": np.array(per_frame_hw_time_us, dtype=np.float64),
        "avg_hw_cycles_per_frame": float(avg_hw_cycles_per_frame),
        "avg_hw_time_us_per_frame": float(avg_hw_time_us_per_frame),
        "hw_model": asdict(hw_model),
    }






### CELL number 30 ###
from typing import Dict, Any, Optional

def run_hybrid_ldpc_grand_adaptive(
    sim_cfg: SimulationConfig,
    dec_cfg_stage1: DecoderConfig,
    grand_cfg: ClusterGrandConfig,
    snapshot_iter: int,
    mc_cfg: AdaptiveMCConfig,
    rng_seed: Optional[int] = None,
    label: Optional[str] = None,
    hw_model: Optional[HardwareTimingModel] = None,
    grand_cfg_boost: Optional[ClusterGrandConfig] = None,
) -> Dict[str, Any]:
    """
    Hybrid decoder (two-stage):

      Stage-1: LDPC (normalized min-sum) with early stopping, up to max_iters.
      Stage-2: If stage-1 does NOT converge (syndrome != 0),
               run GRAND over the union-of-clusters induced by the snapshot syndrome
               at iteration = snapshot_iter.

    Hardware timing model:
      - NO CPU wall-time is used.
      - Stage-1 cycles are computed from iter_used and the code edge count.
        The last VN->CN pass is charged iff the software executed it:
           * early_stop=False  -> always executed
           * early_stop=True   -> executed only if the stage-1 decode did NOT converge
                                (syn_w != 0).
      - Stage-2 cycles use the *evaluated* (batch-true) GRAND counters:
            total_v2c_edge_visits_evaluated
            total_unique_checks_toggled_evaluated
            num_batches_evaluated
        plus GRAND front-end metadata:
            llr_sort_len, search_size, patterns_generated, sum_pattern_weights_generated,
            cluster_unsat_edges, cluster_pair_edges, positions_packed_evaluated.
    """
    if hw_model is None:
        hw_model = HW_MODEL

    if grand_cfg_boost is None and ("GRAND_USE_BOOST" in globals()) and GRAND_USE_BOOST and ("grand_cfg_awgn_boost" in globals()):
        grand_cfg_boost = grand_cfg_awgn_boost

    if label is None:
        label = f"hyb: LDPC({dec_cfg_stage1.max_iters})+GRAND (snap={snapshot_iter})"

    if rng_seed is None:
        rng_seed = sim_cfg.rng_seed_global + 900 + int(snapshot_iter)

    rng = np.random.default_rng(rng_seed)

    N = int(sim_cfg.code.N)
    max_frames = int(mc_cfg.max_frames)
    target_fe = int(mc_cfg.target_frame_errors)

    # ---- Stage-1 (LDPC) stats ----
    total_bit_errs_stage1 = 0
    frame_errs_stage1 = 0
    total_iters_stage1 = 0

    # ---- Final (LDPC+GRAND) stats ----
    total_bit_errs_after = 0
    frame_errs_after = 0

    # ---- Hardware cycles ----
    total_hw_cycles_stage1 = 0
    total_hw_cycles_grand = 0
    total_hw_cycles_total = 0

    per_frame_iters_stage1 = []
    per_frame_stage1_failed = []

    per_frame_hw_cycles_stage1 = []
    per_frame_hw_cycles_grand = []
    per_frame_hw_cycles_total = []

    # GRAND logs (evaluated workload + front-end)
    per_frame_patterns_tested = []
    per_frame_patterns_evaluated = []
    per_frame_grand_edge_visits_eval = []
    per_frame_grand_checks_toggled_eval = []
    per_frame_grand_num_batches = []
    per_frame_grand_llr_sort_len = []
    per_frame_grand_search_size = []
    per_frame_grand_patterns_generated = []
    per_frame_grand_sumw_generated = []
    per_frame_grand_positions_packed = []
    per_frame_cluster_unsat_edges = []
    per_frame_cluster_pair_edges = []

    # Optional Receiver-3/pre-solver logs
    per_frame_pre_solver_attempted = []
    per_frame_pre_solver_success = []
    per_frame_peel_candidate_size = []
    per_frame_peel_residual_vars = []
    per_frame_peel_dense_xor_ops = []

    n_frames = 0
    frame_id = 0

    while True:
        if frame_id >= max_frames:
            break
        if frame_errs_after >= target_fe:
            break

        # ---- Generate frame ----
        frame = run_single_frame(sim_cfg, frame_id, rng)

        # ---- Stage-1 LDPC ----
        ldpc_min_sum_decoder_frame(frame, sim_cfg, dec_cfg_stage1)

        it1 = int(frame.iter_used if frame.iter_used is not None else 0)
        total_iters_stage1 += it1
        per_frame_iters_stage1.append(it1)

        be1 = int(frame.error_positions_final.size)
        total_bit_errs_stage1 += be1
        if be1 > 0:
            frame_errs_stage1 += 1

        syn_w = int(frame.syndrome_final.sum())
        stage1_failed = (syn_w != 0)
        per_frame_stage1_failed.append(bool(stage1_failed))

        # ---- Stage-1 HW cycles ----
        final_vn2cn_executed_stage1 = bool((not dec_cfg_stage1.early_stop) or (syn_w != 0))

        hw_c_stage1 = ldpc_hw_cycles_frame(
            it1,
            sim_cfg.code,
            hw_model,
            final_vn2cn_executed=final_vn2cn_executed_stage1,
        )

        # Defaults for stage-2
        hw_c_grand = 0
        be_after = be1

        # Defaults for GRAND logs
        pt = 0
        pe = 0
        evis = 0
        ctog = 0
        nb = 0
        llrs = 0
        ss = 0
        pg = 0
        sw = 0
        posp = 0
        cu_e = 0
        cu_p = 0

        ps_attempt = 0
        ps_success = 0
        peel_cand = 0
        peel_res_vars = 0
        peel_xor = 0

        # ---- Stage-2 GRAND ----
        if stage1_failed:
            try:
                res = run_local_rescue_with_optional_presolver(
                    frame=frame,
                    sim_cfg=sim_cfg,
                    snapshot_iter=int(snapshot_iter),
                    cfg=grand_cfg,
                )
            except Exception as e:
                print(f"[WARN] GRAND failed to run on frame {frame_id} at snap={snapshot_iter}: {e}")
                res = None

            # If the first GRAND attempt exhausted its search and failed, optionally boost.
            if (res is not None) and (not bool(res.success)) and (grand_cfg_boost is not None):
                pt1 = int(getattr(res, "patterns_tested", 0))
                pg1 = int(getattr(res, "patterns_generated", 0))
                cap1 = int(getattr(grand_cfg, "max_patterns", pt1))

                exhausted = (pg1 > 0) and (pt1 >= min(pg1, cap1))

                if exhausted:
                    try:
                        res2 = run_local_rescue_with_optional_presolver(
                            frame=frame,
                            sim_cfg=sim_cfg,
                            snapshot_iter=int(snapshot_iter),
                            cfg=grand_cfg_boost,
                        )
                    except Exception as e:
                        print(f"[WARN] BOOST GRAND failed on frame {frame_id} at snap={snapshot_iter}: {e}")
                        res2 = None

                    if res2 is not None:
                        # Combine HW cycles (you actually executed both searches)
                        hw_c_grand = grand_hw_cycles_from_result(res,  sim_cfg, hw_model) + \
                                    grand_hw_cycles_from_result(res2, sim_cfg, hw_model)

                        # Combine logs (so tails reflect total work)
                        pt2 = int(getattr(res2, "patterns_tested", 0))
                        pe2 = int(getattr(res2, "patterns_evaluated", pt2))

                        pt = int(getattr(res, "patterns_tested", 0)) + pt2
                        pe = int(getattr(res, "patterns_evaluated", int(getattr(res, "patterns_tested", 0)))) + pe2

                        # Prefer boosted result for success/failure
                        if bool(res2.success):
                            be_after = int(res2.final_bit_errors)
                        else:
                            be_after = be1

                        # Replace res with res2 so metadata below reflects the boosted attempt
                        res = res2

            if res is not None:
                # If hw_c_grand wasn't already set by the boost-combine path, compute it
                if hw_c_grand == 0:
                    hw_c_grand = grand_hw_cycles_from_result(res, sim_cfg, hw_model)

                if bool(res.success):
                    be_after = int(res.final_bit_errors)
                else:
                    be_after = be1

                # Logs (evaluated workload is the correct one for hardware time)
                if pt == 0:
                    pt = int(getattr(res, "patterns_tested", 0))
                if pe == 0:
                    pe = int(getattr(res, "patterns_evaluated", pt))

                evis = int(getattr(res, "total_v2c_edge_visits_evaluated",
                                getattr(res, "total_v2c_edge_visits", 0)))
                ctog = int(getattr(res, "total_unique_checks_toggled_evaluated",
                                getattr(res, "total_unique_checks_toggled", 0)))
                nb = int(getattr(res, "num_batches_evaluated", 0))

                # Front-end meta (from the final attempt)
                llrs = int(getattr(res, "llr_sort_len", 0))
                ss = int(getattr(res, "search_size", 0))
                pg = int(getattr(res, "patterns_generated", 0))
                sw = int(getattr(res, "sum_pattern_weights_generated", 0))
                posp = int(getattr(res, "positions_packed_evaluated", 0))

                # Cluster proxies
                cu_e = int(getattr(res, "cluster_unsat_edges", 0))
                cu_p = int(getattr(res, "cluster_pair_edges", 0))

                # Optional Receiver-3 / pre-solver metadata
                ps_attempt = int(getattr(res, "pre_solver_attempted", 0))
                ps_success = int(getattr(res, "pre_solver_success", 0))
                peel_cand = int(getattr(res, "peel_candidate_size", 0))
                peel_res_vars = int(getattr(res, "peel_residual_vars", 0))
                peel_xor = int(getattr(res, "peel_dense_xor_ops", 0))

        # ---- Accumulate final stats ----
        total_bit_errs_after += be_after
        if be_after > 0:
            frame_errs_after += 1

        hw_c_total = int(hw_c_stage1) + int(hw_c_grand)

        total_hw_cycles_stage1 += int(hw_c_stage1)
        total_hw_cycles_grand += int(hw_c_grand)
        total_hw_cycles_total += int(hw_c_total)

        per_frame_hw_cycles_stage1.append(int(hw_c_stage1))
        per_frame_hw_cycles_grand.append(int(hw_c_grand))
        per_frame_hw_cycles_total.append(int(hw_c_total))

        per_frame_patterns_tested.append(pt)
        per_frame_patterns_evaluated.append(pe)
        per_frame_grand_edge_visits_eval.append(evis)
        per_frame_grand_checks_toggled_eval.append(ctog)
        per_frame_grand_num_batches.append(nb)

        per_frame_grand_llr_sort_len.append(llrs)
        per_frame_grand_search_size.append(ss)
        per_frame_grand_patterns_generated.append(pg)
        per_frame_grand_sumw_generated.append(sw)
        per_frame_grand_positions_packed.append(posp)

        per_frame_cluster_unsat_edges.append(cu_e)
        per_frame_cluster_pair_edges.append(cu_p)

        per_frame_pre_solver_attempted.append(ps_attempt)
        per_frame_pre_solver_success.append(ps_success)
        per_frame_peel_candidate_size.append(peel_cand)
        per_frame_peel_residual_vars.append(peel_res_vars)
        per_frame_peel_dense_xor_ops.append(peel_xor)

        n_frames += 1
        frame_id += 1

    if n_frames <= 0:
        return {
            "label": label,
            "snr_db": float(sim_cfg.channel.snr_db),
            "n_frames": 0,
            "ber_ldpc": 0.0,
            "fer_ldpc": 0.0,
            "ber_after": 0.0,
            "fer_after": 0.0,
            "ldpc_iters_hybrid_avg": 0.0,
            "avg_hw_cycles_stage1_per_frame": 0.0,
            "avg_hw_cycles_grand_per_frame": 0.0,
            "avg_hw_cycles_total_per_frame": 0.0,
            "avg_hw_time_stage1_us_per_frame": 0.0,
            "avg_hw_time_grand_us_per_frame": 0.0,
            "avg_hw_time_total_us_per_frame": 0.0,
            "hw_model": asdict(hw_model),
        }

    # Stage-1 metrics
    ber_ldpc = total_bit_errs_stage1 / (n_frames * N)
    fer_ldpc = frame_errs_stage1 / n_frames
    avg_iters_stage1 = total_iters_stage1 / n_frames

    # Final metrics
    ber_after = total_bit_errs_after / (n_frames * N)
    fer_after = frame_errs_after / n_frames

    # HW averages
    avg_hw_cycles_stage1 = total_hw_cycles_stage1 / n_frames
    avg_hw_cycles_grand = total_hw_cycles_grand / n_frames
    avg_hw_cycles_total = total_hw_cycles_total / n_frames

    avg_hw_time_stage1_us = cycles_to_us(avg_hw_cycles_stage1, hw_model)
    avg_hw_time_grand_us = cycles_to_us(avg_hw_cycles_grand, hw_model)
    avg_hw_time_total_us = cycles_to_us(avg_hw_cycles_total, hw_model)

    print(f"\n=== Adaptive HYBRID Monte-Carlo (channel={sim_cfg.channel.name}) ===")
    print(f"Decoder label                 : {label}")
    print(f"SNR (dB)                      : {sim_cfg.channel.snr_db:.2f}")
    print(f"Frames simulated              : {n_frames}")
    print(f"Final frame errors            : {frame_errs_after} (target={target_fe})")
    print(f"Stage-1 BER                   : {ber_ldpc:.3e}")
    print(f"Stage-1 FER                   : {fer_ldpc:.3e}")
    print(f"Final BER (LDPC+GRAND)        : {ber_after:.3e}")
    print(f"Final FER (LDPC+GRAND)        : {fer_after:.3e}")
    print(f"Avg stage-1 iters/frame       : {avg_iters_stage1:.2f}")
    print(f"Avg HW time/frame (stage-1)   : {avg_hw_time_stage1_us:.2f} µs")
    print(f"Avg HW time/frame (GRAND)     : {avg_hw_time_grand_us:.2f} µs")
    print(f"Avg HW time/frame (total)     : {avg_hw_time_total_us:.2f} µs")

    return {
        "label": label,
        "snr_db": float(sim_cfg.channel.snr_db),
        "n_frames": int(n_frames),

        # Stage-1 only
        "ber_ldpc": float(ber_ldpc),
        "fer_ldpc": float(fer_ldpc),
        "ldpc_iters_hybrid_avg": float(avg_iters_stage1),

        # Final hybrid
        "ber_after": float(ber_after),
        "fer_after": float(fer_after),

        # HW averages
        "avg_hw_cycles_stage1_per_frame": float(avg_hw_cycles_stage1),
        "avg_hw_cycles_grand_per_frame": float(avg_hw_cycles_grand),
        "avg_hw_cycles_total_per_frame": float(avg_hw_cycles_total),

        "avg_hw_time_stage1_us_per_frame": float(avg_hw_time_stage1_us),
        "avg_hw_time_grand_us_per_frame": float(avg_hw_time_grand_us),
        "avg_hw_time_total_us_per_frame": float(avg_hw_time_total_us),

        # Per-frame HW cycles
        "per_frame_hw_cycles_stage1": np.array(per_frame_hw_cycles_stage1, dtype=np.int64),
        "per_frame_hw_cycles_grand": np.array(per_frame_hw_cycles_grand, dtype=np.int64),
        "per_frame_hw_cycles_total": np.array(per_frame_hw_cycles_total, dtype=np.int64),

        # Per-frame stage-1 logs
        "per_frame_iters_stage1": np.array(per_frame_iters_stage1, dtype=np.int32),
        "per_frame_stage1_failed": np.array(per_frame_stage1_failed, dtype=np.bool_),

        # Per-frame GRAND logs (evaluated)
        "per_frame_patterns_tested": np.array(per_frame_patterns_tested, dtype=np.int32),
        "per_frame_patterns_evaluated": np.array(per_frame_patterns_evaluated, dtype=np.int32),
        "per_frame_grand_edge_visits_evaluated": np.array(per_frame_grand_edge_visits_eval, dtype=np.int64),
        "per_frame_grand_checks_toggled_evaluated": np.array(per_frame_grand_checks_toggled_eval, dtype=np.int64),
        "per_frame_grand_num_batches_evaluated": np.array(per_frame_grand_num_batches, dtype=np.int32),

        # Front-end meta
        "per_frame_grand_llr_sort_len": np.array(per_frame_grand_llr_sort_len, dtype=np.int32),
        "per_frame_grand_search_size": np.array(per_frame_grand_search_size, dtype=np.int32),
        "per_frame_grand_patterns_generated": np.array(per_frame_grand_patterns_generated, dtype=np.int64),
        "per_frame_grand_sum_pattern_weights_generated": np.array(per_frame_grand_sumw_generated, dtype=np.int64),
        "per_frame_grand_positions_packed_evaluated": np.array(per_frame_grand_positions_packed, dtype=np.int64),

        # Cluster proxies
        "per_frame_cluster_unsat_edges": np.array(per_frame_cluster_unsat_edges, dtype=np.int64),
        "per_frame_cluster_pair_edges": np.array(per_frame_cluster_pair_edges, dtype=np.int64),

        # Optional Receiver-3 / pre-solver logs
        "per_frame_pre_solver_attempted": np.array(per_frame_pre_solver_attempted, dtype=np.int8),
        "per_frame_pre_solver_success": np.array(per_frame_pre_solver_success, dtype=np.int8),
        "per_frame_peel_candidate_size": np.array(per_frame_peel_candidate_size, dtype=np.int32),
        "per_frame_peel_residual_vars": np.array(per_frame_peel_residual_vars, dtype=np.int32),
        "per_frame_peel_dense_xor_ops": np.array(per_frame_peel_dense_xor_ops, dtype=np.int64),

        # Stage-2 configuration (kept in the raw pickle; CSV summaries stay unchanged)
        "grand_selection_mode": str(getattr(grand_cfg, "selection_mode", "llr")),
        "grand_llr_source": str(getattr(grand_cfg, "llr_source", "posterior")),
        "grand_sv_check_cover_k": int(getattr(grand_cfg, "sv_check_cover_k", 0)),
        "grand_sv_epsilon": float(getattr(grand_cfg, "sv_epsilon", 0.0)),
        "grand_pre_solver_mode": str(getattr(grand_cfg, "pre_solver_mode", "none")),
        "grand_peel_candidate_ratio": float(getattr(grand_cfg, "peel_candidate_ratio", 1.0)),
        "grand_peel_max_bits": int(getattr(grand_cfg, "peel_max_bits", 0) or 0),

        "hw_model": asdict(hw_model),
    }






### CELL number 30-B ##########################################################################################################################
# Publication-run overrides (multi-SNR + realistic adaptive MC stopping)
# Stop rule per SNR: 200 frame errors OR 160000 frames cap.

import os

# Option A (default): multi-SNR sweep in one job (EDIT this list as needed)
snr_sweep_global = [-5.5, -5, -4.5, -4, -3.5, -3.25, -3, -2.75, -2.5 ]

# Option B (optional): run ONE SNR per Slurm job by exporting SNR_DB
# Example: export SNR_DB=3.0
snr_env = os.environ.get("SNR_DB", "").strip()
if snr_env:
    snr_sweep_global = [float(snr_env)]

mc_cfg = AdaptiveMCConfig(
    target_frame_errors=200,
    min_frames=0,
    max_frames=160000,
)


### CELL number 31 ###
import csv
import pickle

def _dist_stats(arr) -> dict:
    """Return mean/p95/p99/max for a 1D array; NaNs if empty."""
    a = np.asarray(arr)
    if a.size == 0:
        return {"mean": np.nan, "p95": np.nan, "p99": np.nan, "max": np.nan}
    a = a.astype(np.float64, copy=False)
    return {
        "mean": float(a.mean()),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }

def _cycles_to_us_arr(cycles_arr, hw_model_dict) -> np.ndarray:
    a = np.asarray(cycles_arr)
    if a.size == 0:
        return np.array([], dtype=np.float64)
    fclk = float(hw_model_dict.get("fclk_mhz", np.nan))
    if not np.isfinite(fclk) or fclk <= 0:
        return np.array([], dtype=np.float64)
    return a.astype(np.float64, copy=False) / fclk  # cycles_to_us == cycles / fclk_mhz

def save_awgn_results(
    results: Dict[float, Dict[str, Any]],
    output_dir: str,
    prefix: str = "awgn_adaptive_hw",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = f"{prefix}_{timestamp}"

    # ---- RAW pickle (unchanged) ----
    pkl_path = os.path.join(output_dir, base_name + "_raw.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ---- Mean-only summary (unchanged) ----
    csv_path = os.path.join(output_dir, base_name + "_summary.csv")
    fieldnames = [
        "snr_db",
        "decoder",
        "ber",
        "fer",
        "avg_iters",
        "avg_hw_time_us",
        "avg_hw_cycles",
        "avg_hw_time_stage1_us",
        "avg_hw_time_grand_us",
        "ber_stage1",
        "fer_stage1",
    ]

    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        for snr in sorted(results.keys()):
            stats_all = results[snr]
            for dec_name, stats in stats_all.items():
                row = {
                    "snr_db": float(snr),
                    "decoder": str(dec_name),
                    "ber": np.nan,
                    "fer": np.nan,
                    "avg_iters": np.nan,
                    "avg_hw_time_us": np.nan,
                    "avg_hw_cycles": np.nan,
                    "avg_hw_time_stage1_us": np.nan,
                    "avg_hw_time_grand_us": np.nan,
                    "ber_stage1": np.nan,
                    "fer_stage1": np.nan,
                }

                if str(dec_name).startswith("ldpc"):
                    row["ber"] = float(stats.get("ber", np.nan))
                    row["fer"] = float(stats.get("fer", np.nan))
                    row["avg_iters"] = float(stats.get("avg_iters", np.nan))
                    row["avg_hw_cycles"] = float(stats.get("avg_hw_cycles_per_frame", np.nan))
                    row["avg_hw_time_us"] = float(stats.get("avg_hw_time_us_per_frame", np.nan))
                    row["avg_hw_time_stage1_us"] = row["avg_hw_time_us"]
                    row["avg_hw_time_grand_us"] = 0.0
                                        # For LDPC-only, "stage-1" == total
                    row["ber_stage1"] = row["ber"]
                    row["fer_stage1"] = row["fer"]
                else:
                    row["ber"] = float(stats.get("ber_after", np.nan))
                    row["fer"] = float(stats.get("fer_after", np.nan))
                    row["avg_iters"] = float(stats.get("ldpc_iters_hybrid_avg", np.nan))
                    row["avg_hw_cycles"] = float(stats.get("avg_hw_cycles_total_per_frame", np.nan))
                    row["avg_hw_time_us"] = float(stats.get("avg_hw_time_total_us_per_frame", np.nan))
                    row["avg_hw_time_stage1_us"] = float(stats.get("avg_hw_time_stage1_us_per_frame", np.nan))
                    row["avg_hw_time_grand_us"] = float(stats.get("avg_hw_time_grand_us_per_frame", np.nan))
                    row["ber_stage1"] = float(stats.get("ber_ldpc", np.nan))
                    row["fer_stage1"] = float(stats.get("fer_ldpc", np.nan))

                writer.writerow(row)

    # ---- NEW: tails + patterns summary ----
    tails_path = os.path.join(output_dir, base_name + "_summary_tails.csv")
    tails_fields = [
        "snr_db", "decoder", "n_frames",
        "ber", "fer", "avg_iters",

        # Total HW cycles/time tails
        "hw_cycles_mean", "hw_cycles_p95", "hw_cycles_p99", "hw_cycles_max",
        "hw_time_us_mean", "hw_time_us_p95", "hw_time_us_p99", "hw_time_us_max",

        # Hybrid decomposition (NaN for LDPC-only)
        "ber_stage1", "fer_stage1", "grand_invocation_rate",
        "stage1_cycles_mean", "stage1_cycles_p95", "stage1_cycles_p99", "stage1_cycles_max",
        "grand_cycles_mean", "grand_cycles_p95", "grand_cycles_p99", "grand_cycles_max",

        # GRAND pattern stats (overall; NaN for LDPC-only)
        "patterns_tested_mean", "patterns_tested_p95", "patterns_tested_p99", "patterns_tested_max",
        "patterns_evaluated_mean", "patterns_evaluated_p95", "patterns_evaluated_p99", "patterns_evaluated_max",

        # GRAND pattern stats conditional on GRAND invoked (stage1_failed==True)
        "patterns_tested_mean_if_grand", "patterns_tested_p95_if_grand", "patterns_tested_p99_if_grand", "patterns_tested_max_if_grand",
        "patterns_evaluated_mean_if_grand", "patterns_evaluated_p95_if_grand", "patterns_evaluated_p99_if_grand", "patterns_evaluated_max_if_grand",

        # Optional Receiver-3 / pre-solver tails
        "pre_solver_attempt_rate", "pre_solver_success_rate_total", "pre_solver_success_rate_if_attempted",
        "peel_candidate_mean", "peel_candidate_p95", "peel_candidate_max",
        "peel_residual_vars_mean", "peel_residual_vars_p95", "peel_residual_vars_max",
    ]

    with open(tails_path, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=tails_fields)
        writer.writeheader()

        for snr in sorted(results.keys()):
            stats_all = results[snr]
            for dec_name, stats in stats_all.items():
                dec_name = str(dec_name)
                hw_model_dict = stats.get("hw_model", {}) if isinstance(stats, dict) else {}

                row = {k: np.nan for k in tails_fields}
                row["snr_db"] = float(snr)
                row["decoder"] = dec_name

                if dec_name.startswith("ldpc"):
                    n_frames = int(stats.get("num_frames", 0))
                    row["n_frames"] = n_frames
                    row["ber"] = float(stats.get("ber", np.nan))
                    row["fer"] = float(stats.get("fer", np.nan))
                    row["avg_iters"] = float(stats.get("avg_iters", np.nan))

                    cyc = np.asarray(stats.get("per_frame_hw_cycles", []), dtype=np.int64)
                    t_us = np.asarray(stats.get("per_frame_hw_time_us", []), dtype=np.float64)
                    cyc_s = _dist_stats(cyc)
                    t_s = _dist_stats(t_us)

                    row["hw_cycles_mean"] = cyc_s["mean"]
                    row["hw_cycles_p95"]  = cyc_s["p95"]
                    row["hw_cycles_p99"]  = cyc_s["p99"]
                    row["hw_cycles_max"]  = cyc_s["max"]

                    row["hw_time_us_mean"] = t_s["mean"]
                    row["hw_time_us_p95"]  = t_s["p95"]
                    row["hw_time_us_p99"]  = t_s["p99"]
                    row["hw_time_us_max"]  = t_s["max"]

                    # LDPC-only: stage1 == total, grand == 0
                    row["ber_stage1"] = row["ber"]
                    row["fer_stage1"] = row["fer"]
                    row["grand_invocation_rate"] = 0.0

                    row["stage1_cycles_mean"] = row["hw_cycles_mean"]
                    row["stage1_cycles_p95"]  = row["hw_cycles_p95"]
                    row["stage1_cycles_p99"]  = row["hw_cycles_p99"]
                    row["stage1_cycles_max"]  = row["hw_cycles_max"]

                    row["grand_cycles_mean"] = 0.0
                    row["grand_cycles_p95"]  = 0.0
                    row["grand_cycles_p99"]  = 0.0
                    row["grand_cycles_max"]  = 0.0

                    row["pre_solver_attempt_rate"] = 0.0
                    row["pre_solver_success_rate_total"] = 0.0
                    row["pre_solver_success_rate_if_attempted"] = np.nan
                    row["peel_candidate_mean"] = 0.0
                    row["peel_candidate_p95"] = 0.0
                    row["peel_candidate_max"] = 0.0
                    row["peel_residual_vars_mean"] = 0.0
                    row["peel_residual_vars_p95"] = 0.0
                    row["peel_residual_vars_max"] = 0.0

                else:
                    n_frames = int(stats.get("n_frames", 0))
                    row["n_frames"] = n_frames
                    row["ber"] = float(stats.get("ber_after", np.nan))
                    row["fer"] = float(stats.get("fer_after", np.nan))
                    row["avg_iters"] = float(stats.get("ldpc_iters_hybrid_avg", np.nan))

                    row["ber_stage1"] = float(stats.get("ber_ldpc", np.nan))
                    row["fer_stage1"] = float(stats.get("fer_ldpc", np.nan))

                    cyc_stage1 = np.asarray(stats.get("per_frame_hw_cycles_stage1", []), dtype=np.int64)
                    cyc_grand  = np.asarray(stats.get("per_frame_hw_cycles_grand",  []), dtype=np.int64)
                    cyc_total  = np.asarray(stats.get("per_frame_hw_cycles_total",  []), dtype=np.int64)

                    # Total tails
                    cyc_s = _dist_stats(cyc_total)
                    row["hw_cycles_mean"] = cyc_s["mean"]
                    row["hw_cycles_p95"]  = cyc_s["p95"]
                    row["hw_cycles_p99"]  = cyc_s["p99"]
                    row["hw_cycles_max"]  = cyc_s["max"]

                    t_us_total = _cycles_to_us_arr(cyc_total, hw_model_dict)
                    t_s = _dist_stats(t_us_total)
                    row["hw_time_us_mean"] = t_s["mean"]
                    row["hw_time_us_p95"]  = t_s["p95"]
                    row["hw_time_us_p99"]  = t_s["p99"]
                    row["hw_time_us_max"]  = t_s["max"]

                    # Decomposition tails
                    s1 = _dist_stats(cyc_stage1)
                    g  = _dist_stats(cyc_grand)
                    row["stage1_cycles_mean"] = s1["mean"]
                    row["stage1_cycles_p95"]  = s1["p95"]
                    row["stage1_cycles_p99"]  = s1["p99"]
                    row["stage1_cycles_max"]  = s1["max"]
                    row["grand_cycles_mean"]  = g["mean"]
                    row["grand_cycles_p95"]   = g["p95"]
                    row["grand_cycles_p99"]   = g["p99"]
                    row["grand_cycles_max"]   = g["max"]

                    # GRAND invocation + pattern stats
                    stage1_failed = np.asarray(stats.get("per_frame_stage1_failed", []), dtype=np.bool_)
                    if stage1_failed.size > 0:
                        row["grand_invocation_rate"] = float(stage1_failed.mean())
                    else:
                        row["grand_invocation_rate"] = np.nan

                    pt = np.asarray(stats.get("per_frame_patterns_tested", []), dtype=np.int64)
                    pe = np.asarray(stats.get("per_frame_patterns_evaluated", []), dtype=np.int64)

                    pt_s = _dist_stats(pt)
                    pe_s = _dist_stats(pe)
                    row["patterns_tested_mean"] = pt_s["mean"]
                    row["patterns_tested_p95"]  = pt_s["p95"]
                    row["patterns_tested_p99"]  = pt_s["p99"]
                    row["patterns_tested_max"]  = pt_s["max"]

                    row["patterns_evaluated_mean"] = pe_s["mean"]
                    row["patterns_evaluated_p95"]  = pe_s["p95"]
                    row["patterns_evaluated_p99"]  = pe_s["p99"]
                    row["patterns_evaluated_max"]  = pe_s["max"]

                    if stage1_failed.size > 0 and pt.size == stage1_failed.size:
                        pt_if = pt[stage1_failed]
                        pe_if = pe[stage1_failed]
                    else:
                        pt_if = np.array([], dtype=np.int64)
                        pe_if = np.array([], dtype=np.int64)

                    pt_if_s = _dist_stats(pt_if)
                    pe_if_s = _dist_stats(pe_if)

                    row["patterns_tested_mean_if_grand"] = pt_if_s["mean"]
                    row["patterns_tested_p95_if_grand"]  = pt_if_s["p95"]
                    row["patterns_tested_p99_if_grand"]  = pt_if_s["p99"]
                    row["patterns_tested_max_if_grand"]  = pt_if_s["max"]

                    row["patterns_evaluated_mean_if_grand"] = pe_if_s["mean"]
                    row["patterns_evaluated_p95_if_grand"]  = pe_if_s["p95"]
                    row["patterns_evaluated_p99_if_grand"]  = pe_if_s["p99"]
                    row["patterns_evaluated_max_if_grand"]  = pe_if_s["max"]

                    # Optional Receiver-3 / pre-solver tails
                    ps_attempt = np.asarray(stats.get("per_frame_pre_solver_attempted", []), dtype=np.int8)
                    ps_success = np.asarray(stats.get("per_frame_pre_solver_success", []), dtype=np.int8)
                    peel_cand = np.asarray(stats.get("per_frame_peel_candidate_size", []), dtype=np.int64)
                    peel_resv = np.asarray(stats.get("per_frame_peel_residual_vars", []), dtype=np.int64)

                    if ps_attempt.size > 0:
                        row["pre_solver_attempt_rate"] = float(ps_attempt.mean())
                    else:
                        row["pre_solver_attempt_rate"] = np.nan

                    if ps_success.size > 0:
                        row["pre_solver_success_rate_total"] = float(ps_success.mean())
                    else:
                        row["pre_solver_success_rate_total"] = np.nan

                    if ps_attempt.size > 0 and ps_success.size == ps_attempt.size and int(ps_attempt.sum()) > 0:
                        row["pre_solver_success_rate_if_attempted"] = float(ps_success[ps_attempt.astype(bool)].mean())
                    else:
                        row["pre_solver_success_rate_if_attempted"] = np.nan

                    peel_cand_s = _dist_stats(peel_cand)
                    row["peel_candidate_mean"] = peel_cand_s["mean"]
                    row["peel_candidate_p95"]  = peel_cand_s["p95"]
                    row["peel_candidate_max"]  = peel_cand_s["max"]

                    peel_resv_s = _dist_stats(peel_resv)
                    row["peel_residual_vars_mean"] = peel_resv_s["mean"]
                    row["peel_residual_vars_p95"]  = peel_resv_s["p95"]
                    row["peel_residual_vars_max"]  = peel_resv_s["max"]

                writer.writerow(row)

    print(f"[save_awgn_results] Wrote raw results : {pkl_path}")
    print(f"[save_awgn_results] Wrote mean summary: {csv_path}")
    print(f"[save_awgn_results] Wrote tails summary: {tails_path}")



### CELL number 32 ###
import re
import zlib
import csv as _csv

def _parse_csv_float_list(s: str):
    s = (s or "").strip()
    if not s:
        return None
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    return out if out else None

def _stable_u32_seed_from_string(s: str) -> int:
    return int(zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF)

def _snr_sweep_from_env(default_list):
    env = os.environ.get("SNR_SWEEP", "").strip()
    parsed = _parse_csv_float_list(env)
    if parsed is not None:
        return parsed
    return list(default_list)

def _mc_cfg_from_env(default_mc_cfg: AdaptiveMCConfig) -> AdaptiveMCConfig:
    return AdaptiveMCConfig(
        target_frame_errors=_env_int("TARGET_FRAME_ERRORS", int(default_mc_cfg.target_frame_errors)),
        min_frames=int(default_mc_cfg.min_frames),
        max_frames=_env_int("MAX_FRAMES", int(default_mc_cfg.max_frames)),
    )

def _make_random_interleaver(N: int, seed: int) -> InterleaverConfig:
    rng = np.random.default_rng(int(seed))
    pattern = rng.permutation(int(N)).astype(np.int32, copy=False)
    return create_interleaver_from_pattern(pattern, name=f"randperm_seed{seed}_N{N}")

def _5g_lifting_size_sets():
    # 3GPP TS 38.212 lifting sizes grouped into 8 sets (set index 0..7)
    return [
        [2, 4, 8, 16, 32, 64, 128, 256],
        [3, 6, 12, 24, 48, 96, 192, 384],
        [5, 10, 20, 40, 80, 160, 320],
        [7, 14, 28, 56, 112, 224],
        [9, 18, 36, 72, 144, 288],
        [11, 22, 44, 88, 176, 352],
        [13, 26, 52, 104, 208],
        [15, 30, 60, 120, 240],
    ]

def _5g_set_index_from_z(Z: int) -> int:
    Z = int(Z)
    for s_idx, zs in enumerate(_5g_lifting_size_sets()):
        if Z in zs:
            return s_idx
    raise ValueError(f"Invalid 5G lifting factor Z={Z}. Not found in TS 38.212 lifting-size sets.")

def _load_5g_bg_entries(csv_path: str):
    """
    Parse 5G basegraph CSV in the same sparse format used by Sionna:
      Row index ; Column index ; Set0 ; Set1 ; ... ; Set7
    Row index can be blank, meaning "same as previous row".
    Returns: (mb, nb, entries) where entries is list of (r, c, shifts[8]).
    """
    entries = []
    cur_r = None
    max_r = -1
    max_c = -1

    with open(csv_path, "r", newline="") as f:
        reader = _csv.reader(f, delimiter=";")
        # Skip first two header lines
        next(reader, None)
        next(reader, None)

        for row in reader:
            if not row or len(row) < 3:
                continue

            r0 = row[0].strip() if len(row) > 0 else ""
            c0 = row[1].strip() if len(row) > 1 else ""
            if r0 != "":
                cur_r = int(float(r0))
            if cur_r is None:
                continue
            if c0 == "":
                continue
            c_ind = int(float(c0))

            shifts = []
            for k in range(8):
                idx = 2 + k
                tok = row[idx].strip() if idx < len(row) else ""
                shifts.append(int(float(tok)) if tok != "" else 0)

            entries.append((cur_r, c_ind, shifts))
            if cur_r > max_r:
                max_r = cur_r
            if c_ind > max_c:
                max_c = c_ind

    mb = max_r + 1
    nb = max_c + 1
    return mb, nb, entries

def build_5g_qc_code_cfg(
    bg: str,
    Z: int,
    csv_dir: str,
    interleaver_seed: int = 2025,
) -> Tuple[CodeConfig, InterleaverConfig]:
    """
    Build a 5G-style QC-LDPC Tanner graph (pure lifted basegraph).

    IMPORTANT: this is NOT full TS 38.212 rate-matching; it is the lifted BG Tanner graph.
    Encoding mode: all-zero (no generator-matrix construction).
    """
    bg = str(bg).strip().lower()
    Z = int(Z)

    if bg not in ("bg1", "bg2"):
        raise ValueError(f"LDPC_5G_BG must be bg1 or bg2, got: {bg}")

    csv_name = "5G_bg1.csv" if bg == "bg1" else "5G_bg2.csv"
    csv_path = os.path.join(str(csv_dir), csv_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"5G basegraph CSV not found: {csv_path}")

    set_idx = _5g_set_index_from_z(Z)
    mb, nb, entries = _load_5g_bg_entries(csv_path)

    N = nb * Z
    M = mb * Z
    K = N - M
    rate = float(K) / float(N)

    # Build checks_to_vars adjacency
    c2v_lists = [[] for _ in range(M)]
    for r, c, shifts in entries:
        p = int(shifts[set_idx]) % Z
        base_check = int(r) * Z
        base_var = int(c) * Z
        for i in range(Z):
            chk = base_check + i
            var = base_var + ((i + p) % Z)
            c2v_lists[chk].append(var)

    checks_to_vars = [np.asarray(lst, dtype=np.int32) for lst in c2v_lists]

    # Build vars_to_checks and edge positions
    v2c_lists = [[] for _ in range(N)]
    v2c_ep_lists = [[] for _ in range(N)]
    for chk in range(M):
        vs = checks_to_vars[chk]
        for local_e, v in enumerate(vs):
            v_int = int(v)
            v2c_lists[v_int].append(chk)
            v2c_ep_lists[v_int].append(local_e)

    vars_to_checks = [np.asarray(lst, dtype=np.int32) for lst in v2c_lists]
    var_to_checks_edge_pos = [np.asarray(lst, dtype=np.int32) for lst in v2c_ep_lists]

    code_name = f"5g_{bg}_Z{Z}_N{N}_K{K}_R{rate:.3f}"
    code_cfg = CodeConfig(
        code_name=code_name,
        N=int(N),
        K=int(K),
        rate=float(rate),
        H_path=None,
        checks_to_vars=checks_to_vars,
        vars_to_checks=vars_to_checks,
        var_to_checks_edge_pos=var_to_checks_edge_pos,
    )
    code_cfg.M = int(M)
    code_cfg.encoder_mode = "all_zero"

    prepare_code_for_fast_decoding(code_cfg)

    interleaver = _make_random_interleaver(N, interleaver_seed)
    return code_cfg, interleaver



# -------------------- Sionna 5G NR LDPC (38.212) --------------------
def _pcm_to_tanner_neighborhoods(pcm) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Convert a parity-check matrix to Tanner-graph neighborhoods.

    Supports:
      - scipy.sparse CSR/CSC/COO matrices
      - dense numpy arrays

    Returns:
      checks_to_vars, vars_to_checks, var_to_checks_edge_pos
    """
    # Lazy import (SciPy may not be needed for non-sparse paths)
    try:
        import scipy.sparse as sp  # type: ignore
    except Exception:
        sp = None  # type: ignore

    if sp is not None and sp.issparse(pcm):
        pcm_csr = pcm.tocsr()
        M, N = pcm_csr.shape
        checks_to_vars: List[np.ndarray] = []
        for c in range(M):
            s = pcm_csr.indptr[c]
            e = pcm_csr.indptr[c + 1]
            checks_to_vars.append(pcm_csr.indices[s:e].astype(np.int32, copy=False))
    else:
        pcm_dense = np.asarray(pcm)
        if pcm_dense.ndim != 2:
            raise ValueError(f"PCM must be 2D, got shape {pcm_dense.shape}")
        M, N = pcm_dense.shape
        checks_to_vars = [np.flatnonzero(pcm_dense[c]).astype(np.int32) for c in range(M)]

    vars_to_checks_lists: List[List[int]] = [[] for _ in range(N)]
    for c, v_arr in enumerate(checks_to_vars):
        for v in v_arr:
            vars_to_checks_lists[int(v)].append(int(c))
    vars_to_checks: List[np.ndarray] = [np.asarray(lst, dtype=np.int32) for lst in vars_to_checks_lists]

    # For each VN->CN edge, store the local edge index position inside checks_to_vars[cn]
    var_to_checks_edge_pos: List[np.ndarray] = []
    for v in range(N):
        cn_list = vars_to_checks[v]
        pos = np.empty(cn_list.shape[0], dtype=np.int32)
        for i, c in enumerate(cn_list):
            # checks_to_vars[c] is small: linear search is fine
            loc = np.where(checks_to_vars[int(c)] == v)[0]
            pos[i] = int(loc[0])
        var_to_checks_edge_pos.append(pos)
    return checks_to_vars, vars_to_checks, var_to_checks_edge_pos


def build_sionna_5g_nr_code_cfg(
    k_info: int,
    n_tx: int,
    num_bits_per_symbol: int = 1,
    code_name_prefix: str = "sionna5g",
) -> Tuple[CodeConfig, InterleaverConfig]:
    """Build a 5G NR LDPC code config using Sionna's 38.212-compliant LDPC5GEncoder.

    This is the *receiver-side* LDPC graph (PCM) used for syndrome checks and GRAND membership tests.
    The transmitted codeword has length `n_tx` (rate-matched), but the decoding graph length is
    `pcm.shape[1]` (mother code incl. punctured + filler bits).
    """
    if not SIONNA_AVAILABLE:
        raise RuntimeError(
            "Sionna/TensorFlow not available. Install sionna-no-rt + tensorflow in your venv. "
            f"Import error was: {_SIONNA_IMPORT_ERROR}"
        )

    qm = int(num_bits_per_symbol)
    enc = LDPC5GEncoder(k=k_info, n=n_tx, num_bits_per_symbol=qm)
    pcm = enc.pcm  # typically sparse
    checks_to_vars, vars_to_checks, var_to_checks_edge_pos = _pcm_to_tanner_neighborhoods(pcm)
    M, N = pcm.shape

    rate_eff = float(k_info) / float(n_tx)
    code_name = f"{code_name_prefix}_k{k_info}_n{n_tx}_qm{qm}"
    code_cfg = CodeConfig(code_name=code_name, N=int(N), K=int(k_info), rate=rate_eff, H_path=None)
    code_cfg.M = int(M)
    code_cfg.checks_to_vars = checks_to_vars
    code_cfg.vars_to_checks = vars_to_checks
    code_cfg.var_to_checks_edge_pos = var_to_checks_edge_pos
    code_cfg.encoder_mode = "all_zero"  # all-zero CW for symmetry

    # Store Sionna-specific metadata as a plain dict (picklable for joblib)
    out_int_inv = None
    if getattr(enc, "out_int_inv", None) is not None:
        try:
            out_int_inv = np.array(enc.out_int_inv.numpy(), dtype=np.int32)
        except Exception:
            out_int_inv = np.array(enc.out_int_inv, dtype=np.int32)

    code_cfg.sionna = {
        "k_info": int(k_info),
        "n_tx": int(n_tx),
        "qm": qm,
        "z": int(getattr(enc, "z", 0)),
        "k_filler": int(getattr(enc, "k_filler", 0)),
        "out_int_inv": out_int_inv,
    }

    # Precompute internal VN positions corresponding to transmitted bits (sanity probes + logging)
    code_cfg.sionna["tx_pos"] = _sionna5g_internal_tx_positions(code_cfg)

    prepare_code_for_fast_decoding(code_cfg)

    # IMPORTANT: do NOT apply any extra random interleaver on top of 5G rate-matching.
    interleaver = create_identity_interleaver(code_cfg.N)
    return code_cfg, interleaver

def build_gallager36_code_cfg(
    N: int,
    interleaver_seed: int = 2025,
    rng_seed_H: int = 2025,
    dv: int = 3,
    dc: int = 6,
) -> Tuple[CodeConfig, InterleaverConfig]:
    """
    Textbook-style (dv,dc) regular LDPC using a configuration-model Tanner graph.

    Encoding mode: all-zero (no generator-matrix construction).
    """
    N = int(N)
    dv = int(dv)
    dc = int(dc)
    if (N * dv) % dc != 0:
        raise ValueError(f"Need (N*dv) divisible by dc. Got N={N}, dv={dv}, dc={dc}.")

    M = (N * dv) // dc
    K = N - M
    rate = float(K) / float(N)

    rng = np.random.default_rng(int(rng_seed_H))
    total_edges = N * dv

    # Socket model: random pairing of variable sockets and check sockets
    var_sockets = np.repeat(np.arange(N, dtype=np.int32), dv)
    chk_sockets = np.repeat(np.arange(M, dtype=np.int32), dc)
    rng.shuffle(var_sockets)
    rng.shuffle(chk_sockets)

    c2v_lists = [[] for _ in range(M)]
    for e in range(total_edges):
        chk = int(chk_sockets[e])
        var = int(var_sockets[e])
        c2v_lists[chk].append(var)

    checks_to_vars = [np.asarray(lst, dtype=np.int32) for lst in c2v_lists]

    v2c_lists = [[] for _ in range(N)]
    v2c_ep_lists = [[] for _ in range(N)]
    for chk in range(M):
        vs = checks_to_vars[chk]
        for local_e, v in enumerate(vs):
            v_int = int(v)
            v2c_lists[v_int].append(chk)
            v2c_ep_lists[v_int].append(local_e)

    vars_to_checks = [np.asarray(lst, dtype=np.int32) for lst in v2c_lists]
    var_to_checks_edge_pos = [np.asarray(lst, dtype=np.int32) for lst in v2c_ep_lists]

    code_name = f"gallager36_N{N}_K{K}_R{rate:.3f}"
    code_cfg = CodeConfig(
        code_name=code_name,
        N=int(N),
        K=int(K),
        rate=float(rate),
        H_path=None,
        checks_to_vars=checks_to_vars,
        vars_to_checks=vars_to_checks,
        var_to_checks_edge_pos=var_to_checks_edge_pos,
    )
    code_cfg.M = int(M)
    code_cfg.encoder_mode = "all_zero"

    prepare_code_for_fast_decoding(code_cfg)

    interleaver = _make_random_interleaver(N, interleaver_seed)
    return code_cfg, interleaver

def _sanitize_prefix(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s if s else "awgn_adaptive_hw"


def _set_blas_threads_env_defaults():
    """Prevent BLAS/OpenMP oversubscription when we also use Numba + multiprocessing."""
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, "1")


def _snr_parallel_plan(snr_sweep: List[float], total_threads: int):
    """Plan SNR-parallelism (processes over SNRs + Numba threads inside each process).

    Environment knobs (all optional):
      - PARALLEL_OVER_SNR: 1/0 (default 1)
      - SNR_PARALLEL_JOBS: number of processes (default auto)
      - SNR_THREADS_PER_WORKER: Numba threads per worker (default auto)
      - SNR_MIN_THREADS_PER_WORKER: minimum threads/worker for auto plan (default 8)
      - JOBLIB_BACKEND: loky|multiprocessing (default loky)
    """
    n_tasks = int(len(snr_sweep))
    total_threads = int(max(1, total_threads))

    if n_tasks <= 1:
        return False, 1, total_threads, "serial"

    if not JOBLIB_AVAILABLE:
        return False, 1, total_threads, "serial"

    if _env_int("PARALLEL_OVER_SNR", 1) != 1:
        return False, 1, total_threads, "serial"

    backend = (os.environ.get("JOBLIB_BACKEND", "loky") or "loky").strip().lower()
    if backend not in ("loky", "multiprocessing"):
        backend = "loky"

    # User overrides
    n_jobs_env = _env_int("SNR_PARALLEL_JOBS", 0)
    t_per_env = _env_int("SNR_THREADS_PER_WORKER", 0)
    min_threads = max(1, _env_int("SNR_MIN_THREADS_PER_WORKER", 8))

    if t_per_env > 0:
        threads_per_job = int(max(1, min(t_per_env, total_threads)))
        n_jobs = int(max(1, min(n_tasks, total_threads // threads_per_job)))
        if n_jobs_env > 0:
            n_jobs = int(max(1, min(n_tasks, n_jobs_env)))
            threads_per_job = int(max(1, total_threads // n_jobs))
    else:
        if n_jobs_env > 0:
            n_jobs = int(max(1, min(n_tasks, n_jobs_env)))
        else:
            n_jobs = int(max(1, min(n_tasks, total_threads // min_threads)))
        threads_per_job = int(max(1, total_threads // n_jobs))

    n_jobs = int(max(1, min(n_tasks, n_jobs)))
    threads_per_job = int(max(1, min(total_threads, threads_per_job)))

    # Parallel only if at least 2 processes
    use_parallel = (n_jobs > 1)
    return use_parallel, n_jobs, threads_per_job, backend


def run_awgn_sweep_for_code(
    code_cfg: CodeConfig,
    interleaver: InterleaverConfig,
    snr_sweep: List[float],
    mc_cfg_local: AdaptiveMCConfig,
    output_dir: str,
    alpha: float = 0.8,
) -> Dict[float, Dict[str, Any]]:
    """Run AWGN sweeps and save outputs via save_awgn_results.

    Parallelism:
      - per-kernel Numba parallelism (already in the decoder + GRAND kernels)
      - OPTIONAL process-level parallelism over SNRs (joblib), with Numba threads
        partitioned across workers to saturate the SLURM allocation.
    """
    _set_blas_threads_env_defaults()

    channel_name = os.environ.get("CHANNEL_NAME", "SIONNA_TDL").strip() or "SIONNA_TDL"

    # Enforce cleaned channel support (fail fast before spawning joblib workers)
    if channel_name.strip().upper() not in ("SIONNA_TDL", "TDL"):
        raise ValueError(
            f"Unsupported CHANNEL_NAME='{channel_name}'. "
            "This cleaned script supports only CHANNEL_NAME=SIONNA_TDL."
        )
    channel_name = "SIONNA_TDL"

    stage1_list_env = str(os.environ.get("STAGE1_ITERS", "4,8,15"))
    stage1_list = [int(x) for x in stage1_list_env.split(",") if x.strip()]
    stage1_list = sorted(set([it for it in stage1_list if it > 0]))


    ldpc_list_env = str(os.environ.get("LDPC_ITERS", "4,8,15,20,100"))
    ldpc_list = [int(x) for x in ldpc_list_env.split(",") if x.strip()]
    ldpc_list = sorted(set([it for it in ldpc_list if it > 0]))

    

    base_seed = _env_int("RNG_SEED_GLOBAL", 12345) + _stable_u32_seed_from_string(code_cfg.code_name)
    run_receiver2 = bool(_env_int("RUN_RECEIVER2", 0))
    run_receiver3 = bool(_env_int("RUN_RECEIVER3", 0))

    results: Dict[float, Dict[str, Any]] = {}

    # Total threads available (as detected in CELL 1); fallback to SLURM/OS
    total_threads = int(globals().get("NUMBA_THREADS", 0) or _detect_num_threads())
    use_parallel, n_jobs, threads_per_job, backend = _snr_parallel_plan(snr_sweep, total_threads)

    if use_parallel:
        print(f"[run_awgn_sweep_for_code] SNR-parallel ON: n_jobs={n_jobs}, threads/worker={threads_per_job}, backend={backend}")
    else:
        # Serial: give Numba the full thread budget
        if NUMBA_AVAILABLE and set_num_threads is not None:
            try:
                set_num_threads(total_threads)
            except Exception:
                pass
        print(f"[run_awgn_sweep_for_code] SNR-parallel OFF (serial); Numba threads={total_threads}")

    def _run_one_snr(snr_db: float):
        snr_db = float(snr_db)

                # In a worker process: partition Numba threads to avoid oversubscription
        if use_parallel and NUMBA_AVAILABLE and set_num_threads is not None:
            try:
                set_num_threads(int(threads_per_job))
            except Exception:
                pass

        # LDPC-only sim config (no snapshots needed)
        sim_cfg_ldpc = SimulationConfig(
            code=code_cfg,
            channel=ChannelConfig(name=channel_name, snr_db=snr_db),
            interleaver=interleaver,
            rng_seed_global=int(base_seed),
            snapshot_iters=[],
        )

        per_snr = {}

        # Scenario 1: Legacy LDPC-only baselines (no GRAND)
        for it in ldpc_list:
            dec_name = f"ldpc{int(it)}"
            seed = int((base_seed + 1_000 + int(round(snr_db * 100.0)) + int(it)) & 0xFFFFFFFF)
            dec_cfg = DecoderConfig(max_iters=int(it), alpha=float(alpha), early_stop=True)
            per_snr[dec_name] = run_ldpc_min_sum_adaptive(
                sim_cfg=sim_cfg_ldpc,
                dec_cfg=dec_cfg,
                mc_cfg=mc_cfg_local,
                rng_seed=seed,
                label=dec_name,
            )

        # Scenario 3: Complete hybrid (Receiver 1 = LLR-ranked GRAND rescue)
        for it in stage1_list:
            dec_name = f"hyb{int(it)}"
            seed = int((base_seed + 10_000 + int(round(snr_db * 100.0)) + int(it)) & 0xFFFFFFFF)
            dec_cfg = DecoderConfig(max_iters=int(it), alpha=float(alpha), early_stop=True)

            sim_cfg_hyb = SimulationConfig(
                code=code_cfg,
                channel=ChannelConfig(name=channel_name, snr_db=snr_db),
                interleaver=interleaver,
                rng_seed_global=int(base_seed),
                snapshot_iters=[int(it)],  # snapshot only what GRAND needs
            )

            per_snr[dec_name] = run_hybrid_ldpc_grand_adaptive(
                sim_cfg=sim_cfg_hyb,
                dec_cfg_stage1=dec_cfg,
                grand_cfg=grand_cfg_awgn,
                snapshot_iter=int(it),
                mc_cfg=mc_cfg_local,
                rng_seed=seed,
                label=dec_name,
                grand_cfg_boost=(grand_cfg_awgn_boost if GRAND_USE_BOOST else None),
            )

        # Scenario 4: Receiver 2 (syndrome-vote + check-cover front-end)
        if run_receiver2:
            for it in stage1_list:
                dec_name = f"hybsv{int(it)}"
                seed = int((base_seed + 20_000 + int(round(snr_db * 100.0)) + int(it)) & 0xFFFFFFFF)
                dec_cfg = DecoderConfig(max_iters=int(it), alpha=float(alpha), early_stop=True)

                sim_cfg_hyb = SimulationConfig(
                    code=code_cfg,
                    channel=ChannelConfig(name=channel_name, snr_db=snr_db),
                    interleaver=interleaver,
                    rng_seed_global=int(base_seed),
                    snapshot_iters=[int(it)],  # snapshot only what GRAND needs
                )

                per_snr[dec_name] = run_hybrid_ldpc_grand_adaptive(
                    sim_cfg=sim_cfg_hyb,
                    dec_cfg_stage1=dec_cfg,
                    grand_cfg=grand_cfg_awgn_sv,
                    snapshot_iter=int(it),
                    mc_cfg=mc_cfg_local,
                    rng_seed=seed,
                    label=dec_name,
                    grand_cfg_boost=(grand_cfg_awgn_sv_boost if GRAND_SV_USE_BOOST else None),
                )

        # Scenario 5: Receiver 3+ (syndrome-vote + peel/weighted-GF(2) pre-solver + GRAND fallback)
        if run_receiver3:
            for it in stage1_list:
                dec_name = f"hybptg{int(it)}"
                seed = int((base_seed + 30_000 + int(round(snr_db * 100.0)) + int(it)) & 0xFFFFFFFF)
                dec_cfg = DecoderConfig(max_iters=int(it), alpha=float(alpha), early_stop=True)

                sim_cfg_hyb = SimulationConfig(
                    code=code_cfg,
                    channel=ChannelConfig(name=channel_name, snr_db=snr_db),
                    interleaver=interleaver,
                    rng_seed_global=int(base_seed),
                    snapshot_iters=[int(it)],
                )

                per_snr[dec_name] = run_hybrid_ldpc_grand_adaptive(
                    sim_cfg=sim_cfg_hyb,
                    dec_cfg_stage1=dec_cfg,
                    grand_cfg=grand_cfg_awgn_ptg,
                    snapshot_iter=int(it),
                    mc_cfg=mc_cfg_local,
                    rng_seed=seed,
                    label=dec_name,
                    grand_cfg_boost=(grand_cfg_awgn_ptg_boost if GRAND_PTG_USE_BOOST else None),
                )

        return snr_db, per_snr

    if use_parallel:
        tasks = Parallel(n_jobs=n_jobs, backend=backend, prefer="processes", batch_size=1)(
            delayed(_run_one_snr)(snr_db) for snr_db in snr_sweep
        )
        for snr_db, per_snr in tasks:
            results[float(snr_db)] = per_snr
    else:
        for snr_db in snr_sweep:
            snr_db_f, per_snr = _run_one_snr(snr_db)
            results[float(snr_db_f)] = per_snr

    prefix = _sanitize_prefix(f"{channel_name.lower()}_{code_cfg.code_name}_hybrid")
    save_awgn_results(results, output_dir=output_dir, prefix=prefix)
    return results


def _run_experiments_main():
    # Guarded entry point for SLURM (safe for joblib multiprocessing)
    if int(float(os.environ.get("RUN_EXPERIMENTS", "0") or "0")) != 1:
        return

    run_codes_env = os.environ.get("RUN_CODES", "").strip()
    run_codes = [c.strip().lower() for c in run_codes_env.split(",") if c.strip()] if run_codes_env else []

    if not run_codes:
        run_codes = ["sionna5g"]

    snr_sweep = _snr_sweep_from_env(snr_sweep_global)
    mc_cfg_local = _mc_cfg_from_env(mc_cfg)

    out_dir = os.environ.get("RESULTS_DIR", "./results")
    os.makedirs(out_dir, exist_ok=True)

    interleaver_seed = _env_int("INTERLEAVER_SEED", 2025)

    print(f"[RUN_EXPERIMENTS] codes={run_codes}  snr_sweep={snr_sweep}  out_dir={out_dir}")
    print(f"[RUN_EXPERIMENTS] mc_cfg={mc_cfg_local}")

    for code_token in run_codes:
        # Only keep the Sionna 5G NR LDPC path (requested).
        if code_token not in ("sionna5g", "5g_sionna", "sionna_5g"):
            print(f"[RUN_EXPERIMENTS] WARNING: unsupported RUN_CODES token '{code_token}' (only 'sionna5g' is supported) -> skipping.")
            continue

        # 5G NR LDPC from Sionna (38.212 compliant, RV=0-style rate-matching)
        k_info = _env_int("SIONNA_5G_K", 1024)
        n_tx = _env_int("SIONNA_5G_N", 2048)
        qm = _env_int("SIONNA_5G_QM", 1)
        code_cfg_local, interleaver_local = build_sionna_5g_nr_code_cfg(
            k_info=k_info,
            n_tx=n_tx,
            num_bits_per_symbol=qm,
            code_name_prefix="sionna5g",
        )

        print(f"[RUN_EXPERIMENTS] Built code: {code_cfg_local.code_name}")
        run_awgn_sweep_for_code(
            code_cfg=code_cfg_local,
            interleaver=interleaver_local,
            snr_sweep=snr_sweep,
            mc_cfg_local=mc_cfg_local,
            output_dir=out_dir,
            alpha=0.8,
        )


if __name__ == "__main__":
    # sbatch/CLI convenience:
    #   python <script.py> <output_dir> <code_token>
    # If args are provided, they override the corresponding environment variables.
    if len(sys.argv) >= 2 and sys.argv[1].strip():
        os.environ["RESULTS_DIR"] = sys.argv[1].strip()
    if len(sys.argv) >= 3 and sys.argv[2].strip():
        os.environ["RUN_CODES"] = sys.argv[2].strip()

    # Default: run when executed as a script
    os.environ.setdefault("RUN_EXPERIMENTS", "1")
    _run_experiments_main()
