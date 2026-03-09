#!/bin/bash

# ===== Slurm resources =====
# NOTE: You can override these at submission time if needed.
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs results

module --force purge
module load StdEnv/2023 python/3.11 scipy-stack/2024a

# ===== Activate a persistent venv (recommended) =====
# Create it once on a login node (or inside salloc) and reuse it.
VENV_DIR="${VENV_DIR:-$SCRATCH/venvs/grand5g}"
if [[ ! -d "$VENV_DIR" ]]; then
  echo "ERROR: venv not found: $VENV_DIR"
  echo "Create it first (see venv setup instructions)."
  exit 1
fi
source "$VENV_DIR/bin/activate"
# Use local node storage for temp/cache (faster than shared FS)
export TMPDIR="${TMPDIR:-${SLURM_TMPDIR:-/tmp}}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$TMPDIR/numba_cache}"
export JOBLIB_TEMP_FOLDER="${JOBLIB_TEMP_FOLDER:-$TMPDIR/joblib}"
mkdir -p "$NUMBA_CACHE_DIR" "$JOBLIB_TEMP_FOLDER"

# ===== Performance knobs =====
export USE_GPU=${USE_GPU:-0}
if [[ "${USE_GPU}" != "1" ]]; then
  export CUDA_VISIBLE_DEVICES="-1"
  export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-2}
fi
export TF_INTRA_OP=${TF_INTRA_OP:-1}
export TF_INTER_OP=${TF_INTER_OP:-1}

export MAX_NB_THREADS=${MAX_NB_THREADS:-${SLURM_CPUS_PER_TASK:-64}}
# Threading knobs consumed by HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py
export LDPC_GRAND_NUM_THREADS=${LDPC_GRAND_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-64}}
export SNR_MIN_THREADS_PER_WORKER=${SNR_MIN_THREADS_PER_WORKER:-16}
export PARALLEL_OVER_SNR=${PARALLEL_OVER_SNR:-1}

# Prevent BLAS/OpenMP oversubscription (Numba uses its own threadpool)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ===== Experiment knobs =====
# Channel selection: SIONNA_TDL
export CHANNEL_NAME=SIONNA_TDL

# 5G NR LDPC (Sionna) params
export SIONNA_5G_K=${SIONNA_5G_K:-1024}
export SIONNA_5G_N=${SIONNA_5G_N:-2048}
export SIONNA_5G_QM=${SIONNA_5G_QM:-1}

# 3GPP TDL channel params (Sionna)
export SIONNA_TDL_MODEL=${SIONNA_TDL_MODEL:-A}
export SIONNA_TDL_DELAY_SPREAD_S=${SIONNA_TDL_DELAY_SPREAD_S:-3e-7}
export SIONNA_TDL_CARRIER_FREQUENCY_HZ=${SIONNA_TDL_CARRIER_FREQUENCY_HZ:-3.5e9}
export SIONNA_TDL_MIN_SPEED=${SIONNA_TDL_MIN_SPEED:-0.0}
export SIONNA_TDL_MAX_SPEED=${SIONNA_TDL_MAX_SPEED:-0.0}

# Simple OFDM mapping used for the TDL channel
export SIONNA_OFDM_FFT_SIZE=${SIONNA_OFDM_FFT_SIZE:-256}
export SIONNA_OFDM_SUBCARRIER_SPACING_HZ=${SIONNA_OFDM_SUBCARRIER_SPACING_HZ:-15000}

# Optional impairment knob (keep at 0.0 for now)
export SIONNA_CFO_HZ=${SIONNA_CFO_HZ:-0.0}

# LLR clamp used for filler-bit insertion
export SIONNA_LLR_MAX=${SIONNA_LLR_MAX:-50.0}

# ===== GRAND knobs =====
# Receiver 1 / baseline GRAND knobs
# Use channel LLRs for GRAND bit-ranking (recommended)
export GRAND_LLR_SOURCE=${GRAND_LLR_SOURCE:-channel}

# Defaults kept close to your current setup so legacy outputs still appear.
export GRAND_MAX_WEIGHT=${GRAND_MAX_WEIGHT:-6}
export GRAND_MAX_PATTERNS=${GRAND_MAX_PATTERNS:-8000}
export GRAND_USE_BOOST=${GRAND_USE_BOOST:-1}
export GRAND_BOOST_MAX_WEIGHT=${GRAND_BOOST_MAX_WEIGHT:-5}
export GRAND_BOOST_MAX_PATTERNS=${GRAND_BOOST_MAX_PATTERNS:-30000}
export GRAND_BATCH_SIZE=${GRAND_BATCH_SIZE:-256}

# Receiver 2: syndrome-vote + check-cover front-end.
# These defaults are intentionally more aggressive than Receiver 1 to maximize BER/BLER gain.
export RUN_RECEIVER2=${RUN_RECEIVER2:-1}
export GRAND_SV_LLR_SOURCE=${GRAND_SV_LLR_SOURCE:-${GRAND_LLR_SOURCE:-channel}}
export GRAND_SV_CHECK_COVER_K=${GRAND_SV_CHECK_COVER_K:-1}
export GRAND_SV_EPSILON=${GRAND_SV_EPSILON:-1e-3}
export GRAND_SV_MAX_WEIGHT=${GRAND_SV_MAX_WEIGHT:-7}
export GRAND_SV_MAX_PATTERNS=${GRAND_SV_MAX_PATTERNS:-20000}
export GRAND_SV_USE_BOOST=${GRAND_SV_USE_BOOST:-1}
export GRAND_SV_BOOST_MAX_WEIGHT=${GRAND_SV_BOOST_MAX_WEIGHT:-7}
export GRAND_SV_BOOST_MAX_PATTERNS=${GRAND_SV_BOOST_MAX_PATTERNS:-120000}
export GRAND_SV_BATCH_SIZE=${GRAND_SV_BATCH_SIZE:-${GRAND_BATCH_SIZE:-256}}

# Sweep + Monte-Carlo limits
# ---- IMPORTANT: avoid accidental overrides from your login-shell environment ----
# Default behavior: FORCE_DEFAULTS=1 -> ignore any inherited env vars for these knobs.
# To intentionally override at submission time:
#   FORCE_DEFAULTS=0 SNR_SWEEP="..." LDPC_ITERS="..." STAGE1_ITERS="..." sbatch run_ms.sbatch
export FORCE_DEFAULTS=${FORCE_DEFAULTS:-1}
if [[ "${FORCE_DEFAULTS}" == "1" ]]; then
  unset SNR_SWEEP LDPC_ITERS STAGE1_ITERS TARGET_FRAME_ERRORS MAX_FRAMES || true
fi

# Scenario iteration sets
export STAGE1_ITERS=${STAGE1_ITERS:-"4,8,15"}         # hybrid stage-1 / hybrid snapshots
export LDPC_ITERS=${LDPC_ITERS:-"4,8,15,20,100"}      # legacy LDPC-only baselines
export SNR_SWEEP=${SNR_SWEEP:-"7,7.5,8,8.5,9,9.5,10,10.5,11"}
export TARGET_FRAME_ERRORS=${TARGET_FRAME_ERRORS:-200}
export MAX_FRAMES=${MAX_FRAMES:-200000}

echo "[SBATCH CONFIG] FORCE_DEFAULTS=${FORCE_DEFAULTS}"
echo "[SBATCH CONFIG] SNR_SWEEP=${SNR_SWEEP}"
echo "[SBATCH CONFIG] LDPC_ITERS=${LDPC_ITERS}"
echo "[SBATCH CONFIG] STAGE1_ITERS=${STAGE1_ITERS}"
echo "[SBATCH CONFIG] TARGET_FRAME_ERRORS=${TARGET_FRAME_ERRORS}  MAX_FRAMES=${MAX_FRAMES}"
echo "[SBATCH CONFIG] RUN_RECEIVER2=${RUN_RECEIVER2}  GRAND_SV_CHECK_COVER_K=${GRAND_SV_CHECK_COVER_K}  GRAND_SV_EPSILON=${GRAND_SV_EPSILON}"
echo "[SBATCH CONFIG] GRAND_SV_MAX_WEIGHT=${GRAND_SV_MAX_WEIGHT}  GRAND_SV_MAX_PATTERNS=${GRAND_SV_MAX_PATTERNS}"
echo "[SBATCH CONFIG] GRAND_SV_BOOST_MAX_WEIGHT=${GRAND_SV_BOOST_MAX_WEIGHT}  GRAND_SV_BOOST_MAX_PATTERNS=${GRAND_SV_BOOST_MAX_PATTERNS}"

# ===== Run =====
SCRIPT=${SCRIPT:-HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py}
OUT_PREFIX=${OUT_PREFIX:-results/${SLURM_JOB_NAME}_${SLURM_JOB_ID}}

python -u "$SCRIPT" "$OUT_PREFIX" sionna5g
