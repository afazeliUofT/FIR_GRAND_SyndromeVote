# Exact replace / add instructions

## Files to replace in your repository root

Replace these existing files with the versions from this package:

- `HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py`
- `run_ms.sbatch`
- `run_ms_fair.sbatch`
- `README.md`

## Files to add in your repository root

Add these new documentation files:

- `UPGRADE_INSTRUCTIONS.md`
- `RECEIVER5_TUNING_NOTES.md`

## Exact shell commands

Assuming your repository is already cloned locally:

```bash
cd /path/to/FIR_GRAND_SyndromeVote
mkdir -p backup_receiver4
cp HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py backup_receiver4/
cp run_ms.sbatch backup_receiver4/
cp run_ms_fair.sbatch backup_receiver4/
cp README.md backup_receiver4/

unzip -o /path/to/FIR_GRAND_receiver5_osd_anchor.zip -d /tmp/fir_receiver5_osd_anchor
cp /tmp/fir_receiver5_osd_anchor/HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py .
cp /tmp/fir_receiver5_osd_anchor/run_ms.sbatch .
cp /tmp/fir_receiver5_osd_anchor/run_ms_fair.sbatch .
cp /tmp/fir_receiver5_osd_anchor/README.md .
cp /tmp/fir_receiver5_osd_anchor/UPGRADE_INSTRUCTIONS.md .
cp /tmp/fir_receiver5_osd_anchor/RECEIVER5_TUNING_NOTES.md .
```

## Exact git workflow

```bash
cd /path/to/FIR_GRAND_SyndromeVote
git checkout -b receiver5-osd-anchor

git add HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py \
        run_ms.sbatch \
        run_ms_fair.sbatch \
        README.md \
        UPGRADE_INSTRUCTIONS.md \
        RECEIVER5_TUNING_NOTES.md

git commit -m "Add receiver5 OSD + anchored restart hybrid upgrade"
```

## How to run

For the demonstration / stress case:

```bash
sbatch run_ms.sbatch
```

For the fairer close-to-current comparison:

```bash
sbatch run_ms_fair.sbatch
```

## New decoder names in the output CSVs

You will continue to see all previous decoders, and in addition you should now see:

- `hybosd4`
- `hybosd8`
- `hybosd15`

## New environment knobs added by the upgraded Python

Main Receiver-5 knobs:

- `RUN_RECEIVER5`
- `GRAND_OSD_LLR_SOURCE`
- `GRAND_OSD_CHECK_COVER_K`
- `GRAND_OSD_RATIO`
- `GRAND_OSD_MAX_BITS`
- `GRAND_OSD_ORDER`
- `GRAND_OSD_ENUM_MAX_BITS`
- `GRAND_OSD_MAX_CANDIDATES`
- `GRAND_OSD_DISAGREEMENT_BITS`
- `GRAND_OSD_RESTART_MAX_CANDIDATES`
- `GRAND_OSD_RESTART_ITERS`
- `GRAND_OSD_RESTART_ALPHA`
- `GRAND_OSD_RESTART_GAIN`
- `GRAND_OSD_RESTART_DUAL_GAIN`
- `GRAND_OSD_RESTART_ABS_FLOOR`
- `GRAND_OSD_RESTART_ANCHOR_ALL`
- `GRAND_OSD_MAX_PATTERNS`
- `GRAND_OSD_BOOST_MAX_PATTERNS`

New channel-estimation stress knobs:

- `SIONNA_CSI_MODE=nr_imperfect`
- `SIONNA_CSI_PHASE_DRIFT_STD_DEG`
- `SIONNA_CSI_AMP_RIPPLE_DB`
- `SIONNA_CSI_DELAY_BIAS_NS`
- `SIONNA_CSI_BLOCK_SC`

## What to check after replacement

1. `python -m py_compile HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py`
2. `grep -n "hybosd" HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py`
3. `grep -n "RUN_RECEIVER5" run_ms.sbatch run_ms_fair.sbatch`
4. After the run completes, confirm the summary CSV contains `hybosd4`, `hybosd8`, and `hybosd15`.
