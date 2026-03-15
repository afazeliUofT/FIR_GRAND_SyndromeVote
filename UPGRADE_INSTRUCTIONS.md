# Exact upgrade instructions

## Files to replace in your repository

Replace these existing files:
- `HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py`
- `run_ms.sbatch`
- `run_ms_fair.sbatch`
- `README.md`

## Files to add

Add these new files:
- `run_ms_scout.sbatch`
- `UPGRADE_INSTRUCTIONS.md`
- `RECEIVER6_NOTES.md`
- `CURRENT_RUN_DIAGNOSIS.md`

## Exact shell steps

```bash
cd /path/to/FIR_GRAND_SyndromeVote

mkdir -p backup_receiver6
cp HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py \
   run_ms.sbatch \
   run_ms_fair.sbatch \
   README.md \
   backup_receiver6/

unzip /path/to/FIR_GRAND_receiver6_softanchor.zip -d /tmp/fir_receiver6_pkg

cp /tmp/fir_receiver6_pkg/FIR_GRAND_receiver6_softanchor/HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py .
cp /tmp/fir_receiver6_pkg/FIR_GRAND_receiver6_softanchor/run_ms.sbatch .
cp /tmp/fir_receiver6_pkg/FIR_GRAND_receiver6_softanchor/run_ms_fair.sbatch .
cp /tmp/fir_receiver6_pkg/FIR_GRAND_receiver6_softanchor/run_ms_scout.sbatch .
cp /tmp/fir_receiver6_pkg/FIR_GRAND_receiver6_softanchor/README.md .
cp /tmp/fir_receiver6_pkg/FIR_GRAND_receiver6_softanchor/UPGRADE_INSTRUCTIONS.md .
cp /tmp/fir_receiver6_pkg/FIR_GRAND_receiver6_softanchor/RECEIVER6_NOTES.md .
cp /tmp/fir_receiver6_pkg/FIR_GRAND_receiver6_softanchor/CURRENT_RUN_DIAGNOSIS.md .

python -m py_compile HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py
bash -n run_ms.sbatch run_ms_fair.sbatch run_ms_scout.sbatch
```

## Recommended run order

```bash
sbatch run_ms_scout.sbatch
sbatch run_ms.sbatch
sbatch run_ms_fair.sbatch
```

## What to look for in the outputs

New decoder names:
- `hybahr4`
- `hybahr8`
- `hybahr15`

New tail behavior to watch:
- nonzero `restart_num_runs_mean`
- nonzero `restart_anchor_bits_mean`
- improved `pre_solver_success_rate_if_attempted`
- final FER moving materially below `fer_stage1` for the 15-iteration hybrid
