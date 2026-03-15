# Receiver-6 repository hotfix

## What changed
This refresh package is the current Receiver-6 repository content plus one functional fix:

- `run_ms_scout.sbatch` now starts with `#!/bin/bash`

That shebang is required by Slurm. Without it, `sbatch` rejects the file before execution.

## What to replace
Replace these files in the repository root with the copies from this package:

- `HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py`
- `run_ms.sbatch`
- `run_ms_fair.sbatch`
- `run_ms_scout.sbatch`
- `README.md`
- `UPGRADE_INSTRUCTIONS.md`
- `RECEIVER6_NOTES.md`
- `CURRENT_RUN_DIAGNOSIS.md`
- `CURRENT_RESULTS_DIGEST.md`
- `CODE_MATCH_CHECKSUMS.md`

Functionally, only `run_ms_scout.sbatch` changed. The rest are included so you can re-sync the repository cleanly from one package.

## Exact shell steps
```bash
cd /path/to/FIR_GRAND_SyndromeVote
mkdir -p backup_receiver6_hotfix
cp HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py \
   run_ms.sbatch \
   run_ms_fair.sbatch \
   run_ms_scout.sbatch \
   README.md \
   UPGRADE_INSTRUCTIONS.md \
   RECEIVER6_NOTES.md \
   CURRENT_RUN_DIAGNOSIS.md \
   CURRENT_RESULTS_DIGEST.md \
   CODE_MATCH_CHECKSUMS.md \
   backup_receiver6_hotfix/ 2>/dev/null || true

unzip /path/to/FIR_GRAND_receiver6_full_fix.zip -d /tmp/fir_receiver6_fix
cp /tmp/fir_receiver6_fix/FIR_GRAND_receiver6_full_fix/* .

python -m py_compile HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py
bash -n run_ms.sbatch run_ms_fair.sbatch run_ms_scout.sbatch
```

## Then run
```bash
sbatch run_ms_scout.sbatch
```
