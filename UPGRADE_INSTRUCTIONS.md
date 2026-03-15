# Upgrade instructions for Receiver-7

## 1) Replace these existing files in the repo root
- `HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py`
- `run_ms.sbatch`
- `run_ms_scout.sbatch`
- `run_ms_fair.sbatch`
- `README.md`
- `UPGRADE_INSTRUCTIONS.md`

## 2) Add these new files
- `RECEIVER7_NOTES.md`
- `CURRENT_RESULTS_REVIEW.md`
- `REMOVE_FROM_REPO.txt`

## 3) Remove these stale files from the current repo root
- `CODE_MATCH_CHECKSUMS.md`
- `CURRENT_RESULTS_DIGEST.md`
- `CURRENT_RUN_DIAGNOSIS.md`
- `REPO_FIX_INSTRUCTIONS.md`
- `RECEIVER6_NOTES.md`

## Exact shell commands

```bash
cd /path/to/FIR_GRAND_SyndromeVote

mkdir -p backup_receiver7
cp HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py    run_ms.sbatch    run_ms_scout.sbatch    run_ms_fair.sbatch    README.md    UPGRADE_INSTRUCTIONS.md    RECEIVER6_NOTES.md    CODE_MATCH_CHECKSUMS.md    CURRENT_RESULTS_DIGEST.md    CURRENT_RUN_DIAGNOSIS.md    REPO_FIX_INSTRUCTIONS.md    backup_receiver7/ 2>/dev/null || true

unzip /path/to/FIR_GRAND_receiver7_basisgrand.zip -d /tmp/fir_receiver7_pkg

cp /tmp/fir_receiver7_pkg/FIR_GRAND_receiver7_basisgrand/HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py .
cp /tmp/fir_receiver7_pkg/FIR_GRAND_receiver7_basisgrand/run_ms.sbatch .
cp /tmp/fir_receiver7_pkg/FIR_GRAND_receiver7_basisgrand/run_ms_scout.sbatch .
cp /tmp/fir_receiver7_pkg/FIR_GRAND_receiver7_basisgrand/run_ms_fair.sbatch .
cp /tmp/fir_receiver7_pkg/FIR_GRAND_receiver7_basisgrand/README.md .
cp /tmp/fir_receiver7_pkg/FIR_GRAND_receiver7_basisgrand/UPGRADE_INSTRUCTIONS.md .
cp /tmp/fir_receiver7_pkg/FIR_GRAND_receiver7_basisgrand/RECEIVER7_NOTES.md .
cp /tmp/fir_receiver7_pkg/FIR_GRAND_receiver7_basisgrand/CURRENT_RESULTS_REVIEW.md .
cp /tmp/fir_receiver7_pkg/FIR_GRAND_receiver7_basisgrand/REMOVE_FROM_REPO.txt .

rm -f CODE_MATCH_CHECKSUMS.md       CURRENT_RESULTS_DIGEST.md       CURRENT_RUN_DIAGNOSIS.md       REPO_FIX_INSTRUCTIONS.md       RECEIVER6_NOTES.md

python -m py_compile HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py
bash -n run_ms.sbatch run_ms_scout.sbatch run_ms_fair.sbatch
```

## 4) Run order

```bash
sbatch run_ms_scout.sbatch
sbatch run_ms.sbatch
sbatch run_ms_fair.sbatch
```

## 5) What to look for in the new results
The main new decoder family is:
- `hybbgr4`
- `hybbgr8`
- `hybbgr15`

Success signs:
- `hybbgr15` clearly below its own `fer_stage1`
- `hybbgr15` beating `ldpc20` in a region where FER is not huge
- ideally `hybbgr15` also beating `ldpc100` at least over part of the mid-FER region
- lower GRAND effort per successful rescue than the current `hybahr15`

## Note
The new `run_ms.sbatch` is intentionally **not** the old extreme-stress setup.
It is still 5G-compatible, but it is milder so that a localized rescue stage has a fair chance to show a meaningful win.
