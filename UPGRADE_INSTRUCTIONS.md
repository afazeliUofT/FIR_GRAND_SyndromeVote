# Upgrade instructions

## 1) From the repo root, overlay the new files

The zip is built so that extracting it **from the repo root** updates files in place.

```bash
cd /path/to/FIR_GRAND_SyndromeVote
unzip -o /path/to/FIR_GRAND_receiver7_overlay.zip
```

## 2) Validate the updated files

```bash
python -m py_compile HW3_PARALLEL_Narval_LDPC_GRAND_MS_PATCH.py
bash -n run_ms.sbatch run_ms_scout.sbatch run_ms_fair.sbatch run_ms_receiver7_winner.sbatch
```

## 3) Clean stale tracked outputs and old notes

```bash
while IFS= read -r item; do
  [[ -z "$item" || "$item" =~ ^# ]] && continue
  rm -rf "$item"
done < REMOVE_FROM_REPO.txt
```

## 4) Recommended run order

If you want the strongest direct Receiver-7-vs-LDPC result first:

```bash
sbatch run_ms_receiver7_winner.sbatch
```

If you also want broader operating-point mapping:

```bash
sbatch run_ms_scout.sbatch
sbatch run_ms.sbatch
sbatch run_ms_fair.sbatch
```

## 5) What changed technically

- `GRAND_RESCUE_SNAPSHOT_ITERS` now lets stage-2 rescue probe several LDPC snapshots.
- `PAIR_DECODER_STREAMS=1` makes decoder FER comparisons use the same frame stream.
- `RUN_RECEIVER1=0/1` lets you disable the plain GRAND hybrid when you only want `ldpc100` vs `hybbgr100`.
- `run_ms_receiver7_winner.sbatch` is the dedicated best-FER comparison run.

## 6) What to expect in outputs

The dedicated winner run should mainly emit:
- `ldpc100`
- `hybbgr100`

and, if you leave `RUN_RECEIVER1=1`, also:
- `hyb100`

With paired streams and the same stage-1 depth, `hybbgr100` should not empirically trail `ldpc100` on FER; it can only match or improve by rescuing some of the residual `ldpc100` failures.
