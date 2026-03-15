# Diagnosis of the current Receiver-5 runs

From the latest checked-in result CSVs:

- No hybrid beats `ldpc100` on FER in either the stressed or fair run.
- No hybrid beats `ldpc20` on FER in the fair run.
- In the stressed run, some hybrids beat `ldpc20`, but FER stays extremely high for everyone, so that is not yet a persuasive regime.

The deeper issue is in the tail metrics:

- for `hybosd15`, `osd_free_dim_mean` is essentially zero,
- `restart_num_runs_mean` is zero,
- `restart_ldpc_iters_mean` is zero,
- `restart_anchor_bits_mean` is zero.

That means Receiver-5's intended OSD + anchored-restart path is usually *not actually engaging*. The local subsystem is often too constrained or inconsistent for the OSD step to generate meaningful candidates.

So the dominant failure mode is likely not a small exact local noise pattern. It looks more like:
- a BP trapping-set / wrong-fixed-point state,
- driven by structured reliability distortion,
- where exact local reprocessing has little room to move.

Receiver-6 addresses that by generating soft local hypotheses even when the local parity subsystem is not nicely solvable, then using anchored full-graph restarts to try to leave the wrong BP basin.
