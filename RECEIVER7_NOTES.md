# Receiver-7 notes

## New control knobs added by this overlay

### `GRAND_RESCUE_SNAPSHOT_ITERS`
Comma-separated rescue schedule for stage-2.
Examples:

```bash
export GRAND_RESCUE_SNAPSHOT_ITERS="stage1"
export GRAND_RESCUE_SNAPSHOT_ITERS="4,8,12,15,20,40,60,80,stage1"
export GRAND_RESCUE_SNAPSHOT_ITERS="20,40,60,80,100"
```

Interpretation:
- integers above the stage-1 max iteration are ignored,
- `stage1` / `final` / `last` all mean the final stage-1 iteration,
- the final stage-1 snapshot is always appended even if omitted.

Why it matters:
- some residuals are easier to rescue at earlier iterations, before BP hardens around a wrong basin,
- the old code only rescued from one final snapshot.

### `PAIR_DECODER_STREAMS`

```bash
export PAIR_DECODER_STREAMS=1
```

When enabled, all decoders at one SNR use the same frame stream.
That makes `ldpc100` vs `hybbgr100` a direct same-frame comparison.

This is especially important when you want the hybrid to be an **absolute FER winner**:
- with the same stage-1 depth and the same frames,
- `hybbgr100` can only match `ldpc100` or improve on some of its failures.

### `RUN_RECEIVER1`

```bash
export RUN_RECEIVER1=0
```

Disables the plain GRAND hybrid (`hyb*`) when you want a narrow `ldpc` vs `Receiver-7` run.

## Recommended practical settings

For the strongest direct comparison:

```bash
export RUN_RECEIVER1=0
export RUN_RECEIVER7=1
export STAGE1_ITERS="100"
export LDPC_ITERS="100"
export PAIR_DECODER_STREAMS=1
export GRAND_RESCUE_SNAPSHOT_ITERS="4,8,12,15,20,40,60,80,100"
```

## Why Receiver-7 should help more than Receiver-6
Receiver-7 explicitly targets grouped, structured residual errors by combining:
- singleton suspects,
- unsatisfied-check neighborhoods,
- short ranked windows,
- channel-vs-posterior disagreement groups,
- and block-debias anchored restarts.

That is a better match for correlated reliability distortions than spending most of stage-2 on raw independent bit-flip patterns.
