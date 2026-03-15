# Current run diagnosis from result CSV parse

## Stress run (`run_ms.sbatch`) best FER by SNR

| snr_db | best legacy | legacy FER | best hybrid | hybrid FER | FER gap vs best legacy |
|---:|---|---:|---|---:|---:|
| 6.8 | ldpc100 | 0.728155 | hybctg15 | 0.753769 | +3.52% |
| 7.1 | ldpc100 | 0.699301 | hybsv15 | 0.748130 | +6.98% |
| 7.4 | ldpc100 | 0.688073 | hybctg15 | 0.724638 | +5.31% |
| 7.7 | ldpc100 | 0.655022 | hybsv15 | 0.712589 | +8.79% |
| 8.0 | ldpc100 | 0.686499 | hybsv15 | 0.717703 | +4.55% |
| 8.3 | ldpc20 | 0.641026 | hybsv15 | 0.678733 | +5.88% |
| 8.6 | ldpc100 | 0.645161 | hybctg15 | 0.659341 | +2.20% |

## Fair run (`run_ms_fair.sbatch`) best FER by SNR

| snr_db | best legacy | legacy FER | best hybrid | hybrid FER | FER gap vs best legacy |
|---:|---|---:|---|---:|---:|
| 7.0 | ldpc100 | 0.028975 | hybosd15 | 0.053533 | +84.75% |
| 7.5 | ldpc100 | 0.022321 | hybptg15 | 0.036269 | +62.48% |
| 8.0 | ldpc100 | 0.015601 | hybosd15 | 0.025928 | +66.20% |
| 8.5 | ldpc100 | 0.010407 | hybosd15 | 0.018662 | +79.32% |
| 9.0 | ldpc100 | 0.006671 | hybptg15 | 0.011950 | +79.14% |
| 9.5 | ldpc100 | 0.004549 | hybosd15 | 0.008094 | +77.94% |

## Fair run vs `ldpc20`

| snr_db | ldpc20 FER | best hybrid | hybrid FER | FER gap vs ldpc20 |
|---:|---:|---|---:|---:|
| 7.0 | 0.040186 | hybosd15 | 0.053533 | +33.21% |
| 7.5 | 0.033187 | hybptg15 | 0.036269 | +9.28% |
| 8.0 | 0.022678 | hybosd15 | 0.025928 | +14.33% |
| 8.5 | 0.017064 | hybosd15 | 0.018662 | +9.37% |
| 9.0 | 0.010642 | hybptg15 | 0.011950 | +12.28% |
| 9.5 | 0.007142 | hybosd15 | 0.008094 | +13.34% |

## `hybosd15` rescue fraction

| run | snr_db | rescue fraction |
|---|---:|---:|
| stress | 6.8 | 0.33% |
| stress | 7.1 | 2.60% |
| stress | 7.4 | 1.64% |
| stress | 7.7 | 0.66% |
| stress | 8.0 | 0.33% |
| stress | 8.3 | 2.91% |
| stress | 8.6 | 0.99% |
| fair | 7.0 | 4.21% |
| fair | 7.5 | 6.02% |
| fair | 8.0 | 4.94% |
| fair | 8.5 | 7.06% |
| fair | 9.0 | 6.72% |
| fair | 9.5 | 9.42% |

## `hybosd15` tail-metric clue

| run | snr_db | pre_solver_success_if_attempted | osd_free_dim_mean | restart_num_runs_mean | restart_anchor_bits_mean |
|---|---:|---:|---:|---:|---:|
| stress | 6.8 | 0.0033 | 0.0000 | 0.0000 | 0.0000 |
| stress | 7.1 | 0.0260 | 0.0000 | 0.0000 | 0.0000 |
| stress | 7.4 | 0.0164 | 0.0000 | 0.0000 | 0.0000 |
| stress | 7.7 | 0.0066 | 0.0000 | 0.0000 | 0.0000 |
| stress | 8.0 | 0.0033 | 0.0000 | 0.0000 | 0.0000 |
| stress | 8.3 | 0.0291 | 0.0000 | 0.0000 | 0.0000 |
| stress | 8.6 | 0.0099 | 0.0000 | 0.0000 | 0.0000 |
| fair | 7.0 | 0.0421 | 0.0000 | 0.0000 | 0.0000 |
| fair | 7.5 | 0.0602 | 0.0000 | 0.0000 | 0.0000 |
| fair | 8.0 | 0.0494 | 0.0000 | 0.0000 | 0.0000 |
| fair | 8.5 | 0.0706 | 0.0000 | 0.0000 | 0.0000 |
| fair | 9.0 | 0.0672 | 0.0000 | 0.0000 | 0.0000 |
| fair | 9.5 | 0.0942 | 0.0000 | 0.0000 | 0.0000 |