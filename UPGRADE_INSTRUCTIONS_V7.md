cd /path/to/FIR_GRAND_SyndromeVote
unzip -o /path/to/FIR_GRAND_errrootfix_v7.zip
chmod +x run_main_quiet.py tools/filter_tf_stderr.py \
  run_ms_receiver7_localbias_scout.sbatch \
  run_ms_receiver7_localbias_winner.sbatch \
  run_ms_receiver7_winner.sbatch \
  run_ms_hybridmeta_blockbias_scout.sbatch \
  run_ms_hybridmeta_blockbias_winner.sbatch

python -m py_compile run_main_quiet.py tools/filter_tf_stderr.py
bash -n run_ms_receiver7_localbias_scout.sbatch \
  run_ms_receiver7_localbias_winner.sbatch \
  run_ms_receiver7_winner.sbatch \
  run_ms_hybridmeta_blockbias_scout.sbatch \
  run_ms_hybridmeta_blockbias_winner.sbatch
