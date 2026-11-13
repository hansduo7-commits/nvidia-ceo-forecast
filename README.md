# NVIDIA CEO Polymarket forecaster

Entry for the make.fun bounty `polymarket-nvidia-ceo-forecast-01`.

`python3 main.py` reads every `*.in` in `$MAKEFUN_INPUT_DIR` (default `/input`) and writes the matching `*.out` to `$MAKEFUN_OUTPUT_DIR` (default `/output`): five quantiles per market for each of the next 240 minutes.

Method: median at a spike-robust anchor (the last price, pulled toward the 60-minute median after a fresh isolated jump); spread from empirical h-minute price changes scaled by sqrt(p(1-p)), pooled across markets (opening minutes skipped), interpolated in sqrt(h), shrunk by 0.7, with a minimum band. Deterministic, standard library only.
