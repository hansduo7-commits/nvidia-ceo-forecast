# NVIDIA CEO Polymarket forecaster

Entry for the make.fun bounty `polymarket-nvidia-ceo-forecast-01`.

`python3 main.py` reads every `*.in` in `$MAKEFUN_INPUT_DIR` (default `/input`) and writes the matching `*.out` to `$MAKEFUN_OUTPUT_DIR` (default `/output`): five quantiles per market for each of the next 240 minutes.

Method: median at the last price, pulled down over the first hour when the prices of these mutually exclusive outcomes sum above 1; spread from empirical h-minute price changes scaled by sqrt(p(1-p)), pooled across markets (opening minutes skipped), interpolated in sqrt(h), scaled per market by its last-12-hour activity relative to the pool, with separate inner (1.0) and outer (0.6) multipliers and a minimum band. Deterministic, standard library only.
