"""Forecaster for make.fun's polymarket-nvidia-ceo-forecast-01 (Python 3.13, standard library only).

For each market it forecasts five quantiles of the Yes price at every minute of the next 240, from the
72 hours of 1-minute prices up to the checkpoint.

Method (deterministic, no randomness):
  * anchor: the price at the checkpoint, unless it is an isolated spike (far from the recent median and only
    a few minutes old), in which case the anchor is pulled toward the recent median, since such spikes revert;
  * spread: empirical quantiles of the market's own h-minute price changes over its history (skipping the
    junk minutes right after a market opens), at a ladder of horizons, interpolated in sqrt(h);
  * shrinkage: those empirical quantiles are blended with a pooled, price-scaled distribution so thin or
    quiet histories still get a sensible spread; then clamped to [0, 1] and made monotone.
"""
import json
import math
import os
import sys

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
LADDER = [1, 2, 4, 8, 15, 30, 60, 120, 180, 240]
SKIP_OPEN = 180          # minutes after a market's first price that are ignored (opening junk at ~0.5)
LOOKBACK = 4320          # minutes of history used for the empirical changes
W_OWN = 0.15             # weight of the market's own empirical quantiles vs the pooled ones
SPIKE_MIN = 15           # recent-median window for spike detection
FLOOR = 0.0005
SPREAD = 0.8             # multiplier on the empirical spread


def quantile(sorted_vals, p):
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    pos = p * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    f = pos - lo
    return sorted_vals[lo] * (1 - f) + sorted_vals[hi] * f


def median(vals):
    s = sorted(vals)
    return quantile(s, 0.5)


def clean_series(prices):
    return prices[SKIP_OPEN:] if len(prices) > SKIP_OPEN + 300 else prices[min(len(prices) // 3, SKIP_OPEN):]


def anchor_of(prices):
    x = prices[-1]
    if len(prices) < SKIP_OPEN:
        return x
    recent = prices[-60:]
    med = median(recent)
    # how many trailing minutes the price has sat at (about) its current level
    run = 0
    for v in reversed(prices):
        if abs(v - x) <= 0.0051:
            run += 1
        else:
            break
    dev = x - med
    if run <= SPIKE_MIN and abs(dev) > max(0.02, 0.25 * med):
        w = run / SPIKE_MIN
        return med + w * dev
    return x


def rel_changes(prices, h):
    """h-minute changes divided by a price scale, so markets at different levels can be pooled."""
    out = []
    n = len(prices)
    step = 1 if n < 2000 else 2
    for i in range(0, n - h, step):
        a = prices[i]
        out.append((prices[i + h] - a) / scale(a))
    return out


def scale(p):
    # moves are larger for prices near the middle; floor so tiny prices still move a tick
    return math.sqrt(max(p, 0.01) * (1 - min(p, 0.99)))


def interp_sqrt(table, h):
    """table: list of (h, [5 values]) at LADDER; interpolate linearly in sqrt(h)."""
    for k in range(len(table) - 1):
        h0, v0 = table[k]
        h1, v1 = table[k + 1]
        if h0 <= h <= h1:
            t = (math.sqrt(h) - math.sqrt(h0)) / (math.sqrt(h1) - math.sqrt(h0))
            return [a + (b - a) * t for a, b in zip(v0, v1)]
    return table[-1][1]


def forecast_case(case):
    H = case["horizon_minutes"]
    markets = case["markets"]
    series = {m["id"]: clean_series(m["prices"][-LOOKBACK:]) for m in markets}

    own = {}
    pooled_lists = {h: [] for h in LADDER}
    for m in markets:
        s = series[m["id"]]
        tbl = []
        for h in LADDER:
            ch = rel_changes(s, h)
            pooled_lists[h].extend(ch)
            sc = sorted(ch)
            tbl.append((h, [quantile(sc, p) for p in LEVELS] if len(sc) >= 30 else None, len(sc)))
        own[m["id"]] = tbl
    pooled = []
    for h in LADDER:
        sc = sorted(pooled_lists[h])
        pooled.append((h, [quantile(sc, p) for p in LEVELS]))

    out = []
    for m in markets:
        prices = m["prices"]
        x = anchor_of(prices)
        sc = scale(x)
        table = []
        for (h, q_own, n), (_, q_pool) in zip(own[m["id"]], pooled):
            if q_own is None:
                q = q_pool
            else:
                w = W_OWN * min(1.0, n / 600)
                q = [w * a + (1 - w) * b for a, b in zip(q_own, q_pool)]
            # never narrower than one tick-ish band at the outer levels
            q = list(q)
            table.append((h, q))
        rows = []
        for h in range(1, H + 1):
            d = interp_sqrt(table, h)
            row = [x + v * sc * SPREAD for v in d]
            row[2] = x + d[2] * sc * 0.0  # median: stay at the anchor
            lo_band = 0.0005 + 0.00004 * h
            row[0] = min(row[0], x - lo_band)
            row[4] = max(row[4], x + lo_band)
            row = [min(1.0, max(0.0, v)) for v in row]
            for k in range(1, 5):
                if row[k] < row[k - 1]:
                    row[k] = row[k - 1]
            rows.append([round(v, 6) for v in row])
        out.append({"id": m["id"], "q": rows})
    return {"forecasts": out}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name), encoding="utf-8") as f:
            case = json.load(f)
        res = forecast_case(case)
        with open(os.path.join(out_dir, name[:-3] + ".out"), "w", encoding="utf-8") as f:
            json.dump(res, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
