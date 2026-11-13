"""Quantile forecaster for make.fun's polymarket-nvidia-ceo-forecast-01.

Python 3.13, standard library only, deterministic.

For every market in a case it writes five quantiles (0.05, 0.25, 0.5, 0.75, 0.95)
of the Yes price at each of the 240 minutes after the checkpoint.

Method
  1. Anchor (the median): the last price, unless the last few minutes are a
     fresh spike far from the 60-minute median; such spikes tend to revert, so
     the anchor is pulled toward that median.
  2. Spread: empirical h-minute price changes, scaled by sqrt(p(1-p)) so that
     markets at different price levels can be pooled. The first minutes of a
     market's history (opening prints around 0.5) are skipped. Quantiles of the
     changes are taken at a ladder of horizons and interpolated in sqrt(h).
  3. The market's own quantiles are blended with the pooled ones (mostly pooled:
     single markets have few, lumpy moves), shrunk by a spread multiplier, given
     a minimum band, clamped to [0, 1] and made monotone.
"""
import json
import math
import os

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
LADDER = (1, 2, 4, 8, 15, 30, 60, 120, 180, 240)
SKIP_OPEN = 180      # opening minutes of each market that are ignored
OWN_WEIGHT = 0.05    # weight of a market's own change distribution vs the pool
SPREAD_MULT = 0.7    # multiplier on the empirical spread (backtest: 0.6-0.8 best, 0.7 kept as a hedge)
SPIKE_WINDOW = 15    # a level held for at most this many minutes may be a spike
MED_WINDOW = 60      # window of the reference median for spike detection


def qtile(sorted_vals, p):
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    pos = p * (n - 1)
    i = int(pos)
    j = min(i + 1, n - 1)
    f = pos - i
    return sorted_vals[i] * (1.0 - f) + sorted_vals[j] * f


def price_scale(p):
    p = min(max(p, 0.01), 0.99)
    return math.sqrt(p * (1.0 - p))


def usable(prices):
    """Drop the opening minutes; keep at least two thirds of a short history."""
    skip = min(SKIP_OPEN, len(prices) // 3)
    return prices[skip:]


def anchor(prices):
    x = prices[-1]
    if len(prices) < MED_WINDOW:
        return x
    med = qtile(sorted(prices[-MED_WINDOW:]), 0.5)
    held = 0
    for v in reversed(prices):
        if abs(v - x) <= 0.005:
            held += 1
        else:
            break
    dev = x - med
    if held <= SPIKE_WINDOW and abs(dev) > max(0.02, 0.25 * med):
        return med + dev * held / SPIKE_WINDOW
    return x


def scaled_changes(prices, h):
    out = []
    for i in range(len(prices) - h):
        a = prices[i]
        out.append((prices[i + h] - a) / price_scale(a))
    return out


def interp(table, h):
    """table: [(h, [5 values])] on LADDER; linear in sqrt(h)."""
    if h <= table[0][0]:
        return table[0][1]
    for (h0, v0), (h1, v1) in zip(table, table[1:]):
        if h <= h1:
            t = (math.sqrt(h) - math.sqrt(h0)) / (math.sqrt(h1) - math.sqrt(h0))
            return [a + (b - a) * t for a, b in zip(v0, v1)]
    return table[-1][1]


def forecast(case):
    horizon = int(case.get("horizon_minutes", 240))
    markets = case["markets"]
    hist = {m["id"]: usable(m["prices"]) for m in markets}

    pool = {h: [] for h in LADDER}
    own = {}
    for m in markets:
        rows = []
        for h in LADDER:
            ch = scaled_changes(hist[m["id"]], h)
            pool[h].extend(ch)
            ch.sort()
            rows.append((len(ch), [qtile(ch, p) for p in LEVELS]))
        own[m["id"]] = rows
    pooled = []
    for h in LADDER:
        s = sorted(pool[h])
        pooled.append([qtile(s, p) for p in LEVELS])

    result = []
    for m in markets:
        x = anchor(m["prices"])
        sc = price_scale(x)
        table = []
        for k, h in enumerate(LADDER):
            n, q_own = own[m["id"]][k]
            w = OWN_WEIGHT * min(1.0, n / 600.0) if n >= 30 else 0.0
            table.append((h, [w * a + (1.0 - w) * b for a, b in zip(q_own, pooled[k])]))
        q = []
        for h in range(1, horizon + 1):
            d = interp(table, h)
            row = [x + v * sc * SPREAD_MULT for v in d]
            row[2] = x
            band = 0.0005 + 0.00004 * h
            row[0] = min(row[0], x - band)
            row[4] = max(row[4], x + band)
            row = [min(1.0, max(0.0, v)) for v in row]
            for k in range(1, 5):
                if row[k] < row[k - 1]:
                    row[k] = row[k - 1]
            q.append([round(v, 6) for v in row])
        result.append({"id": m["id"], "q": q})
    return {"forecasts": result}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name), encoding="utf-8") as f:
            case = json.load(f)
        with open(os.path.join(out_dir, name[:-3] + ".out"), "w", encoding="utf-8") as f:
            json.dump(forecast(case), f, separators=(",", ":"))


if __name__ == "__main__":
    main()
