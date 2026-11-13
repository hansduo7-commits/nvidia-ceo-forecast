"""Quantile forecaster for make.fun's polymarket-nvidia-ceo-forecast-01.

Python 3.13, standard library only, deterministic.

For every market in a case it writes five quantiles (0.05, 0.25, 0.5, 0.75, 0.95)
of the Yes price at each of the 240 minutes after the checkpoint.

Method
  1. Median: the last price, pulled down over the first hour by the board's
     excess when the prices of these mutually exclusive outcomes sum above 1.
  2. Spread: empirical h-minute price changes, scaled by sqrt(p(1-p)) so that
     markets at different price levels can be pooled. The first minutes of a
     market's history (opening prints around 0.5) are skipped. Quantiles of the
     changes are taken at a ladder of horizons and interpolated in sqrt(h).
  3. The market's own quantiles are blended with the pooled ones (mostly pooled:
     single markets have few, lumpy moves), scaled by the market's last-12-hour
     activity, with separate multipliers for the inner and outer quantiles, a
     minimum band, clamped to [0, 1] and made monotone.
"""
import json
import math
import os

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
LADDER = (1, 2, 4, 8, 15, 30, 60, 120, 180, 240)
SKIP_OPEN = 360      # opening minutes of each market that are ignored
OWN_WEIGHT = 0.05    # weight of a market's own change distribution vs the pool
MULT_INNER = 1.0     # multiplier on the empirical 25%/75% offsets
MULT_OUTER = 0.6     # multiplier on the empirical 5%/95% offsets
SUM_POW = 1.0        # strength of the pull when the board's prices sum above 1
REGIME_MIN = 720     # recent window (12 h) for a market's activity level
REGIME_POW = 0.5     # spread scales with (recent activity / pooled activity) ** REGIME_POW


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


def regime_gain(markets, hist):
    """Per-market spread gain: how active the market has been over the last
    12 hours (mean |15-minute scaled change|) relative to all markets' whole
    history, lightly smoothed, raised to REGIME_POW and bounded."""
    all_moves = []
    recent = {}
    for m in markets:
        s = hist[m["id"]]
        a = [abs(s[i + 15] - s[i]) / price_scale(s[i]) for i in range(len(s) - 15)]
        all_moves.extend(a)
        r = a[-REGIME_MIN:]
        recent[m["id"]] = sum(r) / len(r) if r else None
    base = sum(all_moves) / len(all_moves) if all_moves else 0.0
    gain = {}
    for k, v in recent.items():
        if v is None or base <= 0:
            gain[k] = 1.0
        else:
            g = ((v + 0.1 * base) / (1.1 * base)) ** REGIME_POW
            gain[k] = min(3.0, max(0.33, g))
    return gain


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
    gain = regime_gain(markets, hist)
    # The markets are mutually exclusive outcomes: when their prices add up to
    # more than 1 the excess tends to bleed out, proportionally, over the next hour.
    board = sum(m["prices"][-1] for m in markets)
    shrink = min(1.0, 1.0 / board) if board > 0 else 1.0
    mult = (MULT_OUTER, MULT_INNER, 0.0, MULT_INNER, MULT_OUTER)

    result = []
    for m in markets:
        # Backtests: pulling a fresh spike toward its recent median hurt, so the
        # anchor is the last price itself.
        x = m["prices"][-1]
        sc = price_scale(x) * gain[m["id"]]
        table = []
        for k, h in enumerate(LADDER):
            n, q_own = own[m["id"]][k]
            w = OWN_WEIGHT * min(1.0, n / 600.0) if n >= 30 else 0.0
            table.append((h, [w * a + (1.0 - w) * b for a, b in zip(q_own, pooled[k])]))
        q = []
        for h in range(1, horizon + 1):
            d = interp(table, h)
            c = x * shrink ** (SUM_POW * min(1.0, h / 60.0))
            row = [c + v * sc * mu for v, mu in zip(d, mult)]
            row[2] = c
            band = 0.0005 + 0.00004 * h
            row[0] = min(row[0], c - band)
            row[4] = max(row[4], c + band)
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
