#!/usr/bin/env python3
"""
Weight Is Not Power: Designing Voting Rules for an Expanded BRICS
Full replication and robustness analysis  (version 3)

Version 3 adds
  - Historical backcast 2000-2025: the rules applied to every year, with annual
    weights and with the proposed design (3-year averages, recalculated every 3 years)
  - Map of acceptable rules: every combination of threshold, blocking-minority size,
    equal-vote share and member-majority requirement tested against the constraints
  - CRA reform: the CRA's current rule and four reform options on its real members
  - Blocking pairs: which pairs of members could jointly block, under every rule

Changes from version 1
  - Proposed rule is now D = 55% of square-root votes + majority of members.
    The original 60% rule is kept as D60 for comparison.
  - New rules G3 / G4: 60% double majority + an EU-style minimum blocking
    minority (a blocking coalition must contain at least 3 / 4 members).
  - Constraint C3 is now relative to group size: the smallest member must keep at
    least MIN_SMALLEST_REL x (100 / n) percent of Banzhaf power.
  - Bloc (coalition) tests run for every design rule, in BRICS-10 and BRICS-10+SAU,
    including a Gulf (UAE + Saudi) bloc.
  - New "constraint scorecard": which rules pass all constraints in which scenario.
  - Missing data can be filled by hand in MANUAL_OVERRIDES (e.g. Iran's services
    trade from UNCTADstat); every override is logged in data_notes.txt.
  - Runs unchanged in Google Colab / Jupyter.

Outputs (./output/): CSV tables, PNG figures, data_notes.txt, results_summary.md

Usage
  pip install numpy pandas matplotlib requests tabulate
  python brics_voting_analysis.py            # live World Bank data
  python brics_voting_analysis.py --offline  # Appendix A1 data only
  Colab / Jupyter: paste into one cell and run (set OFFLINE below if needed)
"""

import argparse
import math
import warnings
import os
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message=".*constant.*")   # Spearman on constant power vectors

# ============================================================================
# 0. SETTINGS - edit these, not the code below
# ============================================================================
OFFLINE = False            # True = use Appendix A1 data only (no internet)
REF_YEAR = 2025            # latest non-missing year <= this is used
AVG_YEARS = 3              # window for the multi-year-average check
HIST_START = 2000          # first year of the historical backcast
RECALC_EVERY = 3           # backcast: weights recalculated every N years (the design)
ALPHA_BASE = 0.5           # weight on GDP in the index (beta = 1 - alpha)
PROPOSED_Q = 0.55          # weighted-vote threshold of the proposed rule D
EPS = 1e-12
OUT = "output"

# Design constraints. These are normative choices: state and justify them in the
# paper BEFORE the results.
#   C1  no single member can pass a decision alone
#   C2  no single member can block a decision alone
#   C3  smallest member keeps >= MIN_SMALLEST_REL x equal share (100/n) of power
#   C4  the largest economy keeps the most power (weight still matters)
#   C5  rank correlation between economic weight and power >= MIN_SPEARMAN
MIN_SMALLEST_REL = 0.5
MIN_SPEARMAN = 0.90

# Fill gaps in World Bank data by hand, in USD billion, e.g. from UNCTADstat or
# IMF BOP. Format: ("ISO3", "variable"): value.  Variables: gdp, gdp_ppp, merch,
# services.  Leave as None to skip. Cite the source in the paper.
MANUAL_OVERRIDES = {
    ("IRN", "services"): 0.0,    # Iran: unreported in WDI and IMF BOP; set to zero (results
                                 # change by <0.3 points for any value up to USD 100 bn)
}

COUNTRIES = {  # ISO3: (name, group)
    "CHN": ("China", "BRICS-5"), "IND": ("India", "BRICS-5"),
    "RUS": ("Russia", "BRICS-5"), "BRA": ("Brazil", "BRICS-5"),
    "ZAF": ("South Africa", "BRICS-5"),
    "IDN": ("Indonesia", "Joined 2024-25"), "ARE": ("UAE", "Joined 2024-25"),
    "EGY": ("Egypt", "Joined 2024-25"), "IRN": ("Iran", "Joined 2024-25"),
    "ETH": ("Ethiopia", "Joined 2024-25"),
    "SAU": ("Saudi Arabia", "Unconfirmed"),
    "THA": ("Thailand", "Partner"), "VNM": ("Vietnam", "Partner"),
    "MYS": ("Malaysia", "Partner"), "KAZ": ("Kazakhstan", "Partner"),
    "NGA": ("Nigeria", "Partner"), "UZB": ("Uzbekistan", "Partner"),
    "CUB": ("Cuba", "Partner"), "BLR": ("Belarus", "Partner"),
    "BOL": ("Bolivia", "Partner"), "UGA": ("Uganda", "Partner"),
}
B5 = ["CHN", "IND", "RUS", "BRA", "ZAF"]
B10 = B5 + ["IDN", "ARE", "EGY", "IRN", "ETH"]
PARTNERS = ["THA", "VNM", "MYS", "KAZ", "NGA", "UZB", "CUB", "BLR", "BOL", "UGA"]
SCENARIOS = {
    "BRICS-5": B5,
    "BRICS-10": B10,
    "BRICS-10+SAU": B10 + ["SAU"],
    "BRICS+20": B10 + PARTNERS,
    "BRICS+20 excl. Cuba": B10 + [p for p in PARTNERS if p != "CUB"],
    "BRICS+20+SAU": B10 + ["SAU"] + PARTNERS,
}
LARGE = ("BRICS+20",)      # 2^20 scenarios: run fewer variants to save time

INDICATORS = {
    "gdp": "NY.GDP.MKTP.CD",          # GDP, current US$
    "gdp_ppp": "NY.GDP.MKTP.PP.CD",   # GDP, PPP, current international $
    "mx": "TX.VAL.MRCH.CD.WT",        # merchandise exports
    "mm": "TM.VAL.MRCH.CD.WT",        # merchandise imports
    "sx": "BX.GSR.NFSV.CD",           # service exports (BoP)
    "sm": "BM.GSR.NFSV.CD",           # service imports (BoP)
}

# Appendix A1 (USD bn): offline fallback so the baseline always replicates
APPENDIX_A1 = {
    "CHN": (19498.0, 6354.7), "IND": (3956.1, 1198.8), "RUS": (2561.3, 722.1),
    "BRA": (2279.9, 642.0), "ZAF": (427.2, 244.5), "IDN": (1445.6, 524.8),
    "ARE": (552.3, 1325.5), "EGY": (365.3, 148.9), "IRN": (362.7, 178.9),
    "ETH": (126.4, 30.6), "THA": (577.0, 684.6), "VNM": (514.7, 926.7),
    "MYS": (472.2, 716.5), "KAZ": (306.2, 143.3), "NGA": (290.8, 96.9),
    "UZB": (147.0, 65.4), "CUB": (107.4, 10.4), "BLR": (93.4, 89.8),
    "BOL": (64.8, 19.7), "UGA": (62.0, 30.9),
}

# CRA (2014 treaty) actual vote shares, BRICS-5 benchmark
CRA_VOTES = {"CHN": 39.95, "BRA": 18.10, "IND": 18.10, "RUS": 18.10, "ZAF": 5.75}

# Rules. alloc: vote allocation; quota: weighted-vote threshold; strict: '>' vs '>=';
# mm: also require more than half of members; min_block: a blocking coalition must
# contain at least this many members (1 = no such clause).
RULES = {
    "P":   dict(label="Proportional, >50%", alloc="prop", quota=0.5, strict=True, mm=False),
    "A":   dict(label="A. Sqrt, >50%", alloc="sqrt", quota=0.5, strict=True, mm=False),
    "B":   dict(label="B. Sqrt, >=2/3", alloc="sqrt", quota=2 / 3, strict=False, mm=False),
    "C":   dict(label="C. Sqrt, optimal quota", alloc="sqrt", quota="optimal", strict=False, mm=False),
    "D":   dict(label=f"D. Double majority {PROPOSED_Q:.0%} (proposed)", alloc="sqrt",
                quota=PROPOSED_Q, strict=False, mm=True),
    "D60": dict(label="D60. Double majority 60% (original draft)", alloc="sqrt", quota=0.60,
                strict=False, mm=True),
    "E":   dict(label="E. 50% equal + 50% sqrt", alloc="hybrid", quota=0.5, strict=True, mm=False),
    "F":   dict(label="F. Sqrt, 25% cap", alloc="cap", quota=0.5, strict=True, mm=False),
    "G3":  dict(label="G3. 60% double majority + blocking minority >=3", alloc="sqrt",
                quota=0.60, strict=False, mm=True, min_block=3),
    "G4":  dict(label="G4. 60% double majority + blocking minority >=4", alloc="sqrt",
                quota=0.60, strict=False, mm=True, min_block=4),
}
MAIN_RULES = ["A", "B", "C", "D", "D60", "E", "F", "G3", "G4"]
DESIGN_RULES = ["D", "D60", "G3", "G4"]      # rules tested against voting blocs
BLOCS = {
    "No blocs (a priori)": None,
    "China-Russia": [["CHN", "RUS"]],
    "China-Russia-Iran": [["CHN", "RUS", "IRN"]],
    "IBSA (India-Brazil-South Africa)": [["IND", "BRA", "ZAF"]],
    "China-Russia + IBSA": [["CHN", "RUS"], ["IND", "BRA", "ZAF"]],
    "Gulf (UAE-Saudi Arabia)": [["ARE", "SAU"]],
}
# CRA (2014 treaty): commitments (USD bn) and vote formula = 5% basic votes shared
# equally + 95% in proportion to commitments (gives China 39.95%, South Africa 5.75%)
CRA_COMMIT = {"CHN": 41, "BRA": 18, "IND": 18, "RUS": 18, "ZAF": 5}
CRA_BASIC = 0.05

CONSTRAINTS = ["C1_no_unilateral_passage", "C2_no_unilateral_veto", "C3_smallest_protected",
               "C4_largest_keeps_most_power", "C5_power_tracks_weight"]


def name(c):
    return "+".join(COUNTRIES[x][0] for x in c.split("+")) if all(
        x in COUNTRIES for x in c.split("+")) else c


# ============================================================================
# 1. DATA
# ============================================================================
def fetch_wdi(codes, start, end):
    import requests
    rows = []
    for key, ind in INDICATORS.items():
        url = (f"https://api.worldbank.org/v2/country/{';'.join(codes)}/indicator/{ind}"
               f"?format=json&date={start}:{end}&per_page=20000")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        js = r.json()
        if len(js) < 2 or js[1] is None:
            print(f"  ! no data returned for {ind}")
            continue
        for d in js[1]:
            if d["value"] is not None:
                rows.append((d["countryiso3code"], key, int(d["date"]), float(d["value"]) / 1e9))
        print(f"  fetched {ind}")
    return pd.DataFrame(rows, columns=["code", "var", "year", "value"])


def build_series(raw):
    wide = raw.pivot_table(index=["code", "year"], columns="var", values="value")
    out = pd.DataFrame(index=wide.index)
    out["gdp"] = wide.get("gdp")
    out["gdp_ppp"] = wide.get("gdp_ppp")
    out["merch"] = wide.get("mx") + wide.get("mm")
    out["services"] = wide.get("sx") + wide.get("sm")
    return out


def latest(series, code, var, ref=REF_YEAR, k=1):
    try:
        s = series.loc[code, var].dropna()
    except KeyError:
        return np.nan, ""
    s = s[s.index <= ref].sort_index()
    if s.empty:
        return np.nan, ""
    s = s.iloc[-k:]
    yrs = f"{s.index.min()}-{s.index.max()}" if len(s) > 1 else str(s.index[0])
    return s.mean(), yrs


def apply_overrides(df, notes):
    for (c, var), val in MANUAL_OVERRIDES.items():
        if val is not None and c in df.index:
            df.loc[c, var] = float(val)
            notes.append(f"{name(c)}: {var} set manually to {val} bn (MANUAL_OVERRIDES)")
    df["total_trade"] = df["merch"] + df["services"]
    return df


def make_dataset(series, k=1, ref=REF_YEAR, max_lag=None, log=True):
    """max_lag: treat a value as missing if its latest year is older than ref - max_lag."""
    recs, notes = [], []
    for c in COUNTRIES:
        rec = {"code": c}
        for var in ["gdp", "gdp_ppp", "merch", "services"]:
            v, y = latest(series, c, var, ref=ref, k=k)
            if max_lag is not None and y and int(y[-4:]) < ref - max_lag:
                v = np.nan
            rec[var] = v
            if k == 1 and log:
                if not y:
                    notes.append(f"{name(c)}: {var} missing")
                elif y != str(REF_YEAR):
                    notes.append(f"{name(c)}: {var} uses {y}")
        recs.append(rec)
    df = pd.DataFrame(recs).set_index("code")
    return apply_overrides(df, notes), notes


def offline_dataset():
    df = pd.DataFrame({c: {"gdp": g, "merch": t} for c, (g, t) in APPENDIX_A1.items()}).T
    df["gdp_ppp"] = np.nan
    df["services"] = np.nan
    df.index.name = "code"
    return apply_overrides(df, [])


# ============================================================================
# 2. WEIGHTS AND VOTES
# ============================================================================
def index_weights(data, members, alpha=ALPHA_BASE, gdp_var="gdp", trade_var="merch"):
    d = data.loc[members]
    if d[[gdp_var, trade_var]].isna().any().any():
        missing = d.index[d[[gdp_var, trade_var]].isna().any(axis=1)].tolist()
        raise ValueError(f"missing {gdp_var}/{trade_var} for {missing}")
    g = d[gdp_var] / d[gdp_var].sum()
    t = d[trade_var] / d[trade_var].sum()
    return (alpha * g + (1 - alpha) * t).values


def sqrt_votes(w):
    s = np.sqrt(w)
    return s / s.sum()


def cap_votes(v, cap):
    v = v.copy()
    fixed = np.zeros(len(v), bool)
    for _ in range(100):
        over = (v > cap + 1e-15) & ~fixed
        if not over.any():
            break
        excess = (v[over] - cap).sum()
        v[over] = cap
        fixed |= over
        v[~fixed] += excess * v[~fixed] / v[~fixed].sum()
    return v


def allocate(w, alloc, cap=0.25, lam=0.0):
    n = len(w)
    if alloc == "sqrt" and lam > 0:        # lam = share of votes distributed equally
        return lam / n + (1 - lam) * sqrt_votes(w)
    return {"prop": lambda: w / w.sum(),
            "sqrt": lambda: sqrt_votes(w),
            "hybrid": lambda: 0.5 / n + 0.5 * sqrt_votes(w),
            "cap": lambda: cap_votes(sqrt_votes(w), cap),
            "equal": lambda: np.full(n, 1 / n)}[alloc]()


def optimal_quota(v):
    return 0.5 * (1 + math.sqrt((v ** 2).sum()))


# ============================================================================
# 3. THE GAME AND POWER INDICES (exact enumeration)
# ============================================================================
def wins(W, C, quota, strict, mm, min_block, n_members):
    """Winning test for coalition(s) with vote share W and member count C."""
    vote_ok = (W > quota + EPS) if strict else (W >= quota - EPS)
    # EU-style clause: if the members outside the coalition are too few to form
    # a blocking minority, the weighted-vote condition is deemed met
    vote_ok = vote_ok | ((n_members - C) < min_block)
    return (vote_ok & (C > n_members / 2)) if mm else vote_ok


@lru_cache(maxsize=8)
def _bits(n):
    masks = np.arange(1 << n, dtype=np.int64)
    bits = np.stack([((masks >> i) & 1).astype(bool) for i in range(n)])
    return masks, bits, bits.sum(axis=0)


def power(votes, counts, rule_args, n_members):
    n = len(votes)
    masks, bits, size = _bits(n)
    W = bits.T.astype(float) @ np.asarray(votes, float)
    C = bits.T.astype(np.int64) @ np.asarray(counts, np.int64)
    win = wins(W, C, n_members=n_members, **rule_args)
    fact = [math.factorial(k) for k in range(n + 1)]
    ss_w = np.array([fact[s - 1] * fact[n - s] / fact[n] if s else 0.0 for s in range(n + 1)])
    eta, ss = np.zeros(n), np.zeros(n)
    for i in range(n):
        idx = masks[bits[i]]
        swing = win[idx] & ~win[idx ^ (1 << i)]
        eta[i] = swing.sum()
        ss[i] = ss_w[size[idx][swing]].sum()
    bz = 100 * eta / eta.sum() if eta.sum() else np.zeros(n)
    return bz, 100 * ss, win


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return (2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()) if x.sum() else np.nan


def run_game(members, weights, rule, quota_override=None, blocs=None, min_block_override=None,
             lam_override=None, mm_override=None):
    r = RULES[rule]
    v = allocate(weights, r["alloc"], lam=lam_override or 0.0)
    quota = r["quota"] if quota_override is None else quota_override
    if quota == "optimal":
        quota = optimal_quota(v)
    args = dict(quota=quota, strict=r["strict"], mm=r["mm"] if mm_override is None else mm_override,
                min_block=r.get("min_block", 1) if min_block_override is None else min_block_override)

    players, pv, pc, pw, used = [], [], [], [], set()
    for b in (blocs or []):
        b = [c for c in b if c in members]
        if len(b) < 2:
            continue
        ix = [members.index(c) for c in b]
        players.append("+".join(b)); pv.append(v[ix].sum()); pc.append(len(b)); pw.append(weights[ix].sum())
        used |= set(b)
    for i, c in enumerate(members):
        if c not in used:
            players.append(c); pv.append(v[i]); pc.append(1); pw.append(weights[i])
    return evaluate(players, np.array(pv), np.array(pc), np.array(pw), args, len(members))


def evaluate(players, pv, pc, pw, args, n_mem):
    """Power indices, veto checks, metrics and constraint tests for one game.
    players: labels; pv: vote shares; pc: member counts; pw: economic weights."""
    bz, ss, win = power(pv, pc, args, n_mem)
    tot_v, tot_c = pv.sum(), pc.sum()
    blocks = np.array([not wins(tot_v - pv[i], tot_c - pc[i], n_members=n_mem, **args)
                       for i in range(len(pv))])
    passes = np.array([bool(wins(pv[i], pc[i], n_members=n_mem, **args)) for i in range(len(pv))])
    single_ix = [i for i in range(len(pv)) if pc[i] == 1]
    blocking_pairs = [f"{name(players[i])}+{name(players[j])}"
                      for a, i in enumerate(single_ix) for j in single_ix[a + 1:]
                      if not wins(tot_v - pv[i] - pv[j], tot_c - 2, n_members=n_mem, **args)]

    per = pd.DataFrame({"player": players, "name": [name(p) for p in players],
                        "index_weight": 100 * pw, "votes": 100 * pv, "banzhaf": bz,
                        "shapley_shubik": ss, "can_block_alone": blocks, "can_pass_alone": passes})

    single = pc == 1                      # constraints refer to individual members
    sm = int(np.argmin(np.where(single, pw, np.inf)))
    equal_share = 100 / n_mem
    m = {
        "quota": args["quota"], "min_block": args["min_block"],
        "gini_power": gini(bz),
        "hhi_power": ((bz / 100) ** 2).sum() * 10000,
        "max_min_ratio": bz.max() / bz.min() if bz.min() > 0 else np.inf,
        "misalignment": 0.5 * np.abs(bz - 100 * pv).sum(),
        "decisiveness": 100 * win.mean(),   # Coleman: share of coalitions that win
        "spearman_w_power": pd.Series(pw).corr(pd.Series(bz), method="spearman"),
        "n_dummies": int((bz < 1e-9).sum()),
        "smallest_member": name(players[sm]),
        "smallest_member_bz": bz[sm],
        "smallest_vs_equal_share": bz[sm] / equal_share,
        "any_pass_alone": bool(passes.any()),
        "any_block_alone": bool(blocks.any()),
        "n_blocking_pairs": len(blocking_pairs),
        "blocking_pairs": "; ".join(blocking_pairs[:6]) + (" ..." if len(blocking_pairs) > 6 else ""),
    }
    m["C1_no_unilateral_passage"] = not m["any_pass_alone"]
    m["C2_no_unilateral_veto"] = not m["any_block_alone"]
    m["C3_smallest_protected"] = m["smallest_vs_equal_share"] >= MIN_SMALLEST_REL
    m["C4_largest_keeps_most_power"] = bool(bz[np.argmax(pw)] >= bz.max() - 1e-9)
    m["C5_power_tracks_weight"] = m["spearman_w_power"] >= MIN_SPEARMAN
    m["passes_all"] = all(m[c] for c in CONSTRAINTS)

    china = per[per.player.str.contains("CHN")].iloc[0] if per.player.str.contains("CHN").any() else per.iloc[0]
    m.update(china_player=china["name"], china_votes=china.votes, china_bz=china.banzhaf,
             china_ss=china.shapley_shubik, china_blocks_alone=bool(china.can_block_alone))
    for c in ["IND", "ZAF", "SAU"]:
        row = per[per.player == c]
        m[f"{c.lower()}_bz"] = row.banzhaf.iloc[0] if len(row) else np.nan
    return per, m


# ============================================================================
# 4. VETO HEADROOM (single member, rules without a blocking-minority clause)
# ============================================================================
def veto_headroom(weights, members, thresholds, big="CHN"):
    """China blocks alone under sqrt votes iff its vote share > 1 - q. Returns the
    index share it would need, holding others' relative weights fixed. Under G3/G4
    no single member can ever block alone, whatever its size."""
    i = members.index(big)
    x0 = weights[i]
    K = np.sqrt(np.delete(weights, i) / (1 - x0)).sum()
    rows = []
    for q in thresholds:
        r = (1 - q) / q * K
        x_req = r ** 2 / (1 + r ** 2)
        mult = (x_req / (1 - x_req)) / (x0 / (1 - x0))
        row = {"threshold": q, "china_index_now": 100 * x0, "china_index_needed": 100 * x_req,
               "relative_weight_multiplier": mult}
        for g in [0.01, 0.02, 0.03]:
            row[f"years_if_outgrows_rest_by_{int(g * 100)}pct"] = (
                math.log(mult) / math.log(1 + g) if mult > 1 else 0.0)
        rows.append(row)
    return pd.DataFrame(rows)



# ============================================================================
# 4b. CRA REFORM
# ============================================================================
def cra_reform():
    """Evaluate the CRA's current rule and reform options on its actual membership.
    Economic weight for C4/C5 = each member's share of CRA commitments."""
    codes = list(CRA_COMMIT)
    n = len(codes)
    cs = np.array([CRA_COMMIT[c] for c in codes], float)
    cs = cs / cs.sum()
    treaty = CRA_BASIC / n + (1 - CRA_BASIC) * cs
    sq = sqrt_votes(cs)
    sm = dict(quota=0.5, strict=True, mm=False, min_block=1)
    dm = dict(quota=PROPOSED_Q, strict=False, mm=True, min_block=1)
    options = [
        ("Status quo: treaty votes, >50%", treaty, sm),
        (f"Treaty votes, {PROPOSED_Q:.0%} + majority of members", treaty, dm),
        ("Treaty formula with 20% basic votes, >50%", 0.20 / n + 0.80 * cs, sm),
        ("Square root of commitments, >50%", sq, sm),
        (f"Square root of commitments, {PROPOSED_Q:.0%} + majority of members", sq, dm),
    ]
    rows, per_rows = [], []
    for label, v, args in options:
        per, m = evaluate(codes, v, np.ones(n, int), cs, args, n)
        rows.append({"option": label, **{k: m[k] for k in [
            "china_votes", "china_bz", "china_ss", "zaf_bz", "n_dummies", "any_block_alone",
            "n_blocking_pairs", "blocking_pairs", "gini_power", "decisiveness"] + CONSTRAINTS + ["passes_all"]}})
        p = per[["name", "votes", "banzhaf", "shapley_shubik", "can_block_alone"]].copy()
        p.insert(0, "option", label)
        per_rows.append(p)
    return pd.DataFrame(rows), pd.concat(per_rows)


# ============================================================================
# 4c. MAP OF ACCEPTABLE RULES
# ============================================================================
def feasibility_map(weight_sets, fine=True):
    """Search the rule space: weighted threshold x minimum blocking minority x
    share of equal votes x member-majority requirement. weight_sets is a list of
    (label, members, weights). Records which rules pass all five constraints."""
    if fine:
        Q = np.round(np.arange(0.50, 0.7501, 0.01), 2)
        MB, LAM, MM = [1, 2, 3, 4], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], [True, False]
    else:  # coarse grid for 20+ member scenarios (2^20 coalitions per game)
        Q = [0.50, 0.55, 0.60, 0.65, 0.70]
        MB, LAM, MM = [1, 3], [0.0, 0.2, 0.4], [True]
    rows = []
    for label, members, w in weight_sets:
        for mm in MM:
            for mb in MB:
                for lam in LAM:
                    for q in Q:
                        _, m = run_game(members, w, "D", quota_override=float(q), min_block_override=mb,
                                        lam_override=lam, mm_override=mm)
                        rows.append({"weight_set": label, "member_majority": mm, "min_block": mb,
                                     "equal_share": lam, "threshold": q, "passes_all": m["passes_all"],
                                     "china_bz": m["china_bz"], "smallest_bz": m["smallest_member_bz"],
                                     "decisiveness": m["decisiveness"],
                                     "n_blocking_pairs": m["n_blocking_pairs"],
                                     **{c: m[c] for c in CONSTRAINTS}})
        print(f"  map: {label} done")
    return pd.DataFrame(rows)


# ============================================================================
# 4d. HISTORICAL BACKCAST
# ============================================================================
def backcast(series, members=B10, rules=("A", "D", "D60")):
    """Apply the rules to every year HIST_START..REF_YEAR, as if today's members had
    adopted them in HIST_START.
      annual: weights from each year's own data
      design: weights recalculated every RECALC_EVERY years from an AVG_YEARS-year
              average ending the year before (publication lag), then held fixed"""
    cache, rows, mrows, notes = {}, [], [], []
    for mode in ["annual", "design"]:
        for Y in range(HIST_START, REF_YEAR + 1):
            if mode == "annual":
                ref, k = Y, 1
            else:
                ref, k = HIST_START + ((Y - HIST_START) // RECALC_EVERY) * RECALC_EVERY - 1, AVG_YEARS
            if (ref, k) not in cache:
                cache[(ref, k)] = make_dataset(series, k=k, ref=ref, max_lag=2, log=False)[0]
            d = cache[(ref, k)]
            if d.loc[members, ["gdp", "merch"]].isna().any().any():
                miss = d.index[d.loc[members, ["gdp", "merch"]].isna().any(axis=1)].tolist() if False else \
                    [c for c in members if d.loc[c, ["gdp", "merch"]].isna().any()]
                notes.append(f"{mode} {Y}: skipped, data missing for {[name(c) for c in miss]}")
                continue
            w = index_weights(d, members)
            for rule in rules:
                per, m = run_game(members, w, rule)
                rows.append({"mode": mode, "year": Y, "data_through": ref, "rule": rule,
                             "china_index": 100 * w[members.index("CHN")],
                             "china_votes": m["china_votes"], "china_bz": m["china_bz"],
                             "smallest_member": m["smallest_member"], "smallest_bz": m["smallest_member_bz"],
                             "china_blocks_alone": m["china_blocks_alone"],
                             "n_blocking_pairs": m["n_blocking_pairs"], "blocking_pairs": m["blocking_pairs"],
                             "n_dummies": m["n_dummies"], "passes_all": m["passes_all"],
                             **{c: m[c] for c in CONSTRAINTS}})
                if rule == "D":
                    for _, r in per.iterrows():
                        mrows.append({"mode": mode, "year": Y, "member": r["name"],
                                      "index_weight": r.index_weight, "votes": r.votes, "banzhaf": r.banzhaf})
    return pd.DataFrame(rows), pd.DataFrame(mrows), notes


# ============================================================================
# 5. MAIN
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args, _ = ap.parse_known_args()          # ignores Colab/Jupyter kernel arguments
    offline = args.offline or OFFLINE
    os.makedirs(f"{OUT}/figures", exist_ok=True)

    # ---------------- data ----------------
    data, data_avg, notes, live = None, None, [], False
    if not offline:
        try:
            print("Fetching World Bank WDI ...")
            raw = fetch_wdi(list(COUNTRIES), min(HIST_START - AVG_YEARS, REF_YEAR - 10), REF_YEAR)
            raw.to_csv(f"{OUT}/wdi_raw.csv", index=False)
            series = build_series(raw)
            data, notes = make_dataset(series, k=1)
            data_avg, _ = make_dataset(series, k=AVG_YEARS)
            live = True
        except Exception as e:
            print(f"  WDI unavailable ({e}); using Appendix A1 data.")
    if data is None:
        data = offline_dataset()
        notes = ["Offline mode: Appendix A1 (nominal GDP, merchandise trade) only. "
                 "Saudi, PPP, services and multi-year checks skipped."]
    data.to_csv(f"{OUT}/dataset_used.csv")

    def available(members, gv="gdp", tv="merch", d="main"):
        d = data if isinstance(d, str) else d
        return (d is not None and all(c in d.index for c in members)
                and not d.loc[members, [gv, tv]].isna().any().any())

    variants = {
        "Baseline (nominal GDP + merch trade)": dict(gv="gdp", tv="merch", alpha=ALPHA_BASE),
        "GDP only (alpha=1)": dict(gv="gdp", tv="merch", alpha=1.0),
        "PPP GDP + merch trade": dict(gv="gdp_ppp", tv="merch", alpha=ALPHA_BASE),
        "Nominal GDP + total trade (incl. services)": dict(gv="gdp", tv="total_trade", alpha=ALPHA_BASE),
        "PPP GDP + total trade": dict(gv="gdp_ppp", tv="total_trade", alpha=ALPHA_BASE),
        f"{AVG_YEARS}-year average": dict(gv="gdp", tv="merch", alpha=ALPHA_BASE, avg=True),
    }
    BASE = "Baseline (nominal GDP + merch trade)"
    summary_rows, member_rows, skipped = [], [], []

    def record(scn, var, rule, per, m, blocs="No blocs (a priori)"):
        summary_rows.append({"scenario": scn, "variant": var, "blocs": blocs, "rule": rule,
                             "rule_label": RULES[rule]["label"], **m})
        p = per.copy()
        p.insert(0, "blocs", blocs); p.insert(0, "rule", rule)
        p.insert(0, "variant", var); p.insert(0, "scenario", scn)
        member_rows.append(p)

    # ---------------- A. scenarios x variants x rules ----------------
    print("Running scenarios ...")
    for scn, members in SCENARIOS.items():
        for var, spec in variants.items():
            if scn.startswith(LARGE) and var not in (BASE, "PPP GDP + merch trade"):
                continue
            d = data_avg if spec.get("avg") else data
            if not available(members, spec["gv"], spec["tv"], d):
                if live:
                    skipped.append(f"{scn} / {var}: data missing")
                continue
            w = index_weights(d, members, spec["alpha"], spec["gv"], spec["tv"])
            for rule in ["P"] + MAIN_RULES:
                per, m = run_game(members, w, rule)
                record(scn, var, rule, per, m)
        print(f"  {scn} done")

    # ---------------- B. CRA benchmark ----------------
    v = np.array([CRA_VOTES[c] for c in B5]) / 100
    bz, ss, _ = power(v, np.ones(5, int), dict(quota=0.5, strict=True, mm=False, min_block=1), 5)
    cra = pd.DataFrame({"member": [name(c) for c in B5], "votes": 100 * v,
                        "banzhaf": bz, "shapley_shubik": ss})
    cra.to_csv(f"{OUT}/cra_benchmark.csv", index=False)

    # ---------------- C. voting blocs ----------------
    print("Bloc scenarios ...")
    bloc_rows = []
    for scn in ["BRICS-10", "BRICS-10+SAU"]:
        members = SCENARIOS[scn]
        if not available(members):
            continue
        w = index_weights(data, members)
        for bname, b in BLOCS.items():
            if b and not all(c in members for grp in b for c in grp):
                continue
            for rule in DESIGN_RULES:
                per, m = run_game(members, w, rule, blocs=b)
                if b is not None:
                    record(scn, BASE, rule, per, m, blocs=bname)
                bloc_rows.append({"scenario": scn, "blocs": bname, "rule": rule,
                                  "china_or_bloc": m["china_player"],
                                  "china_or_bloc_votes": m["china_votes"],
                                  "china_or_bloc_bz": m["china_bz"],
                                  "china_or_bloc_blocks_alone": m["china_blocks_alone"],
                                  "any_player_blocks_alone": m["any_block_alone"],
                                  "smallest_member_bz": m["smallest_member_bz"]})
    blocs_df = pd.DataFrame(bloc_rows)
    blocs_df.to_csv(f"{OUT}/bloc_scenarios.csv", index=False)

    # ---------------- D. threshold sweeps ----------------
    print("Threshold sweeps ...")
    fine = np.round(np.arange(0.50, 0.7501, 0.01), 2)
    sweep = []
    for scn in ["BRICS-10", "BRICS-10+SAU", "BRICS+20"]:
        members = SCENARIOS[scn]
        if not available(members):
            continue
        w = index_weights(data, members)
        grid_q = fine if not scn.startswith(LARGE) else [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
        for mb in [1, 3]:
            for q in grid_q:
                _, m = run_game(members, w, "D", quota_override=float(q), min_block_override=mb)
                _, mcr = run_game(members, w, "D", quota_override=float(q), min_block_override=mb,
                                  blocs=[["CHN", "RUS"]]) if scn != "BRICS+20" else (None, {})
                sweep.append({"scenario": scn, "min_block": mb, "threshold": q,
                              "china_bz": m["china_bz"], "china_ss": m["china_ss"],
                              "india_bz": m["ind_bz"], "smallest_member": m["smallest_member"],
                              "smallest_bz": m["smallest_member_bz"],
                              "smallest_vs_equal_share": m["smallest_vs_equal_share"],
                              "gini_power": m["gini_power"], "decisiveness": m["decisiveness"],
                              "china_blocks_alone": m["china_blocks_alone"],
                              "china_russia_bloc_blocks": mcr.get("china_blocks_alone", np.nan),
                              "passes_all": m["passes_all"]})
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(f"{OUT}/threshold_sweep.csv", index=False)

    # ---------------- E. alpha sweep and alpha x threshold grid ----------------
    print("Alpha grid ...")
    alphas = np.round(np.arange(0, 1.0001, 0.1), 1)
    thr = np.round(np.arange(0.50, 0.7501, 0.025), 3)
    grid, alpha_rows = [], []
    for a in alphas:
        w = index_weights(data, B10, alpha=float(a))
        for q in thr:
            _, m = run_game(B10, w, "D", quota_override=float(q))
            grid.append({"alpha": a, "threshold": q, "china_bz": m["china_bz"],
                         "smallest_bz": m["smallest_member_bz"],
                         "china_blocks_alone": m["china_blocks_alone"], "passes_all": m["passes_all"]})
        for rule in ["A", "D", "D60", "G3"]:
            per, m = run_game(B10, w, rule)
            alpha_rows.append({"alpha": a, "rule": rule, "china_votes": m["china_votes"],
                               "china_bz": m["china_bz"], "smallest_bz": m["smallest_member_bz"],
                               "uae_bz": per.loc[per.player == "ARE", "banzhaf"].iloc[0],
                               "gini_power": m["gini_power"], "passes_all": m["passes_all"]})
    grid = pd.DataFrame(grid)
    grid.to_csv(f"{OUT}/alpha_threshold_grid.csv", index=False)
    pd.DataFrame(alpha_rows).to_csv(f"{OUT}/alpha_sweep.csv", index=False)

    # ---------------- F. veto headroom ----------------
    head = []
    for scn in ["BRICS-10", "BRICS-10+SAU", "BRICS+20"]:
        if available(SCENARIOS[scn]):
            h = veto_headroom(index_weights(data, SCENARIOS[scn]), SCENARIOS[scn], fine)
            h.insert(0, "scenario", scn)
            head.append(h)
    head = pd.concat(head)
    head.to_csv(f"{OUT}/veto_headroom.csv", index=False)

    # ---------------- G. CRA reform ----------------
    print("CRA reform ...")
    cra_opts, cra_members = cra_reform()
    cra_opts.to_csv(f"{OUT}/cra_reform.csv", index=False)
    cra_members.to_csv(f"{OUT}/cra_reform_members.csv", index=False)

    # ---------------- H. map of acceptable rules ----------------
    print("Map of acceptable rules ...")
    wsets = []
    for scn in ["BRICS-10", "BRICS-10+SAU"]:
        for var in [BASE, "PPP GDP + merch trade", "GDP only (alpha=1)", f"{AVG_YEARS}-year average"]:
            spec = variants[var]
            d = data_avg if spec.get("avg") else data
            if available(SCENARIOS[scn], spec["gv"], spec["tv"], d):
                wsets.append((f"{scn} | {var}", SCENARIOS[scn],
                              index_weights(d, SCENARIOS[scn], spec["alpha"], spec["gv"], spec["tv"])))
    fmap = feasibility_map(wsets, fine=True)
    if available(SCENARIOS["BRICS+20"]):
        fmap = pd.concat([fmap, feasibility_map(
            [("BRICS+20 | " + BASE, SCENARIOS["BRICS+20"], index_weights(data, SCENARIOS["BRICS+20"]))],
            fine=False)])
    fmap.to_csv(f"{OUT}/rule_map_all.csv", index=False)
    keys = ["member_majority", "min_block", "equal_share", "threshold"]
    n_sets = fmap.weight_set.nunique()
    b10base = fmap[fmap.weight_set == f"BRICS-10 | {BASE}"].set_index(keys)
    rmap = (fmap.groupby(keys).agg(sets_tested=("passes_all", "size"), sets_passed=("passes_all", "sum"))
            .join(b10base[["china_bz", "smallest_bz", "decisiveness", "n_blocking_pairs"]]
                  .add_prefix("brics10_")))
    rmap["passes_every_set"] = rmap.sets_passed == rmap.sets_tested
    rmap = rmap.reset_index()
    rmap.to_csv(f"{OUT}/rule_map_summary.csv", index=False)
    robust = rmap[rmap.passes_every_set & (rmap.sets_tested == n_sets)]

    # ---------------- I. historical backcast ----------------
    hist, hist_members, hist_notes = pd.DataFrame(), pd.DataFrame(), []
    if live:
        print("Historical backcast ...")
        hist, hist_members, hist_notes = backcast(series)
        hist.to_csv(f"{OUT}/backcast.csv", index=False)
        hist_members.to_csv(f"{OUT}/backcast_members_ruleD.csv", index=False)

    # ---------------- save main tables ----------------
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(f"{OUT}/summary_all_rules.csv", index=False)
    pd.concat(member_rows).to_csv(f"{OUT}/member_level_results.csv", index=False)
    apriori = summary[summary.blocs == "No blocs (a priori)"]
    scorecard = (apriori.assign(col=apriori.scenario + " | " + apriori.variant)
                 .pivot_table(index="rule", columns="col", values="passes_all", aggfunc="first")
                 .reindex(["P"] + MAIN_RULES))
    scorecard.to_csv(f"{OUT}/constraint_scorecard.csv")

    # ---------------- figures ----------------
    print("Figures ...")
    for scn in sweep.scenario.unique():
        s = sweep[(sweep.scenario == scn) & (sweep.min_block == 1)]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(s.threshold * 100, s.china_bz, "-o", ms=3, color="#08519c", label="China (Banzhaf)")
        ax.plot(s.threshold * 100, s.china_ss, "--o", ms=3, color="#3182bd", label="China (Shapley-Shubik)")
        ax.plot(s.threshold * 100, s.india_bz, "-o", ms=3, color="#e6550d", label="India (Banzhaf)")
        ax.plot(s.threshold * 100, s.smallest_bz, "-o", ms=3, color="#636363", label="Smallest member (Banzhaf)")
        ax.axvline(PROPOSED_Q * 100, color="#08519c", lw=0.8, ls=":")
        veto = s[s.china_blocks_alone]
        if len(veto):
            ax.axvspan(veto.threshold.min() * 100 - 0.5, 75.5, color="#fcbba1", alpha=0.4,
                       label="China can block alone")
        cr = s[s.china_russia_bloc_blocks == True]
        if len(cr) and len(cr) != len(veto):
            ax.axvspan(cr.threshold.min() * 100 - 0.5, 75.5, color="#fee0d2", alpha=0.4,
                       label="China-Russia bloc can block")
        ax.set_xlabel("Weighted-vote threshold (plus majority of members), %")
        ax.set_ylabel("Voting power, %")
        ax.set_title(f"Voting power by double-majority threshold, {scn}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{OUT}/figures/threshold_{scn.replace('+', 'plus').replace(' ', '_')}.png", dpi=200)
        plt.close(fig)

    for col, title in [("china_bz", "China's Banzhaf power (%)"),
                       ("smallest_bz", "Smallest member's Banzhaf power (%)")]:
        piv = grid.pivot(index="alpha", columns="threshold", values=col)
        veto = grid.pivot(index="alpha", columns="threshold", values="china_blocks_alone")
        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="Blues")
        ax.set_xticks(range(len(piv.columns)), [f"{c * 100:.1f}" for c in piv.columns], rotation=45, fontsize=7)
        ax.set_yticks(range(len(piv.index)), [f"{a:.1f}" for a in piv.index], fontsize=8)
        for yi in range(piv.shape[0]):
            for xi in range(piv.shape[1]):
                ax.text(xi, yi, f"{piv.values[yi, xi]:.1f}" + ("*" if veto.values[yi, xi] else ""),
                        ha="center", va="center", fontsize=6)
        ax.set_xlabel("Weighted-vote threshold, % (double majority)")
        ax.set_ylabel("alpha (weight on GDP)")
        ax.set_title(f"{title}, BRICS-10  (* = China can block alone)")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(f"{OUT}/figures/heatmap_{col}.png", dpi=200)
        plt.close(fig)

    h10 = head[head.scenario == "BRICS-10"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(h10.threshold * 100, h10.china_index_needed, "-o", ms=3, color="#08519c",
            label="China index share needed to block alone")
    ax.axhline(h10.china_index_now.iloc[0], ls=":", color="grey", label="China's current index share")
    ax.axvline(PROPOSED_Q * 100, color="#08519c", lw=0.8, ls=":")
    ax.set_xlabel("Weighted-vote threshold, %")
    ax.set_ylabel("China's index share, %")
    ax.set_title("Veto headroom under square-root votes, BRICS-10")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/veto_headroom.png", dpi=200)
    plt.close(fig)

    # map of acceptable rules (member-majority on): one panel per blocking-minority size
    fine_sets = fmap[fmap.weight_set.str.startswith("BRICS-10")]
    nf = fine_sets.weight_set.nunique()
    cnt = (fine_sets[fine_sets.member_majority].groupby(["min_block", "equal_share", "threshold"])
           .passes_all.sum().reset_index())
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
    for ax, mb in zip(axes, [1, 2, 3, 4]):
        piv = cnt[cnt.min_block == mb].pivot(index="equal_share", columns="threshold", values="passes_all")
        im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="Greens", vmin=0, vmax=nf)
        ax.set_xticks(range(0, len(piv.columns), 5), [f"{c * 100:.0f}" for c in piv.columns[::5]])
        ax.set_yticks(range(len(piv.index)), [f"{l:.1f}" for l in piv.index])
        if mb == 1 and PROPOSED_Q in list(piv.columns):
            ax.plot(list(piv.columns).index(PROPOSED_Q), 0, "r*", ms=12)
        ax.set_title(f"Blocking minority >= {mb} member{'s' if mb > 1 else ''}", fontsize=9)
        ax.set_xlabel("Weighted-vote threshold, %")
    axes[0].set_ylabel("Share of votes distributed equally")
    fig.colorbar(im, ax=axes, label=f"Scenarios passed (of {nf})")
    fig.suptitle("Rules satisfying all five constraints (double majority; red star = proposed rule)", fontsize=10)
    fig.savefig(f"{OUT}/figures/rule_map.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    if len(hist):
        for mode in ["design", "annual"]:
            h = hist[hist["mode"] == mode]
            fig, ax = plt.subplots(figsize=(9, 4.8))
            hA, hD, h60 = (h[h.rule == r].set_index("year") for r in ["A", "D", "D60"])
            ax.plot(hD.index, hD.china_index, color="#9ecae1", label="China: index share")
            ax.plot(hD.index, hD.china_votes, color="#6baed6", ls="--", label="China: square-root vote share")
            ax.plot(hA.index, hA.china_bz, color="#bdbdbd", label="China: power, Rule A (>50%)")
            ax.plot(hD.index, hD.china_bz, color="#08519c", lw=2, label=f"China: power, Rule D ({PROPOSED_Q:.0%})")
            ax.plot(hD.index, hD.smallest_bz, color="#636363", lw=2, label="Smallest member: power, Rule D")
            for yr in hD.index[hD.china_blocks_alone]:
                ax.axvspan(yr - 0.5, yr + 0.5, color="#fcbba1", alpha=0.5)
            ax.set_ylabel("%")
            ax.set_title(f"BRICS-10 counterfactual, {HIST_START}-{REF_YEAR} ({mode} weights)")
            ax.legend(fontsize=7, ncol=2)
            fig.tight_layout()
            fig.savefig(f"{OUT}/figures/backcast_china_{mode}.png", dpi=200)
            plt.close(fig)
        hm = hist_members[hist_members["mode"] == "design"].pivot(index="year", columns="member", values="banzhaf")
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for col in hm.columns:
            ax.plot(hm.index, hm[col], label=col, lw=2 if col == "China" else 1)
        ax.axhline(MIN_SMALLEST_REL * 100 / len(B10), ls=":", color="grey", lw=0.8, label="C3 floor")
        ax.set_ylabel("Banzhaf power under Rule D, %")
        ax.set_title(f"Every member's power under Rule D, {HIST_START}-{REF_YEAR} (design weights)")
        ax.legend(fontsize=7, ncol=3)
        fig.tight_layout()
        fig.savefig(f"{OUT}/figures/backcast_members.png", dpi=200)
        plt.close(fig)

    b10 = apriori[(apriori.scenario == "BRICS-10") & (apriori.variant == BASE)
                  & apriori.rule.isin(MAIN_RULES)].set_index("rule").loc[MAIN_RULES].reset_index()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(b10))
    ax.bar(x - 0.2, b10.china_bz, 0.4, color="#6baed6", label="China voting power")
    ax.bar(x + 0.2, b10.smallest_member_bz, 0.4, color="#bdbdbd", label="Smallest member's voting power")
    for i, r in b10.iterrows():
        ax.text(i - 0.2, r.china_bz + 0.8, f"{r.china_bz:.1f}", ha="center", fontsize=7)
        ax.text(i + 0.2, r.smallest_member_bz + 0.8, f"{r.smallest_member_bz:.1f}", ha="center", fontsize=7)
        if r.china_blocks_alone:
            ax.text(i - 0.2, r.china_bz / 2, "veto", ha="center", fontsize=7, color="white")
    ax.axhline(b10.china_votes.iloc[0], ls=":", color="grey", lw=0.8)
    ax.set_xticks(x, b10.rule)
    ax.set_ylabel("Banzhaf voting power, %")
    ax.set_title("Voting power under alternative decision rules, BRICS-10")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figures/rules_comparison_BRICS-10.png", dpi=200)
    plt.close(fig)

    # ---------------- markdown summary ----------------
    cols = ["rule_label", "china_votes", "china_bz", "china_ss", "ind_bz", "zaf_bz", "sau_bz",
            "smallest_member", "smallest_member_bz", "smallest_vs_equal_share", "gini_power",
            "decisiveness", "china_blocks_alone"] + CONSTRAINTS + ["passes_all"]
    with open(f"{OUT}/data_notes.txt", "w") as f:
        f.write("\n".join(notes + [f"SKIPPED: {s}" for s in skipped]) + "\n")
    with open(f"{OUT}/results_summary.md", "w") as f:
        f.write(f"# Results summary (v2)\n\nData: {'World Bank WDI (live)' if live else 'Appendix A1 (offline)'}. "
                f"Proposed rule D = {PROPOSED_Q:.0%} of square-root votes + majority of members.\n\n")
        f.write("Constraints: C1 no unilateral passage; C2 no unilateral veto; C3 smallest member >= "
                f"{MIN_SMALLEST_REL} x equal share; C4 largest economy keeps most power; "
                f"C5 Spearman(weight, power) >= {MIN_SPEARMAN}.\n\n")
        f.write("Data notes:\n\n" + "\n".join(f"- {n}" for n in notes + [f"SKIPPED: {s}" for s in skipped]) + "\n\n")
        f.write("## Constraint scorecard (True = passes all five)\n\n")
        f.write(scorecard.T.to_markdown() + "\n\n")
        f.write("## CRA reform options (BRICS-5 CRA, actual commitments)\n\n")
        f.write(cra_opts.round(2).to_markdown(index=False) + "\n\n")
        f.write(cra_members.pivot(index="name", columns="option", values="banzhaf").round(1).to_markdown() + "\n\n")
        f.write(f"## Map of acceptable rules\n\n{len(robust)} rule(s) pass all five constraints in all "
                f"{n_sets} weight sets tested (BRICS-10 and BRICS-10+SAU under four weight definitions"
                f"{', plus BRICS+20 on a coarse grid' if n_sets > len(wsets) else ''}).\n\n")
        f.write("Passing share by threshold (member majority on, no blocking-minority clause, no equal votes):\n\n")
        f.write(rmap[(rmap.member_majority) & (rmap.min_block == 1) & (rmap.equal_share == 0)]
                .round(1).to_markdown(index=False) + "\n\n")
        f.write("Robust rules (pass in every set), first 40:\n\n" + robust.head(40).round(1).to_markdown(index=False) + "\n\n")
        if len(hist):
            f.write(f"## Historical backcast, BRICS-10 {HIST_START}-{REF_YEAR}\n\n")
            agg = hist.groupby(["mode", "rule"]).agg(
                years=("year", "size"), years_china_veto=("china_blocks_alone", "sum"),
                years_any_pair_can_block=("n_blocking_pairs", lambda x: int((x > 0).sum())),
                years_passing_all=("passes_all", "sum"), min_smallest_bz=("smallest_bz", "min"),
                china_bz_first=("china_bz", "first"), china_bz_last=("china_bz", "last"),
                china_index_first=("china_index", "first"), china_index_last=("china_index", "last"))
            f.write(agg.round(1).to_markdown() + "\n\n")
            f.write("Years failing any constraint:\n\n")
            fails = hist[~hist.passes_all][["mode", "year", "rule", "china_bz", "smallest_member", "smallest_bz",
                                           "china_blocks_alone", "blocking_pairs"] + CONSTRAINTS]
            f.write((fails.round(1).to_markdown(index=False) if len(fails) else "None.") + "\n\n")
            if hist_notes:
                f.write("Backcast notes:\n\n" + "\n".join(f"- {n}" for n in hist_notes) + "\n\n")
        f.write("## Voting blocs (design rules)\n\n" + blocs_df.round(1).to_markdown(index=False) + "\n\n")
        f.write("## Threshold sweep, key rows\n\n")
        key = sweep[sweep.threshold.isin([0.55, 0.60, 0.65, 0.70])]
        f.write(key.round(2).to_markdown(index=False) + "\n\n")
        f.write("## Veto headroom (no blocking-minority clause)\n\n")
        f.write(head[head.threshold.isin([0.55, 0.60, 0.65, 0.70])].round(1).to_markdown(index=False) + "\n\n")
        f.write("## CRA benchmark (BRICS-5, simple majority)\n\n" + cra.round(2).to_markdown(index=False) + "\n\n")
        for (scn, var), g in apriori.groupby(["scenario", "variant"], sort=False):
            f.write(f"## {scn} - {var}\n\n" + g[cols].round(2).to_markdown(index=False) + "\n\n")

    print("\nBRICS-10 baseline (paper v1: A 41.0 / B 29.1 / C 25.8 / D60 26.0 / E 21.4 / F 30.0):")
    print(b10[["rule", "china_votes", "china_bz", "china_ss", "smallest_member_bz",
               "decisiveness", "china_blocks_alone", "passes_all"]].round(1).to_string(index=False))
    print("\nCRA reform options:")
    print(cra_opts[["option", "china_bz", "zaf_bz", "n_dummies", "any_block_alone", "passes_all"]]
          .round(1).to_string(index=False))
    print(f"\nRules passing all constraints in all {n_sets} weight sets: {len(robust)}")
    if len(hist):
        print("\nBackcast (design weights), Rule D: years China could block alone =",
              int(hist[(hist["mode"] == "design") & (hist.rule == "D")].china_blocks_alone.sum()),
              "| years failing a constraint =",
              int((~hist[(hist["mode"] == "design") & (hist.rule == "D")].passes_all).sum()))
    if skipped:
        print("\nSkipped (see data_notes.txt):\n  " + "\n  ".join(skipped))
    print(f"\nAll outputs written to ./{OUT}/")


if __name__ == "__main__":
    main()
