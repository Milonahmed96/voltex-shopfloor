"""
Voltex ShopFloor Analyst
data_pipeline.py — Metrics Computation and StoreSnapshot Builder

Transforms raw daily metrics into structured StoreSnapshot objects
ready for LLM reasoning. Pre-computes all comparisons, baselines,
and anomaly flags so the analyst receives clean, structured context.

Key outputs:
  StoreSnapshot — full performance context for one store on one date
  NetworkSummary — network-wide benchmarks for peer comparison
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
from pydantic import BaseModel, Field
from typing import Optional


# ─────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────

class CategoryMetrics(BaseModel):
    """Performance metrics for a single product category."""
    category          : str
    units_today       : int
    revenue_today     : float
    units_wow         : float   # week-over-week % change
    revenue_wow       : float
    units_vs_network  : float   # % above/below network average for this tier
    voltcare_attach   : float   # today's attach rate
    voltcare_vs_network: float  # vs network average
    voltcare_trend    : float   # 4-week trend (positive = improving)


class StoreSnapshot(BaseModel):
    """
    Complete performance context for one store on one date.
    This is the structured input the LLM reasoning layer receives.
    """
    # Identity
    store_id     : str
    store_name   : str
    city         : str
    region       : str
    tier         : str
    snapshot_date: str
    week_num     : int

    # Today's headline metrics
    footfall_today          : int
    footfall_vs_baseline    : float   # % vs 4-week average
    footfall_vs_network     : float   # % vs network tier average
    conversion_today        : float
    conversion_vs_baseline  : float
    conversion_vs_network   : float
    transactions_today      : int
    revenue_today           : float
    revenue_vs_baseline     : float
    revenue_vs_network      : float

    # Service attach rates
    voltmobile_attach_today : float
    voltmobile_vs_network   : float
    voltinstall_wg_today    : float
    voltinstall_wg_vs_network: float

    # Customer satisfaction
    nps_today           : int
    nps_vs_baseline     : float
    nps_7day_avg        : float
    nps_weekend_gap     : Optional[float]   # weekend NPS vs weekday NPS

    # Staffing
    staffing_ratio_today: float
    staffing_vs_baseline: float

    # Category breakdown
    categories: list[CategoryMetrics]

    # Active promotions
    active_promotions: list[str]

    # Anomaly flags — pre-computed signals worth the LLM's attention
    anomalies: list[str]

    # Peer benchmarks — top 3 stores in same region
    peer_stores: list[dict]

    # 4-week trend summary
    trend_summary: dict


class NetworkSummary(BaseModel):
    """Network-wide benchmarks by tier."""
    date          : str
    tier_benchmarks: dict   # tier -> metric -> value
    top_stores    : list[dict]
    bottom_stores : list[dict]


# ─────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """Load all data files and return as DataFrame + lists."""
    with open("data/daily_metrics.json") as f:
        metrics = json.load(f)

    with open("data/stores.json") as f:
        stores = json.load(f)

    with open("data/promotions.json") as f:
        promotions = json.load(f)

    df = pd.DataFrame(metrics)
    df["date"]     = pd.to_datetime(df["date"])
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

    # Flatten nested dicts into columns for easier computation
    for cat in ["computing", "tv_audio", "white_goods", "phones", "gaming", "smart_home"]:
        df[f"rev_{cat}"]  = df["category_revenue"].apply(lambda x: x.get(cat, 0))
        df[f"units_{cat}"] = df["category_units"].apply(lambda x: x.get(cat, 0))
        df[f"vc_{cat}"]   = df["voltcare_attach"].apply(lambda x: x.get(cat, 0))

    df["voltinstall_wg"] = df["voltinstall_attach"].apply(lambda x: x.get("white_goods", 0))
    df["voltinstall_tv"] = df["voltinstall_attach"].apply(lambda x: x.get("tv_audio", 0))

    return df, stores, promotions


# ─────────────────────────────────────────────
# NETWORK BENCHMARKS
# ─────────────────────────────────────────────

def compute_network_benchmarks(df: pd.DataFrame, stores: list[dict]) -> dict:
    """
    Compute network-wide average metrics by tier.
    Used for peer comparison in every StoreSnapshot.
    """
    store_tiers = {s["id"]: s["tier"] for s in stores}
    df["tier"]  = df["store_id"].map(store_tiers)

    benchmarks = {}
    for tier in ["large", "medium", "small"]:
        tier_df = df[df["tier"] == tier]
        benchmarks[tier] = {
            "footfall"          : tier_df["footfall"].mean(),
            "conversion_rate"   : tier_df["conversion_rate"].mean(),
            "total_revenue"     : tier_df["total_revenue"].mean(),
            "nps"               : tier_df["nps"].mean(),
            "voltmobile_attach" : tier_df["voltmobile_attach"].mean(),
            "voltinstall_wg"    : tier_df["voltinstall_wg"].mean(),
            "staffing_ratio"    : tier_df["staffing_ratio"].mean(),
            "vc_computing"      : tier_df["vc_computing"].mean(),
            "vc_tv_audio"       : tier_df["vc_tv_audio"].mean(),
            "vc_white_goods"    : tier_df["vc_white_goods"].mean(),
            "vc_phones"         : tier_df["vc_phones"].mean(),
            "vc_gaming"         : tier_df["vc_gaming"].mean(),
            "vc_smart_home"     : tier_df["vc_smart_home"].mean(),
        }

    return benchmarks


# ─────────────────────────────────────────────
# ANOMALY DETECTION
# ─────────────────────────────────────────────

def detect_anomalies(
    today       : pd.Series,
    baseline    : pd.DataFrame,
    network_avg : dict,
    store_info  : dict,
) -> list[str]:
    """
    Detects statistically meaningful deviations worth flagging.
    Returns a list of human-readable anomaly descriptions.
    """
    anomalies = []
    tier = store_info["tier"]

    # Conversion rate anomaly (vs baseline)
    if len(baseline) >= 7:
        base_conv = baseline["conversion_rate"].mean()
        if today["conversion_rate"] < base_conv * 0.80:
            drop = (1 - today["conversion_rate"] / base_conv) * 100
            anomalies.append(
                f"Conversion rate {today['conversion_rate']:.1%} is {drop:.0f}% below "
                f"4-week baseline ({base_conv:.1%})"
            )
        elif today["conversion_rate"] > base_conv * 1.25:
            gain = (today["conversion_rate"] / base_conv - 1) * 100
            anomalies.append(
                f"Conversion rate {today['conversion_rate']:.1%} is {gain:.0f}% above "
                f"4-week baseline ({base_conv:.1%}) — investigate what is driving this"
            )

    # VoltCare computing attach anomaly
    base_vc_comp = network_avg.get("vc_computing", 0.32)
    if today["vc_computing"] < base_vc_comp * 0.70:
        gap = (base_vc_comp - today["vc_computing"]) * 100
        anomalies.append(
            f"Computing VoltCare attach {today['vc_computing']:.1%} is "
            f"{gap:.0f}pp below network average ({base_vc_comp:.1%})"
        )
    elif today["vc_computing"] > base_vc_comp * 1.25:
        gap = (today["vc_computing"] - base_vc_comp) * 100
        anomalies.append(
            f"Computing VoltCare attach {today['vc_computing']:.1%} is "
            f"{gap:.0f}pp above network average — share best practice"
        )

    # NPS anomaly
    if len(baseline) >= 7:
        base_nps = baseline["nps"].mean()
        if today["nps"] < base_nps - 12:
            anomalies.append(
                f"NPS score {today['nps']} is {base_nps - today['nps']:.0f} points below "
                f"4-week average ({base_nps:.0f})"
            )

    # Staffing anomaly
    if today["staffing_ratio"] < 0.80:
        anomalies.append(
            f"Staffing at {today['staffing_ratio']:.0%} of scheduled hours — "
            f"potential service impact"
        )

    # Revenue anomaly
    if len(baseline) >= 7:
        base_rev = baseline["total_revenue"].mean()
        if today["total_revenue"] < base_rev * 0.75:
            drop = (1 - today["total_revenue"] / base_rev) * 100
            anomalies.append(
                f"Revenue £{today['total_revenue']:,.0f} is {drop:.0f}% below "
                f"4-week average (£{base_rev:,.0f})"
            )

    # White goods VoltInstall attach
    base_vi_wg = network_avg.get("voltinstall_wg", 0.45)
    if today["voltinstall_wg"] < base_vi_wg * 0.70:
        gap = (base_vi_wg - today["voltinstall_wg"]) * 100
        anomalies.append(
            f"White goods VoltInstall attach {today['voltinstall_wg']:.1%} is "
            f"{gap:.0f}pp below network average ({base_vi_wg:.1%})"
        )

    # Footfall vs conversion divergence
    net_footfall = network_avg.get("footfall", 900)
    net_conv     = network_avg.get("conversion_rate", 0.24)
    footfall_vs_net = (today["footfall"] - net_footfall) / net_footfall
    conv_vs_net     = (today["conversion_rate"] - net_conv) / net_conv

    if footfall_vs_net > 0.15 and conv_vs_net < -0.15:
        anomalies.append(
            f"High footfall ({today['footfall']:,}) but low conversion "
            f"({today['conversion_rate']:.1%}) — customers browsing but not buying"
        )

    return anomalies


# ─────────────────────────────────────────────
# STORE SNAPSHOT BUILDER
# ─────────────────────────────────────────────

def build_store_snapshot(
    store_id    : str,
    snapshot_date: str,
    df          : pd.DataFrame,
    stores      : list[dict],
    promotions  : list[dict],
    benchmarks  : dict,
) -> StoreSnapshot:
    """
    Builds a complete StoreSnapshot for one store on one date.
    """
    store_info = next(s for s in stores if s["id"] == store_id)
    tier       = store_info["tier"]
    net_avg    = benchmarks[tier]

    # Today's data
    today_mask = (df["store_id"] == store_id) & (df["date_str"] == snapshot_date)
    today_rows = df[today_mask]

    if today_rows.empty:
        raise ValueError(f"No data for store {store_id} on {snapshot_date}")

    today = today_rows.iloc[0]

    # 4-week baseline (28 days before snapshot date, same store)
    snap_dt   = pd.to_datetime(snapshot_date)
    base_mask = (
        (df["store_id"] == store_id) &
        (df["date"] < snap_dt) &
        (df["date"] >= snap_dt - timedelta(days=28))
    )
    baseline = df[base_mask]

    # Helper: % change vs baseline
    def vs_baseline(metric: str) -> float:
        if baseline.empty or baseline[metric].mean() == 0:
            return 0.0
        return float((today[metric] - baseline[metric].mean()) / baseline[metric].mean())

    # Helper: % vs network average
    def vs_network(metric: str, net_key: str) -> float:
        net_val = net_avg.get(net_key, 0)
        if net_val == 0:
            return 0.0
        return float((today[metric] - net_val) / net_val)

    # ── Category metrics
    categories_out = []
    cat_map = {
        "computing"  : "Computing",
        "tv_audio"   : "TV & Audio",
        "white_goods": "White Goods",
        "phones"     : "Phones",
        "gaming"     : "Gaming",
        "smart_home" : "Smart Home",
    }

    for cat in ["computing", "tv_audio", "white_goods", "phones", "gaming", "smart_home"]:
        # Week-over-week: same day last week
        last_week_dt   = snap_dt - timedelta(days=7)
        last_week_mask = (
            (df["store_id"] == store_id) &
            (df["date"].dt.strftime("%Y-%m-%d") == last_week_dt.strftime("%Y-%m-%d"))
        )
        last_week = df[last_week_mask]

        units_wow = 0.0
        rev_wow   = 0.0
        if not last_week.empty:
            lw = last_week.iloc[0]
            if lw[f"units_{cat}"] > 0:
                units_wow = float((today[f"units_{cat}"] - lw[f"units_{cat}"]) / lw[f"units_{cat}"])
            if lw[f"rev_{cat}"] > 0:
                rev_wow = float((today[f"rev_{cat}"] - lw[f"rev_{cat}"]) / lw[f"rev_{cat}"])

        # Units vs network
        net_rev_avg = df[df["store_id"].isin(
            [s["id"] for s in stores if s["tier"] == tier]
        )][f"rev_{cat}"].mean()
        units_vs_net = float(
            (today[f"rev_{cat}"] - net_rev_avg) / net_rev_avg
        ) if net_rev_avg > 0 else 0.0

        # VoltCare trend (4-week)
        vc_trend = 0.0
        if len(baseline) >= 14:
            early = baseline.head(14)[f"vc_{cat}"].mean()
            late  = baseline.tail(14)[f"vc_{cat}"].mean()
            vc_trend = float(late - early)

        categories_out.append(CategoryMetrics(
            category          = cat_map[cat],
            units_today       = int(today[f"units_{cat}"]),
            revenue_today     = round(float(today[f"rev_{cat}"]), 2),
            units_wow         = round(units_wow, 4),
            revenue_wow       = round(rev_wow, 4),
            units_vs_network  = round(units_vs_net, 4),
            voltcare_attach   = round(float(today[f"vc_{cat}"]), 4),
            voltcare_vs_network= round(float(today[f"vc_{cat}"] - net_avg.get(f"vc_{cat}", 0.3)), 4),
            voltcare_trend    = round(vc_trend, 4),
        ))

    # ── NPS weekend gap
    nps_weekend_gap = None
    if len(baseline) >= 14:
        wd_nps = baseline[~baseline["is_weekend"]]["nps"].mean()
        wk_nps = baseline[baseline["is_weekend"]]["nps"].mean()
        if not np.isnan(wd_nps) and not np.isnan(wk_nps):
            nps_weekend_gap = round(float(wk_nps - wd_nps), 1)

    # ── Peer stores (top 3 in same region by revenue, excluding this store)
    region_stores = [
        s["id"] for s in stores
        if s["region"] == store_info["region"] and s["id"] != store_id
    ]
    peer_perf = []
    for pid in region_stores:
        peer_today = df[(df["store_id"] == pid) & (df["date_str"] == snapshot_date)]
        if not peer_today.empty:
            pr = peer_today.iloc[0]
            peer_info = next(s for s in stores if s["id"] == pid)
            peer_perf.append({
                "store_id"        : pid,
                "store_name"      : peer_info["name"],
                "tier"            : peer_info["tier"],
                "revenue"         : round(float(pr["total_revenue"]), 2),
                "conversion_rate" : round(float(pr["conversion_rate"]), 4),
                "nps"             : int(pr["nps"]),
                "vc_computing"    : round(float(pr["vc_computing"]), 4),
                "staffing_ratio"  : round(float(pr["staffing_ratio"]), 3),
            })

    peer_perf = sorted(peer_perf, key=lambda x: x["revenue"], reverse=True)[:3]

    # ── 4-week trend summary
    trend_summary = {}
    if not baseline.empty:
        for metric in ["total_revenue", "conversion_rate", "nps", "voltmobile_attach"]:
            if len(baseline) >= 7:
                early_avg = baseline.head(len(baseline) // 2)[metric].mean()
                late_avg  = baseline.tail(len(baseline) // 2)[metric].mean()
                trend_summary[metric] = {
                    "early_avg": round(float(early_avg), 4),
                    "late_avg" : round(float(late_avg), 4),
                    "direction": "improving" if late_avg > early_avg else "declining",
                    "change_pct": round(float((late_avg - early_avg) / early_avg * 100
                                              if early_avg != 0 else 0), 1),
                }

    # ── Anomalies
    anomalies = detect_anomalies(today, baseline, net_avg, store_info)

    # ── Active promotions
    active_promos = []
    for p in promotions:
        start = pd.to_datetime(p["start_date"])
        end   = pd.to_datetime(p["end_date"])
        if start <= snap_dt <= end and store_id in p["stores"]:
            active_promos.append(p["name"])

    return StoreSnapshot(
        store_id                 = store_id,
        store_name               = store_info["name"],
        city                     = store_info["city"],
        region                   = store_info["region"],
        tier                     = tier,
        snapshot_date            = snapshot_date,
        week_num                 = int(today["week_num"]),
        footfall_today           = int(today["footfall"]),
        footfall_vs_baseline     = round(vs_baseline("footfall"), 4),
        footfall_vs_network      = round(vs_network("footfall", "footfall"), 4),
        conversion_today         = round(float(today["conversion_rate"]), 4),
        conversion_vs_baseline   = round(vs_baseline("conversion_rate"), 4),
        conversion_vs_network    = round(vs_network("conversion_rate", "conversion_rate"), 4),
        transactions_today       = int(today["transactions"]),
        revenue_today            = round(float(today["total_revenue"]), 2),
        revenue_vs_baseline      = round(vs_baseline("total_revenue"), 4),
        revenue_vs_network       = round(vs_network("total_revenue", "total_revenue"), 4),
        voltmobile_attach_today  = round(float(today["voltmobile_attach"]), 4),
        voltmobile_vs_network    = round(vs_network("voltmobile_attach", "voltmobile_attach"), 4),
        voltinstall_wg_today     = round(float(today["voltinstall_wg"]), 4),
        voltinstall_wg_vs_network= round(vs_network("voltinstall_wg", "voltinstall_wg"), 4),
        nps_today                = int(today["nps"]),
        nps_vs_baseline          = round(vs_baseline("nps"), 4),
        nps_7day_avg             = round(float(
            baseline.tail(7)["nps"].mean() if len(baseline) >= 7 else today["nps"]
        ), 1),
        nps_weekend_gap          = nps_weekend_gap,
        staffing_ratio_today     = round(float(today["staffing_ratio"]), 3),
        staffing_vs_baseline     = round(vs_baseline("staffing_ratio"), 4),
        categories               = categories_out,
        active_promotions        = active_promos,
        anomalies                = anomalies,
        peer_stores              = peer_perf,
        trend_summary            = trend_summary,
    )


# ─────────────────────────────────────────────
# AVAILABLE DATES HELPER
# ─────────────────────────────────────────────

def get_available_dates(df: pd.DataFrame) -> list[str]:
    return sorted(df["date_str"].unique().tolist())


def get_store_list(stores: list[dict]) -> list[dict]:
    return [{"id": s["id"], "name": s["name"], "tier": s["tier"],
             "city": s["city"], "region": s["region"]} for s in stores]


# ─────────────────────────────────────────────
# MAIN — smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    df, stores, promotions = load_data()

    print("Computing network benchmarks...")
    benchmarks = compute_network_benchmarks(df, stores)

    # Test snapshot for S06 Reading on week 7 (after the staff change)
    test_store = "S06"
    test_date  = "2026-02-16"  # week 7

    print(f"\nBuilding snapshot: {test_store} on {test_date}")
    snapshot = build_store_snapshot(
        test_store, test_date, df, stores, promotions, benchmarks
    )

    print(f"\n{'='*55}")
    print(f"STORE SNAPSHOT — {snapshot.store_name}")
    print(f"{'='*55}")
    print(f"Date: {snapshot.snapshot_date} | Week {snapshot.week_num} | {snapshot.tier.title()} store")
    print(f"\nHeadline metrics:")
    print(f"  Revenue:    £{snapshot.revenue_today:,.0f}  ({snapshot.revenue_vs_network:+.1%} vs network)")
    print(f"  Footfall:   {snapshot.footfall_today:,}  ({snapshot.footfall_vs_network:+.1%} vs network)")
    print(f"  Conversion: {snapshot.conversion_today:.1%}  ({snapshot.conversion_vs_network:+.1%} vs network)")
    print(f"  NPS:        {snapshot.nps_today}  (7-day avg: {snapshot.nps_7day_avg:.0f})")
    print(f"  Staffing:   {snapshot.staffing_ratio_today:.0%}")

    print(f"\nComputing VoltCare attach: {snapshot.categories[0].voltcare_attach:.1%}")
    print(f"  vs network: {snapshot.categories[0].voltcare_vs_network:+.1%}")

    print(f"\nAnomalies detected ({len(snapshot.anomalies)}):")
    for a in snapshot.anomalies:
        print(f"  ⚠ {a}")

    print(f"\nActive promotions: {snapshot.active_promotions or 'None'}")
    print(f"\nPeer stores ({len(snapshot.peer_stores)}):")
    for p in snapshot.peer_stores:
        print(f"  {p['store_name']}: £{p['revenue']:,.0f} | conv {p['conversion_rate']:.1%} | NPS {p['nps']}")

    print(f"\nTrend summary:")
    for metric, trend in snapshot.trend_summary.items():
        arrow = "↑" if trend["direction"] == "improving" else "↓"
        print(f"  {metric}: {arrow} {trend['change_pct']:+.1f}%")

    print(f"\ndata_pipeline.py working correctly.")
    print(f"Run: python analyst.py  to test the LLM reasoning layer")