"""
Voltex ShopFloor Analyst
generate_data.py — Synthetic Store Network Data Generator

Generates 12 weeks of daily operational data for 20 Voltex stores.
Embeds 6 realistic operational problems that the analyst should detect.

Output:
  data/stores.json        — store master data
  data/daily_metrics.json — daily performance metrics per store
  data/promotions.json    — promotional calendar
"""

import json
import random
import numpy as np
from pathlib import Path
from datetime import date, timedelta

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

RANDOM_SEED   = 42
NUM_WEEKS     = 12
START_DATE    = date(2026, 1, 5)   # Monday

CATEGORIES = [
    "computing",
    "tv_audio",
    "white_goods",
    "phones",
    "gaming",
    "smart_home",
]

# ─────────────────────────────────────────────
# STORE MASTER DATA
# ─────────────────────────────────────────────

STORES = [
    # London — 4 stores
    {"id": "S01", "name": "Voltex Westfield",        "city": "London",       "region": "London",     "tier": "large",  "sqft": 28000},
    {"id": "S02", "name": "Voltex Oxford Street",    "city": "London",       "region": "London",     "tier": "large",  "sqft": 24000},
    {"id": "S03", "name": "Voltex Croydon",          "city": "London",       "region": "London",     "tier": "medium", "sqft": 14000},
    {"id": "S04", "name": "Voltex Bromley",          "city": "London",       "region": "London",     "tier": "medium", "sqft": 12000},

    # South East — 3 stores
    {"id": "S05", "name": "Voltex Brighton",         "city": "Brighton",     "region": "South East", "tier": "large",  "sqft": 18000},
    {"id": "S06", "name": "Voltex Reading",          "city": "Reading",      "region": "South East", "tier": "medium", "sqft": 13000},
    {"id": "S07", "name": "Voltex Guildford",        "city": "Guildford",    "region": "South East", "tier": "medium", "sqft": 11000},

    # Midlands — 4 stores
    {"id": "S08", "name": "Voltex Bullring",         "city": "Birmingham",   "region": "Midlands",   "tier": "large",  "sqft": 26000},
    {"id": "S09", "name": "Voltex Nottingham",       "city": "Nottingham",   "region": "Midlands",   "tier": "large",  "sqft": 19000},
    {"id": "S10", "name": "Voltex Leicester",        "city": "Leicester",    "region": "Midlands",   "tier": "medium", "sqft": 12000},
    {"id": "S11", "name": "Voltex Coventry",         "city": "Coventry",     "region": "Midlands",   "tier": "small",  "sqft": 7000},

    # North West — 3 stores
    {"id": "S12", "name": "Voltex Trafford Centre", "city": "Manchester",   "region": "North West", "tier": "large",  "sqft": 25000},
    {"id": "S13", "name": "Voltex Liverpool",        "city": "Liverpool",    "region": "North West", "tier": "medium", "sqft": 15000},
    {"id": "S14", "name": "Voltex Preston",          "city": "Preston",      "region": "North West", "tier": "medium", "sqft": 10000},

    # North East — 3 stores
    {"id": "S15", "name": "Voltex Newcastle",        "city": "Newcastle",    "region": "North East", "tier": "medium", "sqft": 14000},
    {"id": "S16", "name": "Voltex Leeds",            "city": "Leeds",        "region": "North East", "tier": "medium", "sqft": 13000},
    {"id": "S17", "name": "Voltex Sheffield",        "city": "Sheffield",    "region": "North East", "tier": "small",  "sqft": 8000},

    # Scotland — 3 stores
    {"id": "S18", "name": "Voltex Glasgow",          "city": "Glasgow",      "region": "Scotland",   "tier": "medium", "sqft": 16000},
    {"id": "S19", "name": "Voltex Edinburgh",        "city": "Edinburgh",    "region": "Scotland",   "tier": "medium", "sqft": 13000},
    {"id": "S20", "name": "Voltex Aberdeen",         "city": "Aberdeen",     "region": "Scotland",   "tier": "small",  "sqft": 7500},
]

# Base performance profiles by tier
TIER_PROFILES = {
    "large" : {"daily_footfall": 1800, "conversion_rate": 0.22, "avg_transaction": 285},
    "medium": {"daily_footfall": 900,  "conversion_rate": 0.24, "avg_transaction": 265},
    "small" : {"daily_footfall": 450,  "conversion_rate": 0.26, "avg_transaction": 245},
}

# Category revenue split (% of total)
CATEGORY_SPLIT = {
    "computing"  : 0.28,
    "tv_audio"   : 0.22,
    "white_goods": 0.20,
    "phones"     : 0.14,
    "gaming"     : 0.10,
    "smart_home" : 0.06,
}

# Base attach rates by category
BASE_ATTACH_RATES = {
    "voltcare": {
        "computing"  : 0.32,
        "tv_audio"   : 0.28,
        "white_goods": 0.35,
        "phones"     : 0.18,
        "gaming"     : 0.12,
        "smart_home" : 0.15,
    },
    "voltmobile": 0.22,   # % of phone sales that also take a SIM
    "voltinstall": {
        "white_goods": 0.45,
        "tv_audio"   : 0.20,
    },
}

# ─────────────────────────────────────────────
# PROMOTIONAL CALENDAR
# ─────────────────────────────────────────────

def build_promotions(start_date: date, num_weeks: int) -> list[dict]:
    """
    Builds a 12-week promotional calendar.
    Some promotions are network-wide, some are regional.
    """
    promotions = []
    end_date = start_date + timedelta(weeks=num_weeks)

    promo_events = [
        {
            "id"         : "P01",
            "name"       : "January Tech Sale",
            "start"      : start_date,
            "end"        : start_date + timedelta(days=13),
            "scope"      : "network",
            "categories" : ["computing", "tv_audio"],
            "uplift"     : 0.25,
            "stores"     : [s["id"] for s in STORES],
        },
        {
            "id"         : "P02",
            "name"       : "VoltCare February Offer",
            "start"      : start_date + timedelta(weeks=4),
            "end"        : start_date + timedelta(weeks=6),
            "scope"      : "network",
            "categories" : ["computing", "white_goods"],
            "uplift"     : 0.15,
            "stores"     : [s["id"] for s in STORES],
        },
        {
            "id"         : "P03",
            "name"       : "Midlands White Goods Event",
            "start"      : start_date + timedelta(weeks=6),
            "end"        : start_date + timedelta(weeks=8),
            "scope"      : "regional",
            "categories" : ["white_goods"],
            "uplift"     : 0.30,
            "stores"     : ["S08", "S09", "S10", "S11"],
        },
        {
            "id"         : "P04",
            "name"       : "Gaming Spring Launch",
            "start"      : start_date + timedelta(weeks=8),
            "end"        : start_date + timedelta(weeks=10),
            "scope"      : "network",
            "categories" : ["gaming"],
            "uplift"     : 0.35,
            "stores"     : [s["id"] for s in STORES],
        },
        {
            "id"         : "P05",
            "name"       : "Scotland Spring Event",
            "start"      : start_date + timedelta(weeks=9),
            "end"        : start_date + timedelta(weeks=11),
            "scope"      : "regional",
            "categories" : ["computing", "phones"],
            "uplift"     : 0.20,
            "stores"     : ["S18", "S19", "S20"],
        },
    ]

    for promo in promo_events:
        if promo["start"] < end_date:
            promotions.append({
                "id"        : promo["id"],
                "name"      : promo["name"],
                "start_date": promo["start"].isoformat(),
                "end_date"  : promo["end"].isoformat(),
                "scope"     : promo["scope"],
                "categories": promo["categories"],
                "uplift"    : promo["uplift"],
                "stores"    : promo["stores"],
            })

    return promotions


def get_active_promotions(store_id: str, current_date: date, promotions: list[dict]) -> list[dict]:
    active = []
    for p in promotions:
        start = date.fromisoformat(p["start_date"])
        end   = date.fromisoformat(p["end_date"])
        if start <= current_date <= end and store_id in p["stores"]:
            active.append(p)
    return active


# ─────────────────────────────────────────────
# EMBEDDED OPERATIONAL PROBLEMS
# ─────────────────────────────────────────────

def get_store_modifiers(store_id: str, current_date: date, week_num: int) -> dict:
    """
    Returns performance modifiers for each store.
    Six embedded problems that the analyst should detect.
    """
    mods = {
        "voltcare_computing_multiplier" : 1.0,
        "conversion_multiplier"         : 1.0,
        "voltcare_attach_bonus"         : 0.0,
        "nps_modifier"                  : 0,
        "staffing_ratio"                : 1.0,
        "recovery_store"                : False,
    }

    # ── PROBLEM 1: S06 Reading — declining laptop (computing) VoltCare attach
    # Staff change happened at the start of week 5. Attach rate drops significantly.
    if store_id == "S06":
        if week_num >= 5:
            mods["voltcare_computing_multiplier"] = 0.55  # drops from ~32% to ~18%

    # ── PROBLEM 2: S03 Croydon — high footfall, poor conversion (browse not buy)
    # Conversion rate is 30% below tier average but footfall is 20% above average
    if store_id == "S03":
        mods["conversion_multiplier"] = 0.70
        # footfall handled separately in generator

    # ── PROBLEM 3: S11 Coventry — small store outperforming on VoltCare attach
    # This store has a colleague who proactively recommends VoltCare at every sale
    if store_id == "S11":
        mods["voltcare_attach_bonus"] = 0.12  # 12 percentage points above baseline

    # ── PROBLEM 4: S08 Bullring and S09 Nottingham — same promo, different execution
    # Both are in the Midlands White Goods Event (P03, weeks 6-8)
    # S08 executes well (+30% as intended), S09 only gets +10% due to poor floor placement
    # This is handled in the promo uplift logic below — S09 gets reduced uplift

    # ── PROBLEM 5: S15 Newcastle — NPS deterioration from weekend understaffing
    # Weekends from week 4 onwards, staffing drops to 70% of scheduled
    if store_id == "S15":
        is_weekend = current_date.weekday() >= 5
        if week_num >= 4 and is_weekend:
            mods["staffing_ratio"] = 0.70
            mods["nps_modifier"]   = -18  # NPS drops sharply on weekends

    # ── PROBLEM 6: S17 Sheffield — recovering store
    # Had poor performance in weeks 1-4 (new manager onboarding)
    # Weeks 5+ show steady improvement — trend is positive but absolute numbers still low
    if store_id == "S17":
        if week_num <= 4:
            mods["conversion_multiplier"] = 0.65
        elif week_num <= 8:
            mods["conversion_multiplier"] = 0.75 + (week_num - 4) * 0.03
        else:
            mods["conversion_multiplier"] = 0.90
        mods["recovery_store"] = True

    return mods


# ─────────────────────────────────────────────
# DAILY METRIC GENERATOR
# ─────────────────────────────────────────────

def generate_daily_metrics(
    stores      : list[dict],
    promotions  : list[dict],
    start_date  : date,
    num_weeks   : int,
    rng         : np.random.Generator,
) -> list[dict]:

    records = []
    end_date = start_date + timedelta(weeks=num_weeks)
    current  = start_date
    week_num = 1

    while current < end_date:
        if current.weekday() == 0:  # Monday
            week_num = ((current - start_date).days // 7) + 1

        is_weekend = current.weekday() >= 5
        day_of_week = current.strftime("%A")

        for store in stores:
            sid     = store["id"]
            profile = TIER_PROFILES[store["tier"]]

            # Get active promotions for this store today
            active_promos = get_active_promotions(sid, current, promotions)

            # Get store-specific modifiers
            mods = get_store_modifiers(sid, current, week_num)

            # ── FOOTFALL
            base_footfall = profile["daily_footfall"]
            weekend_mult  = 1.45 if is_weekend else 1.0
            week_trend    = 1.0 + 0.02 * np.sin(week_num * np.pi / 6)  # gentle seasonal wave
            noise         = rng.normal(1.0, 0.08)

            # S03 Croydon has higher footfall (problem 2)
            footfall_boost = 1.20 if sid == "S03" else 1.0

            footfall = max(50, int(
                base_footfall * weekend_mult * week_trend * noise * footfall_boost
            ))

            # ── CONVERSION RATE
            base_conversion = profile["conversion_rate"]
            conversion = base_conversion * mods["conversion_multiplier"] * rng.normal(1.0, 0.05)
            conversion = float(np.clip(conversion, 0.05, 0.55))

            # ── TRANSACTIONS
            transactions = max(10, int(footfall * conversion))

            # ── REVENUE BY CATEGORY
            avg_tx  = profile["avg_transaction"] * rng.normal(1.0, 0.06)
            revenue = transactions * avg_tx

            category_revenue = {}
            category_units   = {}

            for cat, split in CATEGORY_SPLIT.items():
                cat_rev   = revenue * split
                promo_mult = 1.0

                # Apply promotional uplift
                for promo in active_promos:
                    if cat in promo["categories"]:
                        # Problem 4: S09 Nottingham gets reduced uplift on P03
                        if sid == "S09" and promo["id"] == "P03":
                            promo_mult *= 1.10
                        else:
                            promo_mult *= (1 + promo["uplift"])

                cat_rev *= promo_mult * rng.normal(1.0, 0.07)
                category_revenue[cat] = round(cat_rev, 2)

                # Estimate units from revenue (rough ASP by category)
                asp = {"computing": 680, "tv_audio": 520, "white_goods": 480,
                       "phones": 380, "gaming": 65, "smart_home": 85}
                category_units[cat] = max(1, int(cat_rev / asp[cat]))

            # ── VOLTCARE ATTACH RATES
            voltcare_attach = {}
            for cat in CATEGORIES:
                base_rate = BASE_ATTACH_RATES["voltcare"][cat]

                # Problem 1: S06 computing attach drops after week 4
                if cat == "computing":
                    base_rate *= mods["voltcare_computing_multiplier"]

                # Problem 3: S11 has a bonus attach rate across all categories
                base_rate += mods["voltcare_attach_bonus"]

                noise_attach = rng.normal(1.0, 0.08)
                voltcare_attach[cat] = float(np.clip(base_rate * noise_attach, 0.02, 0.75))

            # ── VOLTMOBILE ATTACH (phones only)
            voltmobile_attach = float(np.clip(
                BASE_ATTACH_RATES["voltmobile"] * rng.normal(1.0, 0.10), 0.05, 0.60
            ))

            # ── VOLTINSTALL ATTACH
            voltinstall_attach = {
                "white_goods": float(np.clip(
                    BASE_ATTACH_RATES["voltinstall"]["white_goods"] * rng.normal(1.0, 0.08),
                    0.10, 0.80
                )),
                "tv_audio": float(np.clip(
                    BASE_ATTACH_RATES["voltinstall"]["tv_audio"] * rng.normal(1.0, 0.10),
                    0.05, 0.50
                )),
            }

            # ── NPS
            base_nps = {
                "large" : 38,
                "medium": 42,
                "small" : 45,
            }[store["tier"]]

            nps = int(np.clip(
                base_nps + mods["nps_modifier"] + rng.normal(0, 6),
                -30, 80
            ))

            # ── STAFFING
            base_staff = {
                "large" : 32,
                "medium": 18,
                "small" : 9,
            }[store["tier"]]

            if is_weekend:
                base_staff = int(base_staff * 1.20)

            scheduled_hours = base_staff * 8
            actual_hours    = int(scheduled_hours * mods["staffing_ratio"] * rng.normal(1.0, 0.04))
            actual_hours    = max(int(scheduled_hours * 0.50), actual_hours)

            # ── BUILD RECORD
            record = {
                "store_id"          : sid,
                "date"              : current.isoformat(),
                "week_num"          : week_num,
                "day_of_week"       : day_of_week,
                "is_weekend"        : is_weekend,
                "footfall"          : footfall,
                "conversion_rate"   : round(conversion, 4),
                "transactions"      : transactions,
                "total_revenue"     : round(sum(category_revenue.values()), 2),
                "category_revenue"  : category_revenue,
                "category_units"    : category_units,
                "voltcare_attach"   : {k: round(v, 4) for k, v in voltcare_attach.items()},
                "voltmobile_attach" : round(voltmobile_attach, 4),
                "voltinstall_attach": {k: round(v, 4) for k, v in voltinstall_attach.items()},
                "nps"               : nps,
                "scheduled_hours"   : scheduled_hours,
                "actual_hours"      : actual_hours,
                "staffing_ratio"    : round(actual_hours / scheduled_hours, 3),
                "active_promotions" : [p["id"] for p in active_promos],
            }

            records.append(record)

        current += timedelta(days=1)

    return records


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("Voltex ShopFloor — Data Generator")
    print("=" * 55)

    random.seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    Path("data").mkdir(exist_ok=True)

    # ── Store master data
    print(f"\nGenerating store master data ({len(STORES)} stores)...")
    stores_path = Path("data/stores.json")
    with open(stores_path, "w") as f:
        json.dump(STORES, f, indent=2)
    print(f"  Saved: {stores_path}")

    # ── Promotions
    print(f"\nBuilding promotional calendar ({NUM_WEEKS} weeks)...")
    promotions = build_promotions(START_DATE, NUM_WEEKS)
    promos_path = Path("data/promotions.json")
    with open(promos_path, "w") as f:
        json.dump(promotions, f, indent=2)
    print(f"  Saved: {promos_path} ({len(promotions)} promotions)")

    # ── Daily metrics
    print(f"\nGenerating daily metrics...")
    print(f"  Stores: {len(STORES)}")
    print(f"  Weeks:  {NUM_WEEKS}")
    print(f"  Days:   {NUM_WEEKS * 7}")
    print(f"  Records: {len(STORES) * NUM_WEEKS * 7:,}")

    metrics = generate_daily_metrics(STORES, promotions, START_DATE, NUM_WEEKS, rng)

    metrics_path = Path("data/daily_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {metrics_path} ({len(metrics):,} records)")

    # ── Verification
    print("\nVerification — embedded problems check:")

    # Problem 1: S06 computing attach drop after week 4
    s06_early = [r for r in metrics if r["store_id"] == "S06" and r["week_num"] <= 4]
    s06_late  = [r for r in metrics if r["store_id"] == "S06" and r["week_num"] >= 5]
    s06_early_attach = np.mean([r["voltcare_attach"]["computing"] for r in s06_early])
    s06_late_attach  = np.mean([r["voltcare_attach"]["computing"] for r in s06_late])
    print(f"  P1 S06 computing VoltCare attach: {s06_early_attach:.1%} (wk1-4) → {s06_late_attach:.1%} (wk5+) {'✓' if s06_late_attach < s06_early_attach * 0.75 else '✗'}")

    # Problem 2: S03 conversion
    s03 = [r for r in metrics if r["store_id"] == "S03"]
    s04 = [r for r in metrics if r["store_id"] == "S04"]
    s03_conv = np.mean([r["conversion_rate"] for r in s03])
    s04_conv = np.mean([r["conversion_rate"] for r in s04])
    print(f"  P2 S03 conversion: {s03_conv:.1%} vs S04 (same tier): {s04_conv:.1%} {'✓' if s03_conv < s04_conv * 0.85 else '✗'}")

    # Problem 3: S11 VoltCare attach bonus
    s11 = [r for r in metrics if r["store_id"] == "S11"]
    s17 = [r for r in metrics if r["store_id"] == "S17"]
    s11_vc = np.mean([r["voltcare_attach"]["computing"] for r in s11])
    s17_vc = np.mean([r["voltcare_attach"]["computing"] for r in s17])
    print(f"  P3 S11 VoltCare attach: {s11_vc:.1%} vs S17 (same tier): {s17_vc:.1%} {'✓' if s11_vc > s17_vc * 1.2 else '✗'}")

    # Problem 4: S08 vs S09 promo uplift
    s08_promo = [r for r in metrics if r["store_id"] == "S08" and "P03" in r["active_promotions"]]
    s09_promo = [r for r in metrics if r["store_id"] == "S09" and "P03" in r["active_promotions"]]
    if s08_promo and s09_promo:
        s08_wg = np.mean([r["category_revenue"]["white_goods"] for r in s08_promo])
        s09_wg = np.mean([r["category_revenue"]["white_goods"] for r in s09_promo])
        print(f"  P4 White goods promo: S08 £{s08_wg:,.0f} vs S09 £{s09_wg:,.0f} {'✓' if s08_wg > s09_wg * 1.1 else '✗'}")

    # Problem 5: S15 weekend NPS
    s15_wd  = [r for r in metrics if r["store_id"] == "S15" and not r["is_weekend"] and r["week_num"] >= 4]
    s15_wk  = [r for r in metrics if r["store_id"] == "S15" and r["is_weekend"] and r["week_num"] >= 4]
    if s15_wd and s15_wk:
        s15_wd_nps = np.mean([r["nps"] for r in s15_wd])
        s15_wk_nps = np.mean([r["nps"] for r in s15_wk])
        print(f"  P5 S15 NPS: weekday {s15_wd_nps:.0f} vs weekend {s15_wk_nps:.0f} {'✓' if s15_wk_nps < s15_wd_nps - 10 else '✗'}")

    # Problem 6: S17 recovery trend
    s17_early = [r for r in metrics if r["store_id"] == "S17" and r["week_num"] <= 4]
    s17_late  = [r for r in metrics if r["store_id"] == "S17" and r["week_num"] >= 9]
    s17_early_conv = np.mean([r["conversion_rate"] for r in s17_early])
    s17_late_conv  = np.mean([r["conversion_rate"] for r in s17_late])
    print(f"  P6 S17 conversion recovery: {s17_early_conv:.1%} (wk1-4) → {s17_late_conv:.1%} (wk9+) {'✓' if s17_late_conv > s17_early_conv * 1.2 else '✗'}")

    print(f"\nData generation complete.")
    print(f"Run: python data_pipeline.py  to build the analytics layer")


if __name__ == "__main__":
    main()