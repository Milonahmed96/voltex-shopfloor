"""
Voltex ShopFloor Analyst — Test Suite

Tests cover:
  - Data generation correctness (embedded problems verified)
  - Pipeline computation accuracy (metrics, anomalies, snapshots)
  - Analyst schema validation (StoreActionBrief Pydantic model)
"""

import json
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_data():
    """Load or generate sample data for tests."""
    data_dir = Path("data")

    if not (data_dir / "daily_metrics.json").exists():
        from generate_data import main as gen_main
        gen_main()

    with open(data_dir / "daily_metrics.json") as f:
        metrics = json.load(f)
    with open(data_dir / "stores.json") as f:
        stores = json.load(f)
    with open(data_dir / "promotions.json") as f:
        promotions = json.load(f)

    return metrics, stores, promotions


@pytest.fixture(scope="session")
def dataframe(sample_data):
    """Return loaded DataFrame with flattened columns."""
    from data_pipeline import load_data
    df, stores, promotions = load_data()
    return df, stores, promotions


@pytest.fixture(scope="session")
def benchmarks(dataframe):
    """Return computed network benchmarks."""
    from data_pipeline import compute_network_benchmarks
    df, stores, _ = dataframe
    return compute_network_benchmarks(df, stores)


# ─────────────────────────────────────────────
# DATA GENERATION TESTS
# ─────────────────────────────────────────────

class TestDataGeneration:

    def test_correct_number_of_records(self, sample_data):
        metrics, stores, _ = sample_data
        expected = 20 * 12 * 7  # 20 stores × 12 weeks × 7 days
        assert len(metrics) == expected, f"Expected {expected} records, got {len(metrics)}"

    def test_all_stores_present(self, sample_data):
        metrics, stores, _ = sample_data
        store_ids_in_data = set(r["store_id"] for r in metrics)
        store_ids_expected = set(s["id"] for s in stores)
        assert store_ids_in_data == store_ids_expected

    def test_all_categories_present(self, sample_data):
        metrics, _, _ = sample_data
        required = ["computing", "tv_audio", "white_goods", "phones", "gaming", "smart_home"]
        first = metrics[0]
        for cat in required:
            assert cat in first["category_revenue"], f"Missing category: {cat}"
            assert cat in first["voltcare_attach"], f"Missing VoltCare attach for: {cat}"

    def test_problem1_voltcare_attach_collapse(self, sample_data):
        """S06 Reading computing VoltCare attach should drop after week 4."""
        metrics, _, _ = sample_data
        s06_early = [r for r in metrics if r["store_id"] == "S06" and r["week_num"] <= 4]
        s06_late  = [r for r in metrics if r["store_id"] == "S06" and r["week_num"] >= 5]

        early_attach = np.mean([r["voltcare_attach"]["computing"] for r in s06_early])
        late_attach  = np.mean([r["voltcare_attach"]["computing"] for r in s06_late])

        assert late_attach < early_attach * 0.75, (
            f"Expected significant attach drop after week 4. "
            f"Early: {early_attach:.1%}, Late: {late_attach:.1%}"
        )

    def test_problem2_croydon_low_conversion(self, sample_data):
        """S03 Croydon should have lower conversion than S04 Bromley (same tier)."""
        metrics, _, _ = sample_data
        s03 = [r for r in metrics if r["store_id"] == "S03"]
        s04 = [r for r in metrics if r["store_id"] == "S04"]

        s03_conv = np.mean([r["conversion_rate"] for r in s03])
        s04_conv = np.mean([r["conversion_rate"] for r in s04])

        assert s03_conv < s04_conv * 0.85, (
            f"S03 conversion {s03_conv:.1%} should be significantly below S04 {s04_conv:.1%}"
        )

    def test_problem3_coventry_high_attach(self, sample_data):
        """S11 Coventry should have higher VoltCare attach than S17 Sheffield (same tier)."""
        metrics, _, _ = sample_data
        s11 = [r for r in metrics if r["store_id"] == "S11"]
        s17 = [r for r in metrics if r["store_id"] == "S17"]

        s11_vc = np.mean([r["voltcare_attach"]["computing"] for r in s11])
        s17_vc = np.mean([r["voltcare_attach"]["computing"] for r in s17])

        assert s11_vc > s17_vc * 1.15, (
            f"S11 VoltCare {s11_vc:.1%} should be above S17 {s17_vc:.1%}"
        )

    def test_problem4_promo_execution_gap(self, sample_data):
        """S08 Bullring should outperform S09 Nottingham on white goods during P03."""
        metrics, _, _ = sample_data
        s08_promo = [r for r in metrics if r["store_id"] == "S08"
                     and "P03" in r["active_promotions"]]
        s09_promo = [r for r in metrics if r["store_id"] == "S09"
                     and "P03" in r["active_promotions"]]

        if s08_promo and s09_promo:
            s08_wg = np.mean([r["category_revenue"]["white_goods"] for r in s08_promo])
            s09_wg = np.mean([r["category_revenue"]["white_goods"] for r in s09_promo])
            assert s08_wg > s09_wg, (
                f"S08 white goods £{s08_wg:,.0f} should exceed S09 £{s09_wg:,.0f} during P03"
            )

    def test_problem5_newcastle_weekend_nps(self, sample_data):
        """S15 Newcastle weekend NPS should be lower than weekday NPS after week 4."""
        metrics, _, _ = sample_data
        s15_wd = [r for r in metrics if r["store_id"] == "S15"
                  and not r["is_weekend"] and r["week_num"] >= 4]
        s15_wk = [r for r in metrics if r["store_id"] == "S15"
                  and r["is_weekend"] and r["week_num"] >= 4]

        wd_nps = np.mean([r["nps"] for r in s15_wd])
        wk_nps = np.mean([r["nps"] for r in s15_wk])

        assert wk_nps < wd_nps - 8, (
            f"S15 weekend NPS {wk_nps:.0f} should be significantly below weekday {wd_nps:.0f}"
        )

    def test_problem6_sheffield_recovery(self, sample_data):
        """S17 Sheffield conversion should improve from early to late weeks."""
        metrics, _, _ = sample_data
        s17_early = [r for r in metrics if r["store_id"] == "S17" and r["week_num"] <= 4]
        s17_late  = [r for r in metrics if r["store_id"] == "S17" and r["week_num"] >= 9]

        early_conv = np.mean([r["conversion_rate"] for r in s17_early])
        late_conv  = np.mean([r["conversion_rate"] for r in s17_late])

        assert late_conv > early_conv * 1.15, (
            f"S17 conversion should recover. Early: {early_conv:.1%}, Late: {late_conv:.1%}"
        )

    def test_revenue_is_positive(self, sample_data):
        metrics, _, _ = sample_data
        for r in metrics:
            assert r["total_revenue"] > 0, f"Non-positive revenue in record: {r['store_id']} {r['date']}"

    def test_conversion_rate_in_range(self, sample_data):
        metrics, _, _ = sample_data
        for r in metrics:
            assert 0 < r["conversion_rate"] < 1, (
                f"Conversion rate out of range: {r['conversion_rate']} for {r['store_id']}"
            )

    def test_voltcare_attach_in_range(self, sample_data):
        metrics, _, _ = sample_data
        for r in metrics:
            for cat, rate in r["voltcare_attach"].items():
                assert 0 <= rate <= 1, (
                    f"VoltCare attach out of range: {rate} for {r['store_id']} {cat}"
                )

    def test_promotions_have_required_fields(self, sample_data):
        _, _, promotions = sample_data
        required = ["id", "name", "start_date", "end_date", "stores", "uplift"]
        for p in promotions:
            for field in required:
                assert field in p, f"Promotion missing field: {field}"

    def test_weekday_flag_correct(self, sample_data):
        metrics, _, _ = sample_data
        for r in metrics[:100]:  # check first 100
            d = date.fromisoformat(r["date"])
            expected_weekend = d.weekday() >= 5
            assert r["is_weekend"] == expected_weekend, (
                f"Weekend flag wrong for {r['date']}"
            )


# ─────────────────────────────────────────────
# PIPELINE TESTS
# ─────────────────────────────────────────────

class TestDataPipeline:

    def test_benchmarks_have_all_tiers(self, benchmarks):
        for tier in ["large", "medium", "small"]:
            assert tier in benchmarks, f"Missing tier in benchmarks: {tier}"

    def test_benchmarks_have_required_metrics(self, benchmarks):
        required = [
            "footfall", "conversion_rate", "total_revenue", "nps",
            "voltmobile_attach", "voltinstall_wg",
        ]
        for tier in ["large", "medium", "small"]:
            for metric in required:
                assert metric in benchmarks[tier], (
                    f"Missing metric {metric} in {tier} benchmarks"
                )

    def test_large_tier_higher_footfall_than_small(self, benchmarks):
        assert benchmarks["large"]["footfall"] > benchmarks["small"]["footfall"]

    def test_store_snapshot_builds_correctly(self, dataframe, benchmarks):
        from data_pipeline import build_store_snapshot
        df, stores, promotions = dataframe
        snapshot = build_store_snapshot("S06", "2026-02-16", df, stores, promotions, benchmarks)

        assert snapshot.store_id    == "S06"
        assert snapshot.store_name  == "Voltex Reading"
        assert snapshot.snapshot_date == "2026-02-16"
        assert snapshot.tier        == "medium"
        assert snapshot.footfall_today > 0
        assert snapshot.revenue_today > 0
        assert 0 < snapshot.conversion_today < 1

    def test_snapshot_has_all_categories(self, dataframe, benchmarks):
        from data_pipeline import build_store_snapshot
        df, stores, promotions = dataframe
        snapshot = build_store_snapshot("S08", "2026-02-16", df, stores, promotions, benchmarks)

        assert len(snapshot.categories) == 6
        category_names = [c.category for c in snapshot.categories]
        for expected in ["Computing", "TV & Audio", "White Goods", "Phones", "Gaming", "Smart Home"]:
            assert expected in category_names

    def test_anomaly_detection_finds_voltcare_gap(self, dataframe, benchmarks):
        """S06 on week 7 should have a VoltCare computing anomaly detected."""
        from data_pipeline import build_store_snapshot
        df, stores, promotions = dataframe
        snapshot = build_store_snapshot("S06", "2026-02-16", df, stores, promotions, benchmarks)

        voltcare_anomalies = [a for a in snapshot.anomalies if "VoltCare" in a or "computing" in a.lower()]
        assert len(voltcare_anomalies) >= 1, (
            f"Expected VoltCare anomaly for S06 week 7. Anomalies: {snapshot.anomalies}"
        )

    def test_anomaly_detection_finds_browse_not_buy(self, dataframe, benchmarks):
        """S03 Croydon should have footfall/conversion divergence anomaly."""
        from data_pipeline import build_store_snapshot
        df, stores, promotions = dataframe

        croydon_dates = df[df["store_id"] == "S03"]["date_str"].tolist()
        found_anomaly = False

        for test_date in croydon_dates[:10]:
            snapshot = build_store_snapshot("S03", test_date, df, stores, promotions, benchmarks)
            if any("browsing" in a.lower() or "conversion" in a.lower() for a in snapshot.anomalies):
                found_anomaly = True
                break

        assert found_anomaly, "Expected browse-not-buy anomaly for S03 Croydon"

    def test_snapshot_peer_stores_in_same_region(self, dataframe, benchmarks):
        from data_pipeline import build_store_snapshot
        df, stores, promotions = dataframe
        snapshot = build_store_snapshot("S06", "2026-02-16", df, stores, promotions, benchmarks)

        store_info     = next(s for s in stores if s["id"] == "S06")
        peer_store_ids = [p["store_id"] for p in snapshot.peer_stores]

        for pid in peer_store_ids:
            peer_info = next(s for s in stores if s["id"] == pid)
            assert peer_info["region"] == store_info["region"], (
                f"Peer store {pid} is in different region"
            )

    def test_snapshot_trend_summary_has_direction(self, dataframe, benchmarks):
        from data_pipeline import build_store_snapshot
        df, stores, promotions = dataframe
        snapshot = build_store_snapshot("S17", "2026-03-16", df, stores, promotions, benchmarks)

        for metric, trend in snapshot.trend_summary.items():
            assert trend["direction"] in ["improving", "declining"], (
                f"Invalid trend direction: {trend['direction']}"
            )

    def test_nps_weekend_gap_is_negative_for_newcastle(self, dataframe, benchmarks):
        """S15 Newcastle should have negative NPS weekend gap (weekends worse)."""
        from data_pipeline import build_store_snapshot
        df, stores, promotions = dataframe

        weekend_dates = df[
            (df["store_id"] == "S15") &
            (df["is_weekend"] == True) &
            (df["week_num"] >= 5)
        ]["date_str"].tolist()

        if weekend_dates:
            snapshot = build_store_snapshot(
                "S15", weekend_dates[0], df, stores, promotions, benchmarks
            )
            if snapshot.nps_weekend_gap is not None:
                assert snapshot.nps_weekend_gap < 0, (
                    f"Expected negative NPS weekend gap for S15, got {snapshot.nps_weekend_gap}"
                )

    def test_invalid_store_raises_error(self, dataframe, benchmarks):
        from data_pipeline import build_store_snapshot
        df, stores, promotions = dataframe
        with pytest.raises((ValueError, StopIteration)):
            build_store_snapshot("S99", "2026-02-16", df, stores, promotions, benchmarks)


# ─────────────────────────────────────────────
# ANALYST SCHEMA TESTS
# ─────────────────────────────────────────────

class TestAnalystSchema:

    VALID_BRIEF = {
        "store_name"          : "Voltex Reading",
        "snapshot_date"       : "2026-02-16",
        "one_line_summary"    : "VoltCare attach crisis costing £4,200 weekly.",
        "trading_context"     : "Week 7 with VoltCare February promotion active.",
        "priority_actions"    : [
            {
                "rank"             : 1,
                "title"            : "Fix VoltCare attach gap",
                "what_is_happening": "Computing VoltCare at 18.6% vs 31.3% network average.",
                "why_it_matters"   : "Costs £4,200 weekly in lost margin.",
                "what_to_do"       : "Run team huddle on VoltCare pitch today.",
                "peer_benchmark"   : "Voltex Guildford achieves 33.4%.",
                "confidence"       : "HIGH",
            }
        ],
        "one_thing_going_well": "VoltMobile attach at 25.1% is exceptional.",
        "watch_list"          : ["NPS trending down 7.5% over 4 weeks"],
        "overall_health"      : "AMBER",
    }

    def test_valid_brief_creates_model(self):
        from analyst import StoreActionBrief
        brief = StoreActionBrief(**self.VALID_BRIEF)
        assert brief.store_name    == "Voltex Reading"
        assert brief.overall_health == "AMBER"
        assert len(brief.priority_actions) == 1

    def test_all_health_values_accepted(self):
        from analyst import StoreActionBrief
        for health in ["GREEN", "AMBER", "RED"]:
            brief = StoreActionBrief(**{**self.VALID_BRIEF, "overall_health": health})
            assert brief.overall_health == health

    def test_invalid_health_rejected(self):
        from analyst import StoreActionBrief
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            StoreActionBrief(**{**self.VALID_BRIEF, "overall_health": "YELLOW"})

    def test_all_confidence_values_accepted(self):
        from analyst import PriorityAction
        for conf in ["HIGH", "MEDIUM", "LOW"]:
            action = PriorityAction(**{
                **self.VALID_BRIEF["priority_actions"][0],
                "confidence": conf,
            })
            assert action.confidence == conf

    def test_invalid_confidence_rejected(self):
        from analyst import PriorityAction
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PriorityAction(**{
                **self.VALID_BRIEF["priority_actions"][0],
                "confidence": "VERY_HIGH",
            })

    def test_missing_required_field_rejected(self):
        from analyst import StoreActionBrief
        from pydantic import ValidationError
        payload = {k: v for k, v in self.VALID_BRIEF.items() if k != "store_name"}
        with pytest.raises(ValidationError):
            StoreActionBrief(**payload)

    def test_priority_action_requires_rank(self):
        from analyst import PriorityAction
        from pydantic import ValidationError
        payload = {k: v for k, v in self.VALID_BRIEF["priority_actions"][0].items()
                   if k != "rank"}
        with pytest.raises(ValidationError):
            PriorityAction(**payload)

    def test_empty_priority_actions_accepted(self):
        from analyst import StoreActionBrief
        brief = StoreActionBrief(**{**self.VALID_BRIEF, "priority_actions": []})
        assert brief.priority_actions == []

    def test_multiple_priority_actions(self):
        from analyst import StoreActionBrief
        action2 = {**self.VALID_BRIEF["priority_actions"][0], "rank": 2, "confidence": "MEDIUM"}
        brief   = StoreActionBrief(**{**self.VALID_BRIEF, "priority_actions":
                                      [self.VALID_BRIEF["priority_actions"][0], action2]})
        assert len(brief.priority_actions) == 2

    def test_watch_list_can_be_empty(self):
        from analyst import StoreActionBrief
        brief = StoreActionBrief(**{**self.VALID_BRIEF, "watch_list": []})
        assert brief.watch_list == []


# ─────────────────────────────────────────────
# ANALYST UNIT TESTS (mocked API)
# ─────────────────────────────────────────────

class TestAnalystUnit:

    MOCK_BRIEF_JSON = json.dumps({
        "store_name"          : "Voltex Reading",
        "snapshot_date"       : "2026-02-16",
        "one_line_summary"    : "VoltCare attach crisis.",
        "trading_context"     : "Week 7 trading.",
        "priority_actions"    : [{
            "rank": 1, "title": "Fix VoltCare",
            "what_is_happening": "18.6% vs 31.3%",
            "why_it_matters"   : "Costs £4,200 weekly",
            "what_to_do"       : "Team huddle today",
            "peer_benchmark"   : "Guildford at 33.4%",
            "confidence"       : "HIGH",
        }],
        "one_thing_going_well": "VoltMobile excellent.",
        "watch_list"          : ["NPS declining"],
        "overall_health"      : "AMBER",
    })

    def test_generate_brief_returns_store_action_brief(self, dataframe, benchmarks):
        from analyst import ShopFloorAnalyst, StoreActionBrief
        from data_pipeline import build_store_snapshot

        df, stores, promotions = dataframe
        snapshot = build_store_snapshot("S06", "2026-02-16", df, stores, promotions, benchmarks)

        with patch("analyst.Anthropic") as mock_anthropic:
            mock_client   = MagicMock()
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text=self.MOCK_BRIEF_JSON)]
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            a     = ShopFloorAnalyst()
            brief = a.generate_brief(snapshot)

        assert isinstance(brief, StoreActionBrief)
        assert brief.store_name == "Voltex Reading"
        assert brief.overall_health == "AMBER"

    def test_graceful_fallback_on_bad_json(self, dataframe, benchmarks):
        from analyst import ShopFloorAnalyst
        from data_pipeline import build_store_snapshot

        df, stores, promotions = dataframe
        snapshot = build_store_snapshot("S06", "2026-02-16", df, stores, promotions, benchmarks)

        with patch("analyst.Anthropic") as mock_anthropic:
            mock_client   = MagicMock()
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="not valid json {{{")]
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            a     = ShopFloorAnalyst()
            brief = a.generate_brief(snapshot)

        assert brief.overall_health == "AMBER"
        assert brief.priority_actions == []