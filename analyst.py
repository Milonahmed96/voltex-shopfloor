"""
Voltex ShopFloor Analyst
analyst.py — LLM Reasoning Layer

Takes a StoreSnapshot and produces a StoreActionBrief.
Uses Claude Sonnet to reason like a retail operations analyst —
identifying root causes, ranking priorities by commercial impact,
and generating specific actionable recommendations.
"""

import os
import json
import re
from typing import Literal
from pydantic import BaseModel, Field
from anthropic import Anthropic
from dotenv import load_dotenv
from data_pipeline import (
    StoreSnapshot,
    load_data,
    compute_network_benchmarks,
    build_store_snapshot,
)

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

CLAUDE_MODEL = "claude-sonnet-4-20250514"


# ─────────────────────────────────────────────
# OUTPUT SCHEMA
# ─────────────────────────────────────────────

class PriorityAction(BaseModel):
    """A single ranked action item for the store manager."""
    rank             : int
    title            : str = Field(description="Short action title, under 8 words")
    what_is_happening: str = Field(description="What the data shows — specific numbers")
    why_it_matters   : str = Field(description="Commercial impact if not addressed")
    what_to_do       : str = Field(description="Specific action the manager can take today")
    peer_benchmark   : str = Field(description="What top stores in the region are doing differently")
    confidence       : Literal["HIGH", "MEDIUM", "LOW"]


class StoreActionBrief(BaseModel):
    """
    The daily AI-generated action brief for a store manager.
    Designed to be read in under 2 minutes at the start of a trading day.
    """
    store_name       : str
    snapshot_date    : str
    one_line_summary : str = Field(
        description="Single sentence capturing the most important thing about today"
    )
    trading_context  : str = Field(
        description="2-3 sentences of context: what week is it, any promotions, "
                    "any external factors worth noting"
    )
    priority_actions : list[PriorityAction] = Field(
        description="2 to 3 ranked priority actions, highest commercial impact first"
    )
    one_thing_going_well: str = Field(
        description="One genuine positive to acknowledge — specific and data-grounded"
    )
    watch_list       : list[str] = Field(
        description="1-2 metrics to monitor today that are not yet critical "
                    "but trending in the wrong direction"
    )
    overall_health   : Literal["GREEN", "AMBER", "RED"] = Field(
        description="GREEN = performing well, AMBER = issues need attention, "
                    "RED = urgent action required"
    )


# ─────────────────────────────────────────────
# SNAPSHOT FORMATTER
# ─────────────────────────────────────────────

def format_snapshot_for_prompt(snapshot: StoreSnapshot) -> str:
    """
    Converts a StoreSnapshot into a clean, structured text block
    for the LLM prompt. Pre-formatted for readability and token efficiency.
    """
    lines = []

    lines.append(f"STORE: {snapshot.store_name} ({snapshot.store_id})")
    lines.append(f"Location: {snapshot.city}, {snapshot.region} | Tier: {snapshot.tier.title()}")
    lines.append(f"Date: {snapshot.snapshot_date} | Week {snapshot.week_num} of trading year")
    lines.append("")

    lines.append("HEADLINE METRICS (vs network average for this tier)")
    lines.append(f"  Revenue:         £{snapshot.revenue_today:,.0f}  ({snapshot.revenue_vs_network:+.1%} vs network, {snapshot.revenue_vs_baseline:+.1%} vs 4-week baseline)")
    lines.append(f"  Footfall:        {snapshot.footfall_today:,}  ({snapshot.footfall_vs_network:+.1%} vs network)")
    lines.append(f"  Conversion rate: {snapshot.conversion_today:.1%}  ({snapshot.conversion_vs_network:+.1%} vs network)")
    lines.append(f"  Transactions:    {snapshot.transactions_today:,}")
    lines.append(f"  NPS today:       {snapshot.nps_today}  (7-day avg: {snapshot.nps_7day_avg:.0f}, {snapshot.nps_vs_baseline:+.1%} vs baseline)")
    if snapshot.nps_weekend_gap is not None:
        lines.append(f"  NPS weekend gap: {snapshot.nps_weekend_gap:+.0f} points (weekend vs weekday)")
    lines.append(f"  Staffing:        {snapshot.staffing_ratio_today:.0%} of scheduled hours")
    lines.append("")

    lines.append("SERVICE ATTACH RATES")
    lines.append(f"  VoltMobile (phones): {snapshot.voltmobile_attach_today:.1%}  ({snapshot.voltmobile_vs_network:+.1%} vs network)")
    lines.append(f"  VoltInstall (white goods): {snapshot.voltinstall_wg_today:.1%}  ({snapshot.voltinstall_wg_vs_network:+.1%} vs network)")
    lines.append("")

    lines.append("CATEGORY BREAKDOWN")
    for cat in snapshot.categories:
        vc_gap = f"{cat.voltcare_vs_network:+.1%}" if cat.voltcare_vs_network != 0 else "at network avg"
        trend  = f"↑" if cat.voltcare_trend > 0.01 else "↓" if cat.voltcare_trend < -0.01 else "→"
        lines.append(
            f"  {cat.category:<14} "
            f"units: {cat.units_today:>4}  "
            f"rev: £{cat.revenue_today:>8,.0f}  "
            f"WoW: {cat.revenue_wow:>+6.1%}  "
            f"VoltCare: {cat.voltcare_attach:.1%} ({vc_gap}) {trend}"
        )
    lines.append("")

    if snapshot.active_promotions:
        lines.append(f"ACTIVE PROMOTIONS: {', '.join(snapshot.active_promotions)}")
        lines.append("")

    lines.append("4-WEEK TRENDS")
    for metric, trend in snapshot.trend_summary.items():
        arrow = "↑" if trend["direction"] == "improving" else "↓"
        lines.append(f"  {metric:<22} {arrow} {trend['change_pct']:+.1f}%  (from {trend['early_avg']:.3f} to {trend['late_avg']:.3f})")
    lines.append("")

    if snapshot.anomalies:
        lines.append(f"PRE-COMPUTED ANOMALIES ({len(snapshot.anomalies)} detected)")
        for a in snapshot.anomalies:
            lines.append(f"  ⚠ {a}")
        lines.append("")

    if snapshot.peer_stores:
        lines.append("TOP PEER STORES (same region, ranked by revenue today)")
        for p in snapshot.peer_stores:
            lines.append(
                f"  {p['store_name']} ({p['tier'].title()}): "
                f"£{p['revenue']:,.0f} | "
                f"conv {p['conversion_rate']:.1%} | "
                f"NPS {p['nps']} | "
                f"VoltCare computing {p['vc_computing']:.1%}"
            )

    return "\n".join(lines)


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior retail operations analyst for Voltex Retail, a UK omnichannel technology and appliances retailer. You analyse daily store performance data and generate concise, actionable briefs for store managers.

Your briefs are read by store managers at the start of a trading day. They have 2 minutes to read it before the store opens. Every word must earn its place.

YOUR ANALYTICAL PRINCIPLES:

1. DATA-GROUNDED SPECIFICITY: Never make generic recommendations. Every priority action must cite specific numbers from the data. "Your computing VoltCare attach is 18.6% vs the network average of 31.3%" is good. "Your attach rates need improvement" is not.

2. ROOT CAUSE FIRST: Do not just describe what is wrong — explain why it is likely happening. A drop in VoltCare attach that coincides with a recent period likely indicates a training or process gap, not a market issue.

3. COMMERCIAL IMPACT: Rank priorities by revenue impact. A 1% improvement in VoltCare attach on computing is worth more than a 5% improvement in smart home attach, because computing has higher volume and higher plan value.

4. PEER BENCHMARKING: When a store is underperforming, always compare to the top stores in their region. "Voltex Brighton is converting at 22.3% vs your 16.8% — what are they doing differently?" is actionable. A raw number without context is not.

5. POSITIVE ACKNOWLEDGEMENT: Always find one genuine thing going well, grounded in data. Managers respond better to feedback that is balanced, not purely critical.

6. WATCH LIST: Flag metrics that are not yet critical but trending wrong. Early warning is more valuable than post-mortem analysis.

VOLTEX BUSINESS CONTEXT:
- VoltCare (warranty and care plans) is a high-margin product — attach rate is a key commercial lever
- VoltMobile SIM attach on phone sales drives recurring revenue
- VoltInstall on white goods and TVs reduces returns and increases satisfaction
- NPS directly correlates with repeat purchase rate — a 10-point NPS drop typically reduces repeat visits by 8-12%
- Staffing below 85% of scheduled hours on high-footfall days creates queue pressure that suppresses conversion

OUTPUT FORMAT: Respond with a single valid JSON object with EXACTLY these field names:
{
  "store_name": "string",
  "snapshot_date": "YYYY-MM-DD string",
  "one_line_summary": "string",
  "trading_context": "string",
  "priority_actions": [
    {
      "rank": 1,
      "title": "string under 8 words",
      "what_is_happening": "string with specific numbers",
      "why_it_matters": "string",
      "what_to_do": "string",
      "peer_benchmark": "string",
      "confidence": "HIGH" or "MEDIUM" or "LOW"
    }
  ],
  "one_thing_going_well": "string",
  "watch_list": ["string", "string"],
  "overall_health": "GREEN" or "AMBER" or "RED"
}
No markdown fences, no explanation, no extra fields. Just the JSON object."""


# ─────────────────────────────────────────────
# ANALYST CLASS
# ─────────────────────────────────────────────

class ShopFloorAnalyst:
    """
    Main analyst class. Initialise once, call generate_brief() per store/date.
    """

    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def generate_brief(self, snapshot: StoreSnapshot) -> StoreActionBrief:
        """
        Takes a StoreSnapshot and returns a structured StoreActionBrief.
        """
        formatted_data = format_snapshot_for_prompt(snapshot)

        user_message = f"""Generate a daily action brief for this store.

Store name: {snapshot.store_name}
Date: {snapshot.snapshot_date}

{formatted_data}

Produce the StoreActionBrief JSON. Use store_name="{snapshot.store_name}" and snapshot_date="{snapshot.snapshot_date}" exactly."""

        response = self.client.messages.create(
            model      = CLAUDE_MODEL,
            max_tokens = 1500,
            system     = SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": user_message}],
        )

        raw = response.content[0].text.strip()
        raw = re.sub(r"```json|```", "", raw).strip()

        try:
            parsed = json.loads(raw)
            return StoreActionBrief(**parsed)
        except Exception as e:
            # Graceful fallback
            return StoreActionBrief(
                store_name          = snapshot.store_name,
                snapshot_date       = snapshot.snapshot_date,
                one_line_summary    = "Unable to generate brief — please review data manually.",
                trading_context     = str(e)[:200],
                priority_actions    = [],
                one_thing_going_well= "System error — manual review required.",
                watch_list          = [],
                overall_health      = "AMBER",
            )


# ─────────────────────────────────────────────
# PRINT HELPER
# ─────────────────────────────────────────────

def print_brief(brief: StoreActionBrief):
    """Pretty-prints a StoreActionBrief to the terminal."""
    health_icon = {"GREEN": "🟢", "AMBER": "🟡", "RED": "🔴"}.get(brief.overall_health, "⚪")

    print(f"\n{'='*65}")
    print(f"DAILY ACTION BRIEF — {brief.store_name}")
    print(f"{brief.snapshot_date}  {health_icon} {brief.overall_health}")
    print(f"{'='*65}")
    print(f"\n{brief.one_line_summary}")
    print(f"\nContext: {brief.trading_context}")

    print(f"\n{'─'*65}")
    print(f"PRIORITY ACTIONS ({len(brief.priority_actions)})")
    print(f"{'─'*65}")
    for action in brief.priority_actions:
        conf_icon = {"HIGH": "✓", "MEDIUM": "~", "LOW": "?"}.get(action.confidence, "?")
        print(f"\n[{action.rank}] {action.title}  {conf_icon} {action.confidence}")
        print(f"  What: {action.what_is_happening}")
        print(f"  Why:  {action.why_it_matters}")
        print(f"  Do:   {action.what_to_do}")
        print(f"  Peer: {action.peer_benchmark}")

    print(f"\n{'─'*65}")
    print(f"GOING WELL")
    print(f"  {brief.one_thing_going_well}")

    if brief.watch_list:
        print(f"\n{'─'*65}")
        print(f"WATCH LIST")
        for item in brief.watch_list:
            print(f"  ⚠ {item}")

    print(f"\n{'='*65}")


# ─────────────────────────────────────────────
# MAIN — smoke test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    df, stores, promotions = load_data()

    print("Computing benchmarks...")
    benchmarks = compute_network_benchmarks(df, stores)

    analyst = ShopFloorAnalyst()

    # Test 1: S06 Reading on week 7 — should flag VoltCare attach collapse
    print("\nTest 1: S06 Reading (week 7 — after staff change)")
    snap1 = build_store_snapshot("S06", "2026-02-16", df, stores, promotions, benchmarks)
    brief1 = analyst.generate_brief(snap1)
    print_brief(brief1)

    # Test 2: S03 Croydon — should flag browse-not-buy pattern
    print("\nTest 2: S03 Croydon (high footfall, low conversion)")
    snap2 = build_store_snapshot("S03", "2026-02-16", df, stores, promotions, benchmarks)
    brief2 = analyst.generate_brief(snap2)
    print_brief(brief2)

    # Test 3: S15 Newcastle on a Saturday — should flag NPS/staffing issue
    print("\nTest 3: S15 Newcastle (Saturday — weekend staffing problem)")
    snap3 = build_store_snapshot("S15", "2026-02-21", df, stores, promotions, benchmarks)
    brief3 = analyst.generate_brief(snap3)
    print_brief(brief3)