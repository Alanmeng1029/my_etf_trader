"""Daily classification recommendation workflow for live-decision review."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from ..config import BACKTEST_CONFIG, DATA_DIR, RECOMMENDATIONS_DIR
from ..utils.data_health import collect_data_health, write_universe_snapshot
from ..utils.logger import setup_logger
from . import recommend_workflow


logger = setup_logger("daily_recommendation", "daily_recommendation.log")


def _first_reference_date(health_df: pd.DataFrame) -> Optional[str]:
    if health_df.empty or "reference_date" not in health_df.columns:
        return None
    values = health_df["reference_date"].dropna()
    return str(values.iloc[0]) if not values.empty else None


def _exclude_reason_counts(health_df: pd.DataFrame) -> Dict[str, int]:
    if health_df.empty or "exclude_reason" not in health_df.columns:
        return {}
    counter = Counter()
    for reason_text in health_df.loc[~health_df["is_eligible"], "exclude_reason"].fillna(""):
        for reason in str(reason_text).split(";"):
            reason = reason.strip()
            if reason:
                counter[reason] += 1
    return dict(counter)


def _serialise_recommendations(recommendations: List[Dict]) -> List[Dict]:
    fields = [
        "rank", "code", "name", "predicted_class",
        "class_0_prob", "class_1_prob", "class_2_prob", "class_3_prob",
        "prediction_date", "model_version",
    ]
    return [{field: rec.get(field) for field in fields} for rec in recommendations]


def build_daily_summary(
    recommendations: List[Dict],
    health_df: pd.DataFrame,
    report_paths: Dict[str, str],
    threshold: float,
    generated_at: Optional[str] = None,
    as_of_date: Optional[str] = None,
    max_data_age_days: Optional[int] = None,
) -> Dict:
    """Build an actionable summary object for the daily recommendation run."""
    generated_at = generated_at or datetime.now().isoformat(timespec="seconds")
    as_of = pd.to_datetime(as_of_date or datetime.now().date())
    max_data_age_days = (
        BACKTEST_CONFIG["MAX_STALENESS_DAYS"]
        if max_data_age_days is None else max_data_age_days
    )

    reference_date = _first_reference_date(health_df)
    data_age_days = None
    stale_market_data = False
    if reference_date:
        data_age_days = int((as_of - pd.to_datetime(reference_date)).days)
        stale_market_data = data_age_days > max_data_age_days

    recommendations_payload = _serialise_recommendations(recommendations)
    eligible_count = int(health_df["is_eligible"].sum()) if not health_df.empty else 0
    excluded_count = int((~health_df["is_eligible"]).sum()) if not health_df.empty else 0

    if stale_market_data:
        actionable = False
        message = f"数据过期 {data_age_days} 天，不给出实盘买入建议"
    elif not recommendations_payload:
        actionable = False
        message = "今日空仓/无推荐"
    else:
        actionable = True
        message = f"生成分类推荐 {len(recommendations_payload)} 只"

    return {
        "generated_at": generated_at,
        "actionable": actionable,
        "message": message,
        "threshold": threshold,
        "data_reference_date": reference_date,
        "data_age_days": data_age_days,
        "eligible_count": eligible_count,
        "excluded_count": excluded_count,
        "exclude_reasons": _exclude_reason_counts(health_df),
        "report_paths": report_paths,
        "recommendations": recommendations_payload,
    }


def write_daily_summary(
    summary: Dict,
    output_dir: str = RECOMMENDATIONS_DIR,
    timestamp: Optional[str] = None,
) -> Dict[str, str]:
    """Write daily recommendation summary as JSON and Markdown."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"daily_recommendation_summary_{timestamp}.json")
    md_path = os.path.join(output_dir, f"daily_recommendation_summary_{timestamp}.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    lines = [
        "# ETF Daily Classification Recommendation",
        "",
        f"- generated_at: {summary.get('generated_at')}",
        f"- actionable: {summary.get('actionable')}",
        f"- message: {summary.get('message')}",
        f"- threshold: {summary.get('threshold')}",
        f"- data_reference_date: {summary.get('data_reference_date')}",
        f"- data_age_days: {summary.get('data_age_days')}",
        f"- eligible_count: {summary.get('eligible_count')}",
        f"- excluded_count: {summary.get('excluded_count')}",
        f"- exclude_reasons: {summary.get('exclude_reasons')}",
        "",
        "## Report Paths",
    ]
    for key, value in summary.get("report_paths", {}).items():
        lines.append(f"- {key}: {value}")

    lines.extend([
        "",
        "## Recommendations",
        "",
        "| rank | code | name | predicted_class | class_0_prob | class_1_prob | class_2_prob | class_3_prob | prediction_date | model_version |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ])
    if summary.get("recommendations"):
        for rec in summary["recommendations"]:
            lines.append(
                f"| {rec.get('rank')} | {rec.get('code')} | {rec.get('name')} | "
                f"{rec.get('predicted_class')} | {rec.get('class_0_prob')} | "
                f"{rec.get('class_1_prob')} | {rec.get('class_2_prob')} | "
                f"{rec.get('class_3_prob')} | {rec.get('prediction_date')} | "
                f"{rec.get('model_version')} |"
            )
    else:
        lines.append("|  |  | 今日空仓/无推荐 |  |  |  |  |  |  |  |")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return {"json": json_path, "md": md_path}


def run_daily_recommendation(
    threshold: float = 0.1,
    skip_data_update: bool = False,
    skip_indicators: bool = False,
) -> int:
    """Run update -> indicators -> classification recommendation -> summary."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_paths: Dict[str, str] = {}
    recommendations: List[Dict] = []

    try:
        if not skip_data_update and not recommend_workflow.run_data_update():
            raise RuntimeError("data_update_failed")
        if not skip_indicators and not recommend_workflow.run_indicators_calc():
            raise RuntimeError("indicators_failed")

        health_df = collect_data_health()
        universe_snapshot_path = write_universe_snapshot(
            health_df,
            "daily_recommendation",
            DATA_DIR,
        )
        report_paths["universe_snapshot"] = universe_snapshot_path

        predictions = recommend_workflow.generate_classification_predictions()
        if not predictions:
            raise RuntimeError("classification_prediction_failed")

        recommendations = recommend_workflow.get_threshold_based_recommendations(
            predictions,
            min_prob_threshold=threshold,
            max_n=5,
        )
        if recommendations:
            report_paths.update(
                recommend_workflow.save_recommendations(
                    recommendations,
                    use_classification=True,
                ) or {}
            )

        summary = build_daily_summary(
            recommendations,
            health_df,
            report_paths,
            threshold=threshold,
        )
        summary_paths = write_daily_summary(summary, timestamp=timestamp)
        report_paths.update(summary_paths)

        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        print("SUMMARY_FILES=" + json.dumps(summary_paths, ensure_ascii=False))
        return 0
    except Exception as exc:
        logger.exception("Daily recommendation failed")
        health_df = collect_data_health()
        summary = build_daily_summary(
            [],
            health_df,
            report_paths,
            threshold=threshold,
        )
        summary["actionable"] = False
        summary["message"] = f"自动化失败：{exc}，不输出实盘买入建议"
        summary_paths = write_daily_summary(summary, timestamp=timestamp)
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        print("SUMMARY_FILES=" + json.dumps(summary_paths, ensure_ascii=False))
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily ETF classification recommendation")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--skip-data-update", action="store_true")
    parser.add_argument("--skip-indicators", action="store_true")
    args = parser.parse_args()
    return run_daily_recommendation(
        threshold=args.threshold,
        skip_data_update=args.skip_data_update,
        skip_indicators=args.skip_indicators,
    )


if __name__ == "__main__":
    sys.exit(main())
