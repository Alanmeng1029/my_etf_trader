from pathlib import Path

import pandas as pd

from my_etf.workflows.daily_recommendation import build_daily_summary, write_daily_summary


def test_build_daily_summary_marks_no_recommendation_as_not_actionable():
    health = pd.DataFrame({
        "code": ["510300", "512780"],
        "reference_date": ["2026-05-13", "2026-05-13"],
        "is_eligible": [True, False],
        "exclude_reason": ["", "stale_data"],
    })

    summary = build_daily_summary(
        recommendations=[],
        health_df=health,
        report_paths={},
        threshold=0.1,
        generated_at="2026-05-13T16:30:00",
        as_of_date="2026-05-13",
    )

    assert summary["actionable"] is False
    assert summary["message"] == "今日空仓/无推荐"
    assert summary["excluded_count"] == 1
    assert summary["exclude_reasons"]["stale_data"] == 1


def test_build_daily_summary_rejects_stale_market_data_even_with_recommendations():
    health = pd.DataFrame({
        "code": ["510300"],
        "reference_date": ["2026-04-01"],
        "is_eligible": [True],
        "exclude_reason": [""],
    })
    recommendations = [{
        "rank": 1,
        "code": "510300",
        "name": "沪深300",
        "predicted_class": 3,
        "class_0_prob": 0.01,
        "class_1_prob": 0.02,
        "class_2_prob": 0.17,
        "class_3_prob": 0.80,
        "prediction_date": "2026-04-01",
        "model_version": "v1",
    }]

    summary = build_daily_summary(
        recommendations=recommendations,
        health_df=health,
        report_paths={"html": "recommendations/example.html"},
        threshold=0.1,
        generated_at="2026-05-13T16:30:00",
        as_of_date="2026-05-13",
        max_data_age_days=10,
    )

    assert summary["actionable"] is False
    assert "数据过期" in summary["message"]
    assert summary["recommendations"][0]["code"] == "510300"


def test_write_daily_summary_creates_markdown_and_json(tmp_path: Path):
    summary = {
        "generated_at": "2026-05-13T16:30:00",
        "actionable": True,
        "message": "生成分类推荐 1 只",
        "threshold": 0.1,
        "data_reference_date": "2026-05-13",
        "data_age_days": 0,
        "eligible_count": 1,
        "excluded_count": 0,
        "exclude_reasons": {},
        "report_paths": {"html": "recommendations/example.html"},
        "recommendations": [{
            "rank": 1,
            "code": "510300",
            "name": "沪深300",
            "predicted_class": 3,
            "class_0_prob": 0.01,
            "class_1_prob": 0.02,
            "class_2_prob": 0.17,
            "class_3_prob": 0.80,
            "prediction_date": "2026-05-13",
            "model_version": "v1",
        }],
    }

    paths = write_daily_summary(summary, output_dir=str(tmp_path), timestamp="20260513_163000")

    assert Path(paths["json"]).exists()
    markdown = Path(paths["md"]).read_text(encoding="utf-8")
    assert "510300" in markdown
    assert "class_3_prob" in markdown
