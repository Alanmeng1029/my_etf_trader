import numpy as np
import pandas as pd

from my_etf.models.train import create_walk_forward_splits, evaluate_classification_calibration


def test_walk_forward_splits_keep_embargo_gap_between_train_and_validation():
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=20, freq="D").strftime("%Y-%m-%d"),
        "close": range(20),
    })

    splits = create_walk_forward_splits(
        df,
        train_days=8,
        validation_days=4,
        step_days=4,
        embargo_days=2,
    )

    assert len(splits) == 2
    first = splits[0]
    assert first["train_end_date"] == "2026-01-08"
    assert first["validation_start_date"] == "2026-01-11"
    assert first["embargo_days"] == 2


def test_classification_calibration_reports_logloss_brier_and_bins():
    y_true = pd.Series([0, 1, 2, 3])
    probabilities = np.array([
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.7, 0.1, 0.1],
        [0.1, 0.1, 0.7, 0.1],
        [0.1, 0.1, 0.1, 0.7],
    ])

    metrics = evaluate_classification_calibration(y_true, probabilities, bins=2)

    assert metrics["logloss"] > 0
    assert metrics["brier_score"] > 0
    assert metrics["class_distribution"] == {"0": 1, "1": 1, "2": 1, "3": 1}
    assert len(metrics["probability_bins"]) == 2
