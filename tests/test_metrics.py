"""
Unit tests for evaluation metrics.
"""

from src.evaluation.metrics import compute_sentiment_metrics, compute_multilabel_complaint_metrics
from src.models.schemas import ComplaintCategory


def test_sentiment_metrics_calculation():
    y_true = ["Positive", "Negative", "Neutral", "Positive"]
    y_pred = ["Positive", "Negative", "Positive", "Positive"]

    metrics = compute_sentiment_metrics(y_true, y_pred)
    assert "macro_f1" in metrics
    assert "accuracy" in metrics
    assert metrics["accuracy"] == 0.75
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_multilabel_complaint_metrics():
    y_true = [
        [ComplaintCategory.BATTERY_CHARGING.value, ComplaintCategory.AUDIO_SOUND.value],
        [ComplaintCategory.NONE.value]
    ]
    y_pred = [
        [ComplaintCategory.BATTERY_CHARGING.value],
        [ComplaintCategory.NONE.value]
    ]

    metrics = compute_multilabel_complaint_metrics(y_true, y_pred)
    assert "macro_f1" in metrics
    assert "micro_f1" in metrics
    assert "hamming_loss" in metrics
    assert 0.0 <= metrics["macro_f1"] <= 1.0
