"""
Unit tests for the LLM-as-Judge Faithfulness Evaluator.
"""

from src.judge.faithfulness_judge import FaithfulnessJudge
from src.models.schemas import ProductExecutiveSummary, AggregatedProductStats


def test_faithfulness_judge_audit():
    judge = FaithfulnessJudge()

    stats = AggregatedProductStats(
        product_id="PROD_TEST",
        product_name="Test Headphones",
        total_reviews=10,
        sentiment_distribution={"Positive": 8, "Negative": 2, "Neutral": 0},
        sentiment_percentages={"Positive": 80.0, "Negative": 20.0, "Neutral": 0.0},
        aspect_sentiment_matrix={"sound_quality": {"Positive": 8, "Negative": 2, "Neutral": 0}},
        complaint_counts={"Audio / Sound Issue": 2},
        top_complaint_categories=[{"category": "Audio / Sound Issue", "count": 2, "percentage_of_reviews": 20.0}],
        representative_quotes={"sound_quality": ["'great bass' (Positive)"]}
    )

    summary = ProductExecutiveSummary(
        product_id="PROD_TEST",
        product_name="Test Headphones",
        total_reviews_analyzed=10,
        executive_verdict="Solid audio performers with slight distortion at high volume.",
        key_strengths=["Exceptional bass response praised by 80% of users."],
        critical_complaints=["Audio / Sound Issue: Distortion occurs at maximum volume."],
        actionable_recommendations=["Refine gain staging in amplifier circuitry."],
        target_audience_verdict="Great for bass lovers."
    )

    audit = judge.audit_summary(summary, stats)
    assert audit.product_id == "PROD_TEST"
    assert 0.0 <= audit.faithfulness_score <= 1.0
    assert audit.faithfulness_rating in ["High", "Moderate", "Low"]
    assert len(audit.verified_claims) > 0
