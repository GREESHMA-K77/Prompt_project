"""
Unit tests for the prompt chain pipeline (Extract -> Aggregate -> Summarise).
"""

from src.pipeline.chain import ReviewAnalyticsPipeline
from src.models.schemas import ReviewAnalysisRecord, AspectDetail, SentimentType, ComplaintCategory


def test_aggregation_and_summary():
    pipeline = ReviewAnalyticsPipeline()

    records = [
        ReviewAnalysisRecord(
            review_id="REV_T1",
            product_id="PROD_TEST",
            product_name="Test Headphones",
            overall_sentiment=SentimentType.POSITIVE,
            aspects=[
                AspectDetail(aspect="sound_quality", sentiment=SentimentType.POSITIVE, evidence="amazing sound"),
                AspectDetail(aspect="battery_life", sentiment=SentimentType.POSITIVE, evidence="lasts 20h")
            ],
            complaint_categories=["None"]
        ),
        ReviewAnalysisRecord(
            review_id="REV_T2",
            product_id="PROD_TEST",
            product_name="Test Headphones",
            overall_sentiment=SentimentType.NEGATIVE,
            aspects=[
                AspectDetail(aspect="connectivity", sentiment=SentimentType.NEGATIVE, evidence="drops constantly")
            ],
            complaint_categories=[ComplaintCategory.CONNECTIVITY_BLUETOOTH.value]
        )
    ]

    stats = pipeline.aggregate_product("PROD_TEST", records)
    assert stats.total_reviews == 2
    assert stats.sentiment_distribution["Positive"] == 1
    assert stats.sentiment_distribution["Negative"] == 1
    assert stats.aspect_sentiment_matrix["sound_quality"]["Positive"] == 1
    assert stats.complaint_counts[ComplaintCategory.CONNECTIVITY_BLUETOOTH.value] == 1

    summary = pipeline.summarise_product(stats)
    assert summary.product_id == "PROD_TEST"
    assert len(summary.key_strengths) > 0
    assert len(summary.critical_complaints) > 0
