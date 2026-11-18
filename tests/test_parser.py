"""
Unit tests for MalformedOutputRepairParser and LangChain parser.
"""

import pytest
from src.parsers.repair_parser import MalformedOutputRepairParser
from src.parsers.langchain_parser import LangChainReviewOutputParser
from src.models.schemas import SentimentType, ComplaintCategory


def test_clean_json_parsing():
    parser = MalformedOutputRepairParser()
    valid_json = """
    {
      "review_id": "REV_001",
      "product_id": "PROD_HEADPHONES",
      "product_name": "SonicPulse Pro",
      "overall_sentiment": "Positive",
      "sentiment_confidence": 0.95,
      "aspects": [
        {"aspect": "sound_quality", "sentiment": "Positive", "evidence": "crisp highs and deep bass"},
        {"aspect": "battery_life", "sentiment": "Neutral", "evidence": "lasts about 12 hours"}
      ],
      "complaint_categories": ["None"],
      "summary_takeaway": "Great sound quality with average battery life."
    }
    """
    record = parser.parse_and_repair(valid_json)
    assert record.review_id == "REV_001"
    assert record.overall_sentiment == SentimentType.POSITIVE
    assert len(record.aspects) == 2
    assert record.is_repaired is False


def test_markdown_and_conversational_repair():
    parser = MalformedOutputRepairParser()
    corrupted = """
    Sure thing! Here is the JSON record you requested:
    ```json
    {
      "review_id": "REV_002",
      "product_id": "PROD_WATCH",
      "overall_sentiment": "Negative",
      "aspects": [
        {"aspect": "battery", "sentiment": "Negative", "evidence": "dies in 4 hours"}
      ],
      "complaint_categories": ["battery"]
    }
    ```
    Let me know if you need anything else!
    """
    record = parser.parse_and_repair(corrupted)
    assert record.review_id == "REV_002"
    assert record.overall_sentiment == SentimentType.NEGATIVE
    assert record.is_repaired is True
    assert any("markdown" in log.lower() for log in record.repair_log)
    assert ComplaintCategory.BATTERY_CHARGING.value in record.complaint_categories


def test_single_quotes_and_trailing_commas():
    parser = MalformedOutputRepairParser()
    corrupted = """
    {
      'review_id': 'REV_003',
      'product_id': 'PROD_MOUSE',
      'overall_sentiment': 'Negative',
      'aspects': [
        {'aspect': 'comfort', 'sentiment': 'Negative', 'evidence': 'causes wrist fatigue',},
      ],
      'complaint_categories': ['Comfort / Fit Issue',],
    }
    """
    record = parser.parse_and_repair(corrupted)
    assert record.review_id == "REV_003"
    assert record.overall_sentiment == SentimentType.NEGATIVE
    assert record.is_repaired is True
    assert len(record.aspects) == 1
    assert record.complaint_categories == [ComplaintCategory.COMFORT_FIT.value]


def test_unclosed_truncated_json():
    parser = MalformedOutputRepairParser()
    truncated = """
    {"review_id": "REV_004", "product_id": "PROD_SPEAKER", "overall_sentiment": "Positive", "aspects": [{"aspect": "sound", "sentiment": "Positive", "evidence": "loud and clear"
    """
    record = parser.parse_and_repair(truncated)
    assert record.review_id == "REV_004"
    assert record.overall_sentiment == SentimentType.POSITIVE
    assert record.is_repaired is True


def test_langchain_parser_interface():
    lc_parser = LangChainReviewOutputParser()
    text = '{"review_id": "REV_005", "product_id": "PROD_01", "overall_sentiment": "Neutral"}'
    record = lc_parser.parse(text)
    assert record.review_id == "REV_005"
    assert "JSON schema" in lc_parser.get_format_instructions()
