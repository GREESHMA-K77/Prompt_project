"""
Data schemas and Pydantic models for Aspect-Based Customer Review Analytics.
Provides strict validation, JSON schema generation, and serialization helpers.
"""

from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class SentimentType(str, Enum):
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    NEUTRAL = "Neutral"


class ComplaintCategory(str, Enum):
    AUDIO_SOUND = "Audio / Sound Issue"
    BATTERY_CHARGING = "Battery / Charging Issue"
    CONNECTIVITY_BLUETOOTH = "Connectivity / Bluetooth Issue"
    COMFORT_FIT = "Comfort / Fit Issue"
    BUILD_DURABILITY = "Build Quality / Durability Flaw"
    PRICE_VALUE = "Overpriced / Poor Value"
    DELIVERY_PACKAGING = "Late Delivery / Packaging Damage"
    CUSTOMER_SUPPORT = "Customer Support Issue"
    SOFTWARE_BUG = "Software / Pairing Bug"
    NONE = "None"


# Canonical list of all allowed complaint categories for multi-label classification
ALL_COMPLAINT_CATEGORIES = [c.value for c in ComplaintCategory]


class AspectDetail(BaseModel):
    aspect: str = Field(
        ...,
        description="The product aspect evaluated (e.g., sound_quality, battery_life, build_quality, comfort, price, connectivity, software)"
    )
    sentiment: SentimentType = Field(
        ...,
        description="Sentiment towards this specific aspect: Positive, Negative, or Neutral"
    )
    evidence: str = Field(
        default="",
        description="Direct quote or snippet from the review supporting this aspect judgment"
    )

    @field_validator("aspect", mode="before")
    @classmethod
    def normalize_aspect(cls, v: Any) -> str:
        if isinstance(v, str):
            clean = v.strip().lower().replace(" ", "_").replace("-", "_")
            return clean if clean else "general"
        return "general"

    @field_validator("sentiment", mode="before")
    @classmethod
    def normalize_sentiment(cls, v: Any) -> SentimentType:
        if isinstance(v, str):
            v_norm = v.strip().capitalize()
            if v_norm in ["Positive", "Pos"]:
                return SentimentType.POSITIVE
            elif v_norm in ["Negative", "Neg"]:
                return SentimentType.NEGATIVE
            elif v_norm in ["Neutral", "Neut", "Mixed"]:
                return SentimentType.NEUTRAL
        return SentimentType.NEUTRAL


class ReviewAnalysisRecord(BaseModel):
    review_id: str = Field(..., description="Unique identifier of the customer review")
    product_id: str = Field(..., description="Unique product identifier")
    product_name: Optional[str] = Field(default="Unknown Product", description="Name of the product")
    overall_sentiment: SentimentType = Field(
        ...,
        description="Overall review sentiment: Positive, Negative, or Neutral"
    )
    sentiment_confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Model confidence score between 0.0 and 1.0"
    )
    aspects: List[AspectDetail] = Field(
        default_factory=list,
        description="List of aspect-level sentiment extractions with evidence snippets"
    )
    complaint_categories: List[str] = Field(
        default_factory=lambda: ["None"],
        description="Multi-label classification of complaints matching canonical complaint categories"
    )
    summary_takeaway: str = Field(
        default="",
        description="Concise 1-2 sentence core takeaway summarizing the review"
    )
    is_repaired: bool = Field(
        default=False,
        description="Flag indicating whether this record had to be auto-repaired from malformed LLM output"
    )
    repair_log: List[str] = Field(
        default_factory=list,
        description="Log of transformations applied to repair malformed input"
    )

    @field_validator("overall_sentiment", mode="before")
    @classmethod
    def normalize_overall_sentiment(cls, v: Any) -> SentimentType:
        if isinstance(v, str):
            clean = v.strip().capitalize()
            if clean in ["Positive", "Pos", "Good"]:
                return SentimentType.POSITIVE
            elif clean in ["Negative", "Neg", "Bad"]:
                return SentimentType.NEGATIVE
            elif clean in ["Neutral", "Mixed", "Neut", "Average"]:
                return SentimentType.NEUTRAL
        return SentimentType.NEUTRAL

    @field_validator("complaint_categories", mode="before")
    @classmethod
    def normalize_complaints(cls, v: Any) -> List[str]:
        if v is None:
            return ["None"]
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            return ["None"]

        normalized: List[str] = []
        canonical_map = {cat.lower(): cat for cat in ALL_COMPLAINT_CATEGORIES}
        # Add fuzzy alias mappings
        canonical_map.update({
            "audio": ComplaintCategory.AUDIO_SOUND.value,
            "sound": ComplaintCategory.AUDIO_SOUND.value,
            "sound quality": ComplaintCategory.AUDIO_SOUND.value,
            "audio quality": ComplaintCategory.AUDIO_SOUND.value,
            "battery": ComplaintCategory.BATTERY_CHARGING.value,
            "charging": ComplaintCategory.BATTERY_CHARGING.value,
            "power": ComplaintCategory.BATTERY_CHARGING.value,
            "bluetooth": ComplaintCategory.CONNECTIVITY_BLUETOOTH.value,
            "connectivity": ComplaintCategory.CONNECTIVITY_BLUETOOTH.value,
            "pairing": ComplaintCategory.CONNECTIVITY_BLUETOOTH.value,
            "comfort": ComplaintCategory.COMFORT_FIT.value,
            "fit": ComplaintCategory.COMFORT_FIT.value,
            "ergonomic": ComplaintCategory.COMFORT_FIT.value,
            "build": ComplaintCategory.BUILD_DURABILITY.value,
            "build quality": ComplaintCategory.BUILD_DURABILITY.value,
            "durability": ComplaintCategory.BUILD_DURABILITY.value,
            "broken": ComplaintCategory.BUILD_DURABILITY.value,
            "hardware": ComplaintCategory.BUILD_DURABILITY.value,
            "price": ComplaintCategory.PRICE_VALUE.value,
            "value": ComplaintCategory.PRICE_VALUE.value,
            "expensive": ComplaintCategory.PRICE_VALUE.value,
            "shipping": ComplaintCategory.DELIVERY_PACKAGING.value,
            "delivery": ComplaintCategory.DELIVERY_PACKAGING.value,
            "packaging": ComplaintCategory.DELIVERY_PACKAGING.value,
            "support": ComplaintCategory.CUSTOMER_SUPPORT.value,
            "customer service": ComplaintCategory.CUSTOMER_SUPPORT.value,
            "software": ComplaintCategory.SOFTWARE_BUG.value,
            "app": ComplaintCategory.SOFTWARE_BUG.value,
            "bug": ComplaintCategory.SOFTWARE_BUG.value,
            "none": ComplaintCategory.NONE.value,
            "no complaints": ComplaintCategory.NONE.value,
            "no complaint": ComplaintCategory.NONE.value,
        })

        for item in v:
            if not isinstance(item, str):
                continue
            item_clean = item.strip().lower()
            matched = False
            for key, canon in canonical_map.items():
                if key == item_clean or key in item_clean:
                    if canon not in normalized:
                        normalized.append(canon)
                    matched = True
                    break
            if not matched and item.strip():
                # keep as is if non-empty
                normalized.append(item.strip())

        if not normalized or (len(normalized) > 1 and ComplaintCategory.NONE.value in normalized):
            if ComplaintCategory.NONE.value in normalized and len(normalized) > 1:
                normalized.remove(ComplaintCategory.NONE.value)
        if not normalized:
            normalized = ["None"]
        return normalized

    def to_flattened_dict(self) -> Dict[str, Any]:
        """Flatten record for CSV export."""
        aspects_str = "; ".join([f"{a.aspect}:{a.sentiment.value} ('{a.evidence}')" for a in self.aspects])
        complaints_str = "; ".join(self.complaint_categories)
        return {
            "review_id": self.review_id,
            "product_id": self.product_id,
            "product_name": self.product_name,
            "overall_sentiment": self.overall_sentiment.value,
            "sentiment_confidence": round(self.sentiment_confidence, 3),
            "aspects_count": len(self.aspects),
            "aspects_detail": aspects_str,
            "complaint_categories": complaints_str,
            "summary_takeaway": self.summary_takeaway,
            "is_repaired": self.is_repaired,
            "repair_log": " | ".join(self.repair_log) if self.repair_log else "None"
        }


class AggregatedProductStats(BaseModel):
    product_id: str
    product_name: str
    total_reviews: int
    sentiment_distribution: Dict[str, int]
    sentiment_percentages: Dict[str, float]
    aspect_sentiment_matrix: Dict[str, Dict[str, int]]
    complaint_counts: Dict[str, int]
    top_complaint_categories: List[Dict[str, Any]]
    representative_quotes: Dict[str, List[str]]


class ProductExecutiveSummary(BaseModel):
    product_id: str
    product_name: str
    total_reviews_analyzed: int
    executive_verdict: str
    key_strengths: List[str]
    critical_complaints: List[str]
    actionable_recommendations: List[str]
    target_audience_verdict: str


class ClaimVerification(BaseModel):
    claim_id: int
    claim_text: str
    status: str = Field(..., description="SUPPORTED, CONTRADICTED, or UNSUPPORTED")
    evidence_found: str = Field(default="", description="Evidence from aggregated data or customer quotes")
    reasoning: str = Field(default="", description="Detailed rationale from the LLM Judge")


class FaithfulnessAudit(BaseModel):
    product_id: str
    product_name: str
    total_claims_extracted: int
    supported_claims_count: int
    contradicted_claims_count: int
    unsupported_claims_count: int
    faithfulness_score: float = Field(..., ge=0.0, le=1.0, description="Score: supported_claims / total_claims")
    faithfulness_rating: str = Field(..., description="High, Moderate, or Low")
    verified_claims: List[ClaimVerification]
    judge_rationale: str
