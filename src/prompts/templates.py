"""
Prompt templates for Zero-Shot, Few-Shot, and Chain-of-Thought (CoT) prompting techniques.
Includes canonical complaint categories and JSON schema instructions.
"""

from src.models.schemas import ALL_COMPLAINT_CATEGORIES

COMPLAINT_CATEGORIES_STR = "\n".join([f"- {cat}" for cat in ALL_COMPLAINT_CATEGORIES])

SYSTEM_BASE = f"""You are an expert E-Commerce Customer Review and Sentiment Intelligence AI.
Your objective is to analyze unstructured customer reviews and extract precise, validated structured analytics records.

Allowed Complaint Categories (Multi-Label Classification):
{COMPLAINT_CATEGORIES_STR}

If the customer has no complaints or only positive feedback, you MUST return ["None"].
"""


ZERO_SHOT_PROMPT = """{system_base}

Review to Analyze:
Review ID: {review_id}
Product ID: {product_id}
Product Name: {product_name}
Review Text: "{review_text}"

Task:
Perform aspect-based sentiment analysis and multi-label complaint classification.
Output ONLY a valid JSON object strictly matching this schema:
{{
  "review_id": "{review_id}",
  "product_id": "{product_id}",
  "product_name": "{product_name}",
  "overall_sentiment": "Positive" | "Negative" | "Neutral",
  "sentiment_confidence": float between 0.0 and 1.0,
  "aspects": [
    {{
      "aspect": "aspect_name",
      "sentiment": "Positive" | "Negative" | "Neutral",
      "evidence": "exact quote from review"
    }}
  ],
  "complaint_categories": ["List of matching categories from the allowed list, or [\\"None\\"]"],
  "summary_takeaway": "1-2 sentence core insight"
}}
Return only raw JSON. Do not include markdown codeblocks or conversational text.
"""


FEW_SHOT_PROMPT = """{system_base}

Here are reference examples demonstrating the exact expected analysis format:

--- Example 1 ---
Review ID: EX_01
Product ID: PROD_HEADPHONES_01
Product Name: AeroSound Pro Wireless ANC Headphones
Review Text: "The sound quality on this is phenomenal, crystal clear highs and deep bass. Battery life easily lasts 30 hours. Outstanding quality!"
Analysis:
{{
  "review_id": "EX_01",
  "product_id": "PROD_HEADPHONES_01",
  "product_name": "AeroSound Pro Wireless ANC Headphones",
  "overall_sentiment": "Positive",
  "sentiment_confidence": 0.98,
  "aspects": [
    {{"aspect": "sound_quality", "sentiment": "Positive", "evidence": "crystal clear highs and deep bass"}},
    {{"aspect": "battery_life", "sentiment": "Positive", "evidence": "easily lasts 30 hours"}}
  ],
  "complaint_categories": ["None"],
  "summary_takeaway": "Exceptional audio reproduction and long-lasting battery life."
}}

--- Example 2 ---
Review ID: EX_02
Product ID: PROD_HEADPHONES_01
Product Name: AeroSound Pro Wireless ANC Headphones
Review Text: "Constant Bluetooth dropouts every 10 minutes. Charging port stopped working after a month. Very disappointing."
Analysis:
{{
  "review_id": "EX_02",
  "product_id": "PROD_HEADPHONES_01",
  "product_name": "AeroSound Pro Wireless ANC Headphones",
  "overall_sentiment": "Negative",
  "sentiment_confidence": 0.95,
  "aspects": [
    {{"aspect": "connectivity", "sentiment": "Negative", "evidence": "Constant Bluetooth dropouts every 10 minutes"}},
    {{"aspect": "battery_life", "sentiment": "Negative", "evidence": "Charging port stopped working after a month"}},
    {{"aspect": "build_quality", "sentiment": "Negative", "evidence": "Charging port stopped working"}}
  ],
  "complaint_categories": ["Connectivity / Bluetooth Issue", "Battery / Charging Issue", "Build Quality / Durability Flaw"],
  "summary_takeaway": "Crippling hardware flaws: persistent wireless disconnects and dead charging port."
}}

--- Example 3 ---
Review ID: EX_03
Product ID: PROD_SMARTWATCH_02
Product Name: PulseFit Ultra GPS Smartwatch
Review Text: "The audio is decent for podcasts, but battery life is strictly average at around 6 hours. Just middle of the road."
Analysis:
{{
  "review_id": "EX_03",
  "product_id": "PROD_SMARTWATCH_02",
  "product_name": "PulseFit Ultra GPS Smartwatch",
  "overall_sentiment": "Neutral",
  "sentiment_confidence": 0.88,
  "aspects": [
    {{"aspect": "sound_quality", "sentiment": "Neutral", "evidence": "decent for podcasts"}},
    {{"aspect": "battery_life", "sentiment": "Neutral", "evidence": "strictly average at around 6 hours"}}
  ],
  "complaint_categories": ["None"],
  "summary_takeaway": "Passable audio and typical battery endurance meeting moderate expectations."
}}

--- Example 4 ---
Review ID: EX_04
Product ID: PROD_KEYBOARD_03
Product Name: ApexKey Pro Mechanical Keyboard
Review Text: "Sound is okay and connection is fine, but for $150 I was hoping for something more premium. Needs better keycaps."
Analysis:
{{
  "review_id": "EX_04",
  "product_id": "PROD_KEYBOARD_03",
  "product_name": "ApexKey Pro Mechanical Keyboard",
  "overall_sentiment": "Neutral",
  "sentiment_confidence": 0.85,
  "aspects": [
    {{"aspect": "sound_quality", "sentiment": "Neutral", "evidence": "Sound is okay"}},
    {{"aspect": "connectivity", "sentiment": "Positive", "evidence": "connection is fine"}},
    {{"aspect": "price_value", "sentiment": "Negative", "evidence": "for $150 I was hoping for something more premium"}}
  ],
  "complaint_categories": ["Overpriced / Poor Value"],
  "summary_takeaway": "Adequate functionality hampered by uncompetitive pricing and underwhelming materials."
}}

--- Target Review to Analyze ---
Review ID: {review_id}
Product ID: {product_id}
Product Name: {product_name}
Review Text: "{review_text}"

Now analyze the target review following the exact structure and schemas above. Return valid JSON only.
"""


COT_PROMPT = """{system_base}

You must use Chain-of-Thought (CoT) reasoning to thoroughly analyze this review before producing the final JSON.

Review to Analyze:
Review ID: {review_id}
Product ID: {product_id}
Product Name: {product_name}
Review Text: "{review_text}"

Perform step-by-step reasoning following these 4 analytical stages:
Stage 1 - Lexical & Sentiment Polarity Scan:
- Identify emotional cues, praises, or frustrations.
- Check if sentiment is overwhelmingly positive, negative, or mixed/neutral.

Stage 2 - Granular Aspect Decomposition:
- Extract all mentioned dimensions (e.g., sound_quality, battery_life, comfort, connectivity, build_quality, price_value, software, customer_service).
- Assign specific sentiment (Positive, Negative, or Neutral) and note exact textual evidence.

Stage 3 - Multi-Label Complaint Category Audit:
- Evaluate against allowed complaint categories:
{complaint_categories_str}
- If any complaint is detected, select all that apply. If no defect, flaw, or frustration is reported, select ["None"].

Stage 4 - Structured JSON Synthesis:
- Compile findings into the final JSON output.

Format your response as:
THOUGHT:
<step-by-step reasoning for Stages 1 to 3>

JSON:
{{
  "review_id": "{review_id}",
  "product_id": "{product_id}",
  "product_name": "{product_name}",
  "overall_sentiment": "Positive" | "Negative" | "Neutral",
  "sentiment_confidence": float between 0.0 and 1.0,
  "aspects": [
    {{
      "aspect": "aspect_name",
      "sentiment": "Positive" | "Negative" | "Neutral",
      "evidence": "quote"
    }}
  ],
  "complaint_categories": ["category_1", "category_2"],
  "summary_takeaway": "concise summary"
}}
"""


def get_prompt(
    technique: str,
    review_id: str,
    product_id: str,
    product_name: str,
    review_text: str
) -> str:
    """Format prompt based on the chosen prompting technique."""
    tech_clean = technique.strip().lower().replace("-", "_")
    if tech_clean in ["zero_shot", "zeroshot", "zero"]:
        return ZERO_SHOT_PROMPT.format(
            system_base=SYSTEM_BASE,
            review_id=review_id,
            product_id=product_id,
            product_name=product_name,
            review_text=review_text
        )
    elif tech_clean in ["few_shot", "fewshot", "few"]:
        return FEW_SHOT_PROMPT.format(
            system_base=SYSTEM_BASE,
            review_id=review_id,
            product_id=product_id,
            product_name=product_name,
            review_text=review_text
        )
    elif tech_clean in ["cot", "chain_of_thought"]:
        return COT_PROMPT.format(
            system_base=SYSTEM_BASE,
            review_id=review_id,
            product_id=product_id,
            product_name=product_name,
            review_text=review_text,
            complaint_categories_str=COMPLAINT_CATEGORIES_STR
        )
    else:
        raise ValueError(f"Unknown prompting technique: {technique}. Choose zero_shot, few_shot, or cot.")
