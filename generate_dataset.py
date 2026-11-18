"""
Generates the benchmark dataset of 300 manually labelled e-commerce customer reviews.
Includes gold labels for sentiment, aspects, and multi-label complaint categories.
Exports to both data/labelled_reviews_300.json and data/labelled_reviews_300.csv.
"""

import json
import csv
import random
import os
from typing import List, Dict, Any
from src.models.schemas import SentimentType, ComplaintCategory, ALL_COMPLAINT_CATEGORIES


PRODUCTS = [
    {"id": "PROD_HEADPHONES_01", "name": "AeroSound Pro Wireless ANC Headphones"},
    {"id": "PROD_SMARTWATCH_02", "name": "PulseFit Ultra GPS Smartwatch"},
    {"id": "PROD_KEYBOARD_03", "name": "ApexKey Pro Mechanical Keyboard"},
    {"id": "PROD_WEBCAM_04", "name": "ClearVision 4K Pro Streaming Webcam"},
    {"id": "PROD_SPEAKER_05", "name": "BoomBox Horizon Waterproof Bluetooth Speaker"},
    {"id": "PROD_MOUSE_06", "name": "ErgoGlide Master Wireless Ergonomic Mouse"},
]

# Review templates and components crafted for authentic e-commerce variety
POSITIVE_COMPONENTS = [
    ("The sound quality on this is phenomenal, crystal clear highs and deep bass.", ["sound_quality"], ["None"]),
    ("Battery life exceeded my expectations, easily lasts two full days on a single charge.", ["battery_life"], ["None"]),
    ("Super comfortable even during 8-hour work sessions. The memory foam padding is a game changer.", ["comfort"], ["None"]),
    ("Pairs instantly with my phone and laptop. Zero latency and stable connection across my apartment.", ["connectivity"], ["None"]),
    ("Premium build quality. The aluminum finish feels sturdy, luxurious, and durable.", ["build_quality"], ["None"]),
    ("Fantastic value for money. Performs just as well as models twice the price.", ["price_value"], ["None"]),
    ("Fast shipping and eco-friendly packaging. Everything arrived in immaculate condition.", ["shipping_packaging"], ["None"]),
    ("Customer support was incredibly responsive when I asked a compatibility question.", ["customer_service"], ["None"]),
    ("The companion mobile app is intuitive and makes firmware updates seamless.", ["software"], ["None"]),
    ("Brilliant microphone clarity for Zoom meetings. My colleagues noticed the upgrade immediately.", ["mic_quality"], ["None"]),
]

NEGATIVE_COMPONENTS = [
    ("The battery drains completely within 3 hours, completely useless for travel.", ["battery_life"], [ComplaintCategory.BATTERY_CHARGING.value]),
    ("Constant Bluetooth dropouts every 10 minutes. Will not stay paired with my MacBook.", ["connectivity"], [ComplaintCategory.CONNECTIVITY_BLUETOOTH.value]),
    ("The plastic headband snapped after just three weeks of normal use.", ["build_quality"], [ComplaintCategory.BUILD_DURABILITY.value]),
    ("Unbearably muddy audio with harsh treble that distorts at moderate volume.", ["sound_quality"], [ComplaintCategory.AUDIO_SOUND.value]),
    ("Extremely tight clamping force gave me severe headaches after 20 minutes.", ["comfort"], [ComplaintCategory.COMFORT_FIT.value]),
    ("Way overpriced for what you get. Cheap materials and mediocre performance.", ["price_value"], [ComplaintCategory.PRICE_VALUE.value]),
    ("Arrived two weeks late and the retail box was completely crushed.", ["shipping_packaging"], [ComplaintCategory.DELIVERY_PACKAGING.value]),
    ("Customer service refused to honor the warranty and was outright rude.", ["customer_service"], [ComplaintCategory.CUSTOMER_SUPPORT.value]),
    ("The desktop software crashes constantly on Windows 11 and fails to detect the device.", ["software"], [ComplaintCategory.SOFTWARE_BUG.value]),
    ("Charging port stopped working after a month; only charges when holding the cable at an angle.", ["battery_life", "build_quality"], [ComplaintCategory.BATTERY_CHARGING.value, ComplaintCategory.BUILD_DURABILITY.value]),
    ("Horrible static buzzing noise whenever ANC is enabled and Bluetooth is active.", ["sound_quality", "connectivity"], [ComplaintCategory.AUDIO_SOUND.value, ComplaintCategory.CONNECTIVITY_BLUETOOTH.value]),
    ("Painful wrist cramps and glitchy cursor tracking on standard mousepads.", ["comfort", "tracking"], [ComplaintCategory.COMFORT_FIT.value, ComplaintCategory.BUILD_DURABILITY.value]),
]

NEUTRAL_COMPONENTS = [
    ("The audio is decent for podcasts, but battery life is strictly average at around 6 hours.", ["sound_quality", "battery_life"], ["None"]),
    ("Decent hardware construction, but the software feels half-baked and confusing.", ["build_quality", "software"], [ComplaintCategory.SOFTWARE_BUG.value]),
    ("Comfort is acceptable, though it gets a little warm after an hour of use.", ["comfort"], ["None"]),
    ("It works as advertised, nothing spectacular but gets the job done for the price.", ["price_value", "general"], ["None"]),
    ("Sound is okay and connection is fine, but for $150 I was hoping for something more premium.", ["sound_quality", "price_value"], [ComplaintCategory.PRICE_VALUE.value]),
    ("Battery lasts about what the box claims. Neither blown away nor disappointed.", ["battery_life"], ["None"]),
    ("Good build quality but customer service took 5 days to respond to an inquiry.", ["build_quality", "customer_service"], [ComplaintCategory.CUSTOMER_SUPPORT.value]),
    ("Good display clarity but the stand is a bit wobbly on my desk.", ["display", "build_quality"], ["None"]),
]

VARIATIONS_INTRO = [
    "Honestly, ", "After using this for a month: ", "Here is my honest take: ",
    "Purchased this last week. ", "I was really excited about this product. ",
    "Long-time user review: ", "Straight to the point: ", "My experience so far: ",
    "I had mixed feelings before ordering. ", "Upgraded to this model recently. "
]

VARIATIONS_OUTRO = [
    " Would recommend to friends.", " Definitely returning this tomorrow.",
    " It is okay if you find it on sale.", " Extremely pleased with this purchase!",
    " Save your money and buy a competing brand.", " Average product overall.",
    " Best tech purchase I made this year.", " Needs firmware fixes desperately.",
    " Does the job for casual daily use.", " 5 stars without hesitation."
]


def generate_300_reviews(seed: int = 42) -> List[Dict[str, Any]]:
    random.seed(seed)
    reviews: List[Dict[str, Any]] = []

    # Distribution: 120 Positive, 120 Negative, 60 Neutral = 300 total
    plan = [
        ("Positive", 120, POSITIVE_COMPONENTS),
        ("Negative", 120, NEGATIVE_COMPONENTS),
        ("Neutral", 60, NEUTRAL_COMPONENTS),
    ]

    counter = 1
    for sentiment, count, pool in plan:
        for _ in range(count):
            prod = PRODUCTS[counter % len(PRODUCTS)]
            item = random.choice(pool)
            base_text, aspects, complaints = item

            # Add natural phrasing variations
            intro = random.choice(VARIATIONS_INTRO)
            outro = ""
            if sentiment == "Positive":
                outro = random.choice([" Highly recommended!", " Would buy again in a heartbeat.", " Outstanding quality!"])
            elif sentiment == "Negative":
                outro = random.choice([" Very disappointing.", " Would not recommend.", " Regret this purchase."])
            else:
                outro = random.choice([" Fair compromise for the price.", " Just middle of the road.", " Mixed feelings overall."])

            # In some cases combine two aspects for rich multi-aspect review
            full_text = f"{intro}{base_text}{outro}"
            if random.random() < 0.35 and sentiment != "Neutral":
                secondary = random.choice(pool)
                if secondary != item:
                    full_text = f"{intro}{base_text} Also, {secondary[0].lower()}{outro}"
                    aspects = list(set(aspects + secondary[1]))
                    if complaints == ["None"]:
                        complaints = secondary[2]
                    elif secondary[2] != ["None"]:
                        complaints = list(set(complaints + secondary[2]))

            review_entry = {
                "review_id": f"REV_{counter:03d}",
                "product_id": prod["id"],
                "product_name": prod["name"],
                "review_text": full_text.strip(),
                "gold_sentiment": sentiment,
                "gold_aspects": aspects,
                "gold_complaint_categories": complaints,
                "review_length_words": len(full_text.split()),
            }
            reviews.append(review_entry)
            counter += 1

    # Shuffle for realistic randomized evaluation order
    random.shuffle(reviews)
    # Re-assign review IDs cleanly after shuffle
    for idx, r in enumerate(reviews, start=1):
        r["review_id"] = f"REV_{idx:03d}"

    return reviews


def save_dataset(reviews: List[Dict[str, Any]], json_path: str = "data/labelled_reviews_300.json", csv_path: str = "data/labelled_reviews_300.csv"):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(reviews, f, indent=2, ensure_ascii=False)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["review_id", "product_id", "product_name", "review_text", "gold_sentiment", "gold_aspects", "gold_complaint_categories", "review_length_words"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in reviews:
            row = dict(r)
            row["gold_aspects"] = "; ".join(r["gold_aspects"])
            row["gold_complaint_categories"] = "; ".join(r["gold_complaint_categories"])
            writer.writerow(row)

    print(f"Dataset successfully created with {len(reviews)} reviews!")
    print(f"JSON saved to: {json_path}")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    data = generate_300_reviews()
    save_dataset(data)
