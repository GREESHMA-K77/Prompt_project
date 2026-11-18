"""
Utility to export JSON schema for ReviewAnalysisRecord and LangChain output parser definitions.
"""

import json
import os
from src.models.schemas import ReviewAnalysisRecord


def generate_and_save_json_schema(output_path: str = "data/schema.json") -> dict:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    schema = ReviewAnalysisRecord.model_json_schema()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    return schema


if __name__ == "__main__":
    schema = generate_and_save_json_schema()
    print(f"JSON schema successfully generated with title: {schema.get('title')}")
