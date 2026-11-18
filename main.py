"""
Main CLI Application for Aspect-Based Customer Review Analytics with Structured Outputs.

Deliverables Covered:
1. JSON schema and extraction pipeline with LangChain / Pydantic models, JSON & CSV export, and auto-repair.
2. Comparison of zero-shot, few-shot, and CoT classification prompts against 300 labelled reviews (Macro F1-score).
3. Prompt chain (extract -> aggregate -> summarise) with LLM-as-Judge faithfulness scoring.
"""

import os
import sys
import json
import argparse
from typing import Optional

from src.models.schemas import ReviewAnalysisRecord
from src.parsers.repair_parser import MalformedOutputRepairParser
from src.llm.client import UnifiedLLMClient
from src.evaluation.evaluator import PromptTechniqueEvaluator
from src.pipeline.chain import ReviewAnalyticsPipeline
from src.judge.faithfulness_judge import FaithfulnessJudge


def banner():
    print("=" * 80)
    print("  ASPECT-BASED CUSTOMER REVIEW ANALYTICS & PROMPT ENGINEERING PIPELINE")
    print("  Validated Structured Outputs | Zero-Shot / Few-Shot / CoT | LLM-as-Judge")
    print("=" * 80)


def cmd_test_repair():
    """Demonstrate automatic self-healing repair on malformed LLM outputs (Deliverable 1)."""
    banner()
    print("\n[DEMO] Testing Automatic Self-Healing Repair on Malformed LLM Outputs\n")

    parser = MalformedOutputRepairParser()

    test_cases = [
        (
            "Case 1: Markdown code fences + Conversational chatter",
            """
            Hello! Here is the JSON you requested:
            ```json
            {
              "review_id": "REV_MAL_01",
              "product_id": "PROD_HEADPHONES",
              "product_name": "AeroSound Pro",
              "overall_sentiment": "Positive",
              "sentiment_confidence": 0.95,
              "aspects": [
                {"aspect": "sound_quality", "sentiment": "Positive", "evidence": "astonishing clarity"}
              ],
              "complaint_categories": ["None"],
              "summary_takeaway": "Superb acoustic fidelity."
            }
            ```
            Hope this was helpful! Let me know if you need more.
            """
        ),
        (
            "Case 2: Single quotes, unquoted keys, and illegal trailing commas",
            """
            {
              review_id: 'REV_MAL_02',
              product_id: 'PROD_WATCH',
              product_name: 'PulseFit Ultra',
              overall_sentiment: 'Negative',
              aspects: [
                {'aspect': 'battery_life', 'sentiment': 'Negative', 'evidence': 'dead after 3 hours',},
              ],
              complaint_categories: ['Battery / Charging Issue',],
              summary_takeaway: 'Battery life is abysmal.',
            }
            """
        ),
        (
            "Case 3: Truncated JSON with unclosed braces/brackets from token limit",
            """
            {"review_id": "REV_MAL_03", "product_id": "PROD_KEYBOARD", "overall_sentiment": "Neutral", "aspects": [{"aspect": "sound_quality", "sentiment": "Neutral", "evidence": "satisfactory clicks"
            """
        )
    ]

    for title, corrupted_input in test_cases:
        print(f"--- {title} ---")
        print("Raw Corrupted LLM Output:")
        print(corrupted_input.strip()[:180] + ("..." if len(corrupted_input) > 180 else ""))
        
        record = parser.parse_and_repair(corrupted_input)
        
        print("\nAuto-Repair Status:", "SUCCESS (REPAIRED)" if record.is_repaired else "CLEAN")
        print("Repair Log Actions Applied:")
        for log_item in record.repair_log:
            print(f"  * {log_item}")
        print("Validated Pydantic Object:")
        print(f"  Review ID: {record.review_id} | Sentiment: {record.overall_sentiment.value} | Complaints: {record.complaint_categories}")
        print("-" * 70 + "\n")


def cmd_evaluate(limit: Optional[int] = None):
    """Run benchmark comparing Zero-Shot, Few-Shot, and CoT on 300 reviews (Deliverable 2)."""
    banner()
    print("\n[BENCHMARK] Evaluating Zero-Shot, Few-Shot, and CoT Prompts against 300 Reviews")
    evaluator = PromptTechniqueEvaluator()
    results = evaluator.run_full_comparison(techniques=["zero_shot", "few_shot", "cot"], limit=limit)

    print("\n" + "=" * 80)
    print("                      PROMPT BENCHMARK SUMMARY REPORT")
    print("=" * 80)
    print(f"{'Technique':<14} | {'Sentiment Macro F1':<19} | {'Accuracy':<10} | {'Complaint Macro F1':<19} | {'Avg Latency'}")
    print("-" * 80)
    for t in ["zero_shot", "few_shot", "cot"]:
        r = results[t]
        s = r["sentiment"]
        c = r["complaints"]
        print(f"{t.replace('_', ' ').title():<14} | {s['macro_f1']:^19.4f} | {s['accuracy']*100:^9.2f}% | {c['macro_f1']:^19.4f} | {r['avg_latency_ms']:>7.2f} ms")
    print("=" * 80)
    print("\nReports and charts successfully generated:")
    print("  - Markdown Report: reports/prompt_comparison_report.md")
    print("  - Predictions CSV: reports/predictions_comparison.csv")
    print("  - Comparison Chart: reports/prompt_comparison_chart.png")
    print("  - Confusion Matrix: reports/sentiment_confusion_matrices.png\n")


def cmd_pipeline(product_id: str = "PROD_HEADPHONES_01"):
    """Run Prompt Chain (Extract -> Aggregate -> Summarise) + LLM Judge (Deliverable 3)."""
    banner()
    print(f"\n[PIPELINE] Running Prompt Chain + LLM-as-Judge for: {product_id}")

    with open("data/labelled_reviews_300.json", "r", encoding="utf-8") as f:
        reviews = json.load(f)

    pipeline = ReviewAnalyticsPipeline()
    chain_result = pipeline.run_product_chain(product_id, reviews)

    # Run LLM-as-Judge
    judge = FaithfulnessJudge()
    audit = judge.audit_summary(
        summary=chain_result["summary"],
        stats=chain_result["stats"]
    )

    print("\n" + "=" * 80)
    print(f"  EXECUTIVE SUMMARY FOR {chain_result['product_name'].upper()}")
    print("=" * 80)
    print(f"Verdict: {chain_result['summary'].executive_verdict}")
    print("\nKey Strengths:")
    for st in chain_result["summary"].key_strengths:
        print(f"  + {st}")
    print("\nCritical Complaints:")
    for cp in chain_result["summary"].critical_complaints:
        print(f"  - {cp}")
    print("\nActionable Recommendations:")
    for rec in chain_result["summary"].actionable_recommendations:
        print(f"  * {rec}")

    print("\n" + "=" * 80)
    print(f"  LLM-AS-JUDGE FAITHFULNESS AUDIT RESULTS")
    print("=" * 80)
    print(f"Faithfulness Score : {audit.faithfulness_score * 100:.1f}% ({audit.faithfulness_rating})")
    print(f"Supported Claims   : {audit.supported_claims_count} / {audit.total_claims_extracted}")
    print(f"Judge Rationale    : {audit.judge_rationale}")
    print("=" * 80 + "\n")


def cmd_extract(input_file: str, output_json: str, output_csv: str):
    """Run extraction pipeline on user-supplied reviews file with auto-repair (Deliverable 1)."""
    banner()
    print(f"\n[EXTRACT] Processing input file: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        reviews = json.load(f)

    pipeline = ReviewAnalyticsPipeline()
    extracted = pipeline.extract_reviews(
        reviews,
        technique="cot",
        output_json_path=output_json,
        output_csv_path=output_csv
    )
    print(f"\nSuccessfully extracted and validated {len(extracted)} records.")
    print(f"  Exported JSON: {output_json}")
    print(f"  Exported CSV : {output_csv}\n")


def cmd_run_all():
    """Run full suite covering Deliverables 1, 2, and 3."""
    banner()
    print("\nExecuting Complete End-to-End Project Suite...\n")
    # 1. Self-healing parser demo
    cmd_test_repair()
    # 2. Benchmark on 300 reviews
    cmd_evaluate()
    # 3. Prompt Chain + Faithfulness Judge
    cmd_pipeline(product_id="PROD_HEADPHONES_01")
    print("\n[ALL DONE] All deliverables have completed successfully!\n")


def main():
    parser = argparse.ArgumentParser(
        description="Aspect-Based Customer Review Analytics with Structured Outputs"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available Commands")

    # Command: evaluate
    eval_p = subparsers.add_parser("evaluate", help="Compare zero-shot, few-shot, and CoT prompts against 300 reviews")
    eval_p.add_argument("--limit", type=int, default=None, help="Optional limit for rapid testing")

    # Command: pipeline
    pipe_p = subparsers.add_parser("pipeline", help="Run Prompt Chain (Extract -> Aggregate -> Summarise) + Faithfulness Judge")
    pipe_p.add_argument("--product", type=str, default="PROD_HEADPHONES_01", help="Target Product ID")

    # Command: test-repair
    subparsers.add_parser("test-repair", help="Test self-healing repair on malformed/corrupted LLM outputs")

    # Command: extract
    ext_p = subparsers.add_parser("extract", help="Extract structured records from a JSON reviews file")
    ext_p.add_argument("--input", type=str, default="data/labelled_reviews_300.json", help="Path to input reviews JSON")
    ext_p.add_argument("--output-json", type=str, default="output/extracted_records.json", help="Destination JSON path")
    ext_p.add_argument("--output-csv", type=str, default="output/extracted_records.csv", help="Destination CSV path")

    # Command: run-all
    subparsers.add_parser("run-all", help="Run all project components and generate all reports")

    args = parser.parse_args()

    if args.command == "evaluate":
        cmd_evaluate(limit=args.limit)
    elif args.command == "pipeline":
        cmd_pipeline(product_id=args.product)
    elif args.command == "test-repair":
        cmd_test_repair()
    elif args.command == "extract":
        cmd_extract(args.input, args.output_json, args.output_csv)
    elif args.command == "run-all" or args.command is None:
        cmd_run_all()


if __name__ == "__main__":
    main()
