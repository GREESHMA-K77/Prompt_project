"""
Automatic Repair and Extraction Parser for Malformed LLM Outputs.
Handles markdown wrappers, conversational prose, invalid single-quotes, unquoted keys,
trailing commas, unclosed brackets/braces, Python booleans, and schema validation.
"""

import re
import json
import ast
import logging
from typing import Tuple, Dict, Any, Optional, List
from pydantic import ValidationError
from src.models.schemas import ReviewAnalysisRecord, AspectDetail, SentimentType

logger = logging.getLogger(__name__)


class MalformedOutputRepairParser:
    """
    Production-grade parser that extracts, sanitizes, repairs, and validates
    structured records from unpredictable LLM outputs.
    """

    def __init__(self, target_model=ReviewAnalysisRecord):
        self.target_model = target_model

    def clean_markdown_and_prose(self, raw_text: str) -> Tuple[str, List[str]]:
        """Strip markdown fences, conversational preambles, and conversational suffixes."""
        repair_log = []
        text = raw_text.strip()

        # Step 1: Remove markdown code block markers
        markdown_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if markdown_block_match:
            text = markdown_block_match.group(1).strip()
            repair_log.append("Stripped markdown code fences (```json ... ```)")
        elif text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
            repair_log.append("Removed unclosed markdown code fence prefix/suffix")

        # Step 2: Extract substring between first '{' and last '}'
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            if first_brace > 0 or last_brace < len(text) - 1:
                repair_log.append(f"Trimmed conversational preamble/postscript ({first_brace} chars before, {len(text) - last_brace - 1} chars after)")
                text = text[first_brace : last_brace + 1]
        elif first_brace != -1 and last_brace == -1:
            # Missing closing brace
            text = text[first_brace:]
            text = text + "}"
            repair_log.append("Appended missing closing brace '}' at end")

        return text, repair_log

    def heuristic_json_fixes(self, json_candidate: str) -> Tuple[str, List[str]]:
        """Apply deterministic regex repairs to common malformed JSON syntaxes."""
        repair_log = []
        fixed = json_candidate

        # Normalize curly / fancy unicode quotes
        if any(q in fixed for q in ["“", "”", "‘", "’"]):
            fixed = fixed.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
            repair_log.append("Normalized fancy/curly Unicode quotes to standard ASCII quotes")

        # Clean trailing commas repeatedly
        while re.search(r",\s*([}\]])", fixed):
            fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
            repair_log.append("Removed illegal trailing comma before bracket/brace")

        # Replace Python booleans/None with JSON standard
        patterns = [
            (r"\bTrue\b", "true", "Replaced Python 'True' with JSON 'true'"),
            (r"\bFalse\b", "false", "Replaced Python 'False' with JSON 'false'"),
            (r"\bNone\b", "null", "Replaced Python 'None' with JSON 'null'"),
        ]
        for pat, repl, log_msg in patterns:
            if re.search(pat, fixed):
                fixed = re.sub(pat, repl, fixed)
                repair_log.append(log_msg)

        # Fix single-quoted JSON keys: {'key': -> {"key":
        if re.search(r"\'([a-zA-Z0-9_]+)\'\s*:", fixed):
            fixed = re.sub(r"\'([a-zA-Z0-9_]+)\'\s*:", r'"\1":', fixed)
            repair_log.append("Converted single-quoted object keys to double-quoted keys")

        # Fix unquoted keys: {key: "value" or , key: "value"
        if re.search(r"([{,\s])([a-zA-Z_][a-zA-Z0-9_]*)\s*:", fixed):
            fixed = re.sub(r"([{,\s])([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', fixed)
            repair_log.append("Added quotes to unquoted JSON keys")

        # Fix single-quoted string values
        if re.search(r":\s*'([^']*)'", fixed):
            fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)
            repair_log.append("Converted single-quoted string values to double-quoted strings")

        # Fix single-quoted array elements: ['item1', 'item2'] -> ["item1", "item2"]
        if re.search(r"\[\s*'([^']*)'", fixed) or re.search(r",\s*'([^']*)'", fixed):
            fixed = re.sub(r"\'([^\']*?)\'", r'"\1"', fixed)
            repair_log.append("Converted remaining single quotes to double quotes")

        # Fix unclosed brackets or braces (stack balancing)
        fixed_balanced, balance_log = self._balance_brackets_and_braces(fixed)
        if balance_log:
            fixed = fixed_balanced
            repair_log.extend(balance_log)

        return fixed, repair_log

    def _balance_brackets_and_braces(self, text: str) -> Tuple[str, List[str]]:
        """Balance unclosed brackets '[' or braces '{' caused by token truncation."""
        repair_log = []
        stack = []
        in_string = False
        escape_next = False

        for char in text:
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue

            if char in "{[":
                stack.append(char)
            elif char == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
            elif char == "]":
                if stack and stack[-1] == "[":
                    stack.pop()

        if in_string:
            text += '"'
            repair_log.append("Closed unclosed string quote before end of output")

        if stack:
            closing = ""
            for opener in reversed(stack):
                if opener == "{":
                    closing += "}"
                elif opener == "[":
                    closing += "]"
            text += closing
            repair_log.append(f"Auto-balanced unclosed structures by appending: '{closing}'")

        return text, repair_log

    def fallback_regex_extractor(self, raw_text: str, default_review_id: str = "UNKNOWN_REV", default_prod_id: str = "UNKNOWN_PROD") -> Dict[str, Any]:
        """Emergency fallback extractor using regex patterns if JSON parsing completely fails."""
        data: Dict[str, Any] = {
            "review_id": default_review_id,
            "product_id": default_prod_id,
            "overall_sentiment": "Neutral",
            "aspects": [],
            "complaint_categories": ["None"],
            "summary_takeaway": "Extracted via emergency regex fallback",
        }

        # Extract review_id
        r_id = re.search(r'["\']?review_id["\']?\s*:\s*["\']([^"\']+)["\']', raw_text, re.IGNORECASE)
        if r_id:
            data["review_id"] = r_id.group(1)

        # Extract sentiment
        sent = re.search(r'["\']?overall_sentiment["\']?\s*:\s*["\'](Positive|Negative|Neutral)["\']', raw_text, re.IGNORECASE)
        if sent:
            data["overall_sentiment"] = sent.group(1).capitalize()
        else:
            lower = raw_text.lower()
            if "positive" in lower and "negative" not in lower:
                data["overall_sentiment"] = "Positive"
            elif "negative" in lower and "positive" not in lower:
                data["overall_sentiment"] = "Negative"

        # Extract complaints
        complaints = re.findall(r'["\']([a-zA-Z0-9_\s/]+(?:Issue|Flaw|Bug|Damage|Mismatch|Problem|None))["\']', raw_text)
        if complaints:
            data["complaint_categories"] = complaints

        return data

    def parse_and_repair(
        self,
        raw_output: str,
        default_review_id: str = "REV_UNKNOWN",
        default_product_id: str = "PROD_UNKNOWN"
    ) -> ReviewAnalysisRecord:
        """
        Main pipeline: Cleans, repairs, parses JSON, and validates into ReviewAnalysisRecord.
        Guarantees a valid ReviewAnalysisRecord return even on heavily corrupted outputs.
        """
        all_repairs: List[str] = []
        is_repaired = False

        # Phase 1: Clean markdown fences & conversational text
        cleaned_text, clean_logs = self.clean_markdown_and_prose(raw_output)
        if clean_logs:
            all_repairs.extend(clean_logs)
            is_repaired = True

        # Phase 2: Attempt standard JSON parse
        parsed_dict = None
        try:
            parsed_dict = json.loads(cleaned_text)
        except json.JSONDecodeError:
            pass

        # Phase 2b: Try ast.literal_eval if standard json failed
        if parsed_dict is None:
            # First clean trailing commas in python literal
            temp_clean = re.sub(r",\s*([}\]])", r"\1", cleaned_text)
            try:
                evaluated = ast.literal_eval(temp_clean)
                if isinstance(evaluated, dict):
                    parsed_dict = evaluated
                    all_repairs.append("Parsed single-quoted Python dict literal using ast.literal_eval")
                    is_repaired = True
            except (ValueError, SyntaxError):
                pass

        # Phase 3: Apply heuristic regex repairs if not yet parsed
        if parsed_dict is None:
            repaired_text, repair_logs = self.heuristic_json_fixes(cleaned_text)
            all_repairs.extend(repair_logs)
            is_repaired = True
            try:
                parsed_dict = json.loads(repaired_text)
            except json.JSONDecodeError:
                # Try ast.literal_eval on repaired
                try:
                    evaluated = ast.literal_eval(repaired_text)
                    if isinstance(evaluated, dict):
                        parsed_dict = evaluated
                        all_repairs.append("Successfully decoded using ast.literal_eval after heuristic fixes")
                except (ValueError, SyntaxError) as err:
                    all_repairs.append(f"Heuristic JSON decode failed ({str(err)}); engaged emergency regex fallback")
                    parsed_dict = self.fallback_regex_extractor(raw_output, default_review_id, default_product_id)

        # Phase 4: Validate against Pydantic schema
        if not isinstance(parsed_dict, dict):
            all_repairs.append("LLM output root was not a JSON object; initialized blank record")
            parsed_dict = {
                "review_id": default_review_id,
                "product_id": default_product_id,
                "overall_sentiment": "Neutral",
            }
            is_repaired = True

        # Ensure minimal required fields
        if "review_id" not in parsed_dict or not parsed_dict["review_id"]:
            parsed_dict["review_id"] = default_review_id
            all_repairs.append(f"Injected default review_id: {default_review_id}")
            is_repaired = True
        if "product_id" not in parsed_dict or not parsed_dict["product_id"]:
            parsed_dict["product_id"] = default_product_id
            all_repairs.append(f"Injected default product_id: {default_product_id}")
            is_repaired = True

        try:
            record = self.target_model(**parsed_dict)
            record.is_repaired = is_repaired
            record.repair_log = all_repairs
            return record
        except ValidationError as val_err:
            all_repairs.append(f"Pydantic validation errors encountered: {str(val_err.error_count())}; resolving defaults")
            record = ReviewAnalysisRecord(
                review_id=str(parsed_dict.get("review_id", default_review_id)),
                product_id=str(parsed_dict.get("product_id", default_product_id)),
                product_name=str(parsed_dict.get("product_name", "Unknown Product")),
                overall_sentiment=SentimentType.NEUTRAL,
                sentiment_confidence=0.5,
                aspects=[],
                complaint_categories=["None"],
                summary_takeaway="Recovered from validation error",
                is_repaired=True,
                repair_log=all_repairs
            )
            return record
