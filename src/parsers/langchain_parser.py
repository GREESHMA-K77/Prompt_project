"""
LangChain-compatible output parser interface.
Can be dropped into any LangChain LCEL chain (e.g., prompt | llm | parser).
"""

from typing import Any, Optional
from src.parsers.repair_parser import MalformedOutputRepairParser
from src.models.schemas import ReviewAnalysisRecord


class LangChainReviewOutputParser:
    """
    LangChain LCEL compatible BaseOutputParser implementation
    featuring automatic self-healing repair.
    """

    def __init__(self, pydantic_object=ReviewAnalysisRecord):
        self.pydantic_object = pydantic_object
        self.repair_parser = MalformedOutputRepairParser(target_model=pydantic_object)

    def parse(self, text: str) -> ReviewAnalysisRecord:
        """Parse the output of an LLM call into a ReviewAnalysisRecord."""
        return self.repair_parser.parse_and_repair(text)

    def get_format_instructions(self) -> str:
        """Return format instructions including JSON schema for the prompt."""
        schema = self.pydantic_object.model_json_schema()
        return (
            "The output should be formatted as a valid JSON instance that conforms to the JSON schema below.\n"
            f"```json\n{schema}\n```\n"
            "Do not output markdown code blocks, do not include intro or outro explanations. Return only valid JSON."
        )

    @property
    def _type(self) -> str:
        return "langchain_self_healing_review_parser"
