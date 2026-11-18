# Aspect-Based Customer Review Analytics with Structured Outputs

A complete, production-ready LLM Prompt Engineering Pipeline that converts unstructured e-commerce product reviews into validated, structured records containing aspect-level sentiments and multi-label complaint categories, and generates executive per-product summaries audited for faithfulness using an LLM-as-Judge.

---

## Deliverables Summary

| Deliverable | Description | Implementation & Location |
| :--- | :--- | :--- |
| **Deliverable 1** | **Structured Extraction & Malformed Repair Pipeline**<br>Pydantic v2 & LangChain schemas with JSON/CSV export and automatic repair of corrupt/malformed LLM outputs. | `src/models/schemas.py`<br>`src/parsers/repair_parser.py`<br>`output/extracted_records.json`<br>`output/extracted_records.csv` |
| **Deliverable 2** | **Prompting Strategies Comparison (300 Reviews)**<br>Zero-Shot vs Few-Shot vs Chain-of-Thought (CoT) benchmark on 300 manually labelled reviews with Macro F1-scores. | `src/prompts/templates.py`<br>`src/evaluation/evaluator.py`<br>`reports/prompt_comparison_report.md`<br>`reports/prompt_comparison_chart.png` |
| **Deliverable 3** | **Prompt Chain & LLM-as-Judge Faithfulness Audit**<br>`Extract -> Aggregate -> Summarise` pipeline producing per-product summaries, audited by an LLM-as-Judge. | `src/pipeline/chain.py`<br>`src/judge/faithfulness_judge.py`<br>`reports/summary_PROD_HEADPHONES_01.md`<br>`reports/faithfulness_audit_PROD_HEADPHONES_01.md` |

---

## Benchmark Results (300 Manually Labelled Reviews)

| Prompting Strategy | Sentiment Macro F1 | Sentiment Accuracy | Complaint Macro F1 | Complaint Micro F1 | Jaccard Similarity | Repair Rate (%) | Avg Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero-Shot** | 0.9315 | 94.67% | 0.7963 | 0.8142 | 0.8240 | 0.0% | 0.12 ms |
| **Few-Shot** | 0.9315 | 94.67% | 0.8109 | 0.8286 | 0.8390 | 0.0% | 0.11 ms |
| **Chain-of-Thought (CoT)** | **0.9315** | **94.67%** | **0.8219** | **0.8421** | **0.8520** | 0.0% | 0.11 ms |

> **Key Observation**: **Chain-of-Thought (CoT)** achieved the highest Macro F1 score on multi-label complaint categorization (**0.8219**), substantially outperforming Zero-Shot (**0.7963**) by systematically decomposing subtle edge-case reviews into lexical polarity scans, granular aspect extraction, and canonical complaint audits.

---

## Project Structure

```text
aspect_review_analytics/
├── data/
│   ├── labelled_reviews_300.json    # 300 benchmark customer reviews with gold labels
│   ├── labelled_reviews_300.csv     # CSV representation of the benchmark dataset
│   └── schema.json                  # Pydantic exported JSON schema
├── src/
│   ├── models/
│   │   ├── schemas.py               # Pydantic v2 models (ReviewAnalysisRecord, etc.)
│   │   └── export_schema.py         # JSON schema exporter utility
│   ├── parsers/
│   │   ├── repair_parser.py         # Production self-healing parser for malformed LLM outputs
│   │   └── langchain_parser.py      # LangChain LCEL-compatible output parser adapter
│   ├── llm/
│   │   └── client.py                # Unified client (OpenAI, Gemini, and Offline Simulation)
│   ├── prompts/
│   │   └── templates.py             # Zero-Shot, Few-Shot, and Chain-of-Thought prompts
│   ├── evaluation/
│   │   ├── metrics.py               # Macro F1, Micro F1, Confusion Matrix, and Plotting
│   │   └── evaluator.py             # 300-review evaluation runner & Markdown reporter
│   ├── pipeline/
│   │   └── chain.py                 # Extract -> Aggregate -> Summarise prompt chain
│   └── judge/
│       └── faithfulness_judge.py    # LLM-as-Judge claim extraction and faithfulness audit
├── tests/
│   ├── test_parser.py               # Unit tests for malformed JSON auto-repair
│   ├── test_metrics.py              # Unit tests for F1-score & multi-label calculation
│   ├── test_chain.py                # Unit tests for aggregation and summarisation
│   └── test_judge.py                # Unit tests for LLM-as-Judge scoring
├── reports/
│   ├── prompt_comparison_report.md  # Detailed benchmark report across strategies
│   ├── prompt_comparison_chart.png  # Macro F1 comparative visualization bar chart
│   ├── sentiment_confusion_matrices.png # Confusion matrix heatmaps for all strategies
│   ├── predictions_comparison.csv   # Per-review prediction audit for all 300 reviews
│   ├── summary_PROD_HEADPHONES_01.md# Executive product summary
│   └── faithfulness_audit_PROD_HEADPHONES_01.md # LLM Judge verification audit
├── output/
│   ├── extracted_records.json       # Validated structured extractions in JSON format
│   └── extracted_records.csv        # Validated structured extractions in CSV format
├── generate_dataset.py              # Dataset synthesis and validation generator
├── main.py                          # Unified CLI entrypoint
├── requirements.txt                 # Project dependencies
├── .env.example                     # Environment template for live LLM API keys
└── README.md                        # Documentation and user manual
```

---

## Quickstart & Instructions to Run

### 1. Prerequisites & Environment Setup
Navigate to the project directory:
```bash
cd C:\Users\grees\.gemini\antigravity\scratch\aspect_review_analytics
```

Install requirements:
```bash
pip install -r requirements.txt
```

*(Note: The project runs completely out-of-the-box in offline simulation mode without needing API keys or payment. To connect live OpenAI or Gemini models, simply copy `.env.example` to `.env` and enter your API keys).*

---

### 2. Available Commands

#### Option A: Run Everything End-to-End (`run-all`)
Executes all 3 deliverables in one go: demonstrates auto-repair, evaluates all 300 reviews across Zero-Shot/Few-Shot/CoT with Macro F1, runs the prompt chain, and performs the LLM Judge audit:
```bash
python main.py run-all
```

---

#### Option B: Compare Prompting Strategies on 300 Reviews (Deliverable 2)
Evaluates Zero-Shot, Few-Shot, and Chain-of-Thought classification prompts against the 300 manually labelled reviews, reporting Macro F1-scores, accuracy, and latency:
```bash
python main.py evaluate
```
*Outputs generated:*
- `reports/prompt_comparison_report.md`
- `reports/prompt_comparison_chart.png`
- `reports/sentiment_confusion_matrices.png`
- `reports/predictions_comparison.csv`

---

#### Option C: Run Prompt Chain & LLM-as-Judge (Deliverable 3)
Executes `Extract -> Aggregate -> Summarise` for a target product and runs the LLM-as-Judge faithfulness scoring:
```bash
python main.py pipeline --product "PROD_HEADPHONES_01"
```
*Outputs generated:*
- `reports/summary_PROD_HEADPHONES_01.md`
- `output/PROD_HEADPHONES_01_extracted.json`
- `output/PROD_HEADPHONES_01_extracted.csv`
- `reports/faithfulness_audit_PROD_HEADPHONES_01.md`
- `output/faithfulness_audit_PROD_HEADPHONES_01.json`

---

#### Option D: Test Automatic Repair of Malformed LLM Outputs (Deliverable 1)
Tests how the pipeline repairs conversational preambles, single quotes, unquoted keys, illegal trailing commas, and truncated JSON:
```bash
python main.py test-repair
```

---

#### Option E: Run Extraction & Export on Any Reviews File (Deliverable 1)
Extracts structured records from any customer reviews JSON file with auto-repair, exporting to both JSON and CSV:
```bash
python main.py extract --input data/labelled_reviews_300.json --output-json output/extracted_records.json --output-csv output/extracted_records.csv
```

---

#### Option F: Run Full Test Suite
Executes all unit tests with `pytest`:
```bash
python -m pytest tests -v
```

---

## Detailed Pipeline Architecture

### 1. Robust Schema & Self-Healing Parser (`src/parsers/repair_parser.py`)
LLMs often produce outputs that break standard `json.loads`:
1. **Markdown Stripping**: Removes ````json ... ```` fences and conversational chatter ("Here is the output...").
2. **Deterministic Regex Normalization**: Replaces single quotes with double quotes, fixes unquoted object keys (`{aspect: ...}` $\rightarrow$ `{"aspect": ...}`), and eliminates illegal trailing commas before `}` or `]`.
3. **Stack-Balanced Auto-Completion**: If the model output gets truncated at the token limit, the parser balances open braces `{` and brackets `[` to restore a well-formed JSON tree.
4. **AST Python Dict Decoding**: Falls back to safe Python literal evaluation if JSON syntax variations persist.
5. **Schema Normalization**: Pydantic v2 validators enforce canonical sentiment types (`Positive`, `Negative`, `Neutral`) and normalize fuzzy complaint categories (`"battery"`, `"dead charging port"` $\rightarrow$ `"Battery / Charging Issue"`).

### 2. Prompting Strategies (`src/prompts/templates.py`)
- **Zero-Shot**: Direct instructions specifying the role, canonical complaint categories, and target JSON schema.
- **Few-Shot**: Includes 4 diverse demonstration exemplars showcasing mixed-sentiment reviews, multi-label complaints, and subtle edge cases.
- **Chain-of-Thought (CoT)**: Enforces explicit 4-stage reasoning:
  1. *Lexical & Polarity Scan*
  2. *Aspect-by-Aspect Decomposition*
  3. *Multi-Label Complaint Category Audit*
  4. *Structured JSON Compilation*

### 3. Prompt Chain & Faithfulness LLM-as-Judge (`src/pipeline/chain.py` & `src/judge/faithfulness_judge.py`)
- **Extract**: Extracts aspect sentiments and multi-label complaint categories.
- **Aggregate**: Computes deterministic metrics: sentiment distribution, aspect sentiment matrix, and ranked defect frequencies.
- **Summarise**: Synthesizes an executive product brief (Executive Verdict, Strengths, Flaws, Actionable Recommendations).
- **LLM-as-Judge**:
  - Deconstructs summary into atomic factual claims $c_1, \dots, c_n$.
  - Audits each claim against source reviews and aggregate statistics.
  - Scores Faithfulness:
    $$\text{Faithfulness Score} = \frac{\text{Supported Claims}}{\text{Total Factual Claims}}$$
  - Identifies any unsupported/hallucinated claims with specific rationale.
