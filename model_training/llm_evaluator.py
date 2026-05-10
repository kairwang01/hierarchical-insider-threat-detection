#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2: LLM Evaluator

This script reads graph-augmented narratives (e.g. stage2_narratives_xgb_first50.txt),
splits them into samples (blocks), and sends each sample to an LLM (e.g. GPT-4o-mini)
with a Chain-of-Thought (CoT) style prompt to obtain:

  - A risk assessment score
  - A short natural-language explanation
  - A recommended action (e.g. escalate / monitor / ignore)

Usage example (from model_training/):

    python llm_evaluator.py \
        --input stage2_narratives_xgb_first50.txt \
        --output llm_predictions_xgb_100.jsonl \
        --model gpt-4o-mini

Notes:
  - Requires the OpenAI Python client (>=1.0.0) and an API key in OPENAI_API_KEY.
  - Use --workers N (default 8) to run N parallel API calls per batch; output order is preserved
    so resume-by-line-count still works. Reduce N if you hit rate limits (429).
  - Use --prompt-style minimal for non-CoT ablation vs default cot (proposal VI.D.2).
"""

import argparse
import json
import os
import pathlib
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Set

from llm_eval_metrics import parse_header_user_date_from_narrative

# One OpenAI client per worker thread (httpx connection reuse, thread-safe usage).
_tls = threading.local()


def load_blocks(path: pathlib.Path, max_samples: int | None = None) -> List[str]:
    """
    Load narrative blocks separated by '=====' lines.
    Each block starts with 'Summary of suspicious behavior...' and ends at the separator line.
    """
    blocks: List[str] = []
    current_lines: List[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "=" * 80:
                # End of current block
                current_lines.append(line)
                block = "".join(current_lines).strip()
                if block:
                    blocks.append(block)
                current_lines = []
                if max_samples is not None and len(blocks) >= max_samples:
                    break
            else:
                current_lines.append(line)

    # If file doesn't end with separator, flush last block
    if current_lines and (max_samples is None or len(blocks) < max_samples):
        block = "".join(current_lines).strip()
        if block:
            blocks.append(block)

    return blocks


def load_key_set_from_split_json(path: pathlib.Path, subset: str) -> Set[Tuple[str, str]]:
    """subset: 'test' | 'train' — keys from stage2_baselines --write-split JSON."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    key = "test_keys" if subset == "test" else "train_keys"
    raw = data.get(key) or []
    out: Set[Tuple[str, str]] = set()
    for row in raw:
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            out.add((str(row[0]).strip(), str(row[1]).strip()[:10]))
    return out


def filter_blocks_by_key_set(blocks: List[str], key_set: Set[Tuple[str, str]]) -> List[str]:
    kept: List[str] = []
    for b in blocks:
        k = parse_header_user_date_from_narrative(b)
        if k is not None and k in key_set:
            kept.append(b)
    return kept


def _few_shot_block() -> str:
    """Short calibration examples (synthetic); keeps format consistent with real narratives."""
    return """
--- CALIBRATION EXAMPLES (same JSON schema as your answer; do not copy these users/dates into real cases) ---

Example 1 (benign / false alarm):
NARRATIVE:
Summary of suspicious behavior for user DEMO_BENIGN on 2010-01-15:
No concrete log events were found for this day in the integrated logs.

Expected judgment: routine screening miss; no evidence. JSON:
{"risk_score": 0.08, "risk_level": "low", "primary_indicators": ["no substantive events"], "explanation": "No file, logon, or cross-source activity is described; treat as non-incident.", "recommended_action": "no action"}

Example 2 (malicious multi-step pattern):
NARRATIVE:
Summary of suspicious behavior for user DEMO_BAD on 2010-06-01: (department: Engineering)
Key events:
- At 2010-06-01 09:10:00, user on host PC1 performed FileOp and accessed "Q4_salaries_confidential.xlsx". [NOTE: user department and resource type are inconsistent]
Historical behavior comparison:
- [NOTE] In the last 30 days, no prior access to resources similar to "Q4_salaries_confidential.xlsx" (financial) was observed
Cross-source chronological narrative:
- 2010-06-01 11:00:00: user connected a USB device on PC1.
- 2010-06-01 11:20:00: user sent an email from PC1 to partner@external.com (size=9000000, attachments=1)

Expected judgment: sensitive financial data + policy conflict + staging/exfiltration chain. JSON:
{"risk_score": 0.92, "risk_level": "high", "primary_indicators": ["cross-department sensitive file", "USB after sensitive access", "large email to external"], "explanation": "Multiple independent signals form a plausible insider exfiltration chain, not explained by normal role duties.", "recommended_action": "escalate to incident response"}
--- END CALIBRATION EXAMPLES ---
"""


def build_cot_prompt(narrative: str) -> List[Dict[str, str]]:
    """
    Build a Chain-of-Thought style chat prompt for a single narrative.
    The model is asked to reason step-by-step, then output a structured JSON decision.
    """
    system_msg = (
        "You are a senior SOC analyst for insider-threat detection. "
        "Narratives come from an automated Stage-1 screener: most flagged windows are benign false positives. "
        "Default to low risk unless several independent, concrete indicators support malicious intent. "
        "Use the narrative's timestamps and cross-source timeline to reason about multi-stage attack chains "
        "(e.g., access to sensitive data → archive/USB/email/web exfil). "
        "Treat bracketed [NOTE: ...] hints as soft signals—they still require alignment with a coherent story. "
        "CRITICAL — anti-hallucination: do not invent hosts, files, paths, email addresses, URLs, USB events, "
        "or timestamps that do not appear in the narrative. Do not claim exfiltration or policy violations "
        "without textual evidence. If the narrative is empty or vague, assign a low risk_score and say so."
    )

    user_msg = f"""
{_few_shot_block()}

You are given ONE real narrative (one user, one calendar day / window). It may include LDAP context, key events,
historical comparison, and a cross-source timeline.

---------------- NARRATIVE START ----------------
{narrative}
---------------- NARRATIVE END ------------------

Reason step by step (internally), covering:
1) Entities: list users, hosts, files/paths, email recipients/domains, URLs, removable media—only what the text states.
2) Relations: who touched what, in what order same-day (build a minimal mental graph).
3) Benign hypothesis: could this be normal job function, IT maintenance, or broad screening noise?
4) Threat hypothesis: is there a plausible insider pattern (misuse of access, policy violation + concealment, staging, exfil)?
5) Evidence checklist: count STRONG indicators (e.g., confirmed sensitive data + exfil vector + inconsistency with history).
   Do NOT treat a single vague keyword as sufficient for high risk.
6) Score: map your conclusion to the rubric below.

RISK SCORE RUBRIC (risk_score must be a JSON number, not a string):
- 0.00–0.20: clearly benign or no substantive activity / obvious false alarm
- 0.21–0.40: weak or ambiguous; only peripheral anomalies
- 0.41–0.60: concerning but incomplete chain; monitor-worthy
- 0.61–0.85: strong multi-indicator pattern consistent with insider misuse
- 0.86–1.00: clear malicious or high-confidence exfiltration / severe policy abuse

Then output ONLY one JSON object (no markdown fences, no commentary) with exactly these fields:
{{
  "risk_score": <float 0-1>,
  "risk_level": "low" | "medium" | "high",
  "primary_indicators": ["short strings"],
  "explanation": "2-4 sentences, plain English, cite specific behaviors",
  "recommended_action": "escalate to incident response | monitor closely | no action"
}}

Rules:
- Output valid JSON: double quotes on keys/strings, risk_score is numeric.
- If the narrative lacks real events, keep risk_score under 0.2.
- In "explanation" and "primary_indicators", only reference facts present in the narrative above.
"""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def build_minimal_prompt(narrative: str) -> List[Dict[str, str]]:
    """
    Direct JSON output without explicit step-by-step CoT (proposal ablation: CoT vs non-CoT).
    Same schema and anti-hallucination rules as CoT, shorter instructions.
    """
    system_msg = (
        "You are a senior SOC analyst for insider-threat detection. "
        "Narratives are from an automated screener; default to low risk without concrete evidence. "
        "CRITICAL — anti-hallucination: do not invent hosts, files, paths, email addresses, URLs, USB events, "
        "or timestamps not in the narrative. If the narrative is empty or vague, assign a low risk_score."
    )
    user_msg = f"""
{_few_shot_block()}

Read the narrative below and output ONLY one JSON object (no markdown fences) with exactly:
{{
  "risk_score": <float 0-1>,
  "risk_level": "low" | "medium" | "high",
  "primary_indicators": ["short strings"],
  "explanation": "2-4 sentences citing only facts from the narrative",
  "recommended_action": "escalate to incident response | monitor closely | no action"
}}

Rubric: 0-0.2 benign/no activity; 0.21-0.4 weak; 0.41-0.6 incomplete chain; 0.61-0.85 strong pattern; 0.86-1.0 clear misuse.

---------------- NARRATIVE START ----------------
{narrative}
---------------- NARRATIVE END ------------------
"""
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def parse_llm_json_output(raw: str) -> Dict[str, Any] | None:
    """Parse model output into a dict; tolerate markdown fences and trailing text."""
    s = (raw or "").strip()
    if not s:
        return None
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
        s = re.sub(r"\s*```\s*$", "", s)
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    i = s.find("{")
    j = s.rfind("}")
    if i != -1 and j > i:
        try:
            obj = json.loads(s[i : j + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _thread_local_openai_client():
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "openai package is not installed. Install with `pip install openai`."
        ) from e
    client = getattr(_tls, "openai_client", None)
    if client is None:
        client = OpenAI()
        _tls.openai_client = client
    return client


def call_openai_chat(model: str, messages: List[Dict[str, str]], temperature: float | None = None) -> str:
    """
    Call OpenAI Chat Completions API and return the model's text output.
    This function assumes OPENAI_API_KEY is set in the environment.
    """
    client = _thread_local_openai_client()
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    # Some hosted models only support the default temperature; they will reject 0.0.
    # If temperature is None, we do not send this arg and let the backend use its default.
    if temperature is not None:
        kwargs["temperature"] = temperature

    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    content = choice.message.content or ""
    return content.strip()


def _process_block(
    idx: int,
    narrative: str,
    model: str,
    temperature: float | None,
    prompt_style: str,
) -> Tuple[int, Dict[str, Any]]:
    """Run one narrative through the LLM; return (index, record)."""
    try:
        if prompt_style == "minimal":
            messages = build_minimal_prompt(narrative)
        else:
            messages = build_cot_prompt(narrative)
        raw_output = call_openai_chat(model, messages, temperature=temperature)
        record: Dict[str, Any] = {
            "index": idx,
            "narrative": narrative,
            "model": model,
            "prompt_style": prompt_style,
            "raw_output": raw_output,
        }
        parsed = parse_llm_json_output(raw_output)
        if parsed is not None:
            record["parsed"] = parsed
        else:
            try:
                record["parsed"] = json.loads(raw_output.strip())
            except Exception as parse_err:
                record["parse_error"] = str(parse_err)
        return idx, record
    except Exception as e:
        return idx, {
            "index": idx,
            "narrative": narrative,
            "model": model,
            "error": str(e),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Evaluator for Stage 2 Narratives")
    parser.add_argument(
        "--input",
        type=str,
        default="stage2_narratives_xgb_first50.txt",
        help="Input narrative file (default: stage2_narratives_xgb_first50.txt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="llm_predictions_xgb_100.jsonl",
        help="Output JSONL file with one JSON per sample (default: llm_predictions_xgb_100.jsonl)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini-2025-08-07",
        help="OpenAI model name (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index for processing (useful for resuming)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel API calls per batch (default: 8). Use 1 for fully serial. Lower if you see 429 rate limits.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 for reproducible scores).",
    )
    parser.add_argument(
        "--no-temperature",
        action="store_true",
        help="Do not send temperature to the API (use model default; needed for some endpoints).",
    )
    parser.add_argument(
        "--keys-json",
        type=str,
        default="",
        help="Optional JSON from stage2_baselines --write-split: only run LLM on those (user,date) windows.",
    )
    parser.add_argument(
        "--keys-subset",
        type=str,
        choices=("test", "train"),
        default="test",
        help="Which key list to use from --keys-json (default: test).",
    )
    parser.add_argument(
        "--prompt-style",
        type=str,
        choices=("cot", "minimal"),
        default="cot",
        help="cot = step-by-step reasoning instructions (default); minimal = direct JSON ablation (no explicit CoT).",
    )

    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    base = pathlib.Path(__file__).parent
    input_path = (base / args.input).resolve()
    output_path = (base / args.output).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading narratives from {input_path} ...")
    blocks = load_blocks(input_path, max_samples=args.max_samples)
    if args.keys_json:
        split_path = (base / args.keys_json).resolve()
        if not split_path.exists():
            raise FileNotFoundError(split_path)
        key_set = load_key_set_from_split_json(split_path, args.keys_subset)
        before = len(blocks)
        blocks = filter_blocks_by_key_set(blocks, key_set)
        print(
            f"  --keys-json {split_path.name} ({args.keys_subset}): {len(blocks)} / {before} blocks match",
            flush=True,
        )
        if not blocks:
            raise RuntimeError("No narrative blocks left after --keys-json filter.")
    print(f"  Loaded {len(blocks)} narrative blocks to process")

    # If resuming, skip already processed lines in output (by counting lines)
    processed = 0
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            processed = sum(1 for _ in f)
        print(f"  Found existing output with {processed} lines; will append from there.")

    start = max(args.start_index, processed)
    if start >= len(blocks):
        print("Nothing to do: all blocks already processed.")
        return

    workers = max(1, args.workers)
    temp_kw: float | None = None if args.no_temperature else float(args.temperature)
    print(
        f"Processing blocks from index {start} to {len(blocks)-1} using model {args.model} "
        f"(prompt_style={args.prompt_style}, {workers} worker(s) per batch, temperature={temp_kw!r}) ..."
    )

    with output_path.open("a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            batch_start = start
            while batch_start < len(blocks):
                batch_end = min(batch_start + workers, len(blocks))
                batch_indices = list(range(batch_start, batch_end))
                print(
                    f"\n=== Batch indices {batch_start}-{batch_end - 1} "
                    f"({len(batch_indices)} blocks, done {batch_start}/{len(blocks)}) ==="
                )

                futures = {
                    executor.submit(
                        _process_block,
                        idx,
                        blocks[idx],
                        args.model,
                        temp_kw,
                        args.prompt_style,
                    ): idx
                    for idx in batch_indices
                }
                results: List[Tuple[int, Dict[str, Any]]] = []
                for fut in as_completed(futures):
                    results.append(fut.result())

                results.sort(key=lambda x: x[0])
                for idx, record in results:
                    if "error" in record:
                        print(f"  Error on block {idx}: {record['error']}")
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                batch_start = batch_end

    print(f"\nDone. Results written to {output_path}")


if __name__ == "__main__":
    main()

