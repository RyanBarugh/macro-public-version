"""
FOMC Forward Guidance Batch Scorer
====================================
Reads fomc_statements.json from the scraper, sends each guidance extract
to Claude API (Option 1: pre-stripped), scores forward guidance tone,
and outputs a scored time series.

Requires: ANTHROPIC_API_KEY environment variable

Usage:
    python fomc_scorer.py
    python fomc_scorer.py --input fomc_statements.json --output fomc_scores.json
    python fomc_scorer.py --runs 3   # ensemble 3 runs per statement
"""

import argparse
import json
import os
import time
from datetime import datetime
from ..engine.secrets import get_secret

import requests

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = """You are scoring a central bank's forward guidance. The current conditions and rate decision have been removed — you see ONLY guidance language, dissent info, and relevant context.

Score from -2.0 (very dovish guidance) to +2.0 (very hawkish guidance):
- -2.0 = clearly signaling aggressive further easing ahead
- -1.0 = leaning toward further easing, door wide open to cuts
-  0.0 = genuinely balanced, pure data-dependency, no lean
- +1.0 = patience, no urgency to ease or tighten
- +2.0 = signaling tightening or extended restrictive stance

Key signals:
- "not pre-committing to a particular rate path" after an easing cycle = keeping the door open to more cuts (dovish lean)
- "stands ready to adjust all instruments" = activist stance (dovish lean)
- "additional adjustments" without "extent and timing" = less presumptive than before (hawkish shift)
- "carefully assess" with no directional lean = patience (hawkish lean)
- A single dissent for a BIGGER cut, while prior doves now vote with majority = the cut satisfied the doves, majority drew the line (hawkish signal)
- Growth revised significantly upward = less urgency to ease (hawkish)
- Inflation at target = mission accomplished, less need for further action (hawkish for the easer, neutral for the holder)

Respond with ONLY a JSON object, no markdown, no backticks:
{"score": <number>, "rationale": "<brief explanation>", "key_changes": ["<notable signal>"]}

Be precise to one decimal place. Use the full range — scores like -1.3, -0.4, 0.3, 0.7, 1.2 are expected. Do not round to the nearest 0.5."""


def score_statement(guidance_text: str, dissents: str, action: str, date: str, api_key: str, prior_rationale: str = None) -> dict:
    """Score a single statement's forward guidance via Claude API."""

    user_msg = f"""Score this FOMC forward guidance from {date} (action: {action}):

GUIDANCE TEXT:
{guidance_text}

DISSENTS:
{dissents}"""

    if prior_rationale:
        user_msg += f"""

PRIOR MEETING'S ASSESSMENT:
{prior_rationale}"""

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    payload = {
        "model": MODEL,
        "max_tokens": 1000,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_msg}],
    }

    resp = requests.post(API_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()

    data = resp.json()
    text = "".join(b.get("text", "") for b in data.get("content", []))
    clean = text.replace("```json", "").replace("```", "").strip()

    return json.loads(clean)


def main():
    parser = argparse.ArgumentParser(description="Score FOMC forward guidance")
    parser.add_argument("--input", default="fomc_statements.json", help="Input JSON from scraper")
    parser.add_argument("--output", default="fomc_scores.json", help="Output scored JSON")
    parser.add_argument("--runs", type=int, default=1, help="Number of scoring runs per statement (for ensembling)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls in seconds")
    args = parser.parse_args()

    secret = get_secret(os.environ["API_SECRET__ANTHROPIC"], os.environ.get("AWS_REGION", "us-east-1"))
    api_key = secret.get("api_key") or secret.get("ANTHROPIC_API_KEY")

    with open(args.input) as f:
        statements = json.load(f)

    print(f"Scoring {len(statements)} statements ({args.runs} run(s) each)...\n")

    results = []
    prior_rationale = None

    for i, stmt in enumerate(statements):
        date = stmt["date"]
        action = stmt["action"]
        guidance = stmt["guidance_text"]
        dissents = stmt["dissents"]

        if not guidance or len(guidance) < 50:
            print(f"[{i+1}/{len(statements)}] {date} — guidance too short, skipping")
            results.append({
                "date": date,
                "action": action,
                "score": None,
                "rationale": "Guidance text too short",
                "scores": [],
            })
            continue

        scores = []
        rationales = []
        key_changes_list = []

        for run in range(args.runs):
            try:
                result = score_statement(guidance, dissents, action, date, api_key, prior_rationale)
                scores.append(result["score"])
                rationales.append(result.get("rationale", ""))
                key_changes_list.append(result.get("key_changes", []))

                if args.runs > 1:
                    time.sleep(args.delay * 0.5)

            except Exception as e:
                print(f"  [ERR] Run {run+1}: {e}")
                scores.append(None)
                rationales.append(str(e))

        # Compute ensemble score (median of valid scores)
        valid_scores = [s for s in scores if s is not None]
        if valid_scores:
            valid_scores.sort()
            median_idx = len(valid_scores) // 2
            final_score = valid_scores[median_idx]
        else:
            final_score = None

        entry = {
            "date": date,
            "action": action,
            "dissents": dissents,
            "score": final_score,
            "rationale": rationales[0] if rationales else "",
            "key_changes": key_changes_list[0] if key_changes_list else [],
            "scores": scores,
        }
        results.append(entry)

        # Compute delta from prior
        delta = None
        if len(results) >= 2 and results[-2]["score"] is not None and final_score is not None:
            delta = final_score - results[-2]["score"]

        delta_str = f"Δ{delta:+.1f}" if delta is not None else ""
        score_str = f"{final_score:+.1f}" if final_score is not None else "N/A"
        print(f"[{i+1}/{len(statements)}] {date} | {action:<12} | score={score_str} {delta_str} | {rationales[0][:60] if rationales else ''}")

        time.sleep(args.delay)
        if rationales and rationales[0] and not rationales[0].startswith("Expecting"):
            prior_rationale = rationales[0]

    # ── Save results ──────────────────────────────────────────────
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nScores saved to {args.output}")

    # ── Print summary table ───────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"{'Date':<12} {'Action':<12} {'Score':>6} {'Delta':>6}  {'Rationale':<50}")
    print("-" * 90)

    prev_score = None
    for r in results:
        s = r["score"]
        delta = ""
        if s is not None and prev_score is not None:
            d = s - prev_score
            delta = f"{d:+.1f}"
        score_str = f"{s:+.1f}" if s is not None else "N/A"
        rat = (r.get("rationale") or "")[:50]
        print(f"{r['date']:<12} {r['action']:<12} {score_str:>6} {delta:>6}  {rat}")
        if s is not None:
            prev_score = s

    # ── Print for easy copy into spreadsheet ──────────────────────
    print(f"\n{'='*40}")
    print("CSV format (date, score, delta):")
    print("date,score,delta")
    prev = None
    for r in results:
        s = r["score"]
        d = ""
        if s is not None and prev is not None:
            d = f"{s - prev:.2f}"
        print(f"{r['date']},{s if s is not None else ''},{d}")
        if s is not None:
            prev = s


if __name__ == "__main__":
    main()