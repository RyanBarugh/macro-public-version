"""
FOMC Statement Scraper — Prototype
====================================
Scrapes FOMC post-meeting statements from federalreserve.gov,
extracts full text and forward guidance paragraphs.

URL pattern: /newsevents/pressreleases/monetary{YYYYMMDD}a.htm
The date is the LAST day of the meeting (announcement day).

Usage:
    python fomc_scraper.py
"""

import json
import re
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup

# ── Known FOMC meeting announcement dates (last day of meeting) ───
# Going back to Jan 2020. These are public and don't change.
FOMC_DATES = [
    # 2020
    "20200129", "20200315", "20200429", "20200610", "20200729",
    "20200916", "20201105", "20201216",
    # 2021
    "20210127", "20210317", "20210428", "20210616", "20210728",
    "20210922", "20211103", "20211215",
    # 2022
    "20220126", "20220316", "20220504", "20220615", "20220727",
    "20220921", "20221102", "20221214",
    # 2023
    "20230201", "20230322", "20230503", "20230614", "20230726",
    "20230920", "20231101", "20231213",
    # 2024
    "20240131", "20240320", "20240501", "20240612", "20240731",
    "20240918", "20241107", "20241218",
    # 2025
    "20250129", "20250319", "20250507", "20250618", "20250730",
    "20250917", "20251029", "20251210",
    # 2026
    "20260128", "20260318",
]

BASE_URL = "https://www.federalreserve.gov/newsevents/pressreleases/monetary{}a.htm"


def fetch_statement(date_str: str, session: requests.Session) -> dict | None:
    """Fetch and parse a single FOMC statement."""
    url = BASE_URL.format(date_str)

    try:
        resp = session.get(url, timeout=30)
        if resp.status_code == 404:
            print(f"  [404] {date_str} — not found, skipping")
            return None
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [ERR] {date_str} — {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # The statement text is in <div id="article"> or the main content area
    # Find all paragraphs in the article body
    article = soup.find("div", {"id": "article"})
    if not article:
        # Fallback: look for the press release content area
        article = soup.find("div", class_="col-xs-12")

    if not article:
        print(f"  [PARSE] {date_str} — could not find article div")
        return None

    paragraphs = []
    for p in article.find_all("p"):
        text = p.get_text(strip=True)
        if text and len(text) > 20:  # skip tiny fragments
            paragraphs.append(text)

    if not paragraphs:
        print(f"  [PARSE] {date_str} — no paragraphs found")
        return None

    full_text = "\n\n".join(paragraphs)

    # ── Extract structured components ─────────────────────────────
    guidance = extract_guidance(paragraphs)
    dissents = extract_dissents(paragraphs)
    action = extract_action(full_text)

    date_obj = datetime.strptime(date_str, "%Y%m%d")

    return {
        "date": date_obj.strftime("%Y-%m-%d"),
        "date_raw": date_str,
        "url": url,
        "full_text": full_text,
        "guidance_text": guidance,
        "dissents": dissents,
        "action": action,
        "paragraph_count": len(paragraphs),
    }


def extract_guidance(paragraphs: list[str]) -> str:
    """
    Extract forward guidance paragraphs.
    
    FOMC statements have a consistent structure:
      P1: Current conditions (growth, jobs, inflation)
      P2: Mandate + risk assessment
      P3: Rate decision + forward guidance + QT
      P4: Monitoring + preparedness to adjust
      P5: Voting
    
    We want P2 (risk framing), P3 (guidance language), P4 (monitoring).
    Skip P1 (backward-looking) and P5 (voting — extracted separately).
    """
    guidance_parts = []

    for p in paragraphs:
        p_lower = p.lower()

        # Skip the conditions paragraph
        if any(phrase in p_lower for phrase in [
            "recent indicators suggest",
            "available indicators suggest",
            "economic activity has",
            "job gains have",
            "the labor market",
            "growth of economic activity",
        ]) and "committee" not in p_lower:
            continue

        # Skip voting paragraph
        if "voting for the monetary policy action" in p_lower:
            continue
        if "voting against" in p_lower and "voting for" in p_lower:
            continue

        # Skip media contact
        if "media inquiries" in p_lower:
            continue

        # Keep everything else — mandate framing, guidance, monitoring
        if any(phrase in p_lower for phrase in [
            "committee seeks",
            "uncertainty about",
            "attentive to the risks",
            "in support of its goals",
            "in considering",
            "additional adjustments",
            "carefully assess",
            "continue to monitor",
            "prepared to adjust",
            "strongly committed",
            "balance of risks",
            "downside risks",
            "upside risks",
        ]):
            guidance_parts.append(p)

    return "\n\n".join(guidance_parts)


def extract_dissents(paragraphs: list[str]) -> str:
    """Extract dissent information from the voting paragraph."""
    for p in paragraphs:
        if "voting" in p.lower() and ("against" in p.lower() or "preferred" in p.lower()):
            # There are dissents
            against_match = re.search(
                r"Voting against.*$", p, re.IGNORECASE | re.DOTALL
            )
            if against_match:
                return against_match.group(0).strip()

    # Check for unanimous
    for p in paragraphs:
        if "voting for the monetary policy action" in p.lower():
            if "against" not in p.lower():
                return "Unanimous."

    return "Unknown."


def extract_action(full_text: str) -> str:
    """Extract the rate decision."""
    text = full_text.lower()

    if "decided to lower" in text:
        match = re.search(r"lower.*?by\s+([\d/\-]+)\s+percentage point", text)
        if match:
            return f"Cut {match.group(1)} pp"
        return "Cut"
    elif "decided to raise" in text or "decided to increase" in text:
        match = re.search(r"(?:raise|increase).*?by\s+([\d/\-]+)\s+percentage point", text)
        if match:
            return f"Hike {match.group(1)} pp"
        return "Hike"
    elif "decided to maintain" in text:
        return "Hold"
    elif "decided to keep" in text or "decided to leave" in text:
        return "Hold"
    elif re.search(r"target range.*?0 to 1/4 percent", text):
        return "Hold (ZLB)"
    else:
        return "Unknown"


def main():
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html",
    })

    results = []
    print(f"Scraping {len(FOMC_DATES)} FOMC statements...\n")

    for i, date_str in enumerate(FOMC_DATES):
        print(f"[{i+1}/{len(FOMC_DATES)}] {date_str}...", end=" ")

        stmt = fetch_statement(date_str, session)
        if stmt:
            results.append(stmt)
            action = stmt["action"]
            dissent_short = "dissent" if "Unanimous" not in stmt["dissents"] else "unanimous"
            guidance_len = len(stmt["guidance_text"])
            print(f"✓ {action} | {dissent_short} | guidance={guidance_len} chars")
        else:
            print("✗")

        # Be polite to the Fed
        time.sleep(0.5)

    print(f"\n{'='*60}")
    print(f"Scraped {len(results)}/{len(FOMC_DATES)} statements successfully")
    print(f"Date range: {results[0]['date']} to {results[-1]['date']}")
    print(f"{'='*60}\n")

    # ── Summary table ─────────────────────────────────────────────
    print(f"{'Date':<12} {'Action':<12} {'Dissents':<40} {'Guidance chars':<15}")
    print("-" * 79)
    for r in results:
        dissent_short = r["dissents"][:37] + "..." if len(r["dissents"]) > 40 else r["dissents"]
        print(f"{r['date']:<12} {r['action']:<12} {dissent_short:<40} {len(r['guidance_text']):<15}")

    # ── Save full results ─────────────────────────────────────────
    output_path = "fomc_statements.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {output_path}")

    # ── Show one example: September 2025 ──────────────────────────
    sep = next((r for r in results if r["date"] == "2025-09-17"), None)
    if sep:
        print(f"\n{'='*60}")
        print("EXAMPLE: September 17, 2025")
        print(f"{'='*60}")
        print(f"Action: {sep['action']}")
        print(f"Dissents: {sep['dissents']}")
        print(f"\n--- EXTRACTED GUIDANCE ---")
        print(sep["guidance_text"])


if __name__ == "__main__":
    main()