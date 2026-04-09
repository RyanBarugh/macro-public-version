"""
ECB Monetary Policy Statement Scraper
=======================================
Scrapes ECB Governing Council monetary policy decisions,
extracts forward guidance paragraphs (pre-stripped), and outputs
ecb_statements.json matching the FOMC scorer input format.

Usage:
    python ecb_scraper.py
    python ecb_scraper.py --output ecb_statements.json
"""

import argparse
import json
import re
import time

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# ── Verified ECB monetary policy decision URLs (from browser console) ─────────
ECB_DECISIONS = [
    ("2020-01-23", "https://www.ecb.europa.eu/press/pr/date/2020/html/ecb.mp200123~ae33d37f6e.en.html"),
    ("2020-03-12", "https://www.ecb.europa.eu/press/pr/date/2020/html/ecb.mp200312~8d3aec3ff2.en.html"),
    ("2020-04-30", "https://www.ecb.europa.eu/press/pr/date/2020/html/ecb.mp200430~1eaa128265.en.html"),
    ("2020-06-04", "https://www.ecb.europa.eu/press/pr/date/2020/html/ecb.mp200604~a307d3429c.en.html"),
    ("2020-07-16", "https://www.ecb.europa.eu/press/pr/date/2020/html/ecb.mp200716~fc5fbe06d9.en.html"),
    ("2020-09-10", "https://www.ecb.europa.eu/press/pr/date/2020/html/ecb.mp200910~f4a8da495e.en.html"),
    ("2020-10-29", "https://www.ecb.europa.eu/press/pr/date/2020/html/ecb.mp201029~4392a355f4.en.html"),
    ("2020-12-10", "https://www.ecb.europa.eu/press/pr/date/2020/html/ecb.mp201210~8c2778b843.en.html"),
    ("2021-01-21", "https://www.ecb.europa.eu/press/pr/date/2021/html/ecb.mp210121~eb9154682e.en.html"),
    ("2021-03-11", "https://www.ecb.europa.eu/press/pr/date/2021/html/ecb.mp210311~35ba71f535.en.html"),
    ("2021-04-22", "https://www.ecb.europa.eu/press/pr/date/2021/html/ecb.mp210422~f075ebe1f0.en.html"),
    ("2021-06-10", "https://www.ecb.europa.eu/press/pr/date/2021/html/ecb.mp210610~b4d5381df0.en.html"),
    ("2021-07-22", "https://www.ecb.europa.eu/press/pr/date/2021/html/ecb.mp210722~48dc3b436b.en.html"),
    ("2021-09-09", "https://www.ecb.europa.eu/press/pr/date/2021/html/ecb.mp210909~2c94b35639.en.html"),
    ("2021-10-28", "https://www.ecb.europa.eu/press/pr/date/2021/html/ecb.mp211028~85474438a4.en.html"),
    ("2021-12-16", "https://www.ecb.europa.eu/press/pr/date/2021/html/ecb.mp211216~1b6d3a1fd8.en.html"),
    ("2022-02-03", "https://www.ecb.europa.eu/press/pr/date/2022/html/ecb.mp220203~90fbe94662.en.html"),
    ("2022-03-10", "https://www.ecb.europa.eu/press/pr/date/2022/html/ecb.mp220310~2d19f8ba60.en.html"),
    ("2022-04-14", "https://www.ecb.europa.eu/press/pr/date/2022/html/ecb.mp220414~d1b76520c6.en.html"),
    ("2022-06-09", "https://www.ecb.europa.eu/press/pr/date/2022/html/ecb.mp220609~122666c272.en.html"),
    ("2022-07-21", "https://www.ecb.europa.eu/press/pr/date/2022/html/ecb.mp220721~53e5bdd317.en.html"),
    ("2022-09-08", "https://www.ecb.europa.eu/press/pr/date/2022/html/ecb.mp220908~c1b6839378.en.html"),
    ("2022-10-27", "https://www.ecb.europa.eu/press/pr/date/2022/html/ecb.mp221027~df1d778b84.en.html"),
    ("2022-12-15", "https://www.ecb.europa.eu/press/pr/date/2022/html/ecb.mp221215~f3461d7b6e.en.html"),
    ("2023-02-02", "https://www.ecb.europa.eu/press/pr/date/2023/html/ecb.mp230202~08a972ac76.en.html"),
    ("2023-03-16", "https://www.ecb.europa.eu/press/pr/date/2023/html/ecb.mp230316~aad5249f30.en.html"),
    ("2023-05-04", "https://www.ecb.europa.eu/press/pr/date/2023/html/ecb.mp230504~cdfd11a697.en.html"),
    ("2023-06-15", "https://www.ecb.europa.eu/press/pr/date/2023/html/ecb.mp230615~d34cddb4c6.en.html"),
    ("2023-07-27", "https://www.ecb.europa.eu/press/pr/date/2023/html/ecb.mp230727~da80cfcf24.en.html"),
    ("2023-09-14", "https://www.ecb.europa.eu/press/pr/date/2023/html/ecb.mp230914~aab39f8c21.en.html"),
    ("2023-10-26", "https://www.ecb.europa.eu/press/pr/date/2023/html/ecb.mp231026~6028cea576.en.html"),
    ("2023-12-14", "https://www.ecb.europa.eu/press/pr/date/2023/html/ecb.mp231214~9846e62f62.en.html"),
    ("2024-01-25", "https://www.ecb.europa.eu/press/pr/date/2024/html/ecb.mp240125~f738889bde.en.html"),
    ("2024-03-07", "https://www.ecb.europa.eu/press/pr/date/2024/html/ecb.mp240307~a5fa52b82b.en.html"),
    ("2024-04-11", "https://www.ecb.europa.eu/press/pr/date/2024/html/ecb.mp240411~1345644915.en.html"),
    ("2024-06-06", "https://www.ecb.europa.eu/press/pr/date/2024/html/ecb.mp240606~2148ecdb3c.en.html"),
    ("2024-07-18", "https://www.ecb.europa.eu/press/pr/date/2024/html/ecb.mp240718~b9e0ddd9d5.en.html"),
    ("2024-09-12", "https://www.ecb.europa.eu/press/pr/date/2024/html/ecb.mp240912~67cb23badb.en.html"),
    ("2024-10-17", "https://www.ecb.europa.eu/press/pr/date/2024/html/ecb.mp241017~aa366eaf20.en.html"),
    ("2024-12-12", "https://www.ecb.europa.eu/press/pr/date/2024/html/ecb.mp241212~2acab6e51e.en.html"),
    ("2025-01-30", "https://www.ecb.europa.eu/press/pr/date/2025/html/ecb.mp250130~530b29e622.en.html"),
    ("2025-03-06", "https://www.ecb.europa.eu/press/pr/date/2025/html/ecb.mp250306~d4340800b3.en.html"),
    ("2025-04-17", "https://www.ecb.europa.eu/press/pr/date/2025/html/ecb.mp250417~42727d0735.en.html"),
    ("2025-06-05", "https://www.ecb.europa.eu/press/pr/date/2025/html/ecb.mp250605~3b5f67d007.en.html"),
    ("2025-07-24", "https://www.ecb.europa.eu/press/pr/date/2025/html/ecb.mp250724~50bc70e13f.en.html"),
    ("2025-09-11", "https://www.ecb.europa.eu/press/pr/date/2025/html/ecb.mp250911~6afb7a9490.en.html"),
    ("2025-10-30", "https://www.ecb.europa.eu/press/pr/date/2025/html/ecb.mp251030~cf0540b5c0.en.html"),
    ("2025-12-18", "https://www.ecb.europa.eu/press/pr/date/2025/html/ecb.mp251218~58b0e415a6.en.html"),
    ("2026-02-05", "https://www.ecb.europa.eu/press/pr/date/2026/html/ecb.mp260205~001d26959b.en.html"),
    ("2026-03-19", "https://www.ecb.europa.eu/press/pr/date/2026/html/ecb.mp260319~3057739775.en.html"),
]


# ── Guidance extraction ───────────────────────────────────────────────────────

GUIDANCE_MARKERS = [
    "the governing council",
    "data-dependent",
    "meeting-by-meeting",
    "not pre-committing",
    "stands ready",
    "determined to ensure",
    "appropriate monetary policy stance",
    "particular rate path",
    "within its mandate",
    "adjust all of its instruments",
    "inflation outlook",
    "interest rate decisions will be based",
    "inflation stabilises",
    "present or lower levels",
    "robustly converge",
    "well ahead of the end",
    "net purchases under",
    "asset purchase programme",
    "reinvest the principal",
    "transmission protection",
    "expects the key ecb interest rates",
    "net asset purchases",
    "pandemic emergency purchase",
]

SKIP_MARKERS = [
    "the president of the ecb will comment",
    "press conference starting",
    "separate press release",
    "media contacts",
    "disclaimer",
    "reproduction is permitted",
    "for media queries",
    "cookie",
    "thank you for letting us know",
]


def is_guidance_paragraph(text: str) -> bool:
    lower = text.lower()
    if any(m in lower for m in SKIP_MARKERS):
        return False
    if len(text) < 40:
        return False
    return any(m in lower for m in GUIDANCE_MARKERS)


def extract_action(text: str) -> str:
    lower = text.lower()
    if "decided to lower" in lower or "decided to reduce" in lower or "decided to decrease" in lower:
        bp = re.search(r"(\d+)\s*basis\s*points?", lower)
        return f"Cut {bp.group(1)}bp" if bp else "Cut"
    if "decided to raise" in lower or "decided to increase" in lower:
        bp = re.search(r"(\d+)\s*basis\s*points?", lower)
        return f"Hike {bp.group(1)}bp" if bp else "Hike"
    if "remain unchanged" in lower or "decided to keep" in lower or "decided to maintain" in lower:
        return "Hold"
    return "Unknown"


def extract_deposit_rate(text: str) -> str:
    match = re.search(r"deposit\s+facility[^.]*?(\d+\.?\d*)\s*%", text.lower())
    return f"{float(match.group(1)):.2f}%" if match else ""


def scrape_statement(session: requests.Session, date: str, url: str) -> dict:
    resp = session.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    main = soup.find("main") or soup.find("div", class_="section") or soup
    paragraphs = main.find_all("p")

    full_text = " ".join(p.get_text(strip=True) for p in paragraphs)
    action = extract_action(full_text)
    rate = extract_deposit_rate(full_text)

    guidance_paras = []
    for p in paragraphs:
        text = p.get_text(strip=True)
        if is_guidance_paragraph(text):
            guidance_paras.append(text)

    guidance = "\n\n".join(guidance_paras)
    if rate:
        guidance += f"\n\nContext: Deposit facility rate at {rate}."

    return {
        "date": date,
        "action": action,
        "guidance_text": guidance,
        "dissents": "ECB does not publish individual votes.",
        "url": url,
    }


def main():
    parser = argparse.ArgumentParser(description="Scrape ECB monetary policy statements")
    parser.add_argument("--output", default="ecb_statements.json", help="Output statements JSON")
    parser.add_argument("--delay", type=float, default=1.5, help="Delay between requests")
    args = parser.parse_args()

    session = requests.Session()
    statements = []

    print(f"Scraping {len(ECB_DECISIONS)} ECB monetary policy decisions...\n")

    for i, (date, url) in enumerate(ECB_DECISIONS):
        print(f"[{i+1}/{len(ECB_DECISIONS)}] {date} — ", end="", flush=True)

        try:
            stmt = scrape_statement(session, date, url)
            statements.append(stmt)
            g_len = len(stmt["guidance_text"])
            print(f"{stmt['action']} | guidance: {g_len} chars")
            if g_len < 50:
                print(f"  ⚠ Short guidance — check extraction markers")
        except requests.exceptions.HTTPError as e:
            print(f"HTTP {e.response.status_code} — URL may be wrong")
            statements.append({
                "date": date, "action": "Unknown", "guidance_text": "",
                "dissents": "ECB does not publish individual votes.", "url": url,
            })
        except Exception as e:
            print(f"ERROR: {e}")
            statements.append({
                "date": date, "action": "Unknown", "guidance_text": "",
                "dissents": "ECB does not publish individual votes.", "url": url,
            })

        time.sleep(args.delay)

    with open(args.output, "w") as f:
        json.dump(statements, f, indent=2)
    print(f"\nSaved {len(statements)} statements to {args.output}")

    # Summary
    print(f"\n{'='*80}")
    print(f"{'Date':<12} {'Action':<12} {'Guidance':>8}  {'Status'}")
    print("-" * 80)
    for s in statements:
        g_len = len(s.get("guidance_text", ""))
        status = "OK" if g_len > 50 else "⚠ SHORT"
        print(f"{s['date']:<12} {s['action']:<12} {g_len:>6} ch  {status}")


if __name__ == "__main__":
    main()