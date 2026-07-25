from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Preview_backfill_aptet import PreviewConfig, run_preview

IMPERIUM_UNIVERSE = [
    "NVDA", "PLTR", "ANET", "AVGO", "TSM", "ASML", "META", "NFLX", "AMZN", "BKNG",
    "LLY", "ISRG", "GEV", "ETN", "CAT", "PH", "TT", "URI", "VST", "CEG",
    "XOM", "FCX", "IBKR", "JPM", "WMT",
]


def build_preview(start_date: str, end_date: str) -> PreviewConfig:
    return PreviewConfig(
        bot_name="Imperium",
        universe=list(IMPERIUM_UNIVERSE),
        fallback_ticker="VOO",
        min_holdings=2,
        max_holdings=8,
        adaptation_speed="balanced",
        risk_off_enabled=True,
        start_date=start_date,
        end_date=end_date,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the local Aptet Imperium preview path.")
    parser.add_argument("--start-date", default="2026-07-20")
    parser.add_argument("--end-date", default="2026-07-25")
    args = parser.parse_args()

    preview = build_preview(args.start_date, args.end_date)
    started = time.perf_counter()
    result = run_preview(preview)
    elapsed_seconds = time.perf_counter() - started

    summary = dict(result.get("summary") or {})
    payload = {
        "success": bool(result.get("success")),
        "elapsedSeconds": elapsed_seconds,
        "input": asdict(preview),
        "startDate": result.get("startDate"),
        "endDate": result.get("endDate"),
        "summary": summary,
        "rowCount": len(result.get("equity") or []),
        "lastMeta": ((result.get("equity") or [])[-1].get("meta") if (result.get("equity") or []) else None),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
