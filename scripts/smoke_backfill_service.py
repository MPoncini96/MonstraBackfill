#!/usr/bin/env python
"""Smoke-test the MonstraBackfill runtime service."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import requests


def build_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


DEFAULT_PATHS = [
    "/backfill/alpha1",
    "/backfill/alpha2",
    "/backfill/gamma1",
    "/backfill/echo1",
]


def classify_response(status_code: int, payload: dict[str, Any] | None) -> str:
    if status_code == 404:
        return "route_missing"
    if status_code == 401:
        return "unauthorized"
    if status_code >= 500:
        return "server_error"
    if not isinstance(payload, dict):
        return "invalid_json"
    if payload.get("success") is True:
        return "success"
    error_code = str(payload.get("errorCode") or "").strip()
    if error_code:
        return error_code
    error = str(payload.get("error") or "").lower()
    if "no active" in error:
        return "no_active_bot_found"
    if "database" in error:
        return "database_not_configured"
    if "temporarily unavailable" in error:
        return "service_unavailable"
    return "unexpected_error"


def request_json(session: requests.Session, base_url: str, path: str, token: str | None, bot_id: str) -> dict[str, Any]:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = session.post(
        base_url.rstrip("/") + path,
        json={"botId": bot_id},
        headers=headers,
        timeout=60,
    )
    try:
        payload = response.json()
    except ValueError:
        payload = None
    return {
        "path": path,
        "statusCode": response.status_code,
        "classification": classify_response(response.status_code, payload),
        "payload": payload,
    }


def get_health(session: requests.Session, base_url: str, token: str | None) -> dict[str, Any]:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = session.get(base_url.rstrip("/") + "/health", headers=headers, timeout=30)
    try:
        payload = response.json()
    except ValueError:
        payload = None
    return {
        "path": "/health",
        "statusCode": response.status_code,
        "classification": "success" if response.ok else "server_error",
        "payload": payload,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("MONSTRA_BACKFILL_SERVICE_URL", "https://monstrabackfill.onrender.com"),
        help="Backfill service base URL.",
    )
    parser.add_argument(
        "--bot-id",
        default="smoke-bot",
        help="Bot id to use for fake/safe backfill probes.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("MONSTRA_PREVIEW_SECRET", ""),
        help="Optional bearer token when the service is protected.",
    )
    args = parser.parse_args()

    session = build_session()
    summary = {
        "baseUrl": args.base_url,
        "health": get_health(session, args.base_url, args.token or None),
        "backfills": [
            request_json(session, args.base_url, path, args.token or None, args.bot_id)
            for path in DEFAULT_PATHS
        ],
    }

    print(json.dumps(summary, indent=2, sort_keys=True))

    bad = [
        item
        for item in summary["backfills"]
        if item["classification"]
        in {"route_missing", "missing_module", "database_not_configured", "server_error", "invalid_json"}
    ]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
