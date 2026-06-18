"""
HTTP API for Monstra bot previews and direct single-bot backfills.

Deploy to Render (or any host with Python 3.11+). Set MONSTRA_PREVIEW_SECRET and send:
  Authorization: Bearer <MONSTRA_PREVIEW_SECRET>

Next.js can call this service when MONSTRA_PREVIEW_SERVICE_URL or
MONSTRA_BACKFILL_SERVICE_URL is configured.

Dispatch is driven by algorithm_registry.ALGORITHM_REGISTRY so that adding a new
algorithm only requires a new AlgorithmEntry there (plus the corresponding
Preview_backfill_<slug> and backfill_<slug> modules).  No if-chains to update.
"""

from __future__ import annotations

import importlib
import logging
import os

# Skip noisy .env warnings in env_loader (DATABASE_URL is optional for preview mode).
os.environ.setdefault("MONSTRA_PREVIEW_SERVICE", "1")
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from algorithm_registry import get_algorithm

logger = logging.getLogger("monstra_backfill")

app = FastAPI(title="MonstraBackfill", version="1.0.0")

_allow_origins = os.environ.get("MONSTRA_PREVIEW_CORS_ORIGINS", "").strip()
if _allow_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in _allow_origins.split(",") if o.strip()],
        allow_credentials=True,
        allow_methods=["POST", "GET"],
        allow_headers=["*"],
    )

_bearer = HTTPBearer(auto_error=False)


def verify_preview_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    secret = os.environ.get("MONSTRA_PREVIEW_SECRET", "").strip()
    if not secret:
        return
    token = ""
    if credentials and credentials.scheme.lower() == "bearer":
        token = credentials.credentials or ""
    if not token:
        auth_header = request.headers.get("authorization") or ""
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
    if token != secret:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _run(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Preview dispatch – registry-driven.

    Adding a new algorithm: add AlgorithmEntry to algorithm_registry.py and
    create Preview_backfill_<slug>.py with build_preview_config / run_preview.
    No changes needed here.
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")

    if get_algorithm(kind) is None:
        raise HTTPException(status_code=400, detail=f"Unknown preview kind: {kind!r}")

    try:
        mod = importlib.import_module(f"Preview_backfill_{kind}")
        preview = mod.build_preview_config(payload)
        return mod.run_preview(preview)
    except ValueError as exc:
        return {"success": False, "error": str(exc)}


def _run_backfill(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Direct backfill dispatch – registry-driven.

    Adding a new algorithm: add AlgorithmEntry to algorithm_registry.py and
    create backfill_<slug>.py with fetch_active_<slug>_bots / backfill_single_bot.
    No changes needed here.
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")

    entry = get_algorithm(kind)
    if entry is None:
        raise HTTPException(status_code=400, detail=f"Unknown backfill kind: {kind!r}")

    bot_id = str(payload.get("botId") or "").strip()
    user_id = str(payload.get("userId") or "").strip()
    config_hash = str(payload.get("configHash") or "").strip()

    if not bot_id:
        raise HTTPException(status_code=400, detail="botId is required")

    logger.info(
        "Backfill started",
        extra={
            "bot_id": bot_id,
            "algorithm": kind,
            "user_id": user_id,
            "config_hash": config_hash,
        },
    )

    try:
        mod = importlib.import_module(f"backfill_{kind}")
        fetch_fn = getattr(mod, f"fetch_active_{kind}_bots")
        bots = fetch_fn()
        match = next((bot for bot in bots if bot.bot_id == bot_id), None)
        if match is None:
            raise ValueError(f"No active {kind} bot found with bot_id={bot_id!r}")
        result = mod.backfill_single_bot(match)
    except (ValueError, RuntimeError) as exc:
        logger.error(
            "Backfill failed",
            extra={
                "bot_id": bot_id,
                "algorithm": kind,
                "user_id": user_id,
                "error": str(exc),
            },
        )
        return {"success": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Backfill unexpected error",
            extra={
                "bot_id": bot_id,
                "algorithm": kind,
                "user_id": user_id,
                "error": str(exc),
            },
        )
        return {"success": False, "error": f"Unexpected backfill error: {exc}"}

    errors = int(result.get("errors") or 0)
    skipped = bool(result.get("skipped"))
    processed_days = int(result.get("processed_days") or result.get("processedDays") or 0)
    if skipped or errors > 0 or processed_days <= 0:
        msg = (
            f"Backfill did not complete cleanly for {bot_id} "
            f"(skipped={skipped}, errors={errors}, processed_days={processed_days})."
        )
        logger.error(
            "Backfill incomplete",
            extra={
                "bot_id": bot_id,
                "algorithm": kind,
                "user_id": user_id,
                "skipped": skipped,
                "errors": errors,
                "processed_days": processed_days,
            },
        )
        return {"success": False, "error": msg, "result": result}

    logger.info(
        "Backfill completed",
        extra={
            "bot_id": bot_id,
            "algorithm": kind,
            "user_id": user_id,
            "processed_days": processed_days,
        },
    )
    return {"success": True, "result": result}


@app.post("/preview/alpha1")
def post_preview_alpha1(
    payload: dict[str, Any],
    _: None = Depends(verify_preview_auth),
) -> dict[str, Any]:
    return _run("alpha1", payload)


@app.post("/preview/alpha2")
def post_preview_alpha2(
    payload: dict[str, Any],
    _: None = Depends(verify_preview_auth),
) -> dict[str, Any]:
    return _run("alpha2", payload)


@app.post("/preview/gamma1")
def post_preview_gamma1(
    payload: dict[str, Any],
    _: None = Depends(verify_preview_auth),
) -> dict[str, Any]:
    return _run("gamma1", payload)


@app.post("/backfill/alpha1")
def post_backfill_alpha1(
    payload: dict[str, Any],
    _: None = Depends(verify_preview_auth),
) -> dict[str, Any]:
    return _run_backfill("alpha1", payload)


@app.post("/backfill/alpha2")
def post_backfill_alpha2(
    payload: dict[str, Any],
    _: None = Depends(verify_preview_auth),
) -> dict[str, Any]:
    return _run_backfill("alpha2", payload)


@app.post("/backfill/gamma1")
def post_backfill_gamma1(
    payload: dict[str, Any],
    _: None = Depends(verify_preview_auth),
) -> dict[str, Any]:
    return _run_backfill("gamma1", payload)
