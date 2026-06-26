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
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from algorithm_registry import get_algorithm
from scripts.analyze_echo_pair import (
    PairAnalysisConfig,
    PairAnalysisError,
    TICKER_PATTERN,
    run_analysis,
)

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


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Error sanitization – never leak infrastructure details to callers
# ---------------------------------------------------------------------------

from backfill_error_sanitizer import sanitize_backfill_error

def _error_payload(message: str) -> dict[str, str]:
    return {"error": message}


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    message = str(exc.detail) if exc.detail else "Request failed."
    return JSONResponse(status_code=exc.status_code, content=_error_payload(message))


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    _: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content=_error_payload(f"Invalid request body: {exc.errors()[0]['msg']}"),
    )


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


def _parse_signal_windows(raw: Any) -> list[int]:
    if not isinstance(raw, list):
        raise PairAnalysisError(
            "signalWindows must be an array of positive integers.",
            "invalid_signal_windows",
            2,
        )

    values: list[int] = []
    for entry in raw:
        try:
            parsed = int(entry)
        except (TypeError, ValueError) as exc:
            raise PairAnalysisError(
                "signalWindows must be an array of positive integers.",
                "invalid_signal_windows",
                2,
            ) from exc
        if parsed <= 0:
            raise PairAnalysisError(
                "signalWindows must be an array of positive integers.",
                "invalid_signal_windows",
                2,
            )
        values.append(parsed)

    unique = sorted(set(values))
    if not unique:
        raise PairAnalysisError(
            "signalWindows must include at least one positive integer.",
            "invalid_signal_windows",
            2,
        )
    return unique


def _build_echo_pair_config(payload: dict[str, Any]) -> PairAnalysisConfig:
    leader = str(payload.get("leader") or "").strip().upper()
    follower = str(payload.get("follower") or "").strip().upper()
    if not leader or not follower:
        raise PairAnalysisError("Both stock symbols are required.", "invalid_ticker", 2)
    if not TICKER_PATTERN.match(leader) or not TICKER_PATTERN.match(follower):
        raise PairAnalysisError(
            "Stock symbols must use standard ticker characters only.",
            "invalid_ticker",
            2,
        )
    if leader == follower:
        raise PairAnalysisError(
            "Leader and follower must be different stocks for Echo pair analysis.",
            "identical_tickers",
            2,
        )

    return PairAnalysisConfig(
        leader=leader,
        follower=follower,
        signal_windows=_parse_signal_windows(payload.get("signalWindows")),
    )


def _run_echo_preview(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")

    try:
        mod = importlib.import_module("Preview_backfill_echo1")
        preview = mod.build_preview_config(payload)
        return mod.run_preview(preview)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Echo preview crashed")
        raise HTTPException(
            status_code=500,
            detail=f"Echo preview failed: {exc}",
        ) from exc


def _run_echo_pair_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")

    try:
        config = _build_echo_pair_config(payload)
        return run_analysis(config)
    except PairAnalysisError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Echo pair analysis crashed")
        raise HTTPException(
            status_code=500,
            detail=f"Echo pair analysis failed: {exc}",
        ) from exc


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
    except ValueError as exc:
        # ValueError is a domain error (e.g. bot not found) – safe to surface.
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
        # Log the real error (includes DATABASE_URL messages, stack traces, etc.)
        # then return only a sanitized message to the caller.
        logger.exception(
            "Backfill failed",
            extra={
                "bot_id": bot_id,
                "algorithm": kind,
                "user_id": user_id,
                "error": str(exc),
            },
        )
        return {"success": False, "error": sanitize_backfill_error(exc)}

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


@app.post("/preview/echo1")
def post_preview_echo1(
    payload: dict[str, Any],
    _: None = Depends(verify_preview_auth),
) -> dict[str, Any]:
    return _run_echo_preview(payload)


@app.post("/preview/echo1-pair-correlation")
def post_preview_echo1_pair_correlation(
    payload: dict[str, Any],
    _: None = Depends(verify_preview_auth),
) -> dict[str, Any]:
    return _run_echo_pair_analysis(payload)


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


@app.post("/backfill/echo1")
def post_backfill_echo1(
    payload: dict[str, Any],
    _: None = Depends(verify_preview_auth),
) -> dict[str, Any]:
    return _run_backfill("echo1", payload)
