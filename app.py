"""
HTTP API for Monstra bot previews and direct single-bot backfills.

Deploy to Render (or any host with Python 3.11+). Set MONSTRA_PREVIEW_SECRET and send:
  Authorization: Bearer <MONSTRA_PREVIEW_SECRET>

Next.js can call this service when MONSTRA_PREVIEW_SERVICE_URL or
MONSTRA_BACKFILL_SERVICE_URL is configured.

Dispatch is driven by algorithm_registry.ALGORITHM_REGISTRY so that adding a new
algorithm only requires a new AlgorithmEntry there (plus the corresponding
Preview_backfill_<slug> and backfill_<slug> modules). No if-chains to update.
"""

from __future__ import annotations

import importlib
import logging
import os
import subprocess
from functools import lru_cache
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from algorithm_registry import get_algorithm
from backfill_error_sanitizer import sanitize_backfill_error
from db import database_status_summary, is_database_configured
from scripts.analyze_echo_pair import (
    PairAnalysisConfig,
    PairAnalysisError,
    TICKER_PATTERN,
    run_analysis,
)

logger = logging.getLogger("monstra_backfill")

SERVICE_NAME = os.environ.get("RENDER_SERVICE_NAME", "").strip() or os.environ.get(
    "MONSTRA_SERVICE_NAME",
    "",
).strip() or "monstra-backfill"

REQUIRED_BACKFILL_MODULES: dict[str, str] = {
    "alpha1": "backfill_alpha1",
    "alpha2": "backfill_alpha2",
    "gamma1": "backfill_gamma1",
    "echo1": "backfill_echo1",
}

REQUIRED_PREVIEW_MODULES: dict[str, str] = {
    "alpha1": "Preview_backfill_alpha1",
    "alpha2": "Preview_backfill_alpha2",
    "gamma1": "Preview_backfill_gamma1",
    "echo1": "Preview_backfill_echo1",
}

app = FastAPI(title="MonstraBackfill", version="1.1.0")

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


@lru_cache(maxsize=1)
def _detect_git_commit() -> str | None:
    for key in ("RENDER_GIT_COMMIT", "GIT_COMMIT", "COMMIT_SHA", "RENDER_GIT_SHA"):
        value = os.environ.get(key, "").strip()
        if value:
            return value[:12]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    value = completed.stdout.strip()
    return value or None


def _classify_exception(exc: BaseException) -> str:
    if isinstance(exc, ModuleNotFoundError):
        return "missing_module"
    message = str(exc).lower()
    if "database_url" in message:
        return "database_not_configured"
    if "connection refused" in message or "could not connect" in message:
        return "database_connection_failed"
    if "timeout" in message or "timed out" in message:
        return "timeout"
    return "unexpected_error"


@lru_cache(maxsize=None)
def _module_status(module_name: str) -> dict[str, Any]:
    try:
        importlib.import_module(module_name)
        return {
            "module": module_name,
            "importable": True,
            "errorCode": None,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "module": module_name,
            "importable": False,
            "errorCode": _classify_exception(exc),
            "error": str(exc),
        }


@lru_cache(maxsize=1)
def _echo_pair_analysis_status() -> dict[str, Any]:
    try:
        run_analysis
        return {
            "module": "scripts.analyze_echo_pair",
            "importable": True,
            "errorCode": None,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "module": "scripts.analyze_echo_pair",
            "importable": False,
            "errorCode": _classify_exception(exc),
            "error": str(exc),
        }


def _registered_routes(prefix: str) -> list[str]:
    paths = sorted(
        {
            route.path
            for route in app.routes
            if getattr(route, "path", "").startswith(prefix)
        }
    )
    return paths


def _readiness_snapshot() -> dict[str, Any]:
    backfill_modules = {
        slug: _module_status(module_name)
        for slug, module_name in REQUIRED_BACKFILL_MODULES.items()
    }
    preview_modules = {
        slug: _module_status(module_name)
        for slug, module_name in REQUIRED_PREVIEW_MODULES.items()
    }
    database = database_status_summary()
    registered_backfill_routes = _registered_routes("/backfill/")
    required_backfill_routes = [f"/backfill/{slug}" for slug in REQUIRED_BACKFILL_MODULES]
    missing_backfill_routes = [
        route for route in required_backfill_routes if route not in registered_backfill_routes
    ]
    all_backfill_modules_importable = all(
        status["importable"] for status in backfill_modules.values()
    )

    return {
        "status": "ok",
        "serviceName": SERVICE_NAME,
        "version": app.version,
        "gitCommit": _detect_git_commit(),
        "database": database,
        "requiredModules": {
            "backfill": backfill_modules,
            "preview": preview_modules,
            "echoPairAnalysis": _echo_pair_analysis_status(),
        },
        "registeredRoutes": {
            "backfill": registered_backfill_routes,
            "preview": _registered_routes("/preview/"),
            "health": ["/health"],
        },
        "readiness": {
            "backfillReady": bool(database["configured"]) and not missing_backfill_routes and all_backfill_modules_importable,
            "previewReady": all(status["importable"] for status in preview_modules.values())
            and _echo_pair_analysis_status()["importable"],
            "missingBackfillRoutes": missing_backfill_routes,
        },
        "queueSupport": {
            "handledByService": False,
            "note": "creator_publish queue jobs are consumed by monstra-backfill-worker, not this web service.",
        },
    }


@app.on_event("startup")
def log_startup_diagnostics() -> None:
    snapshot = _readiness_snapshot()
    logger.info(
        "MonstraBackfill startup diagnostics service=%s commit=%s db_configured=%s backfill_ready=%s preview_ready=%s backfill_routes=%s",
        snapshot["serviceName"],
        snapshot["gitCommit"] or "unknown",
        snapshot["database"]["configured"],
        snapshot["readiness"]["backfillReady"],
        snapshot["readiness"]["previewReady"],
        ",".join(snapshot["registeredRoutes"]["backfill"]),
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return _readiness_snapshot()


def _run(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
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

    if not is_database_configured():
        logger.error(
            "Backfill request rejected because database is not configured",
            extra={"bot_id": bot_id, "algorithm": kind, "user_id": user_id},
        )
        return {
            "success": False,
            "error": "Backfill service is not ready. Database configuration is missing.",
            "errorCode": "database_not_configured",
        }

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
    except ModuleNotFoundError as exc:
        logger.exception(
            "Backfill module import failed",
            extra={
                "bot_id": bot_id,
                "algorithm": kind,
                "user_id": user_id,
                "error": str(exc),
            },
        )
        return {
            "success": False,
            "error": f"Backfill service is not ready for {kind}.",
            "errorCode": "missing_module",
            "module": exc.name,
        }
    except ValueError as exc:
        logger.error(
            "Backfill failed",
            extra={
                "bot_id": bot_id,
                "algorithm": kind,
                "user_id": user_id,
                "error": str(exc),
            },
        )
        return {"success": False, "error": str(exc), "errorCode": "no_active_bot_found"}
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Backfill failed",
            extra={
                "bot_id": bot_id,
                "algorithm": kind,
                "user_id": user_id,
                "error": str(exc),
            },
        )
        return {
            "success": False,
            "error": sanitize_backfill_error(exc),
            "errorCode": _classify_exception(exc),
        }

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
        return {
            "success": False,
            "error": msg,
            "errorCode": "backfill_incomplete",
            "result": result,
        }

    logger.info(
        "Backfill completed",
        extra={
            "bot_id": bot_id,
            "algorithm": kind,
            "user_id": user_id,
            "processed_days": processed_days,
        },
    )
    return {"success": True, "status": "completed", "result": result}


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
