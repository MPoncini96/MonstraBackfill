"""
HTTP API for Monstra bot previews (alpha1, alpha2, beta1, gamma1).

Deploy to Render (or any host with Python 3.11+). Set MONSTRA_PREVIEW_SECRET and send:
  Authorization: Bearer <MONSTRA_PREVIEW_SECRET>

Next.js can call this service when MONSTRA_PREVIEW_SERVICE_URL is configured.
"""

from __future__ import annotations

import os

# Skip noisy .env warnings in env_loader (this service does not require DATABASE_URL).
os.environ.setdefault("MONSTRA_PREVIEW_SERVICE", "1")
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import Preview_backfill_alpha1 as preview_alpha1
import Preview_backfill_alpha2 as preview_alpha2
import Preview_backfill_beta1 as preview_beta1
import Preview_backfill_gamma1 as preview_gamma1

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
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")

    try:
        if kind == "alpha1":
            preview = preview_alpha1.build_preview_config(payload)
            return preview_alpha1.run_preview(preview)
        if kind == "alpha2":
            preview = preview_alpha2.build_preview_config(payload)
            return preview_alpha2.run_preview(preview)
        if kind == "beta1":
            preview = preview_beta1.build_preview_config(payload)
            return preview_beta1.run_preview(preview)
        if kind == "gamma1":
            preview = preview_gamma1.build_preview_config(payload)
            return preview_gamma1.run_preview(preview)
    except ValueError as exc:
        return {"success": False, "error": str(exc)}
    raise HTTPException(status_code=400, detail="Unknown preview kind")


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


@app.post("/preview/beta1")
def post_preview_beta1(
    payload: dict[str, Any],
    _: None = Depends(verify_preview_auth),
) -> dict[str, Any]:
    return _run("beta1", payload)


@app.post("/preview/gamma1")
def post_preview_gamma1(
    payload: dict[str, Any],
    _: None = Depends(verify_preview_auth),
) -> dict[str, Any]:
    return _run("gamma1", payload)
