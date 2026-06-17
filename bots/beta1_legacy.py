# =============================================================================
# LEGACY SNAPSHOT — bots/beta1_legacy.py
# =============================================================================
# This file is a preserved copy of bots/beta1.py as it existed BEFORE the
# composite-scoring redesign of the Beta1 strategy (captured 2026-06-16).
#
# PURPOSE:
#   - Historical reference for the original confirmation-threshold approach.
#   - Comparison baseline against the new composite scoring framework.
#   - Rollback target if the new strategy needs to be reverted.
#
# The original Beta1 approach uses a two-consecutive-week confirmation signal:
#   - BUY  when r2 > threshold AND r1 > threshold (two positive weeks)
#   - SELL when r2 < threshold AND r1 < threshold (two negative weeks)
#   - Positions are scored by strength_score = r2 + r1 and ranked by score.
#
# DO NOT MODIFY THIS FILE.
# Future Beta1 development must target bots/beta1.py, NOT this file.
# Changes here would corrupt the historical snapshot.
# =============================================================================

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from beta1_legacy_shared import (
    Beta1Config,
    normalize_universe as _normalize_universe,
    resolve_beta1_state_start_date,
    run_beta1_reconstruction,
    safe_float as _safe_float,
    safe_int as _safe_int,
    clean_ticker as _clean_ticker,
)
from bot_identity import BOT_TYPE_BETA1

ENTRY_THRESHOLD = 0.0
EXIT_THRESHOLD = 0.0
MAX_HOLDINGS = 4
FALLBACK_TICKER = "VOO"
STATE_START_DATE = resolve_beta1_state_start_date()
logger = logging.getLogger("beta1_legacy_live")


class Beta1ConfigError(RuntimeError):
    pass


def _resolve_download_window() -> tuple[str, str]:
    end = datetime.now(timezone.utc).date() + timedelta(days=1)
    return STATE_START_DATE, end.isoformat()


def _build_live_config(bot_id: str, use_db_config: bool = True) -> Beta1Config:
    if not use_db_config:
        raise Beta1ConfigError(
            "Live beta1 requires DB-backed config. Pass an explicit Beta1Config to helper utilities instead."
        )

    try:
        from db import get_beta1_config
    except Exception as exc:
        raise Beta1ConfigError(f"Could not import beta1 config loader for {bot_id}: {exc}") from exc

    try:
        config_data: dict[str, Any] | None = get_beta1_config(bot_id)
    except Exception as exc:
        raise Beta1ConfigError(f"Could not load beta1 config for {bot_id}: {exc}") from exc

    if not config_data:
        raise Beta1ConfigError(f"No active beta1 config found for {bot_id}.")

    universe = _normalize_universe(config_data.get("universe"))
    if not universe:
        raise Beta1ConfigError(f"Active beta1 config for {bot_id} has an empty or invalid universe.")

    return Beta1Config(
        bot_id=bot_id,
        universe=universe,
        max_holdings=_safe_int(config_data.get("max_holdings"), MAX_HOLDINGS),
        entry_threshold=_safe_float(config_data.get("entry_threshold"), ENTRY_THRESHOLD),
        exit_threshold=_safe_float(config_data.get("exit_threshold"), EXIT_THRESHOLD),
        fallback_ticker=_clean_ticker(config_data.get("fallback_ticker"), FALLBACK_TICKER),
    )


def _latest_target_weights(config: Beta1Config, start: str, end: str) -> tuple[dict[str, float], str | None, str]:
    reconstruction = run_beta1_reconstruction(
        universe=config.universe,
        start=start,
        end=end,
        entry_threshold=config.entry_threshold,
        exit_threshold=config.exit_threshold,
        max_holdings=config.max_holdings,
        fallback_ticker=config.fallback_ticker,
    )
    return reconstruction.final_weights, reconstruction.final_asof, reconstruction.apply_date


def run_beta1(bot_id: str = "beta1", use_db_config: bool = True) -> dict:
    ts = datetime.now(timezone.utc)

    try:
        config = _build_live_config(bot_id, use_db_config=use_db_config)
        start, end = _resolve_download_window()
        logger.info(
            "Running live beta1_legacy bot_id=%s state_start=%s end=%s universe=%s",
            bot_id,
            start,
            end,
            ",".join(config.universe),
        )
        target_weights, asof, applied_to = _latest_target_weights(config, start, end)

        signal = "REBALANCE"
        note = f"As of {asof}: weekly beta1 rebalance"
        logger.info(
            "Completed live beta1_legacy bot_id=%s state_start=%s asof=%s apply_date=%s weights=%s",
            bot_id,
            start,
            asof,
            applied_to,
            target_weights,
        )

        return {
            "bot_id": bot_id,
            "bot_type": BOT_TYPE_BETA1,
            "ts": ts,
            "signal": signal,
            "note": note,
            "payload": {
                "asof": asof,
                "apply_date": applied_to,
                "interval": "W-FRI",
                "state_start_date": start,
                "entry_threshold": config.entry_threshold,
                "exit_threshold": config.exit_threshold,
                "max_holdings": config.max_holdings,
                "fallback_ticker": config.fallback_ticker,
                "target_weights": target_weights,
                },
        }

    except Beta1ConfigError as exc:
        logger.error(
            "Refusing live beta1_legacy bot_id=%s state_start=%s reason=%s",
            bot_id,
            STATE_START_DATE,
            exc,
        )
        return {
            "bot_id": bot_id,
            "bot_type": BOT_TYPE_BETA1,
            "ts": ts,
            "signal": "HOLD",
            "note": f"beta1 refused to run: {exc}",
            "payload": {
                "interval": "W-FRI",
                "state_start_date": STATE_START_DATE,
                "target_weights": {},
            },
        }
    except Exception as exc:
        logger.exception(
            "Live beta1_legacy failed bot_id=%s state_start=%s",
            bot_id,
            STATE_START_DATE,
        )
        return {
            "bot_id": bot_id,
            "bot_type": BOT_TYPE_BETA1,
            "ts": ts,
            "signal": "HOLD",
            "note": f"beta1 error: {exc}",
            "payload": {
                "interval": "W-FRI",
                "state_start_date": STATE_START_DATE,
                "target_weights": {},
            },
        }
