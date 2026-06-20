#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bots.echo1 import (  # noqa: E402
    DEFAULT_DISCOVERY_LOOKBACK,
    DEFAULT_MIN_SAMPLES,
    Echo1Config,
    download_echo1_prices,
)

TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")
PAIR_WINDOWS = {
    "corr3m": 63,
    "corr6m": 126,
    "corr1y": 252,
    "corr2y": 504,
}
MAX_SCATTER_POINTS = 220


@dataclass(frozen=True)
class PairAnalysisConfig:
    leader: str
    follower: str
    signal_windows: list[int]
    discovery_lookback: int = DEFAULT_DISCOVERY_LOOKBACK
    min_samples: int = DEFAULT_MIN_SAMPLES


class PairAnalysisError(Exception):
    def __init__(self, message: str, error_code: str, exit_code: int = 1):
        super().__init__(message)
        self.error_code = error_code
        self.exit_code = exit_code


def _emit_json(payload: dict[str, Any], *, target: Any = sys.stdout) -> None:
    target.write(json.dumps(payload))
    target.write("\n")
    target.flush()


def _normalize_ticker(value: str) -> str:
    return str(value or "").strip().upper()


def _parse_signal_windows(raw: str) -> list[int]:
    values: list[int] = []
    for part in str(raw or "").split(","):
        token = part.strip()
        if not token:
            continue
        try:
            parsed = int(token)
        except ValueError as exc:
            raise PairAnalysisError(
                f"Signal windows must be positive integers. Received '{token}'.",
                "invalid_signal_windows",
                2,
            ) from exc
        if parsed <= 0:
            raise PairAnalysisError(
                "Signal windows must be positive integers.",
                "invalid_signal_windows",
                2,
            )
        values.append(parsed)

    unique = sorted(set(values))
    if not unique:
        raise PairAnalysisError(
            "Signal windows must include at least one positive integer.",
            "invalid_signal_windows",
            2,
        )
    return unique


def parse_args() -> PairAnalysisConfig:
    parser = argparse.ArgumentParser(description="Analyze one Echo leader-follower stock pair.")
    parser.add_argument("--leader", required=True)
    parser.add_argument("--follower", required=True)
    parser.add_argument("--signal-windows", required=True)
    args = parser.parse_args()

    leader = _normalize_ticker(args.leader)
    follower = _normalize_ticker(args.follower)
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
        signal_windows=_parse_signal_windows(args.signal_windows),
    )


def _build_aligned_frame(
    prices: pd.DataFrame,
    leader: str,
    follower: str,
    lag_days: int,
    return_period: int,
) -> pd.DataFrame:
    leader_returns = prices[leader].pct_change(return_period, fill_method=None)
    follower_returns = prices[follower].shift(-lag_days).pct_change(return_period, fill_method=None)
    aligned = pd.concat(
        [
            leader_returns.rename("leader_move"),
            follower_returns.rename("follower_move"),
        ],
        axis=1,
    ).dropna()
    return aligned


def _rank_pair_rows(
    prices: pd.DataFrame,
    cfg: PairAnalysisConfig,
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for lag_days in cfg.signal_windows:
        for return_period in cfg.signal_windows:
            if lag_days < return_period:
                continue

            row: dict[str, Any] = {
                "leader": cfg.leader,
                "follower": cfg.follower,
                "lag_days": lag_days,
                "return_period": return_period,
                **{key: np.nan for key in PAIR_WINDOWS.keys()},
            }
            skip = False

            for label, length in PAIR_WINDOWS.items():
                aligned = _build_aligned_frame(
                    prices.tail(length),
                    cfg.leader,
                    cfg.follower,
                    lag_days,
                    return_period,
                )
                if len(aligned) < cfg.min_samples:
                    corr = np.nan
                else:
                    corr = aligned["leader_move"].corr(aligned["follower_move"])

                row[label] = corr

                if label == "corr3m" and (pd.isna(corr) or abs(corr) < threshold):
                    skip = True
                    break

            if not skip:
                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    corr_cols = list(PAIR_WINDOWS.keys())
    abs_corrs = df[corr_cols].abs()
    df["raw_strength_score"] = abs_corrs.sum(axis=1, skipna=True)
    df["avg_abs_corr"] = abs_corrs.mean(axis=1, skipna=True)
    df["corr_std"] = abs_corrs.std(axis=1, skipna=True).fillna(1)
    df["valid_windows"] = df[corr_cols].notna().sum(axis=1)
    df["same_sign_windows"] = df[corr_cols].apply(
        lambda row: max((row.dropna() > 0).sum(), (row.dropna() < 0).sum())
        if row.dropna().size > 0 else 0,
        axis=1,
    )
    df["sign_consistency"] = df["same_sign_windows"] / df["valid_windows"].replace(0, np.nan)
    df["stability_score"] = 1 / (1 + df["corr_std"])
    df["consistency_score"] = (
        df["sign_consistency"].fillna(0)
        * df["stability_score"]
        * (df["valid_windows"] / len(corr_cols))
    )
    df["score"] = df["avg_abs_corr"].fillna(0) * df["consistency_score"].fillna(0)

    return df.sort_values(by="score", ascending=False, na_position="last").reset_index(drop=True)


def _sample_points(aligned: pd.DataFrame) -> list[dict[str, Any]]:
    if aligned.empty:
        return []

    if len(aligned) <= MAX_SCATTER_POINTS:
        sampled = aligned
    else:
        positions = np.linspace(0, len(aligned) - 1, MAX_SCATTER_POINTS).astype(int)
        sampled = aligned.iloc[positions]

    points: list[dict[str, Any]] = []
    for date, row in sampled.iterrows():
        leader_move = float(row["leader_move"])
        follower_move = float(row["follower_move"])
        if math.isfinite(leader_move) and math.isfinite(follower_move):
            points.append(
                {
                    "leaderMove": leader_move,
                    "followerMove": follower_move,
                    "date": pd.Timestamp(date).date().isoformat(),
                }
            )
    return points


def run_analysis(cfg: PairAnalysisConfig) -> dict[str, Any]:
    runtime_cfg = Echo1Config(
        universe=[cfg.leader, cfg.follower],
        discovery_lookback=cfg.discovery_lookback,
        lag_range=list(cfg.signal_windows),
        return_period_range=list(cfg.signal_windows),
        min_samples=cfg.min_samples,
    )
    prices = download_echo1_prices(runtime_cfg.universe, runtime_cfg.discovery_lookback)

    if prices.empty:
        raise PairAnalysisError(
            "No market data was returned for this pair.",
            "market_data_unavailable",
        )

    prices = prices.dropna(how="all").sort_index()
    prices.index = pd.to_datetime(prices.index)
    prices = prices[~prices.index.duplicated(keep="last")]
    prices.columns = [str(column).upper() for column in prices.columns]
    prices = prices.tail(cfg.discovery_lookback + 10)

    if cfg.leader not in prices.columns or cfg.follower not in prices.columns:
        raise PairAnalysisError(
            "Adjusted close data was unavailable for one or both stocks.",
            "market_data_unavailable",
        )

    ranked = _rank_pair_rows(prices, cfg, threshold=0.10)
    used_relaxed_threshold = False
    if ranked.empty:
        ranked = _rank_pair_rows(prices, cfg, threshold=0.0)
        used_relaxed_threshold = True

    if ranked.empty:
        return {
            "success": True,
            "leader": cfg.leader,
            "follower": cfg.follower,
            "bestLag": None,
            "bestReturnPeriod": None,
            "correlation": None,
            "sampleSize": 0,
            "points": [],
            "noCandidates": True,
            "summaryMessage": "No valid Echo pair setup cleared the selected signal windows.",
            "usedRelaxedThreshold": used_relaxed_threshold,
            "corr3m": None,
            "corr6m": None,
            "corr1y": None,
            "corr2y": None,
        }

    best = ranked.iloc[0]
    best_lag = int(best["lag_days"])
    best_return_period = int(best["return_period"])
    aligned = _build_aligned_frame(prices, cfg.leader, cfg.follower, best_lag, best_return_period)
    if aligned.empty or len(aligned) < cfg.min_samples:
        raise PairAnalysisError(
            "Not enough aligned samples were available for this pair.",
            "insufficient_samples",
        )

    correlation = aligned["leader_move"].corr(aligned["follower_move"])

    return {
        "success": True,
        "leader": cfg.leader,
        "follower": cfg.follower,
        "bestLag": best_lag,
        "bestReturnPeriod": best_return_period,
        "correlation": None if pd.isna(correlation) else float(correlation),
        "sampleSize": int(len(aligned)),
        "points": _sample_points(aligned),
        "noCandidates": False,
        "summaryMessage": None,
        "usedRelaxedThreshold": used_relaxed_threshold,
        "corr3m": None if pd.isna(best["corr3m"]) else float(best["corr3m"]),
        "corr6m": None if pd.isna(best["corr6m"]) else float(best["corr6m"]),
        "corr1y": None if pd.isna(best["corr1y"]) else float(best["corr1y"]),
        "corr2y": None if pd.isna(best["corr2y"]) else float(best["corr2y"]),
    }


def main() -> int:
    try:
        config = parse_args()
        _emit_json(run_analysis(config))
        return 0
    except PairAnalysisError as exc:
        payload = {"success": False, "error": str(exc), "errorCode": exc.error_code}
        _emit_json(payload, target=sys.stderr)
        _emit_json(payload)
        return exc.exit_code
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        payload = {
            "success": False,
            "error": f"Echo pair analysis crashed: {exc}",
            "errorCode": "unexpected_error",
        }
        _emit_json(payload, target=sys.stderr)
        _emit_json(payload)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
