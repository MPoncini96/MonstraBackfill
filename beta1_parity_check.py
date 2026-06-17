from __future__ import annotations

import json

import pandas as pd

import backfill_beta1
import beta1_shared
from beta1_shared import Beta1Config
from bots.beta1 import _latest_target_weights


def _build_fixture() -> pd.DataFrame:
    dates = pd.to_datetime(
        [
            "2025-01-03",
            "2025-01-10",
            "2025-01-17",
            "2025-01-24",
            "2025-01-31",
            "2025-02-07",
            "2025-02-14",
        ]
    )
    return pd.DataFrame(
        {
            "AAA": [100.0, 103.0, 107.0, 110.0, 113.0, 117.0, 121.0],
            "BBB": [100.0, 102.0, 101.0, 105.0, 109.0, 112.0, 111.0],
            "CCC": [100.0, 99.0, 101.0, 104.0, 107.0, 111.0, 114.0],
            "VOO": [100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0],
        },
        index=dates,
    )


def _extract_final_weights(weights_history: pd.DataFrame) -> dict[str, float]:
    if weights_history.empty:
        return {}
    row = weights_history.iloc[-1]
    result: dict[str, float] = {}
    for ticker, weight in row.items():
        value = float(weight)
        if abs(value) > 1e-12:
            result[str(ticker)] = value
    return result


def main() -> int:
    fixture = _build_fixture()
    original_download = beta1_shared.download_weekly_close

    def fake_download_weekly_close(tickers: list[str], start: str, end: str) -> pd.DataFrame:
        return fixture.loc[:, tickers]

    beta1_shared.download_weekly_close = fake_download_weekly_close
    try:
        state_start = beta1_shared.resolve_beta1_state_start_date()
        config = Beta1Config(
            bot_id="parity-fixture",
            universe=["AAA", "BBB", "CCC"],
            entry_threshold=0.0,
            exit_threshold=0.0,
            max_holdings=2,
            fallback_ticker="VOO",
        )

        backfill_result = backfill_beta1.run_backtest(
            universe=config.universe,
            start=state_start,
            end="2025-02-21",
            entry_threshold=config.entry_threshold,
            exit_threshold=config.exit_threshold,
            max_holdings=config.max_holdings,
            fallback_ticker=config.fallback_ticker,
        )
        live_weights, asof, apply_date = _latest_target_weights(
            config,
            start=state_start,
            end="2025-02-21",
        )
    finally:
        beta1_shared.download_weekly_close = original_download

    backfill_weights = _extract_final_weights(backfill_result.weights_history)
    if live_weights != backfill_weights:
        print(
            json.dumps(
                {
                    "status": "mismatch",
                    "state_start": state_start,
                    "live_weights": live_weights,
                    "backfill_weights": backfill_weights,
                    "live_asof": asof,
                    "live_apply_date": apply_date,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1

    print(
        json.dumps(
            {
                "status": "ok",
                "state_start": state_start,
                "live_weights": live_weights,
                "backfill_weights": backfill_weights,
                "live_asof": asof,
                "live_apply_date": apply_date,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
