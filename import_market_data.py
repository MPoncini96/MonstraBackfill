import logging
from typing import Iterable, Optional

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values


DEFAULT_MARKET_DATA_TABLE = 'trading."Market_Data"'
logger = logging.getLogger(__name__)


def ensure_market_data_table(
    conn: psycopg2.extensions.connection,
    table: str = DEFAULT_MARKET_DATA_TABLE,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                ticker text NOT NULL,
                price_date date NOT NULL,
                closing_price double precision NOT NULL,
                inserted_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (ticker, price_date)
            )
            """
        )
        cur.execute(
            f"""
            ALTER TABLE {table}
            ADD COLUMN IF NOT EXISTS inserted_at timestamptz NOT NULL DEFAULT now()
            """
        )


def import_prices_to_market_data(
    conn: psycopg2.extensions.connection,
    prices: pd.DataFrame,
    only_new: bool = True,
    verbose: bool = False,
    table: str = DEFAULT_MARKET_DATA_TABLE,
):
    """
    Efficiently upsert price DataFrame into trading."Market_Data".
    - prices: DataFrame with index as datetime/date, columns as tickers, values as price.
    - only_new: If True, skips rows already present in DB (minimizes writes).
    """
    if prices.empty:
        if verbose:
            print("No prices to import.")
        return 0

    ensure_market_data_table(conn, table=table)

    if not pd.api.types.is_datetime64_any_dtype(prices.index):
        prices.index = pd.to_datetime(prices.index)
    prices.index = prices.index.date

    from datetime import datetime, timezone

    now_ts = datetime.now(timezone.utc)
    rows = [
        (str(ticker), date, float(price), now_ts)
        for date, row in prices.iterrows()
        for ticker, price in row.items()
        if pd.notna(price)
    ]
    if not rows:
        if verbose:
            print("No valid price rows to import.")
        return 0

    with conn.cursor() as cur:
        if only_new:
            keys = set((ticker, date) for (ticker, date, _, _) in rows)
            cur.execute(
                f"""
                SELECT ticker, price_date FROM {table}
                WHERE (ticker, price_date) IN %s
                """,
                (tuple(keys),),
            )
            existing = set(cur.fetchall())
            rows = [r for r in rows if (r[0], r[1]) not in existing]
            if not rows:
                if verbose:
                    print("All price rows already present.")
                return 0

        execute_values(
            cur,
            f'''
            INSERT INTO {table} (ticker, price_date, closing_price, inserted_at)
            VALUES %s
            ON CONFLICT (ticker, price_date) DO UPDATE
            SET closing_price = EXCLUDED.closing_price,
                inserted_at = EXCLUDED.inserted_at
            ''',
            rows,
            page_size=500,
        )
        if verbose:
            print(f"Upserted {len(rows)} price rows into {table}.")
    conn.commit()
    return len(rows)


def persist_prices_to_market_data(
    prices: pd.DataFrame,
    *,
    only_new: bool = False,
    verbose: bool = False,
    table: str = DEFAULT_MARKET_DATA_TABLE,
    database_url: Optional[str] = None,
) -> int:
    conn: psycopg2.extensions.connection | None = None
    try:
        if database_url:
            conn = psycopg2.connect(database_url)
        else:
            from db import get_conn

            conn = get_conn()

        rows_written = import_prices_to_market_data(
            conn,
            prices,
            only_new=only_new,
            verbose=verbose,
            table=table,
        )
        conn.commit()
        return rows_written
    except Exception:
        if conn is not None:
            conn.rollback()
        raise
    finally:
        if conn is not None:
            conn.close()


def combine_price_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    normalized_frames: list[pd.DataFrame] = []
    for frame in frames:
        if frame is None or frame.empty:
            continue
        normalized = frame.copy()
        normalized.index = pd.to_datetime(normalized.index)
        normalized = normalized.sort_index()
        normalized = normalized[~normalized.index.duplicated(keep="last")]
        normalized = normalized.dropna(how="all")
        normalized.columns = [str(col).strip().upper() for col in normalized.columns]
        normalized_frames.append(normalized)

    if not normalized_frames:
        return pd.DataFrame()

    combined = pd.concat(normalized_frames, axis=1)
    combined = combined.T.groupby(level=0).last().T
    combined = combined.sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.dropna(how="all")


def refresh_ticker_performance_safe() -> None:
    conn: psycopg2.extensions.connection | None = None
    logger.info("Refreshing trading.ticker_performance from Market_Data...")
    try:
        from db import get_conn

        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT trading.refresh_ticker_performance()")
        conn.commit()
        logger.info("Finished refreshing trading.ticker_performance")
    except Exception:
        if conn is not None:
            conn.rollback()
        logger.exception("Failed to refresh trading.ticker_performance")
    finally:
        if conn is not None:
            conn.close()
