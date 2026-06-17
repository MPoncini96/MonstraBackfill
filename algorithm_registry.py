"""
algorithm_registry.py – Central algorithm registry for Monstra Worker.

Each entry in ALGORITHM_REGISTRY describes one algorithm family (e.g. "alpha1").
The registry is the single source of truth for:
  - slug / bot_type string (must match trading.* table names)
  - display name and human-readable alias
  - worker timing offset (minute past the hour to run)
  - fallback bot id list (used when the DB is unavailable)
  - backfill eligibility
  - live-signal runner import path
  - active-bot-id DB query

Adding a new algorithm requires ONE new entry here plus a backfill dispatcher
registration (see BACKFILL_DISPATCHERS below).  Everything else – the main
loop, run_all_bots, feed refresh – is driven from this table.

Usage:
    from algorithm_registry import ALGORITHM_REGISTRY, BACKFILL_DISPATCHERS
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# AlgorithmEntry dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AlgorithmEntry:
    """Metadata for one algorithm family.

    Attributes
    ----------
    slug:
        Canonical string key used everywhere: DB table names, bot_type column,
        API routes, registry lookups.  Must be lowercase, no spaces.
    display_name:
        Human-readable algorithm name shown in logs and error messages.
    alias:
        Short marketing alias (e.g. "Force", "Phantom").  Used in UI copy.
    worker_minute_offset:
        Which minute past the hour the worker fires this algorithm's signal
        cycle (1 = xx:01 UTC, 2 = xx:02, …).  Must be unique across entries.
    fallback_bot_ids:
        Bot ids used when the DB is unavailable (official roster only).
    db_table:
        PostgreSQL table in the ``trading`` schema that stores bot configs.
        Usually the same as ``slug``; override if they differ.
    backfill_enabled:
        True means a backfill dispatcher exists for this algorithm.
    live_enabled:
        True means the worker runs live signals for this algorithm.
    brokerage_eligible:
        True means Alpaca live trading is supported for this algorithm.
    user_creatable:
        True means regular users can create bots with this algorithm.
    status:
        "active" | "labs" | "hidden" | "deprecated"
    """

    slug: str
    display_name: str
    alias: str
    worker_minute_offset: int
    fallback_bot_ids: list[str] = field(default_factory=list)
    db_table: str = ""           # defaults to slug in __post_init__
    backfill_enabled: bool = True
    live_enabled: bool = True
    brokerage_eligible: bool = False
    user_creatable: bool = True
    status: str = "active"       # "active" | "labs" | "hidden" | "deprecated"

    def __post_init__(self) -> None:
        # Allow db_table to be set independently; fall back to slug.
        if not self.db_table:
            object.__setattr__(self, "db_table", self.slug)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALGORITHM_REGISTRY: tuple[AlgorithmEntry, ...] = (
    AlgorithmEntry(
        slug="alpha1",
        display_name="Force",
        alias="Force",
        worker_minute_offset=1,
        fallback_bot_ids=["bellitus", "bellator", "imperium", "medicus", "vitalis", "vectura", "vis"],
        backfill_enabled=True,
        live_enabled=True,
        brokerage_eligible=True,
        user_creatable=True,
        status="active",
    ),
    AlgorithmEntry(
        slug="alpha2",
        display_name="Oracle",
        alias="Oracle",
        worker_minute_offset=2,
        fallback_bot_ids=["cyclus", "viator"],
        backfill_enabled=True,
        live_enabled=True,
        brokerage_eligible=False,
        user_creatable=True,
        status="active",
    ),
    AlgorithmEntry(
        slug="beta1",
        display_name="Phantom",
        alias="Phantom",
        worker_minute_offset=3,
        fallback_bot_ids=["beta1"],
        backfill_enabled=False,
        live_enabled=False,
        brokerage_eligible=False,
        user_creatable=False,
        status="deprecated",
    ),
    AlgorithmEntry(
        slug="gamma1",
        display_name="Vex",
        alias="Vex",
        worker_minute_offset=4,
        fallback_bot_ids=[],
        backfill_enabled=True,
        live_enabled=True,
        brokerage_eligible=False,
        user_creatable=True,
        status="active",
    ),
    # ---------------------------------------------------------------------------
    # PLACEHOLDER — hidden, not run in production
    # ---------------------------------------------------------------------------
    # delta1 is a stub that proves the registry supports future algorithm families.
    # Status "hidden" means it never appears in creation flows or live loops.
    # To promote: set status="active", live_enabled=True, backfill_enabled=True,
    # create the backfill_delta1.py + bots/delta1.py modules, add to BotType union
    # in TypeScript, and follow the full checklist in ADDING_NEW_ALGORITHM.md.
    AlgorithmEntry(
        slug="delta1",
        display_name="Apex",
        alias="Apex",
        worker_minute_offset=5,
        fallback_bot_ids=[],
        backfill_enabled=False,  # no backfill_delta1.py yet
        live_enabled=False,      # not run in main_loop until promoted
        brokerage_eligible=False,
        user_creatable=False,
        status="hidden",
    ),
    # -------------------------------------------------------------------------
    # beta1_v2 — Phantom v2 (composite scoring)
    # Momentum Score + Confirmation Score + Risk Penalty
    #
    # Status: "labs" — registered for internal routing and DB isolation only.
    # live_enabled=False  — no live runner yet (bots/beta1_v2.py not created).
    #                       Worker will skip this slot entirely.
    # backfill_enabled=False — no backfill_beta1_v2.py yet.
    # user_creatable=False  — create/preview routes not wired yet.
    #
    # Promotion checklist (do not promote until composite scoring is validated):
    #   1. Implement bots/beta1_v2.py and beta1_v2_shared.py
    #   2. Implement backfill_beta1_v2.py
    #   3. Register live runner in worker.py and backfill dispatcher in
    #      process_backfill_jobs.py
    #   4. Wire create-beta1-v2 and preview-beta1-v2 API routes
    #   5. Flip live_enabled=True, backfill_enabled=True, user_creatable=True,
    #      status="active"
    # -------------------------------------------------------------------------
    AlgorithmEntry(
        slug="beta1_v2",
        display_name="Phantom",
        alias="Phantom",
        worker_minute_offset=6,
        fallback_bot_ids=[
            "imperium_v2",
            "regina_v2",
            "bellitus_v2",
            "rapax_v2",
            "bellator_v2",
            "vectura_v2",
            "vis_v2",
        ],
        db_table="beta1_v2",
        backfill_enabled=True,
        live_enabled=True,
        brokerage_eligible=False,
        user_creatable=False,
        status="active",
    ),
)

# ---------------------------------------------------------------------------
# Convenience lookups (built once at import time)
# ---------------------------------------------------------------------------

_REGISTRY_BY_SLUG: dict[str, AlgorithmEntry] = {e.slug: e for e in ALGORITHM_REGISTRY}
_REGISTRY_BY_OFFSET: dict[int, AlgorithmEntry] = {e.worker_minute_offset: e for e in ALGORITHM_REGISTRY}

# Validate uniqueness at import time so bugs surface immediately.
assert len(_REGISTRY_BY_SLUG) == len(ALGORITHM_REGISTRY), \
    "Duplicate slug detected in ALGORITHM_REGISTRY"
assert len(_REGISTRY_BY_OFFSET) == len(ALGORITHM_REGISTRY), \
    "Duplicate worker_minute_offset detected in ALGORITHM_REGISTRY"


def get_algorithm(slug: str) -> AlgorithmEntry | None:
    """Return the AlgorithmEntry for *slug*, or None if unknown."""
    return _REGISTRY_BY_SLUG.get(slug.strip().lower())


def get_active_algorithms() -> list[AlgorithmEntry]:
    """Return all algorithms whose status is 'active'."""
    return [e for e in ALGORITHM_REGISTRY if e.status == "active"]


def get_live_algorithms() -> list[AlgorithmEntry]:
    """Return algorithms that run live signals (active and live_enabled)."""
    return [e for e in ALGORITHM_REGISTRY if e.status == "active" and e.live_enabled]


def active_slugs() -> list[str]:
    """Return slug strings for all active algorithms in registration order."""
    return [e.slug for e in get_active_algorithms()]


# ---------------------------------------------------------------------------
# Backfill dispatcher registry
# ---------------------------------------------------------------------------
# Each entry maps a slug → a callable(bot_id: str) -> dict that runs the
# backfill for a single bot.  The callables are registered lazily so that
# importing algorithm_registry does NOT pull in heavy backfill modules.

_BACKFILL_DISPATCHER_FACTORIES: dict[str, Callable[[], Callable[[str], dict[str, Any]]]] = {}


def register_backfill_dispatcher(
    slug: str,
    factory: Callable[[], Callable[[str], dict[str, Any]]],
) -> None:
    """Register a lazy factory that returns the backfill dispatcher for *slug*.

    The factory is called once on first use and the result is cached.

    Example (in process_backfill_jobs.py or similar)::

        from algorithm_registry import register_backfill_dispatcher

        def _alpha1_dispatcher_factory():
            from backfill_alpha1 import fetch_active_alpha1_bots, backfill_single_bot
            from bot_identity import strip_bot_type_suffix

            def run(bot_id: str) -> dict:
                bots = fetch_active_alpha1_bots()
                want = strip_bot_type_suffix(bot_id)
                matched = [b for b in bots if strip_bot_type_suffix(b.bot_id) == want]
                if not matched:
                    raise ValueError(f"No active alpha1 bot: {bot_id!r}")
                return backfill_single_bot(matched[0])

            return run

        register_backfill_dispatcher("alpha1", _alpha1_dispatcher_factory)
    """
    slug_normalized = slug.strip().lower()
    if slug_normalized not in _REGISTRY_BY_SLUG:
        raise ValueError(
            f"Cannot register backfill dispatcher for unknown slug {slug!r}. "
            f"Add an AlgorithmEntry to ALGORITHM_REGISTRY first."
        )
    _BACKFILL_DISPATCHER_FACTORIES[slug_normalized] = factory


_cached_dispatchers: dict[str, Callable[[str], dict[str, Any]]] = {}


def get_backfill_dispatcher(slug: str) -> Callable[[str], dict[str, Any]] | None:
    """Return the backfill dispatcher for *slug*, or None if none registered."""
    slug_normalized = slug.strip().lower()
    if slug_normalized in _cached_dispatchers:
        return _cached_dispatchers[slug_normalized]
    factory = _BACKFILL_DISPATCHER_FACTORIES.get(slug_normalized)
    if factory is None:
        return None
    dispatcher = factory()
    _cached_dispatchers[slug_normalized] = dispatcher
    return dispatcher


# ---------------------------------------------------------------------------
# Live-signal runner registry
# ---------------------------------------------------------------------------
# Maps slug -> callable(bot_id: str) -> dict  for live signal generation.

_LIVE_RUNNER_FACTORIES: dict[str, Callable[[], Callable[[str], dict[str, Any]]]] = {}


def register_live_runner(
    slug: str,
    factory: Callable[[], Callable[[str], dict[str, Any]]],
) -> None:
    """Register a lazy factory that returns the live-signal runner for *slug*.

    Example (in worker.py)::

        from algorithm_registry import register_live_runner

        register_live_runner("alpha1", lambda: __import__("bots.alpha1", fromlist=["run_alpha1"]).run_alpha1)
    """
    slug_normalized = slug.strip().lower()
    if slug_normalized not in _REGISTRY_BY_SLUG:
        raise ValueError(
            f"Cannot register live runner for unknown slug {slug!r}. "
            f"Add an AlgorithmEntry to ALGORITHM_REGISTRY first."
        )
    _LIVE_RUNNER_FACTORIES[slug_normalized] = factory


_cached_runners: dict[str, Callable[[str], dict[str, Any]]] = {}


def get_live_runner(slug: str) -> Callable[[str], dict[str, Any]] | None:
    """Return the live-signal runner for *slug*, or None if none registered."""
    slug_normalized = slug.strip().lower()
    if slug_normalized in _cached_runners:
        return _cached_runners[slug_normalized]
    factory = _LIVE_RUNNER_FACTORIES.get(slug_normalized)
    if factory is None:
        return None
    runner = factory()
    _cached_runners[slug_normalized] = runner
    return runner
