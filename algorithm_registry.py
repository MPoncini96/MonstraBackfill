"""
algorithm_registry.py â€“ Central algorithm registry for Monstra Worker.

Each entry in ALGORITHM_REGISTRY describes one algorithm family (e.g. "force").
The registry is the single source of truth for:
  - slug / bot_type string (must match trading.* table names)
  - display name and human-readable alias
  - worker timing offset (minute past the hour to run)
  - fallback bot id list (used when the DB is unavailable)
  - backfill eligibility
  - live-signal runner import path
  - active-bot-id DB query

Adding a new algorithm requires ONE new entry here plus a backfill dispatcher
registration (see BACKFILL_DISPATCHERS below).  Everything else â€“ the main
loop, run_all_bots, feed refresh â€“ is driven from this table.

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
        Short marketing alias (e.g. "Force", "Oracle").  Used in UI copy.
    worker_minute_offset:
        Which minute past the hour the worker fires this algorithm's signal
        cycle (1 = xx:01 UTC, 2 = xx:02, â€¦).  Must be unique across entries.
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
    repo_tier:
        Publishing tier for the "download this algorithm" effort: "open" =
        full strategy source published as-is. "thin_runner" = a public repo
        that calls a hosted Monstra engine, scoring logic stays server-side.
        "closed" = no repo planned. "undecided" = tiering decision not made
        yet. Independent of `status`: an algorithm can be "active" in
        production while still "undecided" on whether/how its code gets
        published.
    repo_status:
        "not_planned" | "planned" | "published" -- whether a public repo
        exists yet for this algorithm, independent of `repo_tier` (a "planned"
        repo can be either tier, it just isn't live at repo_url yet).
    repo_url:
        GitHub URL once repo_status is "published". None otherwise.
    repo_python_version:
        Minimum Python version required to run the published repo (e.g.
        "3.12+"), or None if repo_status is "not_planned".
    repo_dependencies:
        Keys into DEPENDENCY_REGISTRY below -- the shared Monstra components
        (e.g. the Vectors core schema/engine) this algorithm's code imports.
        Reference by key rather than duplicating each dependency's name/
        status/url per algorithm, since several algorithms share the same
        dependency (e.g. the adaptive-universe engine).
    repo_bring_your_own:
        Free-text list of things the algorithm needs that Monstra doesn't
        provide at all (a market data source, a brokerage connection).
        Per-algorithm, not shared -- two algorithms can have genuinely
        different bring-your-own requirements.
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
    repo_tier: str = "undecided"     # "open" | "thin_runner" | "closed" | "undecided"
    repo_status: str = "not_planned"  # "not_planned" | "planned" | "published"
    repo_url: str | None = None
    repo_python_version: str | None = None
    repo_dependencies: tuple[str, ...] = ()
    repo_bring_your_own: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Allow db_table to be set independently; fall back to slug.
        if not self.db_table:
            object.__setattr__(self, "db_table", self.slug)


# ---------------------------------------------------------------------------
# Dependency registry
# ---------------------------------------------------------------------------
# Shared Monstra components an algorithm repo can depend on, keyed once here
# rather than duplicated per algorithm -- several algorithms share the same
# dependency (e.g. every adaptive-universe algorithm points at
# "adaptive_universe"), so its status/url only needs updating in one place.
# Field names mirror AlgorithmEntry's repo_* fields (and the TS counterpart,
# algorithmFamilyRegistry.ts's RepoCardData) -- a dependency is really just a
# repo in its own right, so it carries the same shape, including its own
# repo_dependencies (adaptive_universe depends on vectors_core).

@dataclass(frozen=True)
class DependencyEntry:
    key: str
    display_name: str
    description: str
    repo_tier: str = "undecided"      # "open" | "thin_runner" | "closed" | "undecided"
    repo_status: str = "not_planned"  # "not_planned" | "planned" | "published"
    repo_url: str | None = None
    repo_python_version: str | None = None
    repo_dependencies: tuple[str, ...] = ()
    repo_bring_your_own: tuple[str, ...] = ()


DEPENDENCY_REGISTRY: tuple[DependencyEntry, ...] = (
    DependencyEntry(
        key="vectors_core",
        display_name="Monstra Vectors (Core)",
        description="vectorframe: a local, yfinance-powered recreation of the scoring engine and per-stock feature contract behind Monstra's Vectors -- the same feature math and composite weights, computed against free market data instead of Monstra's live Alpaca/Postgres pipeline.",
        repo_tier="open",
        repo_status="published",
        repo_url="https://github.com/MPoncini96/vectorframe",
        repo_python_version="3.10+",
        repo_bring_your_own=("Market data -- bring your own (yfinance, bundled)",),
    ),
    DependencyEntry(
        key="adaptive_universe",
        display_name="Adaptive Universe Engine",
        description="universeframe: a local, pandas-native recreation of the shared universe-construction engine (ranking, tag/sector filtering) that adaptive-universe algorithms like Draco, Aptet, Force, and Echo plug their own scoring into -- ships generic example scorers, not any bot's own tuned formula.",
        repo_tier="open",
        repo_status="published",
        repo_url="https://github.com/MPoncini96/universeframe",
        repo_python_version="3.10+",
        repo_dependencies=("vectors_core",),
    ),
)

_DEPENDENCY_BY_KEY: dict[str, DependencyEntry] = {d.key: d for d in DEPENDENCY_REGISTRY}
assert len(_DEPENDENCY_BY_KEY) == len(DEPENDENCY_REGISTRY), \
    "Duplicate key detected in DEPENDENCY_REGISTRY"
for _dep in DEPENDENCY_REGISTRY:
    for _dep_key in _dep.repo_dependencies:
        assert _dep_key in _DEPENDENCY_BY_KEY, \
            f"{_dep.key}: unknown repo_dependencies key {_dep_key!r}, add it to DEPENDENCY_REGISTRY first"
del _dep, _dep_key


def get_dependency(key: str) -> DependencyEntry | None:
    """Return the DependencyEntry for *key*, or None if unknown."""
    return _DEPENDENCY_BY_KEY.get(key.strip().lower())


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Shared by every adaptive-universe algorithm's repo_* fields below (force,
# aptet, echo1, draco all call into vector_universe.py) so the same
# dependency set / bring-your-own list doesn't drift across four separate
# literals.
_ADAPTIVE_UNIVERSE_REPO_DEPENDENCIES: tuple[str, ...] = ("vectors_core", "adaptive_universe")
_STANDARD_REPO_BRING_YOUR_OWN: tuple[str, ...] = (
    "Market data -- bring your own",
    "Brokerage -- optional",
)

ALGORITHM_REGISTRY: tuple[AlgorithmEntry, ...] = (
    AlgorithmEntry(
        slug="force",
        display_name="Force",
        alias="Force",
        worker_minute_offset=1,
        fallback_bot_ids=["bellitus", "bellator", "imperium", "medicus", "vitalis", "vectura", "vis"],
        db_table="alpha1",   # DB table keeps the legacy name trading.alpha1
        backfill_enabled=True,
        live_enabled=True,
        brokerage_eligible=True,
        user_creatable=True,
        status="active",
        repo_tier="open",
        repo_status="planned",
        repo_python_version="3.12+",
        repo_dependencies=_ADAPTIVE_UNIVERSE_REPO_DEPENDENCIES,
        repo_bring_your_own=_STANDARD_REPO_BRING_YOUR_OWN,
    ),
    # ---------------------------------------------------------------------------
    # echo1 â€” EchoNetworkPairs lead-lag pair strategy
    # ---------------------------------------------------------------------------
    # Promoted to live: backfill_echo1.py, bots/echo1.py (run_echo1 live runner),
    # the "echo1" BotType union entry in TypeScript, and trading.echo1 /
    # trading.echo1_pairs_cache tables already exist (see ADDING_NEW_ALGORITHM.md).
    # live_enabled=True is required for get_live_algorithms() to include echo1 â€”
    # without it, main_loop()/run_all_bots() silently skip Echo even though the
    # runner is registered via register_live_runner("echo1", ...) in worker.py.
    # user_creatable stays False: Echo is only deployed as the official "Dux" bot,
    # not yet exposed in the user bot-creation flow.
    AlgorithmEntry(
        slug="echo1",
        display_name="Echo",
        alias="Echo",
        worker_minute_offset=3,
        fallback_bot_ids=[],
        backfill_enabled=True,
        live_enabled=True,
        brokerage_eligible=False,
        user_creatable=False,
        status="active",
        repo_tier="open",
        repo_status="planned",
        repo_python_version="3.12+",
        repo_dependencies=_ADAPTIVE_UNIVERSE_REPO_DEPENDENCIES,
        repo_bring_your_own=_STANDARD_REPO_BRING_YOUR_OWN,
    ),
    AlgorithmEntry(
        slug="aptet",
        display_name="Aptet",
        alias="Aptet",
        worker_minute_offset=5,
        fallback_bot_ids=[],
        backfill_enabled=True,
        live_enabled=True,
        brokerage_eligible=False,
        user_creatable=True,
        status="active",
        repo_tier="open",
        repo_status="planned",
        repo_python_version="3.12+",
        repo_dependencies=_ADAPTIVE_UNIVERSE_REPO_DEPENDENCIES,
        repo_bring_your_own=_STANDARD_REPO_BRING_YOUR_OWN,
    ),
    # ---------------------------------------------------------------------------
    # PLACEHOLDER â€” hidden, not run in production
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
        worker_minute_offset=6,
        fallback_bot_ids=[],
        backfill_enabled=False,  # no backfill_delta1.py yet
        live_enabled=False,      # not run in main_loop until promoted
        brokerage_eligible=False,
        user_creatable=False,
        status="hidden",
    ),
    # ---------------------------------------------------------------------------
    # draco — multi-timeframe log-linear regression mean-reversion / trend-quality
    # ---------------------------------------------------------------------------
    # First official bot: Vectura (a new "draco" variation of the existing
    # official "vectura" identity, alongside its existing alpha1/Force variation
    # - see officialBotRegistry.ts). User-creatable via /create/draco
    # (universe/benchmark/fallback customizable; other params fixed to
    # dracoParamsSchema defaults), same as the aptet/echo1 pattern.
    # Locked regression targets and portfolio/risk state persist in
    # trading.draco_state (see bots/draco.py + migrations/create_draco_tables.sql),
    # never held only in worker process memory.
    # ---------------------------------------------------------------------------
    # bibo — ranks top_n by tag score (bibo_tag_scoring.compute_tag_score: MIN
    # of a bot's selected tags' trading."Vector" composite scores) instead of
    # Force's trailing return, reusing Force's tiered weights / incumbency
    # margin / kill switch mechanics (see bots/bibo.py). Untradable/delisted
    # candidates are skipped before ranking (inactive_assets).
    #
    # Promoted 2026-10-20: 3 official Bibo variations (<bot>_bibo for dux,
    # imperium, regina), chosen from scripts/backtest_bibo_real_vector_history.py
    # (Dux = Echo Dux's tags; Imperium = the "Ignition" combo; Regina = the
    # "Current" combo). Enrolled via
    # NextJS_Monsta/scripts/sync-bibo-official-bots.ts -- that script MUST be
    # run (and trading.bibo / trading.official counts verified) before this
    # entry is deployed, see ADDING_NEW_ALGORITHM.md §10. Live runner is
    # registered in algorithm_plugins.py; backfill dispatcher in
    # process_backfill_jobs.py. Backfilled history can only start at the first
    # trading.vector_history snapshot (2026-09-06), since tag scores have no
    # earlier history. Official-only and paper-only, same posture as Tide.
    # ---------------------------------------------------------------------------
    AlgorithmEntry(
        slug="bibo",
        display_name="Bibo",
        alias="Bibo",
        worker_minute_offset=8,
        fallback_bot_ids=[],
        backfill_enabled=True,
        live_enabled=True,
        brokerage_eligible=False,
        user_creatable=False,
        status="active",
    ),
    # ---------------------------------------------------------------------------
    # oracle â€” LLM-generated allocations (OpenAI now, other providers later)
    # ---------------------------------------------------------------------------
    # A distinct algorithm from the retired "alpha2" (removed from the product
    # 2026-09-21), which also carried the "Oracle" alias historically:
    # trading.oracle is its own table (see
    # NextJS_Monsta/migrations/20260914_A_create_oracle_tables.sql).
    #
    # Each cycle: build_universe-style tag/sector context + trading."Vector"
    # composite scores for the bot's configured universe, plus current paper
    # holdings, are sent to an LLM (see bots/oracle.py) which returns a
    # proposed {ticker: weight} allocation. That proposal is NEVER trusted
    # directly -- it passes through a policy/coherence gate
    # (bots/oracle.py::apply_policy_gate) that drops any ticker outside the
    # closed candidate universe, caps per-name weight, enforces min/max
    # holdings, and renormalizes. On any failure (bad JSON, API error, gate
    # rejection) the bot holds its current position -- same "empty
    # target_weights, never fabricate a fallback trade" convention as
    # bots/aptet.py's run_aptet().
    #
    # backfill_enabled=False permanently, not just for now: LLM completions
    # aren't deterministically replayable the way a price-history calculation
    # is, so there is no historical backfill path for this algorithm, full
    # stop -- unlike bibo/delta1 below, backfill_enabled is not expected to
    # flip True later.
    #
    # Phase 1 (worker side only): scaffolding + the policy gate. Phase 2
    # (2026-09-15): promoted to status="active" / live_enabled=True --
    # OracleAlgorithm is registered in algorithm_plugins.py, 16 official bot
    # variations exist in officialBotRegistry.ts (one per existing brand,
    # mirroring that brand's universe/tags where a sibling algorithm exists),
    # and the TypeScript-side wiring in ADDING_NEW_ALGORITHM.md
    # (algorithmFamilyRegistry.ts, botIdentity.ts BotType union,
    # officialBotRegistry.ts, officialBotAdmin.ts) is done. Still
    # brokerage_eligible=False and user_creatable=False -- official-bot-only,
    # paper trading only, same as draco/echo1.
    AlgorithmEntry(
        slug="oracle",
        display_name="Oracle",
        alias="Oracle",
        worker_minute_offset=4,
        fallback_bot_ids=[],
        backfill_enabled=False,
        live_enabled=True,
        brokerage_eligible=False,
        user_creatable=False,
        status="active",
    ),
    # ---------------------------------------------------------------------------
    # tide — long-hold Force variant: composite Sharpe/return/low-vol score
    # (not raw trailing return), equal-weighted top-N, signal-driven rebalance
    # (trades only when its holdings set changes, not on a fixed schedule),
    # plus a 20% per-position stop-loss and a 25% portfolio drawdown circuit
    # breaker (21-trading-day cooldown). See bots/tide.py and the research
    # write-up at /labs/tide for the full methodology and backtest evidence.
    # ---------------------------------------------------------------------------
    # Promoted 2026-10-20: 12 official Tide variations (<bot>_tide for
    # annona, bellator, bellitus, cyclus, imperium, medicus, metalli, regina,
    # thesauro, vectura, vis, vitalis), each copying its Force variation's
    # universe/cash equivalent/top_n from officialBotRegistry.ts. Enrolled via
    # NextJS_Monsta/scripts/sync-tide-official-bots.ts -- that script MUST be
    # run (and trading.tide / trading.official counts verified) before this
    # entry is deployed, see ADDING_NEW_ALGORITHM.md §10. Live runner is
    # registered in algorithm_plugins.py; backfill dispatcher in
    # process_backfill_jobs.py. Official-only (user_creatable=False) and
    # paper-only (brokerage_eligible=False), same posture as Oracle. Tide's
    # universe is the fixed trading.tide.universe list -- its fit scores
    # (trading.tide_fitness) are still standalone/offline, not wired into the
    # weekly adaptive-universe refresh cron that feeds Force/Aptet/Draco.
    AlgorithmEntry(
        slug="tide",
        display_name="Tide",
        alias="Tide",
        worker_minute_offset=9,
        fallback_bot_ids=[],
        backfill_enabled=True,
        live_enabled=True,
        brokerage_eligible=False,
        user_creatable=False,
        status="active",
    ),
    AlgorithmEntry(
        slug="draco",
        display_name="Draco",
        alias="Draco",
        worker_minute_offset=7,
        fallback_bot_ids=["vectura"],
        backfill_enabled=True,
        live_enabled=True,
        brokerage_eligible=False,
        user_creatable=True,
        status="active",
        # Pilot for the per-algorithm repo effort: fully open tier, chosen
        # specifically to validate the open path (real strategy source, not
        # a thin runner) since Draco is lower-stakes than Force/Aptet.
        # repo_url stays None until the actual monstra-draco repo is
        # published; it also depends on monstra-vectors-core existing (see
        # vector_universe.py's BOT_FEATURE_KEYS["draco"]) for a fork to
        # actually run end-to-end, not just be readable.
        repo_tier="open",
        repo_status="planned",
        repo_url=None,
        repo_python_version="3.12+",
        # Provider-agnostic on purpose: these algorithms ask Monstra for
        # price history (market.history(symbol, period=...)), not any
        # specific vendor's API, so they stay portable to whichever data
        # source a forker already has -- Alpaca, yfinance, IBKR, or anything
        # else.
        repo_dependencies=_ADAPTIVE_UNIVERSE_REPO_DEPENDENCIES,
        repo_bring_your_own=_STANDARD_REPO_BRING_YOUR_OWN,
    ),
    # ---------------------------------------------------------------------------
    # haven -- fixed-tag downside-risk MIN-score ranking (relative strength +
    # downside aware, never momentum/aggression), scoped per bot by
    # sectors/industries borrowed from already-enrolled bot identities
    # ---------------------------------------------------------------------------
    # Ranks candidates by strategy_engine.bibo_tag_scoring.compute_tag_score
    # (MIN of "relative strength" + "downside aware", fixed for every Haven
    # bot -- see strategy_engine.haven.HAVEN_TAGS) instead of Force's
    # trailing price return or the original 5-weight composite design
    # (strategy_engine.haven_scoring, now only used by build_algo_universe.
    # py's display/labeling sleeves, not the live engine). Reuses Force's
    # tiered rank_weights / incumbency margin mechanics -- same relationship
    # Bibo has to Force.
    #
    # Promoted 2026-10-24: 7 official Haven variations (annona, bellator,
    # cyclus, medicus, thesauro, vectura, vis), each reusing its existing
    # (sectors, industries) scope from trading.adaptive_universe -- single
    # sector, multi-sector, industry-only, or fully open/"sector ignorant"
    # (cyclus). metalli explicitly excluded: its backtest showed -19.6%
    # return / 19.6% max drawdown in scripts/backtest_haven_live_bot_scopes.py,
    # the worst of the 8 tested scopes by a wide margin. Verified non-zero
    # via trading.haven / trading.official (7 rows each, alg_name='haven')
    # per ADDING_NEW_ALGORITHM.md Β§10.
    #
    # KNOWN GAP at promotion time: the NextJS_Monsta side (BotType union in
    # botIdentity.ts, algorithmFamilyRegistry.ts, officialBotRegistry.ts,
    # create-flow pages, refresh_monstra_feed.py's UPSERT_SQL UNION ALL) has
    # NOT been updated. run_all_bots()/get_live_algorithms() are fully
    # registry-driven and do not depend on any of that -- these bots will
    # generate real signals and paper-trade correctly via the same
    # run_paper_trading_safe() path every other algorithm uses -- but they
    # may not render correctly (or at all) in the app UI, and won't appear
    # in trading.monstra_feed, until that side is done.
    AlgorithmEntry(
        slug="haven",
        display_name="Haven",
        alias="Haven",
        worker_minute_offset=10,
        fallback_bot_ids=["annona", "bellator", "cyclus", "medicus", "thesauro", "vectura", "vis", "rapax"],
        backfill_enabled=False,  # no backfill_haven.py yet
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
for _entry in ALGORITHM_REGISTRY:
    for _dep_key in _entry.repo_dependencies:
        assert _dep_key in _DEPENDENCY_BY_KEY, \
            f"{_entry.slug}: unknown repo_dependencies key {_dep_key!r}, add it to DEPENDENCY_REGISTRY first"
del _entry, _dep_key


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
# Each entry maps a slug â†’ a callable(bot_id: str) -> dict that runs the
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
    # Backfill jobs are keyed by bot_type, which for Force is still its legacy
    # table name "alpha1" (registry slug "force", db_table "alpha1"), so accept
    # a db_table name as well as a slug.
    known = set(_REGISTRY_BY_SLUG) | {e.db_table for e in ALGORITHM_REGISTRY}
    if slug_normalized not in known:
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




