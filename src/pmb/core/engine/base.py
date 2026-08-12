from __future__ import annotations

from pathlib import Path

from pmb.config import Config
from pmb.core.events import (
    EventStore,
)
from pmb.core.sqlite_helper import patch_global_sqlite3

# Patch sqlite3.connect once at engine import - every subsequent
# sqlite3.connect() in PMB code automatically gets busy_timeout +
# WAL pragmas. Without this, concurrent writes from MCP + dashboard +
# seed scripts on the same DB fail with 'database is locked'.
patch_global_sqlite3()
from pmb.core.engine.ambient import AmbientMixin
from pmb.core.engine.batch import BatchMixin
from pmb.core.engine.dedup import DedupMixin
from pmb.core.engine.embed import EmbedMixin
from pmb.core.engine.goals import GoalsMixin
from pmb.core.engine.graph import GraphMixin
from pmb.core.engine.health import HealthMixin
from pmb.core.engine.lessons import LessonsMixin
from pmb.core.engine.overview import OverviewMixin
from pmb.core.engine.reasoning import ReasoningMixin
from pmb.core.engine.recall import RecallMixin
from pmb.core.engine.write import WriteMixin
from pmb.core.recall_cache import RecallCache
from pmb.core.search import HybridSearch
from pmb.core.workspace import Workspace, detect_workspace
from pmb.graph.entities import make_extractor
from pmb.graph.store import GraphStore
from pmb.reasoning.pamvr import (
    VOCAB_BRIDGES as _PAMVR_DEFAULT_BRIDGES,
)
from pmb.reasoning.vocab_miner import (
    merge_bridges as _merge_vocab_bridges,
)
from pmb.reasoning.vocab_miner import (
    mine_workspace as _mine_workspace_bridges,
)
from pmb.signals.session import SessionTracker


class Engine(
    WriteMixin,
    LessonsMixin,
    OverviewMixin,
    BatchMixin,
    GoalsMixin,
    DedupMixin,
    EmbedMixin,
    GraphMixin,
    RecallMixin,
    ReasoningMixin,
    HealthMixin,
    AmbientMixin,
):
    """
    Main orchestrator. Holds the workspace + storage + search index.

    Created per-workspace. On creation it auto-detects the workspace and
    initializes storage if it's the first time.
    """

    def __init__(
        self,
        workspace: Workspace | None = None,
        cwd: Path | None = None,
        pmb_home: Path | None = None,
        embedding_model: str | None = None,
        bm25_weight: float | None = None,
        rerank_model: str | None = None,
        config_overrides: dict | None = None,
    ):
        self.workspace = workspace or detect_workspace(cwd=cwd, pmb_home=pmb_home)
        self.workspace.ensure_dirs()
        self.workspace.save_meta()

        # Layered config (workspace > global > defaults). Explicit kwargs
        # become overrides so legacy callers keep working unchanged.
        overrides: dict = dict(config_overrides or {})
        if embedding_model is not None:
            overrides["embedding.model"] = embedding_model
        if bm25_weight is not None:
            overrides["recall.bm25_weight"] = bm25_weight
        # rerank_model truthy enables reranker through config
        if rerank_model:
            overrides["recall.rerank_model"] = rerank_model
            overrides["recall.rerank"] = True
        self.config = Config(
            workspace_dir=self.workspace.storage_dir,
            pmb_home=self.workspace.pmb_home,
            overrides=overrides,
        )

        self.events = EventStore(self.workspace.db_path)
        # Pick the right model id depending on backend (fastembed uses the
        # same canonical names but advertises its catalogue separately).
        emb_backend = self.config.get("embedding.backend")
        if emb_backend == "fastembed":
            emb_model = self.config.get("embedding.fastembed_model")
        elif emb_backend == "ollama":
            emb_model = self.config.get("embedding.ollama_model")
        elif emb_backend == "openai":
            emb_model = self.config.get("embedding.openai_model")
        else:
            emb_model = self.config.get("embedding.model")
        emb_base_url = self.config.get("embedding.ollama_url")
        # Improvement #1: reranker is needed if EITHER the always-on
        # `recall.rerank` flag is True OR the gated `recall.rerank_when_close`
        # is True. We pass the model name to HybridSearch in either case;
        # the rerank logic in `recall()` decides per-query whether to fire.
        _rerank_needed = self.config.get("recall.rerank") or self.config.get(
            "recall.rerank_when_close"
        )
        self.search = HybridSearch(
            vector_path=self.workspace.vector_path,
            model_name=emb_model,
            embedding_backend=emb_backend,
            embedding_base_url=emb_base_url,
            bm25_weight=self.config.get("recall.bm25_weight"),
            rerank_model_name=(self.config.get("recall.rerank_model") if _rerank_needed else None),
        )
        self.session_tracker = SessionTracker(self.workspace)
        # Graph layer: same DB file, extra tables (migration v2)
        self.graph = GraphStore(self.workspace.db_path)
        # Pluggable extractor - `graph.extractor` config picks the backend.
        # Defaults to fast regex; users can switch to "spacy" or "llm:claude"
        # / "llm:ollama" / "llm:codex" without recall code changes.
        self.entity_extractor = make_extractor(self.config)
        # Cache populated by record_batch's pre-extraction step. Maps the
        # CLEANED content text → ExtractedEntities. _index_event_in_graph
        # consults it FIRST so when an LLM backend is configured, a batch of
        # N events pays one CLI call total instead of N. Cleared after each
        # record_batch. Empty (and ignored) for non-batch writes.
        self._extract_cache: dict = {}
        # Recall LRU cache - "working memory" buffer in front of the hybrid
        # pipeline. Invalidated on every event write.
        self.recall_cache = RecallCache(
            max_size=self.config.get("recall.cache_size"),
            ttl_seconds=self.config.get("recall.cache_ttl_seconds"),
        )

        # PPR graph cache (HippoRAG-style multi-hop boost). Built lazily on
        # first recall that needs it; invalidated when new events/edges land.
        self._ppr_graph = None
        self._ppr_graph_generation = -1

        # BM25 reload is deferred to the first `self.search.search()` /
        # `.add()` call: the load takes ~50-200 ms on populated workspaces
        # and ~22 s on the very first call in a fresh process (because of
        # the underlying `import lancedb`). Engine() must not pay that cost
        # for read-only CLI commands that never touch search.

        # Improvement W: deferred-embed queue. Writes don't block on the
        # sentence-transformers model load (~50s cold start). Instead we
        # write SQLite immediately and let a background worker embed
        # whenever the model becomes ready.
        self._embed_queue: list[tuple[str, str]] = []
        self._embed_worker_started = False
        self._embed_queue_lock = None  # lazy threading.Lock

        # Hardening H3: durable embed queue (SQLite-backed) for crash-
        # safety. Pending embeds survive process restart. Inits lazily
        # so the read-only CLI commands don't pay the cost.
        self._durable_embed_queue = None  # set by _ensure_durable_queue()

        # Improvement CC: serialize record_batch entries - when multiple
        # async batches run concurrently, they race on self._batch_defer
        # and self._batch_pending. A Lock makes batches process one at a
        # time inside the engine; the MCP caller still returns instantly
        # because each record_batch_async spawns its own thread.
        import threading as _threading

        self._batch_lock = _threading.Lock()

        # In-flight async write counter - lets callers (CLI scripts,
        # tests, fixtures) block until daemon threads actually finish.
        # Without this, `record_batch_async` spawns a daemon thread that
        # gets killed on process exit, silently dropping writes.
        self._async_writes_in_flight = 0
        self._async_writes_cv = _threading.Condition(_threading.Lock())

        # D4: recall singleflight - collapse concurrent identical top-level
        # recalls (one Event per in-flight key; result carried on the Event).
        self._sf_lock = _threading.Lock()
        self._sf_inflight: dict = {}

        # B4: durable write outbox. record_batch_async enqueues a row into the
        # write_outbox SQLite table synchronously, then a single lazy drainer
        # thread replays pendings - so a crash between accept and write loses
        # nothing. State here; logic in BatchMixin. Lazy: a read-only CLI that
        # never calls record_batch_async starts no thread.
        self._outbox_lock = _threading.Lock()
        self._outbox_wake = _threading.Event()
        self._outbox_thread = None
        self._outbox_recovered = False

        # Improvement #5: bulk-import mode. When True, every per-item
        # cross-cutting step (dedup L1+L2, graph indexing, temporal
        # parsing, causation edges, L2.5 queue) is skipped. Only the
        # SQLite row + embedding land. Caller runs `pmb regraph` later.
        # Default False so normal record_batch behaviour is unchanged.
        self._bulk_mode = False
        self._bulk_collected_ulids: list[str] = []

        # Improvement #6: deferred touch buffer. Under concurrent recalls
        # each call writes access_count+1 / last_accessed / importance via
        # `apply_recall_updates` - a SQLite write transaction. With 8+
        # parallel recalls these all queue on the write lock, exploding p95.
        # Solution: enqueue touches in memory, flush every ~250ms in a
        # daemon thread. Single flush coalesces touches from many recalls,
        # so 16 concurrent recalls = 1 lock acquisition instead of 16.
        # On engine.close() we drain the buffer.
        self._touch_buffer: dict[str, float] = {}  # ulid -> last_accessed
        self._touch_imp_buffer: dict[str, float] = {}  # ulid -> latest importance
        self._touch_tier_buffer: dict[str, str] = {}   # S9: ulid -> new tier
        self._touch_lock = _threading.Lock()
        self._touch_flusher_started = False

        # PAMVR - Predicate-Aware Multi-View Reranking. Cache the flag
        # at init time so the recall hot-path doesn't pay for a config
        # lookup per candidate.
        self._pamvr_enabled = bool(self.config.get("recall.pamvr_enabled"))
        # X2: per-workspace base-score channel weights (default identity →
        # byte-identical recall). Cached once so the hot path pays no parse.
        from pmb.reasoning import scoring as _scoring
        self._channel_weights = _scoring.channel_weights(self.config)
        # v0.9 SAE: built lazily on first WARM use (never loads the model on the
        # cold path); None until then.
        self._anchor_index = None
        # Query-worthiness classifier (SAE pattern, new axis) - lazy + warm-only.
        self._query_worthiness = None

        # v0.9 E1: apply this workspace's cached corpus-derived stopwords to the
        # live tokenizer (gated; empty cache / off → no change). The tick
        # refreshes the cache; here we just load what the last tick computed.
        try:
            if self.config.get("lang.corpus_stopwords"):
                from pmb.core.text_match import apply_corpus_stopwords
                from pmb.maintenance.corpus_stopwords import load_corpus_stopwords
                apply_corpus_stopwords(load_corpus_stopwords(self))
        except Exception:
            pass

        # Auto VOCAB_BRIDGES (Improvement TT). Mine the user's own lexicon
        # from workspace events via PMI co-occurrence so PAMVR adapts to
        # any domain (not just coding). Hand-curated VOCAB_BRIDGES stay as
        # fallback; mined bridges extend them. Refreshed lazily - see
        # `_maybe_refresh_vocab_bridges()`.
        self._auto_bridges_enabled = bool(self.config.get("recall.auto_vocab_bridges"))
        # Improvement WW: write-time atomic fact extraction.
        self._atomic_extract_enabled = bool(self.config.get("write.atomic_fact_extract"))
        self._vocab_bridges: dict[str, list[str]] = dict(_PAMVR_DEFAULT_BRIDGES)
        self._vocab_bridges_cache_path = self.workspace.storage_dir / "vocab_bridges.json"
        self._vocab_bridges_last_event_count = -1
        if self._auto_bridges_enabled:
            try:
                self._refresh_vocab_bridges()
            except Exception:
                # Auto-mining is a best-effort enhancement, never crash init.
                pass

    def _refresh_vocab_bridges(self, force: bool = False) -> None:
        """Mine workspace events → merge with hand-curated bridges.

        Called at init and lazily before recall when enough new events have
        landed. The mining itself is ~50 ms per 1000 events.
        """
        mined = _mine_workspace_bridges(
            db_path=self.workspace.db_path,
            cache_path=self._vocab_bridges_cache_path,
            force=force,
            window=int(self.config.get("recall.auto_vocab_window") or 6),
            min_count=int(self.config.get("recall.auto_vocab_min_count") or 3),
            min_pmi=float(self.config.get("recall.auto_vocab_min_pmi") or 2.0),
            max_bridges_per_key=int(self.config.get("recall.auto_vocab_max_per_key") or 8),
            refresh_threshold=int(self.config.get("recall.auto_vocab_refresh_after") or 50),
        )
        merged = _merge_vocab_bridges(_PAMVR_DEFAULT_BRIDGES, mined)
        self._vocab_bridges = merged
        try:
            # Best-effort: track event count so we know when to re-mine.
            import sqlite3 as _sql

            with _sql.connect(str(self.workspace.db_path)) as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM events WHERE archived_at IS NULL"
                ).fetchone()
                self._vocab_bridges_last_event_count = int(row[0] or 0)
        except Exception:
            pass

    def _maybe_refresh_vocab_bridges(self) -> None:
        """Called from recall() - cheap if cache is fresh, mines if stale."""
        if not self._auto_bridges_enabled:
            return
        try:
            import sqlite3 as _sql

            with _sql.connect(str(self.workspace.db_path)) as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM events WHERE archived_at IS NULL"
                ).fetchone()
                n = int(row[0] or 0)
            threshold = int(self.config.get("recall.auto_vocab_refresh_after") or 50)
            if n - self._vocab_bridges_last_event_count >= threshold:
                self._refresh_vocab_bridges()
        except Exception:
            pass

    def stats(self) -> dict:
        sess = self.session_tracker.current(auto_create=False)
        return {
            "workspace": {
                "id": self.workspace.id,
                "name": self.workspace.name,
                "root": str(self.workspace.root),
                "source": self.workspace.source,
                "created_at": self.workspace.created_at,
            },
            "events": self.events.stats(self.workspace.id),
            "search_index_size": self.search.size(),
            "current_session": sess.to_dict() if sess else None,
        }

    def sync_git(
        self,
        since_timestamp: float | None = None,
        max_commits: int | None = None,
    ) -> dict:
        """Capture git commits into memory. Imported lazily.

        `max_commits` caps a single walk; ``None`` uses the signal default and
        ``0`` means uncapped.
        """
        from pmb.signals.git import GitSync

        return GitSync(self).sync(
            since_timestamp=since_timestamp, max_commits=max_commits,
        )

    def session_start(self, name: str | None = None) -> dict:
        return self.session_tracker.start(name).to_dict()

    def session_end(self) -> dict | None:
        sess = self.session_tracker.end()
        out = sess.to_dict() if sess else None
        # Zero-command auto-distill: if enabled, distill durable lessons from
        # the session that just ended. Off the recall path; never crashes end.
        if out and self.config.get("lessons.auto_distill_on_session_end"):
            try:
                d = self.distill_lessons(session_id=out.get("id"))
                out["distilled_lessons"] = d.get("n_recorded", 0)
            except Exception:
                pass
        return out

    def session_current(self) -> dict | None:
        sess = self.session_tracker.current(auto_create=False)
        return sess.to_dict() if sess else None

    def close(self):
        """Close open resources (LanceDB, native handles).

        Drain the three async side-channels first, in dependency order, so a
        process that exits right after close() doesn't silently drop in-flight
        work. Each wait is bounded, so close() can't hang on a stuck worker:

          1. async batch writes  - record_batch_async daemon threads; these
             may themselves enqueue graph + touch work, so drain them first.
          2. deferred graph index - the graph.async_llm background worker.
          3. touch buffer         - access_count / importance reinforcement.

        Each drain returns immediately when nothing is pending, so the common
        close() stays ~instant.
        """
        try:
            self.wait_for_writes(timeout=30.0)
        except Exception:
            pass
        try:
            self.wait_for_graph_queue(timeout_seconds=30.0)
        except Exception:
            pass
        try:
            self._drain_touch_buffer()
        except Exception:
            pass
        try:
            self.search._lance = None
            self.search._table_obj = None
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
