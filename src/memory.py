"""
src/memory.py
=============
Historical episode memory using FAISS vector indexing and sentence-transformers
(all-MiniLM-L6-v2).

Enforces strict pre-cutoff filtering to guarantee zero data leakage during
what-if scenario retrieval.

Memory schema per episode:
--------------------------
* scenario_type: str ('price_up', 'demand_surge', 'supply_delay')
* magnitude: float (e.g. % change or stockout duration)
* pre_event_trend: float (linear regression slope of pre-event sales normalized)
* seasonality: dict (day-of-week profile, weekend vs weekday ratio)
* multiplier_7: float (mean sales in first 7 days / baseline)
* multiplier_14: float (mean sales in first 14 days / baseline)
* multiplier_28: float (mean sales over full 28 days / baseline)
* scenario_text: str (plain-language description)
* cutoff_date: pd.Timestamp (start of episode)
* end_date: pd.Timestamp (cutoff_date + 28 days)

API:
----
>>> memory = EpisodeMemory()
>>> memory.build_from_episodes(episodes_df, daily_df)
>>> results = memory.retrieve(query_scenario="10% price bump on dairy",
...                           cutoff_date="2012-06-01",
...                           k=3)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.leakage_guard import assert_no_leakage, filter_before_cutoff

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
HORIZON_DAYS = 28


# ═══════════════════════════════════════════════════════════════════════════════
# Historical Episode Data Structure
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class HistoricalEpisode:
    """
    Structured record of a past what-if episode stored in memory.

    Attributes
    ----------
    episode_id:
        Unique identifier.
    series_id:
        Time-series identifier (item_id_store_id).
    scenario_type:
        One of 'price_up', 'demand_surge', 'supply_delay'.
    magnitude:
        Quantitative scale of the intervention (% price rise, % demand lift, or zero days).
    pre_event_trend:
        Normalized daily slope over the 28 days prior to the cutoff.
    seasonality:
        Seasonality summary (e.g. weekend_ratio, peak_dow).
    multiplier_7:
        Realized outcome multiplier over the first 7 days (mean actuals[:7] / baseline).
    multiplier_14:
        Realized outcome multiplier over the first 14 days (mean actuals[:14] / baseline).
    multiplier_28:
        Realized outcome multiplier over the full 28 days (mean actuals[:28] / baseline).
    scenario_text:
        Plain-language scenario description.
    cutoff_date:
        Start date of the episode.
    end_date:
        End date of the 28-day outcome window (cutoff_date + 28 days).
    meta:
        Additional episode metadata.
    similarity_score:
        Cosine similarity score populated during retrieval.
    """

    episode_id: str
    series_id: str
    scenario_type: str
    magnitude: float
    pre_event_trend: float
    seasonality: Dict[str, float]
    multiplier_7: float
    multiplier_14: float
    multiplier_28: float
    scenario_text: str
    cutoff_date: pd.Timestamp
    end_date: pd.Timestamp
    meta: Dict[str, Any] = field(default_factory=dict)
    similarity_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert episode to a serializable dictionary."""
        d = asdict(self)
        d["cutoff_date"] = str(self.cutoff_date.date())
        d["end_date"] = str(self.end_date.date())
        return d


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Extraction from Pre-Cutoff Data
# ═══════════════════════════════════════════════════════════════════════════════


def compute_episode_features(
    series_pre_cutoff: pd.DataFrame,
    actuals: np.ndarray,
    cutoff_date: pd.Timestamp,
    scenario_type: str,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract trend, seasonality, magnitude, and outcome multipliers from historical
    and outcome data with zero leakage.

    Parameters
    ----------
    series_pre_cutoff:
        Daily series dated strictly prior to cutoff_date.
    actuals:
        Ground truth actual sales array for the 28-day window following cutoff.
    cutoff_date:
        Episode cutoff date.
    scenario_type:
        Episode scenario type.
    meta:
        Episode detector metadata dictionary.

    Returns
    -------
    dict with magnitude, pre_event_trend, seasonality, multiplier_7, multiplier_14, multiplier_28.
    """
    # 1. Leakage guard assertion
    assert_no_leakage(series_pre_cutoff, date_col="date", cutoff=cutoff_date)

    sales = series_pre_cutoff["sales"].to_numpy(dtype=float)
    dates = pd.to_datetime(series_pre_cutoff["date"])

    # Baseline demand: mean of the prior 28 days
    lookback = min(28, len(sales))
    pre_window = sales[-lookback:] if lookback > 0 else np.array([1.0])
    baseline_mean = float(np.mean(pre_window)) if len(pre_window) > 0 else 1.0
    baseline_mean = max(baseline_mean, 1e-4)

    # Magnitude
    if scenario_type == "price_up":
        magnitude = float(meta.get("pct_change", 10.0))
    elif scenario_type == "demand_surge":
        magnitude = float(meta.get("pct_change", 20.0))
    elif scenario_type == "supply_delay":
        magnitude = float(meta.get("stockout_days", 3.0))
    else:
        magnitude = float(meta.get("pct_change", 0.0))

    # Pre-event trend: normalized linear slope over prior 28 days
    if len(pre_window) >= 7:
        x = np.arange(len(pre_window), dtype=float)
        # y = alpha + beta * x
        x_mean = np.mean(x)
        y_mean = np.mean(pre_window)
        denom = float(np.sum((x - x_mean) ** 2))
        beta = float(np.sum((x - x_mean) * (pre_window - y_mean)) / denom) if denom > 0 else 0.0
        normalized_trend = float(beta / baseline_mean)
    else:
        normalized_trend = 0.0

    # Seasonality: weekend vs weekday ratio and peak day-of-week
    if len(series_pre_cutoff) >= 14:
        dow = dates.dt.dayofweek
        dow_means = [float(np.mean(sales[dow == d])) if np.any(dow == d) else 0.0 for d in range(7)]
        weekday_mean = float(np.mean(dow_means[:5]))
        weekend_mean = float(np.mean(dow_means[5:]))
        weekend_ratio = float(weekend_mean / max(weekday_mean, 1e-4))
        peak_dow = int(np.argmax(dow_means))
    else:
        weekend_ratio = 1.0
        peak_dow = 5

    seasonality = {
        "weekend_ratio": round(weekend_ratio, 3),
        "peak_dow": peak_dow,
    }

    # Outcome multipliers at +7, +14, and +28 days
    actuals = np.asarray(actuals, dtype=float)
    m7 = float(np.mean(actuals[:min(7, len(actuals))]) / baseline_mean)
    m14 = float(np.mean(actuals[:min(14, len(actuals))]) / baseline_mean)
    m28 = float(np.mean(actuals[:min(28, len(actuals))]) / baseline_mean)

    return {
        "magnitude": round(magnitude, 2),
        "pre_event_trend": round(normalized_trend, 4),
        "seasonality": seasonality,
        "multiplier_7": round(m7, 4),
        "multiplier_14": round(m14, 4),
        "multiplier_28": round(m28, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Episode Memory with FAISS Indexing
# ═══════════════════════════════════════════════════════════════════════════════


class EpisodeMemory:
    """
    FAISS-indexed semantic memory of historical what-if episodes.

    Provides semantic nearest-neighbor retrieval strictly filtered to past
    episodes relative to any query's cutoff date.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        embedder: Optional[Any] = None,
    ) -> None:
        self.model_name = model_name
        self._embedder = embedder
        self.episodes: List[HistoricalEpisode] = []
        # Inner Product on normalized vectors = Cosine Similarity
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(EMBEDDING_DIM)

    @property
    def embedder(self) -> SentenceTransformer:
        """Lazy load sentence-transformers model."""
        if self._embedder is None:
            logger.info("Loading SentenceTransformer model '%s'...", self.model_name)
            self._embedder = SentenceTransformer(self.model_name)
        return self._embedder

    def __len__(self) -> int:
        return len(self.episodes)

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2 normalize vectors for cosine similarity search."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vectors / norms).astype(np.float32)

    def add_episode(self, episode: HistoricalEpisode, vector: np.ndarray) -> None:
        """Add a single episode and its pre-computed embedding to the memory."""
        vec = self._normalize(vector.reshape(1, -1))
        self.index.add(vec)
        self.episodes.append(episode)

    def build_from_episodes(
        self,
        episodes_df: pd.DataFrame,
        daily_df: pd.DataFrame,
    ) -> None:
        """
        Build the FAISS index and episode memory records from the detected episodes.

        Parameters
        ----------
        episodes_df:
            Parquet DataFrame containing detected what-if episodes.
        daily_df:
            M5 daily sales DataFrame with columns date, sales, series_id.
        """
        logger.info("Indexing %d episodes into memory...", len(episodes_df))
        series_groups = {sid: grp for sid, grp in daily_df.groupby("series_id")}

        # 1. Extract metadata and compute features for all episodes
        texts: List[str] = []
        valid_episodes: List[HistoricalEpisode] = []

        for idx, row in episodes_df.iterrows():
            sid = row["series_id"]
            stype = row["scenario_type"]
            cutoff = pd.Timestamp(row["cutoff_date"])
            end_date = cutoff + pd.Timedelta(days=HORIZON_DAYS)
            actuals = np.asarray(row["actuals"], dtype=float)
            stext = str(row["scenario_text"])

            # Extract meta_* columns
            meta = {}
            for col in row.index:
                if col.startswith("meta_") and pd.notna(row[col]):
                    meta[col[len("meta_"):]] = row[col]

            if sid not in series_groups:
                continue

            raw_series = series_groups[sid]
            # Strict pre-cutoff slice
            pre_series = filter_before_cutoff(raw_series, date_col="date", cutoff=cutoff)
            if len(pre_series) == 0:
                continue

            feats = compute_episode_features(
                series_pre_cutoff=pre_series,
                actuals=actuals,
                cutoff_date=cutoff,
                scenario_type=stype,
                meta=meta,
            )

            ep = HistoricalEpisode(
                episode_id=str(row.get("episode_id", f"ep_{idx}")),
                series_id=sid,
                scenario_type=stype,
                magnitude=feats["magnitude"],
                pre_event_trend=feats["pre_event_trend"],
                seasonality=feats["seasonality"],
                multiplier_7=feats["multiplier_7"],
                multiplier_14=feats["multiplier_14"],
                multiplier_28=feats["multiplier_28"],
                scenario_text=stext,
                cutoff_date=cutoff,
                end_date=end_date,
                meta=meta,
            )
            valid_episodes.append(ep)
            texts.append(stext)

        # 2. Batch embed text descriptions
        logger.info("Embedding %d episode descriptions with %s...", len(texts), self.model_name)
        embeddings = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        norm_embeddings = self._normalize(embeddings.astype(np.float32))

        # 3. Add to FAISS index
        self.index.reset()
        self.index.add(norm_embeddings)
        self.episodes = valid_episodes
        logger.info("Episode memory built: %d episodes indexed.", len(self.episodes))

    def retrieve(
        self,
        query_scenario: Union[str, Dict[str, Any]],
        cutoff_date: Union[pd.Timestamp, str],
        k: int = 3,
        require_full_outcome: bool = True,
    ) -> List[HistoricalEpisode]:
        """
        Retrieve top-k semantically relevant historical episodes strictly observed
        prior to cutoff_date.

        Parameters
        ----------
        query_scenario:
            Plain text query (or scenario dict with 'scenario_text' or 'name').
        cutoff_date:
            Scenario cutoff date. ONLY episodes observed strictly before this date
            will be returned.
        k:
            Number of nearest episodes to return.
        require_full_outcome:
            If True, enforces that the episode's entire 28-day outcome window
            (end_date) was completed before cutoff_date.
            If False, enforces that the episode's cutoff_date is strictly before cutoff_date.

        Returns
        -------
        List[HistoricalEpisode] of length <= k, sorted by similarity_score descending.
        """
        if len(self.episodes) == 0:
            return []

        # Parse query text
        if isinstance(query_scenario, dict):
            query_text = str(query_scenario.get("scenario_text") or query_scenario.get("description") or query_scenario.get("name", ""))
        else:
            query_text = str(query_scenario)

        query_cutoff = pd.Timestamp(cutoff_date)

        # 1. Embed query
        q_emb = self.embedder.encode([query_text], convert_to_numpy=True)
        q_norm = self._normalize(q_emb.astype(np.float32))

        # 2. Query FAISS index for all candidates
        n_candidates = len(self.episodes)
        distances, indices = self.index.search(q_norm, n_candidates)

        matched: List[HistoricalEpisode] = []

        for sim, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.episodes):
                continue

            candidate = self.episodes[idx]

            # 3. STRICT ZERO-LEAKAGE DATE FILTER
            if require_full_outcome:
                # The full 28-day outcome must have been completed prior to the query cutoff
                if candidate.end_date > query_cutoff:
                    continue
            else:
                # The candidate episode cutoff must be strictly before query cutoff
                if candidate.cutoff_date >= query_cutoff:
                    continue

            # Clone episode and attach similarity score
            ep_copy = HistoricalEpisode(
                episode_id=candidate.episode_id,
                series_id=candidate.series_id,
                scenario_type=candidate.scenario_type,
                magnitude=candidate.magnitude,
                pre_event_trend=candidate.pre_event_trend,
                seasonality=candidate.seasonality,
                multiplier_7=candidate.multiplier_7,
                multiplier_14=candidate.multiplier_14,
                multiplier_28=candidate.multiplier_28,
                scenario_text=candidate.scenario_text,
                cutoff_date=candidate.cutoff_date,
                end_date=candidate.end_date,
                meta=candidate.meta.copy(),
                similarity_score=round(float(sim), 4),
            )
            matched.append(ep_copy)

            if len(matched) >= k:
                break

        return matched

    def save(self, directory: Union[Path, str]) -> None:
        """Save FAISS index and episode metadata to disk."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        faiss_path = dir_path / "episodes.faiss"
        meta_path = dir_path / "episodes_meta.json"

        faiss.write_index(self.index, str(faiss_path))

        meta_list = [ep.to_dict() for ep in self.episodes]
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_list, f, indent=2)

        logger.info("Saved episode memory to %s", dir_path)

    def load(self, directory: Union[Path, str]) -> None:
        """Load FAISS index and episode metadata from disk."""
        dir_path = Path(directory)
        faiss_path = dir_path / "episodes.faiss"
        meta_path = dir_path / "episodes_meta.json"

        if not faiss_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Missing memory files in {dir_path}")

        self.index = faiss.read_index(str(faiss_path))

        with open(meta_path, "r", encoding="utf-8") as f:
            meta_list = json.load(f)

        self.episodes = []
        for d in meta_list:
            ep = HistoricalEpisode(
                episode_id=d["episode_id"],
                series_id=d["series_id"],
                scenario_type=d["scenario_type"],
                magnitude=d["magnitude"],
                pre_event_trend=d["pre_event_trend"],
                seasonality=d["seasonality"],
                multiplier_7=d["multiplier_7"],
                multiplier_14=d["multiplier_14"],
                multiplier_28=d["multiplier_28"],
                scenario_text=d["scenario_text"],
                cutoff_date=pd.Timestamp(d["cutoff_date"]),
                end_date=pd.Timestamp(d["end_date"]),
                meta=d.get("meta", {}),
            )
            self.episodes.append(ep)

        logger.info("Loaded %d episodes from %s", len(self.episodes), dir_path)
