"""
tests/test_memory.py
====================
Tests for the FAISS-indexed episode memory and strict pre-cutoff retrieval.

Verifies:
1. Feature extraction: magnitude, pre-event trend, seasonality, multipliers at +7, +14, +28 days.
2. Embedding and semantic similarity search with sentence-transformers (all-MiniLM-L6-v2).
3. CRITICAL: Zero data leakage date filter enforcement during retrieval:
   - Episodes occurring on or after query cutoff MUST NEVER be returned.
   - Out-of-bounds queries return an empty list without error.
4. Save / reload persistence of FAISS index and metadata.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.leakage_guard import DataLeakageError
from src.memory import (
    EMBEDDING_DIM,
    EpisodeMemory,
    HistoricalEpisode,
    compute_episode_features,
)


def _make_dummy_daily_series(
    start_date: str = "2011-01-01",
    n_days: int = 400,
    base_sales: float = 10.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a dummy time series spanning over a year."""
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    rng = np.random.default_rng(seed)
    sales = rng.poisson(lam=base_sales, size=n_days).astype(float)
    prices = np.full(n_days, 5.0, dtype=float)

    return pd.DataFrame({
        "date": dates,
        "item_id": "FOODS_1_001",
        "store_id": "CA_1",
        "series_id": "FOODS_1_001_CA_1",
        "sales": sales,
        "sell_price": prices,
    })


class TestFeatureExtraction:
    """Tests for compute_episode_features."""

    def test_multipliers_and_trend_calculated(self) -> None:
        df = _make_dummy_daily_series(n_days=100)
        cutoff = pd.Timestamp("2011-04-01")
        pre = df[df["date"] < cutoff]
        actuals = np.full(28, 15.0)  # +50% higher than base_sales=10

        feats = compute_episode_features(
            series_pre_cutoff=pre,
            actuals=actuals,
            cutoff_date=cutoff,
            scenario_type="demand_surge",
            meta={"pct_change": 20.0},
        )

        assert feats["magnitude"] == 20.0
        assert "weekend_ratio" in feats["seasonality"]
        assert "peak_dow" in feats["seasonality"]
        # Multipliers should be around 1.5 since actuals (15) is 1.5x baseline (10)
        assert 1.3 <= feats["multiplier_7"] <= 1.7
        assert 1.3 <= feats["multiplier_14"] <= 1.7
        assert 1.3 <= feats["multiplier_28"] <= 1.7

    def test_leakage_in_feature_extraction_raises(self) -> None:
        df = _make_dummy_daily_series(n_days=100)
        # Deliberately include future rows
        leaky_cutoff = pd.Timestamp("2011-02-01")
        actuals = np.zeros(28)

        with pytest.raises(DataLeakageError):
            compute_episode_features(
                series_pre_cutoff=df,  # contains dates up to April!
                actuals=actuals,
                cutoff_date=leaky_cutoff,
                scenario_type="price_up",
                meta={},
            )


class TestEpisodeMemoryRetrieval:
    """Tests for FAISS indexing and semantic retrieval."""

    @pytest.fixture
    def populated_memory(self) -> EpisodeMemory:
        """Create and populate an EpisodeMemory with 3 typed episodes across time."""
        mem = EpisodeMemory()

        ep1 = HistoricalEpisode(
            episode_id="ep_1",
            series_id="FOODS_1_001_CA_1",
            scenario_type="price_up",
            magnitude=10.0,
            pre_event_trend=0.01,
            seasonality={"weekend_ratio": 1.2, "peak_dow": 6},
            multiplier_7=0.91,
            multiplier_14=0.90,
            multiplier_28=0.89,
            scenario_text="What-if scenario: price increase for FOODS_1_001 at CA_1. Sell price rose by 10%.",
            cutoff_date=pd.Timestamp("2011-06-01"),
            end_date=pd.Timestamp("2011-06-29"),
        )
        ep2 = HistoricalEpisode(
            episode_id="ep_2",
            series_id="FOODS_1_001_CA_1",
            scenario_type="demand_surge",
            magnitude=20.0,
            pre_event_trend=0.05,
            seasonality={"weekend_ratio": 1.1, "peak_dow": 5},
            multiplier_7=1.22,
            multiplier_14=1.20,
            multiplier_28=1.18,
            scenario_text="What-if scenario: unexpected demand surge for FOODS_1_001 at CA_1 with 20% sales spike.",
            cutoff_date=pd.Timestamp("2011-09-01"),
            end_date=pd.Timestamp("2011-09-29"),
        )
        ep3 = HistoricalEpisode(
            episode_id="ep_3",
            series_id="FOODS_1_001_CA_1",
            scenario_type="supply_delay",
            magnitude=4.0,
            pre_event_trend=-0.02,
            seasonality={"weekend_ratio": 1.0, "peak_dow": 4},
            multiplier_7=0.40,
            multiplier_14=0.70,
            multiplier_28=0.85,
            scenario_text="What-if scenario: supply disruption and stockout for FOODS_1_001 with 4 zero-sales days.",
            cutoff_date=pd.Timestamp("2012-02-01"),
            end_date=pd.Timestamp("2012-02-29"),
        )

        texts = [ep1.scenario_text, ep2.scenario_text, ep3.scenario_text]
        embs = mem.embedder.encode(texts, convert_to_numpy=True)

        for ep, vec in zip([ep1, ep2, ep3], embs):
            mem.add_episode(ep, vec)

        return mem

    def test_semantic_match_ranking(self, populated_memory: EpisodeMemory) -> None:
        """A query about price rises should retrieve the price_up episode as top match."""
        # Query late enough so all episodes are in the past
        query_cutoff = pd.Timestamp("2013-01-01")
        results = populated_memory.retrieve(
            query_scenario="Price increase of 10% on food items",
            cutoff_date=query_cutoff,
            k=3,
        )

        assert len(results) == 3
        assert results[0].scenario_type == "price_up"
        assert results[0].similarity_score is not None
        assert results[0].similarity_score > results[1].similarity_score

    def test_strict_date_filter_enforcement(self, populated_memory: EpisodeMemory) -> None:
        """
        CRITICAL TEST: When query cutoff is 2011-10-01, ep_3 (from 2012-02-01)
        MUST NOT be returned, even if requested k=3!
        """
        query_cutoff = pd.Timestamp("2011-10-01")
        results = populated_memory.retrieve(
            query_scenario="Supply disruption or stockout",
            cutoff_date=query_cutoff,
            k=5,
        )

        # Only ep_1 (Jun 2011) and ep_2 (Sep 2011) completed before Oct 2011
        assert len(results) == 2
        returned_ids = {r.episode_id for r in results}
        assert "ep_3" not in returned_ids
        for r in results:
            assert r.end_date <= query_cutoff
            assert r.cutoff_date < query_cutoff

    def test_cutoff_prior_to_all_episodes_returns_empty(self, populated_memory: EpisodeMemory) -> None:
        """Query with a cutoff earlier than the first episode returns an empty list."""
        early_cutoff = pd.Timestamp("2010-01-01")
        results = populated_memory.retrieve(
            query_scenario="Price increase",
            cutoff_date=early_cutoff,
            k=5,
        )
        assert results == []

    def test_save_and_load_persistence(
        self, populated_memory: EpisodeMemory, tmp_path: Path
    ) -> None:
        """Memory index and metadata must persist and reload faithfully."""
        save_dir = tmp_path / "memory"
        populated_memory.save(save_dir)

        loaded_mem = EpisodeMemory()
        loaded_mem.load(save_dir)

        assert len(loaded_mem) == len(populated_memory)
        res = loaded_mem.retrieve(
            query_scenario="demand surge spike",
            cutoff_date="2013-01-01",
            k=1,
        )
        assert len(res) == 1
        assert res[0].scenario_type == "demand_surge"
