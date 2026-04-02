"""Tests for the consolidated QA type system."""

from __future__ import annotations

from cgft.qa_generation.cgft_models import (
    CgftPipelineConfig,
    GenerationTask,
    PlatformConfig,
    TargetsConfig,
    build_generation_tasks,
)
from cgft.qa_generation.corpus_capabilities import CorpusCapabilities


class TestGenerationTaskReasoningMode:
    def test_reasoning_mode_field(self):
        task = GenerationTask(
            task_id="t1",
            qa_type="multi_hop",
            target_hop_count=2,
            seed_chunk_id="c1",
            reasoning_mode="factual",
        )
        assert task.reasoning_mode == "factual"


class TestBuildConsolidatedTasks:
    def _make_cfg(self, **kwargs) -> CgftPipelineConfig:
        targets_kwargs = {
            "total_samples": 100,
            "primary_type_distribution": {"lookup": 0.30, "multi_hop": 0.70},
            "reasoning_mode_distribution": {
                "factual": 0.50,
                "inference": 0.30,
                "sequential": 0.20,
            },
            "hop_distribution": {2: 0.60, 3: 0.40},
        }
        targets_kwargs.update(kwargs)
        return CgftPipelineConfig(
            platform=PlatformConfig(api_key="test"),
            targets=TargetsConfig(**targets_kwargs),
        )

    def test_allocates_correct_totals(self):
        cfg = self._make_cfg()
        tasks = build_generation_tasks(cfg, seed_chunk_ids=[f"c{i}" for i in range(20)])
        assert len(tasks) == 100

        lookup_count = sum(1 for t in tasks if t.qa_type == "lookup")
        multi_hop_count = sum(1 for t in tasks if t.qa_type == "multi_hop")
        assert lookup_count == 30
        assert multi_hop_count == 70

    def test_all_lookup_have_hop_1(self):
        cfg = self._make_cfg()
        tasks = build_generation_tasks(cfg, seed_chunk_ids=[f"c{i}" for i in range(20)])
        for t in tasks:
            if t.qa_type == "lookup":
                assert t.target_hop_count == 1
                assert t.reasoning_mode == ""

    def test_multi_hop_have_reasoning_modes(self):
        cfg = self._make_cfg()
        tasks = build_generation_tasks(cfg, seed_chunk_ids=[f"c{i}" for i in range(20)])
        multi_hop = [t for t in tasks if t.qa_type == "multi_hop"]
        modes = {t.reasoning_mode for t in multi_hop}
        assert "factual" in modes
        assert "inference" in modes
        assert "sequential" in modes

    def test_reasoning_mode_distribution(self):
        cfg = self._make_cfg()
        tasks = build_generation_tasks(cfg, seed_chunk_ids=[f"c{i}" for i in range(20)])
        multi_hop = [t for t in tasks if t.qa_type == "multi_hop"]
        factual = sum(1 for t in multi_hop if t.reasoning_mode == "factual")
        inference = sum(1 for t in multi_hop if t.reasoning_mode == "inference")
        sequential = sum(1 for t in multi_hop if t.reasoning_mode == "sequential")

        # 70 multi_hop * 0.50 factual = 35
        assert factual == 35
        # 70 * 0.30 = 21
        assert inference == 21
        # 70 * 0.20 = 14
        assert sequential == 14

    def test_hop_distribution(self):
        cfg = self._make_cfg()
        tasks = build_generation_tasks(cfg, seed_chunk_ids=[f"c{i}" for i in range(20)])
        multi_hop = [t for t in tasks if t.qa_type == "multi_hop"]
        hop_2 = sum(1 for t in multi_hop if t.target_hop_count == 2)
        hop_3 = sum(1 for t in multi_hop if t.target_hop_count == 3)

        # 70 * 0.60 = 42
        assert hop_2 == 42
        # 70 * 0.40 = 28
        assert hop_3 == 28

    def test_all_multi_hop_have_hop_gte_2(self):
        cfg = self._make_cfg()
        tasks = build_generation_tasks(cfg, seed_chunk_ids=[f"c{i}" for i in range(20)])
        for t in tasks:
            if t.qa_type == "multi_hop":
                assert t.target_hop_count >= 2

    def test_seed_chunks_cycle(self):
        cfg = self._make_cfg(total_samples=10)
        tasks = build_generation_tasks(cfg, seed_chunk_ids=["c0", "c1", "c2"])
        # All seed chunks should be used
        used_seeds = {t.seed_chunk_id for t in tasks}
        assert used_seeds == {"c0", "c1", "c2"}


class TestCorpusCapabilitiesReasoningModes:
    def test_basic_modes(self):
        caps = CorpusCapabilities(
            has_document_ids=True,
            has_section_headers=False,
            has_sequential_links=False,
            has_dates=False,
        )
        modes = caps.available_reasoning_modes
        assert "factual" in modes
        assert "inference" in modes
        assert "sequential" not in modes
        assert "synthesis" not in modes

    def test_with_sequential(self):
        caps = CorpusCapabilities(
            has_document_ids=True,
            has_section_headers=False,
            has_sequential_links=True,
            has_dates=False,
        )
        modes = caps.available_reasoning_modes
        assert "sequential" in modes
