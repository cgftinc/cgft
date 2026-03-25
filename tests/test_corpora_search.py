"""Tests for CorporaSearch — pickle-safe search client."""

from __future__ import annotations

import pickle

import cloudpickle
import pytest

from cgft.corpus.corpora.search import CorporaSearch
from cgft.corpus.search_client import SearchClient


class TestConformance:
    def test_isinstance(self):
        cs = CorporaSearch(api_key="t", corpus_name="t", base_url="http://t")
        assert isinstance(cs, SearchClient)

    def test_pickle_roundtrip(self):
        cs = CorporaSearch(api_key="k", corpus_name="cn", base_url="http://b")
        data = cloudpickle.dumps(cs)
        restored = pickle.loads(data)
        assert restored._api_key == "k"
        assert restored._corpus_name == "cn"
        assert restored._client is None

    def test_get_params_masks_key(self):
        cs = CorporaSearch(api_key="sk_long_secret", corpus_name="c", base_url="http://b")
        params = cs.get_params()
        assert params["backend"] == "corpora"
        assert "secret" not in params["api_key"]


class TestAvailableModes:
    def test_lexical_only(self):
        cs = CorporaSearch(api_key="t", corpus_name="t", base_url="http://t")
        assert cs.available_modes == ["lexical"]


class TestModeValidation:
    def test_vector_raises(self):
        cs = CorporaSearch(api_key="t", corpus_name="t", base_url="http://t")
        with pytest.raises(ValueError, match="lexical"):
            cs.search("query", mode="vector")

    def test_hybrid_raises(self):
        cs = CorporaSearch(api_key="t", corpus_name="t", base_url="http://t")
        with pytest.raises(ValueError, match="lexical"):
            cs.search("query", mode="hybrid")


class TestEmbed:
    def test_always_none(self):
        cs = CorporaSearch(api_key="t", corpus_name="t", base_url="http://t")
        assert cs.embed("hello") is None
