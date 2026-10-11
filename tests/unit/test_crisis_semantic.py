"""The embedding-similarity crisis layer (safety.crisis_semantic) and its wiring."""

from pathlib import Path

import pytest

from fakes import BagOfWordsEmbedder
from kudiguard.api.services import build_crisis_similarity
from kudiguard.config import Settings
from kudiguard.languages import LanguageRegistry
from kudiguard.safety.abuse import ScreenMode
from kudiguard.safety.crisis_semantic import DEFAULT_THRESHOLD, CrisisSimilarity


def test_scores_against_the_exemplars(registry: LanguageRegistry) -> None:
    similarity = CrisisSimilarity(registry, BagOfWordsEmbedder())
    assert similarity.mode is ScreenMode.SHADOW and similarity.threshold == DEFAULT_THRESHOLD
    close = similarity.score("I want to kill myself")  # an exemplar, word for word
    assert close.hit and close.score == pytest.approx(1.0, abs=1e-5) and close.ms >= 0
    far = similarity.score("How do I open a savings account for my daughter?")
    assert not far.hit and far.score < DEFAULT_THRESHOLD


def test_validation(registry: LanguageRegistry, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        CrisisSimilarity(registry, BagOfWordsEmbedder(), threshold=1.0)
    empty = LanguageRegistry(
        [lang.model_copy(update={"abuse": lang.abuse.model_copy(
            update={"crisis_exemplars": []})}) for lang in registry.enabled],
        {lang.code: registry.lexicon_for(lang.code) for lang in registry.enabled},
    )  # fmt: skip
    with pytest.raises(ValueError, match="no crisis_exemplars"):
        CrisisSimilarity(empty, BagOfWordsEmbedder())


def settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg, arg-type]


def test_wiring(registry: LanguageRegistry, monkeypatch: pytest.MonkeyPatch) -> None:
    assert (
        build_crisis_similarity(settings(crisis_semantic_mode="off"), registry, BagOfWordsEmbedder()) is None
    )
    assert build_crisis_similarity(settings(), registry, None) is None  # no embedding model
    built = build_crisis_similarity(settings(crisis_semantic_mode="enforce", crisis_semantic_threshold=0.9),
                                    registry, BagOfWordsEmbedder())  # fmt: skip
    assert built is not None and built.mode is ScreenMode.ENFORCE and built.threshold == 0.9

    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("model missing")

    monkeypatch.setattr(BagOfWordsEmbedder, "embed_documents", broken)
    assert build_crisis_similarity(settings(), registry, BagOfWordsEmbedder()) is None
