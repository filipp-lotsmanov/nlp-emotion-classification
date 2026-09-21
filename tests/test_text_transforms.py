"""Tests for per-model text normalisation.

The DeBERTa English emotion checkpoint is trained on a twelve-step normalised
corpus; the DistilRoBERTa member of the same ensemble is trained on raw text.
Feeding either one the other's convention creates a train/serve skew, so the
transform is a property of the model, not of the stage. These tests pin that,
and pin the vendored cleaner against silent drift.

Nothing here loads a model or needs torch: `vea.text_clean` is stdlib-only and
`resolve_preprocess` returns a plain callable.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from vea.config import MODEL_REGISTRY, TEXT_TRANSFORMS, resolve_preprocess
from vea.text_clean import AFTER_DEDUPE, BEFORE_DEDUPE, FINALLY, PIPELINE, clean_text

CLEAN_PY = Path(__file__).resolve().parents[1] / "src" / "vea" / "text_clean.py"


class TestVendoredCleanerIsUnmodified:
    """`text_clean.py` is vendored verbatim from the dataset build.

    It must not be edited here. The build record pins its step order with a test
    on the other side, and any divergence means the pipeline normalises
    differently from the training data - the exact skew this module exists to
    remove.
    """

    def test_step_order_is_the_published_order(self):
        assert [f.__name__ for f in PIPELINE] == [
            "normalize_quotes",
            "normalize_placeholders",
            "strip_emoji",
            "expand_slang",
            "mask_entities",
            "collapse_whitespace",
            "mark_shouting",
            "collapse_whitespace",
            "normalize_punctuation",
            "squeeze_repeats",
            "drop_repeated_placeholders",
            "collapse_whitespace",
        ]

    def test_pipeline_is_the_three_phases_concatenated(self):
        assert PIPELINE == BEFORE_DEDUPE + AFTER_DEDUPE + FINALLY

    def test_file_digest_is_recorded(self):
        # Fails loudly if someone edits the vendored file. Update the constant
        # only when deliberately re-vendoring from the dataset build, and say so
        # in docs/PROVENANCE.md when you do.
        #
        # Hashed over LF-normalised bytes, because a Windows checkout can hand
        # this file back with CRLF and that is a different digest for the same
        # code. `.gitattributes` marks the file `-text` so it should not happen
        # - this is the belt to that braces, and it keeps the test measuring
        # the content rather than whoever's `core.autocrlf`.
        digest = hashlib.sha256(CLEAN_PY.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        assert digest == "c108e3379a960f700d7c8238a52a34fe132e75823f1f7109ea7dd87753f39f38", (
            f"src/vea/text_clean.py has changed (sha256 {digest}). It is vendored "
            "verbatim from the dataset build; edit it there and re-vendor."
        )


class TestKnownBugsAreReproduced:
    """The training data contains these artefacts, so inference must too."""

    def test_the_url_bug_survives_the_port(self):
        # `:/` is on the emoticon list and emoticons are stripped before URLs
        # are masked, so https:// arrives as https/ and never matches the URL
        # pattern. 1,658 of 1,857 source URLs are mangled this way. Fixing it
        # only at inference would create a fresh mismatch.
        assert clean_text("Check https://example.com now") == "check https/example.com now"

    def test_a_bare_www_url_still_masks(self):
        # www.x.com has no `://`, so the emoticon strip leaves it intact and the
        # URL pattern matches. Asymmetric, and that asymmetry is in the data.
        assert "[URL]" in clean_text("Check www.example.com now")

    def test_shouting_becomes_a_caps_token(self):
        # The strongest single predictor in the error analysis (56.8% error rate
        # vs 6.9%) depends on this token existing.
        assert clean_text("That was AMAZING") == "that was [CAPS] amazing"

    def test_slang_expansion_beats_the_caps_marker(self):
        # A shouted word that is also slang is replaced before mark_shouting
        # sees it, so it loses its [CAPS] marker. Consistent across training and
        # inference, but it means the shouting feature is absent for all 176
        # slang entries.
        assert "[CAPS]" not in clean_text("That was OMG")
        assert clean_text("That was OMG") == "that was oh my god"

    def test_documentary_register_is_mangled_by_the_slang_list(self):
        # Documented limitation, not a defect to fix here: the slang list was
        # built for social text, and this pipeline analyses Russian travel and
        # nature television. The transform is still applied because the model
        # was trained with it; the semantic damage is the same in training and
        # inference. See docs/PROVENANCE.md.
        assert (
            clean_text("The goat climbed the roof") == "the greatest of all time climbed the roof"
        )
        assert clean_text("Waiting for the sub") == "waiting for the subscribe"


class TestPerModelSelection:
    def test_only_the_deberta_checkpoint_declares_a_transform(self):
        declaring = {n for n, s in MODEL_REGISTRY.items() if s.preprocess}
        assert declaring == {"emotion-en-deberta"}

    def test_the_other_ensemble_member_gets_raw_text(self):
        # DistilRoBERTa (Hartmann) was trained on raw GoEmotions text.
        # Normalising its input would degrade it.
        assert MODEL_REGISTRY["emotion-en-distilroberta"].preprocess is None

    def test_neither_the_russian_nor_the_va_models_are_normalised(self):
        for name in ("emotion-ru-rubert-tiny2", "va-xlmroberta-large"):
            assert MODEL_REGISTRY[name].preprocess is None, name

    def test_every_declared_transform_resolves(self):
        for name in TEXT_TRANSFORMS:
            assert callable(resolve_preprocess(name))

    def test_none_resolves_to_identity(self):
        raw = "The GOAT was AMAZING!!!! https://x.com"
        assert resolve_preprocess(None)(raw) == raw

    def test_an_undeclared_transform_is_rejected(self):
        with pytest.raises(KeyError, match="Unknown text transform"):
            resolve_preprocess("super_emotion_v99")

    def test_the_declared_transform_actually_normalises(self):
        fn = resolve_preprocess("super_emotion_v1")
        assert fn("That was AMAZING") == "that was [CAPS] amazing"
