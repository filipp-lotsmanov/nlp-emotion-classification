# Licensing

The repository is CC BY-SA 4.0. That is not a free choice — it is forced by one
dataset, and it has consequences worth understanding before you publish
anything, hand the project to a client, or put a checkpoint on the Hub.

## Why the whole repository is CC BY-SA 4.0

The English emotion classifier is fine-tuned on
[`cirimus/super-emotion`](https://huggingface.co/datasets/cirimus/super-emotion),
distributed under CC BY-SA 4.0. ShareAlike propagates to derived works, and a
fine-tuned checkpoint is a derived work of its training data. So:

- `emotion-en-deberta` weights are CC BY-SA 4.0 and cannot be relicensed as MIT.
- Any distribution that bundles those weights must carry a compatible licence.
- The pipeline *code* could be MIT on its own. It is covered by the repository
  licence because the repository is set up to produce and ship those weights.

If you split the repository so no CC BY-SA 4.0 data or weights ship with the
code, the authors can relicense the code.

One upstream question this page does not settle: about 83% of the English
corpus is `dair-ai/emotion` (CARER), which `cirimus/super-emotion` redistributes
under the label `ISEAR` — see `docs/PROVENANCE.md` section 7. That dataset
carries its own upstream licence, which has not been assessed here. Check it
before relying on CC BY-SA 4.0 as the only term attached to the corpus or to
`emotion-en-deberta`.

## The non-commercial constraint

Stage 5B translation uses **NLLB-200**, which Meta releases under
**CC-BY-NC-4.0** — non-commercial.

This is a harder limit than the ShareAlike one. The model card names a client
("Content Intelligence Agency", media analytics), and any commercial deployment
of this pipeline as it stands is not permitted by the NLLB licence. Options:

- Replace stage 5B with a permissively licensed translator (Helsinki-NLP
  `opus-mt-ru-en` is Apache-2.0; the archive's `russian_model_tuning/` had
  already experimented with MarianMT).
- Or drop stage 5B and the English branch entirely and run Russian-only —
  which also removes stages 6B and 7B.
- Or keep NLLB and treat the project as research-only. Say so explicitly.

Whichever you choose, this belongs in the report as a stated limitation. It is
the kind of thing an assessor looks for and a client's lawyer certainly would.

## Per-model licences

Declared alongside each entry in `MODEL_REGISTRY` (`src/vea/config.py`), so the
licence travels with the reference rather than living only in prose:

| Model | Stage | Licence | Commercial use |
| --- | --- | --- | --- |
| Whisper large-v3 (faster-whisper) | 4 | MIT | Yes |
| `paraphrase-multilingual-MiniLM-L12-v2` | 5A | Apache-2.0 | Yes |
| `facebook/nllb-200-*` | 5B | **CC-BY-NC-4.0** | **No** |
| `va-xlmroberta-large` (published checkpoint, fetched) | 6A, 6B | MIT (mirror) | Yes by its licence; training-data terms not assessed |
| `Djacon/rubert-tiny2-russian-emotion-detection` | 7A | MIT | Yes |
| `j-hartmann/emotion-english-distilroberta-base` | 7B | MIT | Yes |
| `tae898/emoberta-large` | None (registered, loaded by no stage) | MIT | Yes |
| `emotion-en-deberta` (retrained, released) | 7B | **CC-BY-SA-4.0** | Yes, with ShareAlike |

The VA regressor is not trained here. It is the published checkpoint from
[gmendes9/multilingual_va_prediction](https://github.com/gmendes9/multilingual_va_prediction)
(Mendes & Martins, ECIR 2023), whose repository is MIT, re-hosted byte-identical
as a release asset of this repository — hence "MIT (mirror)" in the registry.
See `docs/PROVENANCE.md` section 11. That licence covers what the authors
released; the checkpoint was trained on 34 psycho-linguistic datasets across
100 languages, and the terms of those datasets have not been assessed here.
Some VA lexica are research-only, so read the upstream dataset list before
relying on the MIT label for commercial use.

If you train a replacement with `training/va_regressor/train.py`, check the
licence *before* you train: EmoBank is CC-BY-SA 4.0, which would add a second
ShareAlike source, and a research-only lexicon would make the whole pipeline
research-only regardless of what you do about NLLB.

`tae898/emoberta-large` stays in the registry only so that outputs produced
before stage 7B was switched to `emotion-en-deberta` remain reproducible; no
stage loads it.

Verify each licence yourself before relying on this table; upstream terms
change, and two of these entries were read from documentation rather than from
the current model cards.

## Attribution you owe

The upstream datasets and models that must be credited are listed in the
Citation section of [ARCHITECTURE.md](ARCHITECTURE.md): Hartmann's DistilRoBERTa
emotion model, the super-emotion dataset, the ru-izard-emotions dataset, the
multilingual VA prediction work of Mendes & Martins, and Microsoft's DeBERTa.
CC BY-SA requires attribution, so this is an obligation rather than a courtesy.

## What is safe to commit

`.gitignore` excludes `models/`, `data/` and weight file extensions. Keep it
that way: committing a CC BY-SA 4.0 checkpoint into a repository whose licence
file says something else is the easiest way to create a licence violation by
accident. Publish checkpoints as release artefacts or on the Hub, each with its
own licence field set correctly.
