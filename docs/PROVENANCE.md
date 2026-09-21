# Provenance

What this repository contains, where each part came from, what was removed, and
which published results can and cannot be trusted. Written during the revival of
the original `fae2-nlpr-group-group-14-1` archive.

## 1. The provenance defect you need to know about

The published English emotion results were not produced by the model they are
attributed to.

- `docs/model_cards/emotion_en_deberta.md` documents a **DeBERTa-V2-base**
  classifier, fine-tuned on a cleaned super-emotion dataset, reporting accuracy
  0.8995 and macro F1 0.8127 on a 64,250-sample held-out set.
- `emotion_classifier_en.py` declared that model as `./models/checkpoint-3600`.
- The orchestrator (`Task12/main.py`) **overrode** that path with
  `tae898/emoberta-large` — a third-party RoBERTa-large trained on MELD — while
  keeping the label `deberta-finetuned`.
- Output files were therefore written as
  `emotion_deberta-finetuned_en_local_*.json`, and the consolidated CSV column
  `emotion_en_deberta`, from EmoBERTa predictions.

Consequences:

1. Any report sentence of the form "our fine-tuned DeBERTa predicted X" about
   pipeline output is wrong. The model-card metrics come from a separate
   evaluation of the real checkpoint, which is not in the archive and was never
   loaded by the pipeline.
2. The "dual-model ensemble" was DistilRoBERTa + EmoBERTa, both third-party.
   The claim that the ensemble validates the group's own model does not hold.
3. The XAI report in `docs/evaluation/interpretability_xai.md` analysed the real
   DeBERTa checkpoint via a local path, so it describes a *different model* from
   the one the pipeline ran.

This is recorded in the code at the point of the defect: see the
`stage_7b_english_emotion` comment in `src/vea/pipeline.py` and the
`emotion-en-emoberta` entry in `src/vea/config.py`. After retraining, switch
that slot's `model` key to `emotion-en-deberta` and rerun the affected videos.

## 2. Missing weights and missing training code

Three local checkpoints are declared. None are present, and for two of them the
code that produced them is absent from the archive entirely.

| Checkpoint | Used by | Training code in archive? | Spec available? | Now |
| --- | --- | --- | --- | --- |
| `va-xlmroberta-large` | Stages 6A, 6B | **No** | Only a citation to an external repo | **Fetched** — the citation *was* the spec; see §11 |
| `emotion-en-deberta` | Stage 7B | **No** | Yes — the model card | **Retrained**, macro F1 0.8162; see §10 |
| `emotion-ru-finetuned` | Nothing (see below) | **Yes**, complete | Yes — grid search results | Still absent, and nothing references it |

Because stages 7A and 7B gate on the emotional-significance flag produced by
stages 6A/6B, the absence of `va-xlmroberta-large` disabled stages 6 through 9 —
four of the nine stages, including both output artefacts. That was the
highest-priority gap, and it closed without any training: the model was a
published third-party checkpoint all along, mirrored by digest. Run
`scripts/fetch_va_checkpoint.sh`, then read **section 11**, which records a
measurement that matters more than the download: the arousal output stage 6A
thresholds on separates this project's own classes at AUC 0.5734.

A fourth observation: the Russian emotion model the group trained
(`training/emotion_ru/`, three architectures, 27-configuration grid search) is
**not used by the pipeline**. Stage 7A loads the third-party
`Djacon/rubert-tiny2-russian-emotion-detection` instead. The grid search's best
macro F1 was around 0.53, which plausibly explains the decision, but the
decision is undocumented — worth stating explicitly in any report rather than
leaving the reader to assume the group's own model is in the pipeline.

## 3. File mapping

### Kept, renamed and repaired

| Now | Was | Changes |
| --- | --- | --- |
| `src/vea/pipeline.py` | `Task12/main.py` | Package-relative imports; no `sys.exit()` at import; model paths via registry |
| `src/vea/stages/download.py` | `Task12/download_module.py` | Import-time `logging.basicConfig` removed |
| `src/vea/stages/scenes.py` | `Task12/scene_detector_module.py` | as above |
| `src/vea/stages/audio.py` | `Task12/audio_preprocess_module.py` | as above |
| `src/vea/stages/transcribe.py` | `Task12/transcribe_module.py` | as above, plus `CUDA_VISIBLE_DEVICES` pin removed |
| `src/vea/stages/align.py` | `Task12/scene_align_module.py` | as above |
| `src/vea/stages/translate.py` | `Task12/translate_module.py` | as above |
| `src/vea/stages/intensity_ru.py` | `Task12/intensity_classifier_ru.py` | as above |
| `src/vea/stages/intensity_en.py` | `Task12/intensity_classifier_en.py` | as above, plus CUDA pin removed |
| `src/vea/stages/emotion_ru.py` | `Task12/emotion_classifier_ru.py` | as above, plus CUDA pin removed |
| `src/vea/stages/emotion_en.py` | `Task12/emotion_classifier_en.py` | as above, plus CUDA pin removed |
| `src/vea/stages/visualize.py` | `Task12/visualization.py` | as above |
| `src/vea/stages/export.py` | `Task12/csv_generator.py` | as above |
| `docs/ARCHITECTURE.md` | `Task12/README.md` | unchanged content |
| `docs/model_cards/emotion_en_deberta.md` | `Task11/Model_card.md` | unchanged content |
| `docs/evaluation/error_analysis.md` | `Task9/README.md` | unchanged content |
| `docs/evaluation/interpretability_xai.md` | `Task10/README.md` | unchanged content |
| `docs/evaluation/prompt_engineering_log.md` | `Task 8/prompt_engineering_log.md` | unchanged content |
| `docs/evaluation/feature_exploration.ipynb` | `Task4/NLP_features.ipynb` | unchanged content |
| `training/emotion_ru/01_build_dataset.ipynb` | `emotion-iteration/Cleaning.ipynb` | unchanged content |
| `training/emotion_ru/train_single.py` | `emotion-iteration/model_1.py` | unchanged |
| `training/emotion_ru/grid_search.py` | `emotion-iteration/model2.py` | unchanged |
| `training/baselines/` | `Task6/{src,models,main.py,train_models_nlp.py}` | unchanged |
| `training/translation/` | `Task7/*.py` | unchanged |
| `training/interpretability/` | `Task10/part{1,2,3}.py` | renamed to describe their method |

All migrated Python files were reformatted once with `ruff format`, which is
AST-preserving, and had unused imports removed. No logic was rewritten.

### Deliberately not carried over

| Dropped | Reason |
| --- | --- |
| `Base_version/` | Superseded first-pass scripts; contained a hardcoded `C:\Users\Alex\...` path |
| `Pipeline_src/` | Earlier flat-module pipeline, entirely superseded by `Task12`; only `main.py` overlapped by name |
| `Pipeline_src/.ipynb_checkpoints/` | Jupyter autosaves committed to git (8 files) |
| `Pipeline_src/batch_output/`, `Pipeline_src/output/` | 35 result spreadsheets; derived data |
| `Base_version/Outputs/` | 3 result spreadsheets; derived data |
| `Task9/*.png`, `Task10/*_visualizations/` | 66 committed PNGs, 11.9 MB, including a literal `class_confusion_analysis (1).png` duplicate |
| `Task 8/task 8 1 try.py`, `task 8 2 try.py` | Superseded drafts; `prompt_engineering.py` among them does not parse (see below) |
| `synthetic_data_try/`, `russian_model_tuning/` | Exploratory scripts pinned to `/home/y2a`; not referenced by any pipeline stage |
| `Task6/uv.lock`, `Task10/uv.lock`, per-task `pyproject.toml` | Three separate environments for one project; replaced by one root `pyproject.toml` |

`Task 8/prompt_engineering.py` was the one file in the archive with a **syntax
error** (`SyntaxError: invalid syntax. Perhaps you forgot a comma?`) — it could
never have run. Its written output, `prompt_engineering_log.md`, is kept.

Recovering anything dropped: everything above still exists in the original
archive and in the classroom repository's git history. Nothing here is the only
copy.

## 4. Defects fixed during the revival

Each of these is now covered by a static guard in
`tests/test_source_hygiene.py`, so it fails CI if reintroduced.

1. **GPU pinned at import time.** Four modules ran
   `os.environ['CUDA_VISIBLE_DEVICES'] = '5'` before importing torch. On any
   machine without six GPUs this hides *every* GPU; the stages then silently
   fell back to CPU rather than failing. A 20-minute job becomes a multi-hour
   one with no error message. Device choice is now `VEA_DEVICE`.
2. **Machine-specific absolute paths in 12 files**, including
   `A:\git\fae2-nlpr-group-group-14-1\Task6\models\saved`,
   `C:\Users\Filip Letmanov\Block A\...`, `C:\Users\Admin\Downloads\...` and
   `/home/y2a`. Nothing outside the original authors' machines could run them.
3. **`logging.basicConfig` at import in 13 modules.** Whichever module imported
   first won; log level was not controllable by the application.
4. **`sys.exit(1)` from a module-level `except ImportError`.** A missing
   dependency killed the interpreter instead of raising, and hid the real
   traceback behind a printed checklist of filenames that no longer matched the
   actual files.
5. **Imports that only worked from one directory.** `from download_module
   import ...` requires the process CWD to be the module directory, which is
   why the pipeline could not be imported or tested.
6. **`pip freeze` as a dependency spec.** 103 pins including 18
   `nvidia-*-cu12` wheels, uninstallable on CPU-only or macOS hosts. Replaced
   by 13 direct dependencies plus `uv.lock`.
7. **Python version contradiction.** `Task12/README.md` claimed Python 3.8+,
   while `Task6` and `Task10` pinned `requires-python = ">=3.13"`. Resolved to
   3.11-3.13, verified against the published wheel matrix of `ctranslate2`,
   `torch`, `transformers`, `scenedetect` and `sentence-transformers`.
8. **Filenames that cannot be extracted on Linux.** Result spreadsheets were
   named after Cyrillic video titles; when unzipped with non-UTF-8 escaping the
   paths exceed the 255-byte filename limit and extraction fails outright. The
   fix is structural: derived outputs are no longer committed.

9. **Host-dependent text encoding, guarded but not originally broken.** The
   stage modules were already correct here: all 56 of their text-I/O calls pass
   `encoding` explicitly and 12 `json.dump` calls set `ensure_ascii=False`.
   That matters more than it looks — the pipeline's transcription, segment and
   emotion JSON files are full of Russian text, and `open(p)` without an
   encoding uses `locale.getpreferredencoding()`, which is UTF-8 on the Linux
   development server and cp1252 on a Western-European Windows install. So an
   unencoded read is a guaranteed `UnicodeDecodeError` on Windows and nowhere
   else. `tests/test_source_hygiene.py::test_text_file_io_declares_an_encoding`
   now pins that property, and CI runs the suite under
   `PYTHONWARNDEFAULTENCODING=1 -W error::EncodingWarning` plus a Windows
   runner, because this class of bug is invisible to Linux-only testing.

   The revival's own first test suite did *not* get this right: three
   `read_text()` calls in the guard tests lacked an encoding and failed on
   Windows against `align.py`, which contains 76 lines of Cyrillic docstring
   examples. Fixed, and the guard above is the reason it cannot recur.

## 5. Known issues not fixed

Left alone deliberately, because fixing them means changing behaviour of code
that currently works. Each is a reasonable next commit.

- **TensorFlow imported for two utility functions.** `training/baselines/models/{lstm,gru,rnn}.py`
  import `tensorflow.keras.preprocessing` for `Tokenizer` and `pad_sequences`
  inside otherwise pure PyTorch models — roughly 600 MB of dependency for
  vocabulary building and padding, which is about 20 lines of numpy. Now an
  opt-in extra (`uv sync --extra tensorflow`) rather than a hard requirement.
- **`downloads/` vs `data/`.** Stage 1's default `base_output` is still the
  literal string `downloads`, while `VEA_DATA_DIR` defaults to `data/`. Threading
  the setting through stage 1 means touching the stage's own path handling.
- **136 f-strings without placeholders** were auto-fixed, but the migrated
  modules still log with `%`-style formatting done eagerly inside f-strings
  rather than lazily. Harmless, just wasteful.
- **No stage-level tests.** The current suite covers configuration, the CLI and
  source hygiene. The stage modules have validation functions
  (`validate_transcription`, `validate_global_scenes`, `validate_local_segments`)
  that are good test targets and need no model download if fed fixture JSON.

## 6. The English emotion dataset: corrections and a ported cleaner

Added after the dataset build record (`dataset.md`) reached this repository.

### The model card's dataset table is wrong

`docs/model_cards/emotion_en_deberta.md` gives a "Dataset Details" class
breakdown that does not match the build record, and cannot be right on its own
terms: it sums to **433,387** while the same card states a total of 428,331 two
lines earlier.

The build record's counts are correct, and the card's *own metrics table*
proves it. Fifteen percent of each build-record count equals that class's
held-out validation support in the card, for all seven classes, within one row,
summing to exactly 64,250:

| Class | Build record | x15% | Card's support |
| --- | ---: | ---: | ---: |
| joy | 147,869 | 22,180.3 | 22,181 |
| sadness | 125,615 | 18,842.2 | 18,842 |
| anger | 57,963 | 8,694.4 | 8,695 |
| fear | 53,351 | 8,002.6 | 8,003 |
| surprise | 15,816 | 2,372.4 | 2,372 |
| disgust | 14,316 | 2,147.4 | 2,147 |
| neutral | 13,401 | 2,010.1 | 2,010 |

The card's dataset table matches none of these and appears to be a garbled
transcription - it reports disgust as 9,151, which is actually the count of
synthetic rows, and fear as 13,401, which is actually neutral's count.
`training/emotion_en_deberta/data_prep.py` uses the build record's numbers and
cites this. **The model card's dataset table needs correcting.**

### A rebuild stops at 419,180 rows, and that is correct

9,151 synthetic Disgust rows were written for the original project and the file
did not survive. A faithful rebuild therefore yields 419,180 rows with 5,165
Disgust, and `data_prep.py` reports that as a match rather than a failure.
Anything else is a cleaning difference, and then the published metrics are not
comparable to whatever is trained.

### Text normalisation is now a property of the model, not the stage

The training corpus went through twelve normalisation steps. Stage 7B applied
none: `emotion_en.py` fed raw NLLB output straight to the tokenizer. A model
trained on normalised text and served raw text is a train/serve skew, and the
error analysis's headline finding - 56.8% error on ALL-CAPS text against 6.9%
without - is measured on a `[CAPS]` token the pipeline never produced.

The cleaner is vendored verbatim as `src/vea/text_clean.py` and attached to the
one model that needs it via `ModelSpec.preprocess`. Per-model rather than
per-stage, because stage 7B is an ensemble whose members disagree:

| Model | Transform | Why |
| --- | --- | --- |
| `emotion-en-deberta` | `super_emotion_v1` | trained on the normalised corpus |
| `emotion-en-distilroberta` | none | third-party, trained on raw GoEmotions text |
| `emotion-ru-rubert-tiny2` | none | third-party, own convention |
| `va-xlmroberta-large` | none | own corpus, own convention |

Normalising DistilRoBERTa's input would have created a fresh skew rather than
removing one.

### Two artefacts ported on purpose

**The `:/` URL bug.** `strip_emoji` runs before `mask_entities`, and `:/` is on
the emoticon list, so `https://x` becomes `https/x` and never matches the URL
pattern. 1,658 of 1,857 source URLs are mangled this way. The training data
contains that, so the inference path must too; fixing it only at inference would
reintroduce a mismatch in the opposite direction.

**Slang expansion on the wrong register.** The 176-entry slang list was built for
social text and this pipeline analyses Russian travel and nature television.
`goat` becomes "greatest of all time", `sub` becomes "subscribe", `rip` becomes
"rest in peace", `op` becomes "overpowered". The transform is applied anyway,
because the model was trained with it and the damage is identical on both sides
- but the semantics diverge, since on Twitter those usually *are* slang and on a
wildlife documentary they are not. This is a real limitation of stage 7B and
belongs in the model card. A measured ablation is the way to settle whether it
costs accuracy.

One further quirk, consistent across both sides: a shouted word that is also
slang is expanded before `mark_shouting` sees it, so it loses its `[CAPS]`
marker. The shouting feature is therefore absent for all 176 slang entries.

### Why the runtime encoding check was removed

CI briefly ran pytest under `-W error::EncodingWarning`. It fired inside
third-party packages - `dill`, via `datasets` - and was silent when those were
not installed, so it failed for reasons unrelated to this repository while
missing its own target. Replaced by
`tests/test_source_hygiene.py::test_text_file_io_declares_an_encoding`, which
AST-scans the package, the tests and the training entry points written here.

## 7. The corpus arrived, and 83% of it is not what it says

`corpora/super_emotion_clean.csv.gz`, delivered by the group member who built
it, is the training corpus for `emotion-en-deberta`. Its integrity is good and
its provenance label is wrong.

### What checks out

Every figure in `docs/dataset_build.md` that can be tested against the file
does. Measured, not assumed, and pinned in
`tests/test_emotion_en_training.py`:

- 419,180 rows, the reproducible total, with all seven class counts exact;
- Disgust at 5,165, which is the published 14,316 minus the 9,151 synthetic rows
  that did not survive;
- rule 2's restoration hitting exactly 5,165 rows, with every Disgust row
  carrying a `disgust` source annotation and none without;
- rule 3's 33 disgust-and-neutral rows all landing on Neutral;
- no `love` surviving `drop-love`;
- the text already in the twelve-step cleaner's output form, idempotent under
  `vea.text_clean.clean_text` on 99.9% of a sample.

That is a corpus whose build record can be trusted. It also settles the dispute
in section 6: the record is authoritative over the model card, and the card's
dataset table stays wrong.

### What does not: the `ISEAR` block is `dair-ai/emotion`

The record credits 416,809 pre-filter rows to ISEAR, and 349,057 rows survive
into the corpus under that name. They are not ISEAR. Published ISEAR is 7,666
self-reported emotion narratives over seven classes including disgust, shame and
guilt. This block is:

| Property | The `ISEAR` block | Every other source |
| --- | ---: | ---: |
| rows containing `.` `,` `!` or `?` | **0.0%** | 63.1% |
| rows containing `feel` or `felt` | 98.7% | — |
| rows beginning `i feel` / `i felt` / `i am feeling` | 34.3% | 7.5% |
| distinct classes | 5 (no Disgust, no Neutral) | 7 |

That is `dair-ai/emotion` (CARER; Saravia et al., EMNLP 2018): collected by
querying Twitter for "i feel" patterns, distributed with punctuation stripped,
and 416,809 rows in its `unsplit` configuration over six classes — sadness, joy,
love, anger, fear, surprise. Remove love, as `drop-love` does, and five remain.
The row count matches the record to the row and the class scheme matches after
the documented love removal.

The misattribution is upstream, in `cirimus/super-emotion`, not in the group's
build.

**Verified against the upstream dataset**, 2026-09-16, on the BUAS server:

```bash
uv run --with datasets python -c "
from datasets import load_dataset
d = load_dataset('dair-ai/emotion', 'unsplit', split='train')
print(len(d), d.features['label'].names)"
```

```
416809 ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
```

416,809 rows, matching the build record's ISEAR figure exactly, over the six
classes that become this corpus's five once `drop-love` runs. So this is not an
inference from text statistics — the row count and the label scheme both come
from the upstream dataset itself.

### Three consequences

**1. The corpus is Twitter, not a register mix.** 97.2% of rows are
Twitter-derived; MELD, the only actual television dialogue, is 2.59%. The
record frames GoEmotions' exclusion as protecting "13,708 lines of actual
television dialogue" from being outnumbered by Reddit — a trade-off the ISEAR
block had already settled 30:1 in the other direction. For a pipeline that
scores translated Russian travel and nature documentaries, "the training data is
almost entirely English tweets" is a much sharper limitation than "ISEAR and
Twitter", and it belongs in the model card's limitations section and in the
report.

**2. The card's external validation is not external.** The card reports 2,000
CARER samples at 92.55% accuracy and 0.8865 macro F1, against its own held-out
89.95% and 0.8127, and reads the gap as "robust generalization to unseen data".
CARER is 83% of the training data. An "external" benchmark that beats the
internal one by 2.6 points is the shape of contamination, not of transfer. The
claim has to go, and nothing in this repository can replace it: there is no
genuinely held-out corpus here. `training/emotion_en_deberta/train.py` prints
the exposure on every run so the mistake is not made twice.

**3. GoEmotions is not the disgust source the record says it is.** The record
keeps GoEmotions for disgust because "disgust is the class the other five
corpora barely have". In the corpus, Disgust comes from SemEval 3,936,
GoEmotions 914 and MELD 315. SemEval supplies four times as much. GoEmotions'
filtering down to 947 rows is still defensible on register grounds; the reason
given for it is not the one the data supports.

### Smaller findings

**178 texts carry conflicting labels.** 444 texts appear more than once, 178 of
them under two labels, 362 rows in total (0.086%). Every conflict involves
Disgust, which is what rule 2 predicts: the same string reached the collapse
twice and the source-annotation restoration fired for one copy. Too small to
affect the loss, large enough to leak across a random split, so the trainer
splits on unique text rather than on rows.

**`token_count` is a whitespace count**, correlating 0.9998 with `text.split()`.
It is not a subword count and must not be used to set a tokeniser `max_length`.

**Placeholder casing is mixed**: `[NUM]` in 3,673 rows and `[num]` in 1,264,
`[URL]` 52 and `[url]` 109, `[NAME]` 357 and `[name]` 148. The cleaner emits the
uppercase forms, so the lowercase ones arrived pre-masked from Crowdflower and
SemEval. Not fixable without diverging from the training data.

**`[TAG]` is the one place the cleaner and the corpus disagree.**
`mark_shouting` protects a placeholder only as a whole token, so `[TAG] hello`
survives while `[TAG]._.; hi` becomes `[tag]._.; hi`. That is the whole of the
0.1% idempotence shortfall, and it affects no realistic pipeline input, since
`[TAG]` is a Twitter-mention artefact.

### The head type, resolved

Section 6 left it open. It is resolved as **single-label cross-entropy**, to
match `src/vea/stages/emotion_en.py`, which applies softmax and argmax. The card
describes "7-dimensional sigmoid probabilities" and a sigmoid classifier head.
For the predicted label the two are identical, both being monotonic in the
logits. For the *confidence* they are not, and stage 8 discards predictions
below 0.25 while stage 9 exports the value — so the card's calibration evidence
(mean confidence 0.8873 when correct against 0.4275 when wrong) describes
numbers the shipped pipeline never produces. The trainer reports the same
statistic on its own validation set so the comparison can be made honestly.

## 8. The Russian corpus, and a model nothing uses

`corpora/ru_izard_emotions.csv.gz` arrived in the same delivery: 24,766 rows
from `Djacon/ru-izard-emotions` over the same seven classes, better balanced
than the English set (Neutral:Surprise 4.8:1 against 11:1) with 2,714 genuine
Disgust rows.

It trains `emotion-ru-finetuned`, which **the shipped pipeline never
references**: stage 7A uses the hub checkpoint
`Djacon/rubert-tiny2-russian-emotion-detection` — the same upstream author.
Training it is therefore optional, and worth doing only to answer whether the
group's own fine-tune beats the off-the-shelf model on this data.

Three things to settle before anything is concluded from it:

- **It is not the data `emotion-ru-finetuned` was trained on.** The group's own
  `training/emotion_ru/01_build_dataset.ipynb` loads the same upstream corpus as
  a **multi-label** frame, drops the `shame` and `guilt` columns, then
  downsamples to exactly 1,996 samples per label and writes
  `balanced_emotions_small.pkl` — roughly 14,000 rows, perfectly balanced. This
  file is 24,766 rows, single-label and unbalanced. So
  `training/emotion_ru/grid_search_results.csv` reports runs on a *different*
  dataset, and its numbers are not comparable to anything trained on this file.
  The 24,766 rows are consistent with the upstream `train` split reduced to
  single-label rows after shame and guilt are dropped, but that is an inference
  from the shape of the file, not a record — ask before relying on it.
- **The text is raw, not cleaned.** No `[CAPS]` tokens, 2,553 rows with
  uppercase outside a bracketed placeholder, 437 with emoji. So a model trained
  on it must be served raw text, which is why
  `MODEL_REGISTRY["emotion-ru-finetuned"].preprocess` stays `None`.
- **234 rows (0.94%) contain no Cyrillic at all** — untranslated English
  fragments, and 109 rows mention an `r/` subreddit. Consistent with the
  upstream corpus being a machine translation of GoEmotions, where short
  interjections pass through untranslated.

## 9. ~~Still missing: the valence-arousal corpus~~ Resolved, see section 11

Neither delivered corpus carries valence or arousal annotations, and this section
recorded that as the project's last hard blocker, to be closed by training a
substitute on EmoBank with `training/va_regressor/train.py`.

**That turned out to be unnecessary.** The checkpoint was never lost — it is a
published third-party model, and a group member had already mirrored it by
digest. It is fetched rather than trained. Section 11 has the details, including
a measurement that matters more than the download does.

`training/va_regressor/train.py` is kept. It is written, tested and proven end to
end on synthetic data, and it is the tool for training a *replacement* if the
published checkpoint's weak arousal (section 11) turns out to be the binding
constraint on stage 6A.

## 10. The DeBERTa checkpoint, retrained

`emotion-en-deberta` was retrained on 2026-09-16 and **reproduces the model
card's metrics, slightly exceeding them**: macro F1 0.8162 against the published
0.8127, accuracy 0.9202 against 0.8995, weighted F1 0.9193 against 0.9028.
130.7 minutes on one RTX A6000. Full comparison in
`docs/model_cards/emotion_en_deberta.md`, section "Reproduction, 2026-09-16".

Two consequences for section 1 of this document.

**The provenance defect is now fixable rather than just documented.** The
"DeBERTa" column in every published output CSV came from EmoBERTa because the
orchestrator overrode the path. A real DeBERTa checkpoint now exists, so
`stage_7b_english_emotion`'s second slot can be switched from
`emotion-en-emoberta` to `emotion-en-deberta` and the affected videos re-run.
Until that switch is made and the videos re-run, every reported "DeBERTa" result
is still EmoBERTa.

**The missing synthetic rows produced a usable finding rather than a gap.** Six
of the seven classes trained on byte-identical data; only Disgust lost rows
(14,316 → 5,165). Disgust is also the only class that materially regressed
(F1 0.7296 → 0.6627), and it moved in a specific direction: precision *up*
0.6064 → 0.6770, recall *down* 0.9157 → 0.6490. So the augmentation was
inflating recall at precision's cost, which is what the card's own reading of
its Disgust numbers — "overprediction and overlap with anger" — was describing
without knowing the cause.

That converts the honest admission in section 6 into evidence. The 9,151 rows
cannot be inspected, but their effect can now be measured, and it was not
flattering. It also justifies the trainer's default of no class weighting:
weighting would reintroduce the same distortion on purpose. A
`--class-weights balanced` run is still worth doing as the controlled
comparison, and its result belongs in the report either way.

Across the six unchanged classes macro F1 rises from 0.8265 to 0.8418, and
Neutral — the card's weakest class, precision 0.4438 — improves to 0.5873 with
no balancing at all.

The retrained held-out supports land within five rows of the published ones on
every unchanged class, and exactly on sadness (18,842) and surprise (2,372).
Two documents written months apart, with the split protocol reconstructed from
nothing but the 15% figure, agreeing to five rows on 62,862 samples: the build
record is sound, and so is the card's metrics table. Its *dataset* table remains
wrong, for the reasons in sections 6 and 7.

## 11. The VA checkpoint was never lost, and its arousal output is weak

A group member's reimplementation
([alex-krasnoshtanov/Emotion-Timeline](https://github.com/alex-krasnoshtanov/Emotion-Timeline))
settled three open questions in this document. The first two are good news; the
third is the most consequential finding about this pipeline's design.

### It is fetched, not trained

Section 2 lists `va-xlmroberta-large` as missing with "only a citation to an
external repo" as its specification. The citation *was* the specification: the
original coursework used the published checkpoint from
[gmendes9/multilingual_va_prediction](https://github.com/gmendes9/multilingual_va_prediction)
(Mendes & Martins, ECIR 2023, MIT) — XLM-RoBERTa trained on 34
psycho-linguistic datasets across 100 languages. It reads Russian without
translation, which is why one checkpoint legitimately serves both stage 6A and
6B.

It is mirrored as a public GitHub release asset, repackaged as safetensors and
pinned by digest, so it loads without unpickling a `.bin` out of a Google Drive
folder. `scripts/fetch_va_checkpoint.sh` downloads it, checks
`f75773cb738a8f279832b5dd8b24209c5b1c3c71d4eb09d97b2981ecc9041332`, deletes a
mismatched file rather than caching it, and runs the contract check.

Its inference contract is exactly what `src/vea/stages/intensity_ru.py` already
assumes: `num_labels=2`, index 0 valence, index 1 arousal, and sigmoid applied
by the caller. The checkpoint declares `XLMRobertaForSequenceClassificationSig`,
a stock XLM-R with sigmoid folded into `forward()`; loaded through the Auto class
it returns raw logits, and the stage applies `torch.sigmoid` itself. Drop-in.

One correction: **it is the BASE checkpoint, not large.** The authors' large
model is 2.09 GB, over GitHub's asset limit. The registry key still reads
`va-xlmroberta-large` because the archive's stage config did; `ref` is now
`xlmroberta-base-va`, which is what actually gets loaded. The two were within
noise of each other on the measurement that chose the base, which is weak
evidence that capacity is not the binding constraint here.

### Arousal, which stage 6A thresholds on, separates at AUC 0.5734

Russell's circumplex makes two testable predictions, and the seven-class labels
this project already has can check both without new annotation: valence should
rank Joy above Anger, Disgust, Fear and Sadness; arousal should rank Anger, Fear
and Surprise above Sadness and Neutral. Held out on 3,715 Russian rows:

| Dimension | Separates | AUC |
| --- | --- | ---: |
| Valence | Joy over Anger/Disgust/Fear/Sadness | **0.8223** |
| Arousal | Anger/Fear/Surprise over Sadness/Neutral | **0.5734** |

0.5 is chance. **Valence works; arousal barely does** — and the original design
picked arousal as its intensity measure, without ever scoring it.

This lands directly on the dependency in section 1. `intensity_ru.py` and
`intensity_en.py` compute a per-video median arousal threshold and set
`is_emotionally_significant = arousal > threshold`. Stages 7A and 7B then read
that flag and skip every segment it is false for — verified in
`emotion_en.py:490` and `emotion_ru.py:449`, which `continue` past any segment
without an arousal prediction. So **the gate deciding which segments get emotion
classified at all is driven by the weaker of the two dimensions, at an AUC of
0.5734 against this project's own labels.**

That is not a reason to remove the gate; a median split on a weak signal still
halves the work and the pipeline was published with it. It is a reason not to
present emotional-significance filtering as a validated step. Three further
measurements from the same source, all worth citing rather than repeating:

- **The five intensity levels are three.** The original's cuts at
  0.2/0.4/0.6/0.8 put 43.8% of rows in level 3 and 8.7% in levels 1 and 5
  combined.
- **Adding valence and arousal to the classifier does nothing.** Stacked on both
  classifiers, fitted on validation, scored on test: accuracy 0.5009 → 0.5036,
  net 10 rows in 3,715, p = 0.3634.
- **Two rules built on the idea do measurably worse than nothing**, each with its
  threshold fitted on validation: "low arousal means Neutral" costs 0.0135, and
  "valence picks the positive candidate" costs 0.0167. The arousal threshold
  landing at 0.82, the top of its range, is the optimiser saying *always choose
  Neutral*.

The stated reason is that valence is close to "is this positive or negative",
which a seven-class emotion classifier already encodes — a coarser view of the
same signal rather than an independent one.

Where valence does earn its place is the opposite case: on the recording they
committed, 26 of 47 scenes are labelled Neutral and those 26 span 91% of the
episode's whole valence range, 0.154 to 0.824. The label is least informative
exactly where valence is most.

### What the contract check said, and what it got wrong

The fetched checkpoint passes the structural contract and valence cleanly, and
it exposed two defects in the check itself rather than in the model.

| Check | Result |
| --- | --- |
| `num_labels == 2` | pass |
| predictions in [0, 1] after sigmoid | pass |
| valence spread across probes > 0.15 | pass, 0.775 |
| arousal spread across probes > 0.15 | pass, 0.806 |
| [ru] positive probes outrank negative on valence | pass, 0.794 vs 0.202 |
| [en] positive probes outrank negative on valence | pass, 0.781 vs 0.590 |
| `id2label` names the two outputs | **failed - the published checkpoint never named them** |
| [ru]/[en] activated probes outrank calm on arousal | **failed on the sadness probes only** |

**Index order is confirmed, not assumed.** The "sitting calmly reading" probe
scores 0.78-0.79 on column 0 and 0.08-0.10 on column 1. A calm positive sentence
is high valence and low arousal, so column 0 is valence; a swap would make that
probe read low on column 0, and it does not.
`corr(valence, arousal) = -0.568` across the probes, so the heads are not
entangled.

**The missing labels are real and now written in.** The checkpoint ships
`id2label = {0: LABEL_0, 1: LABEL_1}`. The pipeline does not read `id2label` -
`intensity_ru.py` indexes `logits[0, 0]` and `logits[0, 1]` directly - so the
order was load-bearing and undocumented at once. `fetch_va_checkpoint.sh` writes
the names into `config.json`, which touches nothing the sha256 covers.

**The arousal failures were the check's fault.** It required
`min(activated) > max(calm)` over two probes per group: per-item perfection on a
dimension measured at AUC 0.5734. Both failures came from a single probe each -
the sadness sentences, at arousal 0.830 (ru) and 0.881 (en). Drop those two and
the check passes unchanged. Russell's circumplex puts sadness at low arousal;
raters, and this model, often read intensely expressed sadness as activated. On
aggregate the model orders the classes as the circumplex predicts anyway
(sadness 0.4930 below fear 0.5339 and anger 0.5231). The gate is now on group
means, which still catches a swapped index and a collapsed head, and individual
probe disagreements are printed rather than hidden.

That last point is worth stating plainly: a check that fails on an expected
condition is a check people learn to ignore, and this is the second time in this
project that one of these probes cried wolf. The first was the VA smoke run.

### Two smaller questions closed

**`src/vea/text_clean.py` has not drifted.** Their `data/clean.py` hashes to
`c108e3379a960f700d7c8238a52a34fe132e75823f1f7109ea7dd87753f39f38`, byte-identical
to the copy vendored here and to the digest pinned in
`tests/test_text_transforms.py`. So the DeBERTa checkpoint retrained in section
10 normalises exactly as their corpus build does, and a rebuild would not shift
under it.

**The Russian corpus's 24,766 rows are fully accounted for**, closing the open
question in section 8. Their `benchmarks/russian/build-record.json`:

| Step | Rows out | Note |
| --- | ---: | --- |
| load | 24,891 | every split of `Djacon/ru-izard-emotions` |
| deduplicate | 24,853 | identical text, keeping the first |
| drop-unlabelled | 24,766 | no label left once guilt and shame go |
| assign-labels | 24,766 | collapse by priority; `enthusiasm` becomes Joy |

All seven class counts match `corpora/ru_izard_emotions.csv.gz` exactly, and
1,115 rows moved from `enthusiasm` to Joy. It is still not the balanced set
`training/emotion_ru/01_build_dataset.ipynb` builds, so section 8's warning about
`grid_search_results.csv` being incomparable stands.

## 12. Stage 4 needs a second CUDA runtime

The first end-to-end run died at stage 4 with:

```
RuntimeError: Library libcublas.so.12 is not found or cannot be loaded
```

Neither package is at fault. **faster-whisper runs on CTranslate2, whose 4.x
wheels are built against CUDA 12** and need `libcublas.so.12` and
`libcudnn.so.9`. torch now resolves to a CUDA 13 build on PyPI and brings
`nvidia-cublas` 13.x, which provides `libcublas.so.13`. Same library, different
soname, so the linker finds nothing usable. One process needs both majors:
CTranslate2 for stage 4, torch for stages 5B through 7B.

`scripts/setup_ctranslate2_cuda.sh` installs `nvidia-cublas-cu12` and
`nvidia-cudnn-cu12` (about 1.4 GB) and puts them on `LD_LIBRARY_PATH`. They
coexist with the CUDA 13 wheels precisely because the sonames differ.

They are installed **outside the project venv**, beside the models. `uv sync`
prunes anything absent from `uv.lock`, so a venv install would disappear on the
next sync and stage 4 would break again with nothing to point at. Adding them to
`pyproject.toml` instead was considered and rejected: it would put a 1.4 GB
CUDA-12 runtime into the dependency set of a project whose torch is CUDA 13,
where it is dead weight for everything except one stage, and would still need
`LD_LIBRARY_PATH` set to have any effect.

This is a property of the environment rather than of the archive, and it will
recur on any fresh container. `setup_server.sh` does not yet call the script.

### What stages 1 to 3 established on the way

The same run confirmed the first third of the pipeline works on real input, on a
2h20m video:

| Stage | Result |
| --- | --- |
| 1 download | video + 16 kHz mono audio via yt-dlp |
| 2 scenes | 1,597 shots, mean 5.27s, range 1.92-45.2s, validation passed |
| 3 audio | RMS normalised -31.7 dB to -17.0 dB, 205 silence regions (160.6s) |

Stage 3 flagged `low_snr_14.4dB` as a quality warning and continued, which is
the right behaviour, and stage 4's VAD then removed 49:37 of the 2:20:13 as
non-speech before failing on the library. So roughly 90 minutes of speech
survives for transcription - worth knowing, because that figure, not the video
length, sets the cost of stages 4 through 7B.

## 13. The first full run, and what it exposed

Stages 1 through 9 completed on a 2h20m Russian video: 311 segments, a timeline
PNG, an 18-column CSV, 100% complete rows. Every model was what it claimed to
be - the first time that has been true of this pipeline.

### `enthusiasm` was being relabelled Neutral

Stage 7A's hub checkpoint, `Djacon/rubert-tiny2-russian-emotion-detection`,
emits **`enthusiasm`**, which was absent from `EMOTION_CONFIG` in
`src/vea/stages/visualize.py`. `normalize_emotion_label` rewrites any
unrecognised label to `"neutral"` and raises nothing, so on this run **45 of 311
segments - 14.5% of the video - were silently relabelled Neutral.**

The correct mapping is not a guess. The build record for that model's own
training corpus (`benchmarks/russian/build-record.json` in the group's
reimplementation) states "collapse by priority; enthusiasm becomes Joy" and
moves 1,115 rows that way. The pipeline was sending it to the opposite end of
the valence axis.

Fixed by adding `enthusiasm` to joy's aliases.
`tests/test_emotion_en_training.py::TestRussianModelLabelsAreRecognised` pins
every label observed on this run, so a future omission fails a test rather than
turning grey on a chart.

Two published figures are affected and must be regenerated, not quoted:

- **The Neutral proportion of 76.2%** includes up to 45 misfiled rows.
- **Stage 8's ensemble agreement** was inflated, because spurious Neutrals
  agreed with the English models' Neutrals.

### Stages 8 and 9 disagreed about their own agreement

Both partition all 311 segments and both compute a three-model vote over the
same three files, in independently written code:

| | Stage 8 | Stage 9 | Recomputed from the CSV |
| --- | ---: | ---: | ---: |
| all three agree | 178 (57.2%) | 150 (48.2%) | **150** |
| two of three | 108 (34.7%) | 119 (38.3%) | **119** |
| all three differ | 25 (8.0%) | 42 (13.5%) | **42** |

Neither was correct, and fixing `enthusiasm` proved it.

Before the fix, stage 8's collapse-to-neutral was manufacturing consensus: 28 of
its 178 "full agreement" segments came from Russian `enthusiasm` matching two
English Neutrals it does not actually match. After it, stage 8 reported 37
disagreements against stage 9's 42 - still five apart.

Recomputing both ways from the CSV settles it:

| | all three agree | two of three | all three differ |
| --- | ---: | ---: | ---: |
| raw strings (what stage 9 does) | 150 | 119 | 42 |
| canonicalised (what stage 8 does) | 150 | 124 | **37** |

**Stage 8 is right once its vocabulary is complete; stage 9 was never right.**
Its vote compares raw lowercased strings, so Russian `enthusiasm` against
English `joy` counts as a disagreement between two names for one class. That is
the whole five-segment gap.

So the two stages failed in opposite directions - stage 8 over-collapsed, taking
any unknown label to Neutral; stage 9 under-collapsed, treating synonyms as
distinct - and a third map in `export.py` disagreed with both, mapping `joy` to
`Happiness` for the CSV while passing `enthusiasm` through untouched, so one
class appeared in the output under two different values.

### One vocabulary, and unknown labels raise

`vea.config` now owns `EMOTION_CLASSES`, `EMOTION_ALIASES`,
`EMOTION_DISPLAY_NAMES`, `canonicalize_emotion()` and `display_emotion()`.
`visualize.py` derives its aliases from it, `export.py` canonicalises before
voting and takes display names from it, and `canonicalize_emotion` **raises** on
an unrecognised label instead of defaulting.

That last point is the structural fix. Defaulting to "neutral" is what hid
`enthusiasm` for this project's entire life: it produces a plausible timeline, a
plausible CSV and no error anywhere. The one place a default survives is the CSV
exporter, which is the last stage and must not lose a completed run - there it
logs an error and writes the label through unchanged.

`tests/test_emotion_en_training.py::TestOneEmotionVocabulary` pins all of it,
including that neither module may reintroduce a literal alias list or its own
display map.

### What the run says about the corpus, not the code

Per-model label distributions over 311 segments:

| Emotion | Russian (rubert-tiny2) | English (DistilRoBERTa) | English (DeBERTa) |
| --- | ---: | ---: | ---: |
| neutral | 175 | 231 | 220 |
| happiness | 51 | 19 | 54 |
| enthusiasm | 45 | — | — |
| anger | 22 | 6 | 10 |
| surprise | 7 | 30 | 9 |
| fear | 7 | 8 | 6 |
| sadness | 4 | 1 | 3 |
| disgust | — | 16 | 9 |

Three things worth reporting from this table:

1. **Every model is Neutral-dominant**, between 56% and 74% of segments. That is
   the register mismatch from section 7 arriving in the output: a corpus that is
   97% Twitter, applied to documentary speech.
2. **The two English models disagree in a specific direction.** DistilRoBERTa
   finds 30 Surprise and 19 Happiness; the retrained DeBERTa finds 9 and 54.
   They also now receive *different text* - DeBERTa gets the twelve-step cleaner,
   DistilRoBERTa raw - so this is not a clean model comparison, and the report
   should not present it as one.
3. **The Russian model finds no Disgust at all**, while both English models do
   (16 and 9). Its label set differs from the English one, which is the
   cross-language comparability problem `training/README.md` warns about, now
   visible in real output.

## 14. Class weighting, measured

The DeBERTa retrain in section 10 used no class weighting. `balanced` was left
in the trainer to be measured against rather than assumed, and this is the
measurement: both checkpoints trained on the same corpus and the same split,
differing only in `--class-weights`.

| | `none` | `balanced` | delta |
|---|---|---|---|
| macro F1 | 0.8162 | **0.8225** | +0.0063 |
| accuracy | **0.9202** | 0.9173 | −0.0029 |
| weighted F1 | 0.9193 | **0.9207** | +0.0014 |

Per class, F1:

| | `none` | `balanced` | delta |
|---|---|---|---|
| anger | 0.9391 | 0.9375 | −0.0016 |
| disgust | 0.6627 | 0.6785 | +0.0158 |
| fear | 0.8531 | 0.8430 | −0.0101 |
| joy | 0.9619 | 0.9619 | 0.0000 |
| neutral | 0.5678 | 0.5808 | +0.0130 |
| sadness | 0.9542 | 0.9593 | +0.0051 |
| surprise | 0.7746 | 0.7964 | +0.0218 |

The mechanism is the expected one, and it is visible per class rather than in
the summary. Weighting buys recall on the weak classes and pays for it in
precision: Neutral recall 0.5496 → 0.7035 while its precision falls 0.5873 →
0.4946; Surprise recall 0.7078 → 0.8280 against precision 0.8553 → 0.7672.
The three strong classes (anger, joy, sadness) barely move, which is what you
would expect of classes that were never starved.

### The prediction, and where it was wrong

Recorded before the comparison was read: *Disgust precision falls toward
0.6064, recall rises toward 0.9157, macro F1 within ±0.01.*

- Macro F1 within ±0.01: **correct** (+0.0063).
- Disgust precision falls: **direction correct**, 0.6770 → 0.6424, not to 0.6064.
- Disgust recall rises: **direction correct, magnitude badly wrong.** 0.6490 →
  0.7189, a gain of 0.07 against a predicted 0.27.

The 0.6064 / 0.9157 pair was the *published card's* Disgust profile, and the
prediction assumed class weighting would recover it. It does not, and section
10 already says why: 9,151 synthetic Disgust rows — 64% of the class — could
not be rebuilt. Reweighting changes how hard the model is pushed toward a
class; it cannot supply examples that are not there. The card's Disgust recall
is not reachable from this corpus by tuning the loss.

### The finding that matters more

**Neutral is the worst class in both models, by a wide margin** — F1 0.5678
and 0.5808 against 0.84–0.96 for everything except Disgust and Surprise. In
the balanced model its precision is 0.4946: on its own validation split,
**more than half of what it labels Neutral is not Neutral.**

Neutral is also the label the pipeline emits most: 74.9% of segments in the
first full run, 77.2% in the second. The class the system outputs most often
is the one the classifier is least able to get right. That is a caveat on
every "N% Neutral" figure this project reports, and a stronger one than the
register mismatch already noted in section 13 — the corpus is 97% Twitter
text, and this is what that costs on the class that documentary speech is
mostly made of.

### What ships

`none` remains the checkpoint at `emotion-en-deberta`, and every run in this
repository used it. The case for switching is +0.0063 macro F1 from a single
seed with no variance estimate, against −0.0029 accuracy and re-running every
video. That is not a difference worth claiming. `balanced` is kept alongside
at `emotion-en-deberta-balanced` as the measured comparison.

Reproduce with:

```bash
uv run python training/emotion_en_deberta/compare_runs.py \
    "$VEA_MODELS_DIR/emotion-en-deberta" \
    "$VEA_MODELS_DIR/emotion-en-deberta-balanced"
```

It reads the `train_config.json` each run writes beside its weights, so the
comparison needs no GPU and cannot silently re-evaluate on a different split.

### Independently reproduced

Every figure above was then recomputed from the weights, by re-deriving the
validation split from each run's recorded seed and `val_fraction` and running
the model over it — `training/emotion_en_deberta/error_analysis.py`, 62,862
rows per checkpoint. The two paths agree to four decimal places on the
headline and on all twenty-one per-class figures:

| | recorded by the trainer | recomputed from the weights |
|---|---|---|
| accuracy | 0.9202 | 0.9202 |
| macro F1 | 0.8162 | 0.8162 |
| weighted F1 | 0.9193 | 0.9193 |

That matters beyond tidiness. Loading either checkpoint emits a transformers
warning claiming the tokenizer has "an incorrect regex pattern" and will
"lead to incorrect tokenization" — citing, while loading a DeBERTa-v2
tokenizer, a discussion about a Mistral model. Bit-identical reproduction is
the evidence that the warning is spurious here: a tokenizer that had actually
changed could not return the same predictions on the same rows.

## 15. What the error analysis found

`training/emotion_en_deberta/error_analysis.py`, both checkpoints, all 62,862
validation rows. Three findings, in order of how much they should change what
the project claims.

### Stage 8's confidence filter does nothing

Stage 8 discards emotion predictions below 0.25 confidence. Measured on the
shipped checkpoint:

| | value |
|---|---|
| predictions below 0.25 | 0.0003 |
| **errors that survive the gate** | **0.9962** |
| accuracy among kept predictions | 0.9205 |
| accuracy overall | 0.9202 |

The filter removes three predictions in ten thousand and four errors in a
thousand. It improves accuracy by 0.0003. It is not quality control; it is a
line of code that looks like quality control, and every timeline this project
has drawn was drawn as if it were one.

The reason is calibration. Mean confidence is 0.9765 on correct predictions
and **0.6461 on incorrect ones** — the model is confidently wrong, and a gate
at 0.25 never meets a wrong answer. The model card claimed 0.8873 against
0.4275; the gap it advertised (0.46) is not the gap that exists (0.33), and
the incorrect-mean it advertised is half a standard deviation below the truth.

The report now prints a threshold sweep — coverage, accuracy, errors removed
and correct answers removed at seven gates — so the number can be chosen from
the curve rather than inherited. A gate is worth raising only while it removes
errors faster than it removes correct predictions.

### Neutral is the weakest class and the one the pipeline emits most

Neutral is 3.2% of the validation split, and the pipeline labels 75-77% of
documentary segments with it. Its precision is 0.5873 and recall 0.5496 — by
far the worst of the seven, and the errors run both ways:

- Neutral predicted as Joy: 407 rows, 20.3% of the whole class
- Neutral predicted as Fear: 217 rows, 10.8%
- Joy predicted as Neutral: 259 rows; Fear as Neutral: 205; Sadness as
  Neutral: 128

So roughly two in five segments the pipeline calls Neutral are something else,
measured in-distribution, on Twitter text. On documentary speech it will be
worse. This is the single largest caveat on the project's headline output and
it belongs next to every Neutral percentage reported.

### Fear absorbs errors from everywhere

The largest single confusion in the matrix is Sadness predicted as Fear (448
rows). Fear is also the top wrong answer for Anger (148), Disgust (77) and
Surprise (276). Under `balanced` this worsens: Sadness→Fear 687, Anger→Fear
309, and Fear's precision falls 0.8426 → 0.8193 as it takes the extra traffic.

That is the mechanism behind section 14's summary. `balanced` buys Neutral
recall (0.5496 → 0.7035) by making Neutral and Fear into sinks, and pays for
it in the precision of both.

### The length effect is the corpus, not the model

Correct predictions average 18.60 words; incorrect ones 14.75. The previous
report measured 18.56 against 15.35 on a different model, and concluded that
short inputs are ambiguous. Two models trained differently reproduce the same
gap to within a third of a word, which says the effect belongs to the data
rather than to either model. The recommendation that followed from it —
widening context — survives; the attribution should be to the corpus.
