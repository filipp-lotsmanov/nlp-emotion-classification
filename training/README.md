# Retraining and reproduction

Two checkpoints the pipeline needs do not exist and no code in the archive
produced them. This directory is where they get rebuilt. Read this page before
writing any training code — the order of work matters, and one of the two has a
much weaker specification than the other.

```bash
uv sync --extra train
```

## What exists and what does not

| Target | Blocks | Training code | Corpus | Status |
| --- | --- | --- | --- | --- |
| `va-xlmroberta-large` | Stages 6A, 6B, and therefore 7-9 | **written** (`va_regressor/train.py`) | **missing** | Needs a corpus |
| `emotion-en-deberta` | Stage 7B ensemble member 2 | **written** (`emotion_en_deberta/train.py`) | `corpora/super_emotion_clean.csv.gz`, verified | **Ready to run** |
| `emotion-ru-finetuned` | Nothing (unused by pipeline) | Complete (`emotion_ru/`) | `corpora/ru_izard_emotions.csv.gz`, not the notebook's set | Optional |

**The VA regressor is the only hard blocker left, and it is blocked on data, not
on code.** Stages 7A and 7B consume the `is_emotionally_significant` flag that
stage 6A produces, and stages 8 and 9 consume stage 7, so nothing downstream
runs without it. `va_regressor/train.py` is written, unit-tested and proven end
to end on synthetic data; it needs a valence-arousal corpus and nothing else.
EmoBank is the recommended choice (1–5 scale, so `--va-min 1 --va-max 5`).
Neither delivered corpus carries valence or arousal annotations.

**The DeBERTa classifier can be trained today**, and should be: it is
independent of the VA blocker and costs a few hours of GPU time.

```bash
./scripts/train_emotion_en.sh --smoke    # ~2 min: proves the loop and the contract
./scripts/train_emotion_en.sh --tmux     # the real run, detached
```

## Order of work

1. `va_regressor/` — unblocks stages 6-9. Verify end to end on one short video.
2. Re-run one video and confirm stage 9 writes a CSV with plausible values.
3. `emotion_en_deberta/` — replaces the EmoBERTa stand-in with the documented
   model, and makes the model card's claims true of the pipeline.
4. Switch `stage_7b_english_emotion`'s second entry from `emotion-en-emoberta`
   to `emotion-en-deberta` in `src/vea/pipeline.py` and re-run.
5. Re-run the evaluation in `docs/evaluation/` against the rebuilt models, and
   update any metric in the model card that does not reproduce.

## The inference contract (read this before training anything)

A retrained checkpoint is only useful if it is **drop-in loadable** by the
existing stage modules. These contracts are read from the inference code, not
from the documentation, and the documentation disagrees with the code in one
place. Match the code.

### VA regressor — `src/vea/stages/intensity_ru.py`

```python
self.model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
...
predictions = torch.sigmoid(outputs.logits)
valence = predictions[0, 0].item()
arousal = predictions[0, 1].item()
```

Therefore the checkpoint must satisfy:

- `num_labels = 2`, and the label **order is `[valence, arousal]`**. Getting
  this backwards produces a plausible-looking timeline that is wrong in a way
  no test will catch. Write the order into `config.json` as
  `id2label = {0: "valence", 1: "arousal"}` and assert it in your eval script.
- Inference applies **sigmoid**, so the model must be trained such that
  `sigmoid(logit)` is the target in `[0, 1]` — not raw regression output. If you
  train plain MSE on unbounded logits, sigmoid at inference will squash your
  predictions toward 0.5 and every video will look emotionally flat. Either
  train on `sigmoid(logits)` with MSE against `[0, 1]` targets, or train raw
  regression and remove the sigmoid from the inference code. Pick one and be
  explicit about it in your model card.
- Tokenizer must be saved alongside the weights, `max_length` 512 is what
  inference uses.
- Base model: XLM-RoBERTa-large, because stage 6B feeds the *same* checkpoint
  English text. A Russian-only encoder would break cross-language comparison.

### English emotion classifier — `src/vea/stages/emotion_en.py`

```python
self.id2label = self.model.config.id2label
probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
predicted_id = torch.argmax(probabilities, dim=-1).item()
```

- Labels come from `config.id2label`. They must be spelled exactly
  `anger, disgust, fear, joy, neutral, sadness, surprise` (lowercase). Anything
  else is mapped through `normalize_emotion_label` in `visualize.py`, whose
  fallback for an unrecognised label is `"neutral"` — so a label-naming mistake
  shows up as an implausibly neutral timeline, not as an error.
- **Specification conflict.** The model card specifies a sigmoid head
  (multi-label). The inference code applies softmax and takes argmax
  (single-label). The predicted *class* is unaffected — softmax and sigmoid are
  both monotonic in the logits, so the argmax is the same — but the reported
  `confidence` is not: softmax over logits trained with BCE is miscalibrated.
  That matters, because stage 8 filters predictions at `min_confidence = 0.25`
  and weights timeline opacity by model agreement. Decide deliberately:
  - train **single-label** with cross-entropy, which matches the inference code
    and makes the confidences meaningful, and correct the model card; or
  - keep the multi-label head and change `emotion_en.py` to apply sigmoid.

  The first is less work and matches what the pipeline already assumes.

## `va_regressor/` — rebuilding the valence-arousal model

The trainer is written and tested: `training/va_regressor/train.py`, launched by
`scripts/train_va.sh`, which picks a free GPU, syncs dependencies, logs to a
timestamped file and runs the contract check on the result.

```bash
./scripts/train_va.sh --smoke        # ~2 min: proves the loop and the contract
./scripts/train_va.sh --dataset data/your_corpus.csv \
    --valence-col V --arousal-col A --va-min 1 --va-max 5 --lang-col lang
./scripts/train_va.sh --dataset data/your_corpus.csv --tmux   # detached
```

The `--smoke` run uses synthetic bilingual data and a smaller encoder. It proves
gradients flow, the sigmoid objective behaves, and the saved checkpoint passes
`verify_checkpoint.py`. It says nothing about VA quality — the templates are
trivially separable. Run it first anyway; it is much cheaper than discovering a
plumbing bug after an hour on the real corpus.

What the script already decides for you, matching the inference contract:
`num_labels=2` with `id2label={0: valence, 1: arousal}`, MSE on
`sigmoid(logits)` against `[0, 1]` targets, bf16 autocast, AdamW with no weight
decay on biases and LayerNorm, linear warmup, gradient clipping at 1.0, dynamic
padding truncated at 128 tokens, best-epoch checkpointing on mean Pearson r,
and per-language metrics when you pass `--lang-col`. It logs an explicit error
if either dimension's prediction standard deviation drops below 0.05, since a
collapsed head is the failure that still looks like a successful run.

**What remains your decision**, because the archive never recorded it — only a
citation, in `docs/ARCHITECTURE.md`:

> Mendes, G., & Martins, B. *Quantifying Valence and Arousal in Text with
> Multilingual Pre-trained Transformers.*
> https://github.com/gmendes9/multilingual_va_prediction

1. **Training data.** The reference work uses EmoBank and multilingual VA
   lexica; the original authors did not write down which subset they used.
   Whatever you pick, record it — this is the single biggest source of
   irreproducibility in the project.
2. **Target normalisation.** VA corpora are usually annotated on a 1-9 or -1..1
   scale. Inference expects `[0, 1]`. Record the exact mapping; an off-by-scale
   error here is invisible downstream.
3. **Objective.** Already implemented as MSE on `sigmoid(logits)`, matching
   inference. Change it only if you also change `intensity_{ru,en}.py`.
4. **Hyperparameters.** Nothing was recorded. The script's defaults (lr 1e-5,
   batch 32, 3 epochs, 10% warmup) are sane starting points for
   XLM-RoBERTa-large, which diverges at higher learning rates. On a 48 GB card
   no gradient accumulation is needed; if training loss goes flat immediately,
   lower the learning rate before changing anything else.

Acceptance criteria before you wire it into the pipeline:

- Held-out Pearson r reported separately for valence and arousal. Report
  correlation, not just MSE: MSE looks fine for a model that predicts the mean.
- A sanity check on hand-written Russian and English sentences at the four
  corners of the VA plane (calm-positive, calm-negative, aroused-positive,
  aroused-negative), asserting the ordering comes out right in **both**
  languages — stage 6B reuses this checkpoint on English.
- A degenerate-output check: if the standard deviation of predicted arousal
  across a real video's segments is near zero, the video-adaptive median
  threshold in stage 6A becomes meaningless and every segment lands on one side
  of it.

## `emotion_en_deberta/` — the DeBERTa classifier

Written and ready. The corpus arrived from the group member who built it and is
committed at `corpora/super_emotion_clean.csv.gz`; `data_prep.py` no longer
rebuilds anything, it validates that file against `docs/dataset_build.md` before
any GPU time is spent.

```bash
uv run python training/emotion_en_deberta/data_prep.py    # 5 s, checks the corpus
./scripts/train_emotion_en.sh --smoke                     # ~2 min, proves the loop
./scripts/train_emotion_en.sh --tmux                       # the real run
uv run python training/verify_checkpoint.py emotion-en models/emotion-en-deberta
```

What the run is configured to do, and why:

- **Base:** `microsoft/deberta-v3-base`. The card says "DeBERTa-V2-Base", but its
  own numbers — vocabulary 128,100, SentencePiece, 12 layers, hidden 768, ~183 M
  parameters, relative p2c/c2p attention — are all v3-base's, and v3 checkpoints
  load through the `DebertaV2` classes in transformers, which is where the "V2"
  comes from. DeBERTa-v1-base has a 50,265 BPE vocabulary and ~140 M parameters.
- **Loss:** cross-entropy, single-label. The card describes a sigmoid head;
  `src/vea/stages/emotion_en.py` applies softmax and argmax. The label is the
  same either way, the confidence is not, and stage 8 thresholds on confidence.
  The code is the contract.
- **3 epochs, batch 16, max length 512, 15% validation** — all from the card.
  lr 2e-5 and 6% warmup are conventional; the card does not record them.
- **Model selection on macro F1.** Accuracy is 64% decided by Joy and Sadness
  alone, so a model answering only those two scores 0.64.
- **Split stratified by class and grouped by text.** 444 texts appear twice and
  178 under two different labels, so a row-wise split can train and validate on
  the same string.
- **No class weighting by default.** The two classes the card reports worst,
  Neutral (precision 0.4438) and Disgust (0.6064 against recall 0.9157), are both
  *over*-predicted, and Disgust is the one the original synthetic balancing
  targeted. Weighting would push further in that direction, so
  `--class-weights balanced` and `--class-weights sqrt` exist to be measured
  rather than assumed.
- **The text is not cleaned again.** The corpus is already in the twelve-step
  cleaner's output form (idempotent on 99.9% of a sample) and the pipeline
  applies `clean_text` at inference. `check_corpus_is_preclean` fails the run if
  the vendored cleaner ever drifts from the one that built the corpus.

Two things the published metrics cannot be compared against:

- **9,151 synthetic Disgust rows are gone** — 2.1% of the published set, 64% of
  that class. The corpus stops at 419,180 rows with 5,165 Disgust, which is the
  correct outcome, not a cleaning failure. Report your own numbers and state the
  difference rather than quoting accuracy 0.8995 / macro F1 0.8127.
- **The card's CARER "external validation" is contaminated.** CARER
  (`dair-ai/emotion`) is 83% of the training data — it is the block the corpus
  labels `ISEAR`, mislabelled upstream. The trainer prints the exposure on every
  run. There is no genuinely held-out corpus for this model in this repository.

Both are written up in `docs/PROVENANCE.md` section 7, and the corrections are
applied to `docs/model_cards/emotion_en_deberta.md`.

## `emotion_ru/` — the Russian classifier (already reproducible)

This one is complete, and is the model to imitate for documentation quality.

```
01_build_dataset.ipynb   Loads Djacon/ru-izard-emotions, drops shame and guilt,
                         downsamples to 1,996 samples per label, writes a pickle
train_single.py          DeepPavlov/rubert-base-cased, lr 2e-5, bs 16, 10 epochs
grid_search.py           3 models x 3 learning rates x 3 batch sizes = 27 runs
grid_search_results.json Per-config macro/micro F1, AUC, per-label F1
```

A third corpus arrived with the English one:
`corpora/ru_izard_emotions.csv.gz`, 24,766 single-label rows from the same
upstream source. It is **not** the set this notebook builds — the notebook works
multi-label and downsamples to 1,996 rows per label, about 14,000 rows balanced —
so `grid_search_results.csv` describes a different dataset and its numbers do not
transfer to anything trained on the new file. Its text is also raw rather than
cleaned, which is why `MODEL_REGISTRY["emotion-ru-finetuned"].preprocess` stays
`None`. See `docs/PROVENANCE.md` section 8.

Note two things when you write this up:

1. The pipeline does **not** use this model. Stage 7A loads the third-party
   `Djacon/rubert-tiny2-russian-emotion-detection` instead. The best grid-search
   macro F1 was around 0.53, which is a reasonable justification, but it is
   currently undocumented — state it explicitly.
2. The notebook drops `shame` and `guilt` to reach seven labels, but the
   remaining Izard labels are not the same seven as the English side. Before
   you compare Russian and English emotions per segment, verify the two label
   sets actually align:

   ```bash
   uv run python -c "
   from transformers import AutoConfig
   for m in ['Djacon/rubert-tiny2-russian-emotion-detection',
             'j-hartmann/emotion-english-distilroberta-base']:
       c = AutoConfig.from_pretrained(m)
       print(m, '->', [c.id2label[i] for i in sorted(c.id2label)])
   "
   ```

   Any Russian label that is not in `EMOTION_CONFIG` in
   `src/vea/stages/visualize.py` (`neutral, joy, fear, anger, surprise,
   sadness, disgust`, plus aliases `happiness/happy -> joy` and `sad ->
   sadness`) is silently rewritten to `neutral` in the timeline, while the CSV
   exporter keeps the raw label. That is how the plot and the CSV can disagree.

## `baselines/` and `translation/`

Coursework from earlier tasks, kept because they are the iteration evidence, not
because the pipeline uses them. `baselines/` trains classical and transformer
models on MELD; `translation/` is the from-scratch transformer translation
experiment superseded by NLLB in stage 5B. Both carry hardcoded Windows paths in
their `__main__` blocks (`A:\git\...`), which is why they are not wired into
anything. See `docs/PROVENANCE.md`.

## Where checkpoints go

Save into `VEA_MODELS_DIR` (default `./models/`) under the `ref` declared in
`MODEL_REGISTRY`:

```
models/
├── xlmroberta-large-va/     # va-xlmroberta-large
├── emotion-en-deberta/      # emotion-en-deberta
└── emotion-ru-rubert/       # emotion-ru-finetuned
```

Each needs the full `save_pretrained` output — `config.json` at minimum, which
is what `vea models` checks for. Then:

```bash
uv run vea models    # should report "All declared local checkpoints are present."
```

`models/` is gitignored. Checkpoints belong in release artefacts or on the Hub,
not in git — and remember the CC BY-SA 4.0 obligation on anything trained on
super-emotion data (see `docs/LICENSING.md`).
