# Training corpora

Two labelled emotion corpora, committed gzipped because the server container has
no usable file browser and these are the only route to getting them onto it.
`pandas.read_csv` reads `.csv.gz` directly, so nothing needs unpacking.

| File | Rows | Language | Trains |
| --- | ---: | --- | --- |
| `super_emotion_clean.csv.gz` | 419,180 | English | `emotion-en-deberta` (stage 7B) |
| `ru_izard_emotions.csv.gz` | 24,766 | Russian | `emotion-ru-finetuned` (not wired in) |

Both are **CC BY-SA 4.0** by inheritance from `cirimus/super-emotion`; the
Russian file is MIT by inheritance from `Djacon/ru-izard-emotions`. See
`docs/LICENSING.md` before redistributing either, or any model trained on them.

Neither file contains valence or arousal annotations, so `va-xlmroberta-large`
(stages 6A and 6B) is still unbuildable from what is in this repository.

## Integrity

The inner (uncompressed) sha256 of each file, checked by
`tests/test_emotion_en_training.py`:

```
super_emotion_clean.csv  eeff9a8c3c613bfd8cff5f5f53bbcb4336aea06522c0ac6952bc630aad175d6b
ru_izard_emotions.csv    b4481810e890d79d9f47a601310a6fc8d875750a13461714828d7deaca276da1
```

Gzip output is not byte-stable across implementations, so the digest is taken
after decompression rather than on the archive.

## `super_emotion_clean.csv.gz`

Columns: `text`, `label`, `source`, `labels_source`, `token_count`.

`label` is one of seven Title-cased names. `labels_source` is a stringified
Python list of the original fine-grained annotations (28 distinct names survive
in the final build). `token_count` is a **whitespace** word count, correlating
0.9998 with `text.split()` — it is not a subword count and must not be used to
set a tokeniser `max_length`.

### It reproduces the published build record exactly

Every figure in `docs/dataset_build.md` that can be checked against this file
does check out. This was verified, not assumed:

| Claim in the record | This file |
| --- | --- |
| reproducible build is 419,180 rows | 419,180 |
| Joy 147,869 | 147,869 |
| Sadness 125,615 | 125,615 |
| Anger 57,963 | 57,963 |
| Fear 53,351 | 53,351 |
| Surprise 15,816 | 15,816 |
| Neutral 13,401 | 13,401 |
| Disgust 14,316 published − 9,151 synthetic = 5,165 | 5,165 |
| rule 2 restores 5,165 rows from the source annotation | 5,165 rows carry `disgust` in `labels_source` and are labelled Disgust |
| rule 3 sends 33 disgust+neutral rows to Neutral | exactly 33 |
| `drop-love` removed every Love row | 0 rows mention `love` |

So this is the genuine published corpus minus the 9,151 synthetic Disgust rows
that did not survive, which is the documented and expected shortfall. The
metrics in `docs/model_cards/emotion_en_deberta.md` are therefore *nearly*
comparable to anything trained here — 2.2% of the training rows, and 64% of the
Disgust class, are absent. Report that difference rather than the card's numbers.

### The text is already cleaned — do not clean it again

`vea.text_clean.clean_text` is idempotent on 99.9% of a 3,000-row sample. The
0.1% that changes is the `[TAG]` placeholder, which the vendored cleaner folds to
`[tag]` and the published build left uppercase. Re-running the cleaner over this
file would therefore shift ~0.1% of rows away from what the published model saw,
for no benefit. `training/emotion_en_deberta/train.py` does not clean, and
asserts idempotence on a sample so a future change to the cleaner is caught.

At inference the pipeline *does* apply `clean_text` (via
`MODEL_REGISTRY["emotion-en-deberta"].preprocess`), which is correct: it turns
raw text into this convention. The residual `[TAG]`/`[tag]` disagreement is real
but affects no realistic input — `[TAG]` is a Twitter-mention artefact and this
pipeline scores translated Russian television dialogue.

### Placeholder casing is mixed, and that is upstream

`[NUM]` appears in 3,673 rows and `[num]` in 1,264; `[URL]` 52 and `[url]` 109;
`[NAME]` 357 and `[name]` 148. The cleaner emits the uppercase forms, so the
lowercase ones arrived already masked from Crowdflower and SemEval. Not fixable
without diverging from the training data.

### 178 texts carry conflicting labels

444 texts appear more than once; 178 of them under two different labels, 362 rows
in total (0.086%). Every conflict involves Disgust, which is what rule 2 would
predict: the same string reached the collapse twice and the source-annotation
restoration fired for one copy. Too small to matter for the loss, but large
enough to leak across a random split, so `train.py` splits on **unique text**,
not on rows.

### The largest source is misattributed

The record's source table credits 416,809 rows to **ISEAR**. Those rows are not
ISEAR. Published ISEAR is 7,666 self-reported emotion narratives across seven
classes including disgust, shame and guilt. The 349,057 rows carrying that label
here are:

- **0.0%** punctuated — not one of them contains `.` `,` `!` or `?`, against
  63.1% of every other source;
- **98.7%** contain `feel` or `felt`, and 34.3% begin `i feel` / `i felt` /
  `i am feeling`, against 7.5% elsewhere;
- labelled with exactly five classes — Anger, Fear, Joy, Sadness, Surprise —
  with **zero** Disgust and **zero** Neutral rows.

That is the signature of **`dair-ai/emotion`** (CARER; Saravia et al., EMNLP
2018), which was collected by querying Twitter for "i feel" patterns, is
distributed stripped of punctuation, and whose `unsplit` configuration is
416,809 rows over six classes — sadness, joy, love, anger, fear, surprise. Drop
love, as `drop-love` does, and five classes remain. The row count matches the
record's ISEAR figure to the row, and the class scheme matches after the
documented love removal.

The misattribution is upstream, in `cirimus/super-emotion`, not in the group's
build. But it changes what the corpus *is*, and the report and model card both
leaned on the wrong description.

**Verified against the upstream dataset** on 2026-09-16:

```bash
uv run --with datasets python -c "
from datasets import load_dataset
d = load_dataset('dair-ai/emotion', 'unsplit', split='train')
print(len(d), d.features['label'].names)"
# 416809 ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
```

416,809 rows matches the build record's ISEAR figure exactly, and the six classes
become this corpus's five once `drop-love` runs. See `docs/PROVENANCE.md` §7.

### Consequence: the corpus is 97% Twitter, not 2.6% television

| Source | Rows | Share | Register |
| --- | ---: | ---: | --- |
| ISEAR *(actually dair-ai/emotion)* | 349,057 | 83.27% | Twitter |
| Crowdflower | 34,416 | 8.21% | Twitter |
| TwitterEmotion | 15,183 | 3.62% | Twitter |
| MELD | 10,836 | 2.59% | **television dialogue** |
| SemEval | 8,741 | 2.09% | Twitter |
| GoEmotions | 947 | 0.23% | Reddit |

97.2% of the training data is Twitter-derived. 2.59% is the actual target
register. The record's framing — that including GoEmotions whole "would have made
a sixth of the training data Reddit comments, against 13,708 lines of actual
television dialogue" — describes a trade-off that the ISEAR block had already
settled 30:1 in the other direction.

The record also says GoEmotions "is kept for disgust" because "disgust is the
class the other five corpora barely have". Against this file, Disgust comes from
SemEval 3,936, GoEmotions 914, MELD 315. SemEval supplies four times as much
disgust as GoEmotions does.

## `ru_izard_emotions.csv.gz`

Columns: `text`, `label`, `source`. Same seven Title-cased labels. Single source,
`Djacon/ru-izard-emotions`.

| Class | Rows |
| --- | ---: |
| Neutral | 7,754 |
| Joy | 4,582 |
| Sadness | 3,789 |
| Disgust | 2,714 |
| Anger | 2,398 |
| Fear | 1,898 |
| Surprise | 1,631 |

Better balanced than the English set — Neutral:Surprise is 4.8:1 against the
English 11:1 — and it has 2,714 genuine Disgust rows where the English set has
5,165 across 80× more data.

**This text is raw, not cleaned.** No `[CAPS]` tokens, 2,553 rows with uppercase
outside a bracketed placeholder, 437 rows containing emoji, 7 containing a bare
`http`. So a model trained on it must be served raw text, and
`MODEL_REGISTRY["emotion-ru-finetuned"].preprocess` must stay `None`.

Two things to check with whoever built it:

- **234 rows (0.94%) contain no Cyrillic at all** — untranslated English
  fragments such as `SHUUUUUT UP THIS IS TOO TRUE` and `HEY! NO OC IN
  r/ComedyCemetery`. 109 rows mention an `r/` subreddit. Consistent with
  `ru-izard-emotions` being a machine translation of GoEmotions, where short
  interjections pass through untranslated.
- **It is not the set the group's own notebook builds.**
  `training/emotion_ru/01_build_dataset.ipynb` loads the same upstream corpus
  multi-label, drops `shame` and `guilt`, and downsamples to 1,996 rows per
  label — about 14,000 rows, balanced. This file is 24,766, single-label and
  unbalanced, so `training/emotion_ru/grid_search_results.csv` describes a
  different dataset and its metrics do not transfer. The count is consistent
  with the upstream `train` split reduced to single-label rows, but no record of
  the filter exists here.

`emotion-ru-finetuned` is not referenced by the shipped pipeline — stage 7A uses
the hub checkpoint `Djacon/rubert-tiny2-russian-emotion-detection` instead, which
is the same upstream author as this corpus — so training it is optional. See
`docs/PROVENANCE.md`.
