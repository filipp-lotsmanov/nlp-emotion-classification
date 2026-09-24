# The dataset: what it is made of, and what building it discarded

No single public corpus covers seven emotions in television dialogue, which is
the register this project needs rather than product reviews. This is how one was
assembled, what assembling it threw away, and which parts of it a reader can
rebuild.

The build pipeline and its build record live outside this repository. What is
here is the build's output, `corpora/super_emotion_clean.csv.gz` (described in
[`corpora/README.md`](../corpora/README.md)), and a check that the committed file
still matches the figures on this page:

```bash
uv run python training/emotion_en_deberta/data_prep.py
```

## Where it comes from

One source: [`cirimus/super-emotion`](https://huggingface.co/datasets/cirimus/super-emotion),
which is itself six corpora mapped onto a shared label set.

| Corpus | Rows | Kept |
| --- | ---: | ---: |
| ISEAR | 416,809 | 416,809 |
| GoEmotions | 52,534 | **1,013** |
| Crowdflower | 39,171 | 39,171 |
| TwitterEmotion | 20,000 | 20,000 |
| MELD | 13,708 | 13,708 |
| SemEval | 10,599 | 10,599 |
| **Total** | **552,821** | **501,300** |

GoEmotions is the one corpus filtered, and almost all of it goes. Its 27
fine-grained Reddit labels map onto seven classes badly enough that including it
whole would have made a sixth of the training data Reddit comments, against
13,708 lines of actual television dialogue from MELD. What it is kept for is
disgust: 1,013 rows that annotate it directly, and disgust is the class the other
five corpora barely have.

## What building it discards

Roughly a quarter of the source corpus. Every drop is accounted for in the build
record (not in this repository), and each step's row count chains into the next.

| Step | Rows out | Removed |
| --- | ---: | ---: |
| load | 552,821 | |
| filter-sources | 501,300 | −51,521 |
| normalise | 501,300 | |
| deduplicate | 456,141 | −45,159 |
| fold-case | 456,141 | |
| filter-length | 454,120 | −2,021 |
| tidy | 454,120 | |
| drop-love | **419,180** | −34,940 |

**Love is the expensive one.** 34,940 rows, 7.7% of what was left, dropped for
having no seven-class equivalent. The original build meant to relabel them to
whatever their source annotation named, and relabelled exactly none: the lookup
was keyed by the capitalised seven-class names while source annotations are
lowercase and fine-grained, so no candidate ever matched. Fixing the case would
not have rescued them either: `love` and `admiration` have no seven-class target
to be relabelled *to*, which is why the class was being removed in the first
place. The step was ill-defined rather than merely broken, and what it came to
was that a fourteenth of the data got dropped.

## The text pipeline

Twelve normalisation steps, in a fixed order, vendored verbatim from the build
as [`src/vea/text_clean.py`](../src/vea/text_clean.py), which the pipeline applies
at inference. The order is pinned by a test (`tests/test_text_transforms.py`)
because changing it changes the dataset.

The step that matters most is `mark_shouting`. It lowercases everything and
inserts a literal `[CAPS]` token before each all-caps word, so `ANGRY` and
`angry` share a token while the shouting survives as a feature of its own. That
choice is why the original project's error analysis (not in this repository)
could later measure shouting at all, and what
it measured was the strongest single predictor of a wrong answer in the whole
study: **56.8% error rate on texts with an ALL-CAPS word against 6.9% without**.
Erasing case would have deleted that finding along with the signal.

### Deduplication happens halfway through the funnel

The pipeline is split in two around it, and that is not cosmetic. Before case is
folded, `Yes indeed` and `yes indeed` are different rows; afterwards they are
the same one. Deduplicating late removes 671 rows that the published build keeps,
and a build that does that no longer reproduces the published class counts.
`src/vea/text_clean.py` keeps the same split, as `BEFORE_DEDUPE` and
`AFTER_DEDUPE`.

### A bug this build reproduces on purpose

`:/` is on the emoticon list, and emoticons are stripped *before* URLs are
masked. So `https://example.com` loses its `://` and arrives at the masking step
as `https/example.com`, which the URL pattern no longer matches.

Of the **1,857** texts in the source corpus containing a URL, only **199** still
look like one when `[URL]` masking runs. The other **1,658** reach the training
data as a mangled fragment. Both figures reproduce the original build exactly.

This is not fixed here. The published model was trained on data with the bug in
it, and this chapter describes that data rather than a cleaner version nobody
used. The fix is two lines (drop `:/` from the emoticon pattern, or mask URLs
first) and it belongs with a retrain, where its effect can be measured instead of
assumed.

## The labels

Seven classes: Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise. Getting
there from 35 source emotion names is three rules.

1. **Priority, not frequency.** A row labelled both Anger and Disgust becomes
   Disgust. The order is `Disgust, Neutral, Fear, Sadness, Surprise, Anger`, and
   Joy is deliberately not in it: Joy is a third of the data, so it wins only
   when a row carries nothing else.
2. **The source annotation overrules the collapse.** A row whose original
   annotation says `disgust` is put back to Disgust even if the upstream
   seven-class mapping routed it to Anger. This recovers 5,165 rows.
3. **Disgust is restored before Neutral,** so a row annotated as both ends up
   Neutral. That is 33 rows, and it is written down because it is exactly the
   sort of detail that silently moves a class count.

## What the result looks like

| Class | Rows | Share |
| --- | ---: | ---: |
| Joy | 147,869 | 34.52% |
| Sadness | 125,615 | 29.33% |
| Anger | 57,963 | 13.53% |
| Fear | 53,351 | 12.46% |
| Surprise | 15,816 | 3.69% |
| Disgust | 14,316 | 3.34% |
| Neutral | 13,401 | 3.13% |
| **Total** | **428,331** | |

Joy outnumbers Neutral eleven to one, and the three smallest classes are Neutral,
Disgust and Surprise. Two of those three are the classes the model then fails on
hardest: Neutral at a 36.8% error rate, Surprise at 22.7%. The exception is Fear,
which is the third-largest class and still fails a quarter of the time. So rarity
explains most of the difficulty here, and where it does not, something else is
going on.

These error rates come from the original project's error analysis, which is not
in this repository. The analysis regenerated against the retrained checkpoint is
[`evaluation/error_analysis.md`](evaluation/error_analysis.md).

## Provenance: the 9,151 rows that cannot be rebuilt

The published training set is 428,331 rows. This build produces **419,180**. The
difference is a set of synthetic Disgust examples written for the original
project, because even after keeping GoEmotions for disgust the class had only
5,165 rows, under 1.3% of the data.

That file did not survive, and no copy is committed anywhere. So:

- The build reaches 419,180 and stops.
- The record stores both numbers, separately labelled. Here,
  `tests/test_emotion_en_training.py` asserts that 419,180 + 9,151 = 428,331 and
  that adding the synthetic rows to the reproducible Disgust count gives the
  published one.

Everything else reproduces to the row. Running the build gave
all seven published class counts exactly, and the whole funnel except for a
single text that the deduplication step removed here and the length filter
removed there. Both discard it, so nothing downstream differs.

## The cross-check

The original error analysis records an evaluation over 64,250 samples (its
per-class supports are the ones in the model card's held-out table) and says
nothing about where they came from. This record says the
training set holds 428,331 rows and says nothing about any evaluation. Those two
documents were written months apart, and:

- 15% of 428,331 is 64,250, exactly;
- 15% of each published class count is that class's held-out support, within one
  row, across all seven classes, which is the rounding a stratified split needs
  to hit an exact total.

Neither number is derivable from the other, so their agreeing is real evidence
that the error analysis describes a model trained on this data. It is also the
only such evidence available, because the raw predictions are gone.
`test_the_card_supports_are_fifteen_percent_of_these_counts` in
`tests/test_emotion_en_training.py` asserts it.

## What this does not establish

- **That the dataset is good.** It establishes what it is. Whether television
  dialogue is well served by a corpus that is three-quarters ISEAR and Twitter is
  a question for the model card to answer.
- **That the seven classes are the right seven.** They are the client's, and the
  collapse rules above lose real distinctions, including every one of the 34,940
  Love rows.
- **That the synthetic Disgust rows are sound.** They cannot be inspected. Any
  Disgust result carries that caveat, and 9,151 of the class's 14,316 rows are
  affected by it.
- **That the labels are correct.** They are the source corpora's labels, put
  through a mapping. No sample of them was re-annotated here.
