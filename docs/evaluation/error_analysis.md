# Error analysis: English emotion classification

Generated 2026-09-24 by `training/emotion_en_deberta/error_analysis.py`.
Every figure below is recomputed from the checkpoints named here, on the
validation split reproduced from each run's own `train_config.json`.

Regenerate with:

```bash
uv run python training/emotion_en_deberta/error_analysis.py \
    "$VEA_MODELS_DIR/emotion-en-deberta" \
    "$VEA_MODELS_DIR/emotion-en-deberta-balanced" \
    --out-json docs/evaluation/error_analysis_report.json \
    --out-md   docs/evaluation/error_analysis.md
```

## Headline

|  | emotion-en-deberta | emotion-en-deberta-balanced |
|---|---|---|
| validation rows | 62,862 | 62,862 |
| errors | 5,017 | 5,201 |
| accuracy | 0.9202 | 0.9173 |
| macro F1 | 0.8162 | 0.8225 |
| weighted F1 | 0.9193 | 0.9207 |

Macro F1 leads. Accuracy on this corpus is largely decided by Joy and
Sadness, so a model answering only those two already scores well.

## Per class

### Precision

| class | emotion-en-deberta | emotion-en-deberta-balanced |
|---|---|---|
| anger | 0.9298 | 0.9396 |
| disgust | 0.6770 | 0.6424 |
| fear | 0.8426 | 0.8193 |
| joy | 0.9513 | 0.9676 |
| neutral | 0.5873 | 0.4946 |
| sadness | 0.9622 | 0.9950 |
| surprise | 0.8553 | 0.7672 |

### Recall

| class | emotion-en-deberta | emotion-en-deberta-balanced |
|---|---|---|
| anger | 0.9487 | 0.9355 |
| disgust | 0.6490 | 0.7189 |
| fear | 0.8637 | 0.8681 |
| joy | 0.9729 | 0.9562 |
| neutral | 0.5496 | 0.7035 |
| sadness | 0.9463 | 0.9260 |
| surprise | 0.7078 | 0.8280 |

### F1

| class | emotion-en-deberta | emotion-en-deberta-balanced |
|---|---|---|
| anger | 0.9391 | 0.9375 |
| disgust | 0.6627 | 0.6785 |
| fear | 0.8531 | 0.8430 |
| joy | 0.9619 | 0.9619 |
| neutral | 0.5678 | 0.5808 |
| sadness | 0.9542 | 0.9593 |
| surprise | 0.7746 | 0.7964 |

## emotion-en-deberta

Class weighting: `none`. Error rate 0.0798 over 62,862 rows.

### Where the errors go

For each true class, the classes its mistakes were predicted as.

| true class | predicted as | count | share of class |
|---|---|---|---|
| anger | fear | 148 | 0.017 |
|  | sadness | 101 | 0.012 |
|  | joy | 62 | 0.007 |
| disgust | fear | 77 | 0.100 |
|  | sadness | 54 | 0.070 |
|  | anger | 51 | 0.066 |
| fear | sadness | 278 | 0.035 |
|  | joy | 235 | 0.029 |
|  | neutral | 205 | 0.026 |
| joy | neutral | 259 | 0.012 |
|  | fear | 124 | 0.006 |
|  | sadness | 91 | 0.004 |
| neutral | joy | 407 | 0.203 |
|  | fear | 217 | 0.108 |
|  | sadness | 138 | 0.069 |
| sadness | fear | 448 | 0.024 |
|  | anger | 248 | 0.013 |
|  | neutral | 128 | 0.007 |
| surprise | fear | 276 | 0.116 |
|  | joy | 250 | 0.105 |
|  | neutral | 93 | 0.039 |

### Confidence

|  | value |
|---|---|
| mean on correct | 0.9765 |
| mean on incorrect | 0.6461 |
| below stage 8's 0.25 threshold | 0.0003 |
| accuracy among kept predictions | 0.9205 |
| errors that survive the threshold | 0.9962 |

Stage 8 drops predictions below the threshold, so the last two rows are
properties of the timeline rather than of the table: a confident error is
one the plot will draw.

#### What a different gate would buy

| threshold | coverage | accuracy kept | errors removed | correct removed |
|---|---|---|---|---|
| 0.25  (stage 8) | 0.9997 | 0.9205 | 0.0038 | 0.0000 |
| 0.50 | 0.9757 | 0.9353 | 0.2093 | 0.0083 |
| 0.60 | 0.9326 | 0.9573 | 0.5011 | 0.0298 |
| 0.70 | 0.9156 | 0.9666 | 0.6169 | 0.0382 |
| 0.80 | 0.8983 | 0.9758 | 0.7273 | 0.0474 |
| 0.90 | 0.8756 | 0.9870 | 0.8569 | 0.0608 |
| 0.95 | 0.8584 | 0.9932 | 0.9272 | 0.0735 |

Read the last two columns together. A gate is only worth raising while it
removes errors faster than it removes correct answers; where those columns
converge, the filter is just discarding timeline.

### Length

|  | correct | incorrect |
|---|---|---|
| mean characters | 94.26 | 76.86 |
| mean words | 18.60 | 14.75 |
| median words | 16.0 | 13.0 |

## emotion-en-deberta-balanced

Class weighting: `balanced`. Error rate 0.0827 over 62,862 rows.

### Where the errors go

For each true class, the classes its mistakes were predicted as.

| true class | predicted as | count | share of class |
|---|---|---|---|
| anger | fear | 309 | 0.036 |
|  | neutral | 114 | 0.013 |
|  | disgust | 56 | 0.006 |
| disgust | fear | 76 | 0.098 |
|  | joy | 48 | 0.062 |
|  | neutral | 40 | 0.052 |
| fear | surprise | 349 | 0.044 |
|  | neutral | 345 | 0.043 |
|  | joy | 183 | 0.023 |
| joy | neutral | 559 | 0.025 |
|  | surprise | 145 | 0.007 |
|  | fear | 133 | 0.006 |
| neutral | fear | 243 | 0.121 |
|  | joy | 222 | 0.111 |
|  | disgust | 44 | 0.022 |
| sadness | fear | 687 | 0.036 |
|  | anger | 294 | 0.016 |
|  | neutral | 222 | 0.012 |
| surprise | neutral | 163 | 0.069 |
|  | joy | 125 | 0.053 |
|  | fear | 83 | 0.035 |

### Confidence

|  | value |
|---|---|
| mean on correct | 0.9777 |
| mean on incorrect | 0.6496 |
| below stage 8's 0.25 threshold | 0.0004 |
| accuracy among kept predictions | 0.9175 |
| errors that survive the threshold | 0.9967 |

Stage 8 drops predictions below the threshold, so the last two rows are
properties of the timeline rather than of the table: a confident error is
one the plot will draw.

#### What a different gate would buy

| threshold | coverage | accuracy kept | errors removed | correct removed |
|---|---|---|---|---|
| 0.25  (stage 8) | 0.9996 | 0.9175 | 0.0033 | 0.0001 |
| 0.50 | 0.9666 | 0.9378 | 0.2734 | 0.0118 |
| 0.60 | 0.9446 | 0.9503 | 0.4330 | 0.0214 |
| 0.70 | 0.9250 | 0.9607 | 0.5601 | 0.0313 |
| 0.80 | 0.8911 | 0.9773 | 0.7558 | 0.0506 |
| 0.90 | 0.8740 | 0.9848 | 0.8396 | 0.0616 |
| 0.95 | 0.8627 | 0.9899 | 0.8948 | 0.0690 |

Read the last two columns together. A gate is only worth raising while it
removes errors faster than it removes correct answers; where those columns
converge, the filter is just discarding timeline.

### Length

|  | correct | incorrect |
|---|---|---|
| mean characters | 94.34 | 76.61 |
| mean words | 18.62 | 14.70 |
| median words | 17.0 | 13.0 |

## Reading this against the pipeline

These figures are measured on the training corpus's own validation split,
which is 97% Twitter text. The pipeline applies the model to translated
Russian documentary speech. In-distribution accuracy here is an upper
bound on what the timeline gets, not an estimate of it - see
`docs/PROVENANCE.md` sections 13 and 14.

