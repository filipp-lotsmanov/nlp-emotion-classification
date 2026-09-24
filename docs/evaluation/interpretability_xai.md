# Explainable AI Analysis of the English Emotion Classifier

Integrated gradients and token-ablation curves for `emotion-en-deberta`, the
DeBERTa-v3-base checkpoint retrained on 2026-09-16. Every number below comes
from `docs/evaluation/xai_report.json`, written by
`training/interpretability/attention_analysis.py` in the same run that produced
the figures in `docs/evaluation/figures/xai/`.

Regenerate with:

```bash
uv run --extra xai python training/interpretability/attention_analysis.py \
    "$VEA_MODELS_DIR/emotion-en-deberta" \
    --n-steps 300 \
    --out-dir docs/evaluation/figures/xai \
    --out-json docs/evaluation/xai_report.json
```

## What this replaced, and why

The previous version of this report is not recoverable and should not be
cited. It was wrong in three independent ways, each of which changed its
conclusions:

1. **It analysed a checkpoint that is not in this repository**, reached through
   an absolute path on one machine. See [PROVENANCE](../PROVENANCE.md).
2. **Its forward pass ran the embedding block twice.** It computed
   `model.deberta.embeddings(input_ids)` - word vectors *plus* position vectors
   plus LayerNorm - then fed that back as `inputs_embeds`, which made
   `DebertaV2Model` add positions and normalise a second time. Attributions
   therefore described a different model from the one the ablation curves
   measured. The symptom was unmissable in hindsight: reported confidences as
   low as 0.0001 for the *predicted* class, which softmax over seven classes
   cannot produce, and attribution mass concentrated on commas and full stops.
3. **It integrated the softmax rather than the logit.** Above roughly 0.97
   confidence the softmax gradient is flat along the whole integration path, so
   IG accumulated numerical noise. Attribution rankings were then unstable
   between step counts - one Joy sentence moved from ABC +1.06 to -0.10, and
   Anger's mean crossed zero - which meant the sign of a result depended on a
   parameter nobody had reason to tune.

The report also asserted that the head is multi-label. It is not:
`training/emotion_en_deberta/train.py` sets
`problem_type="single_label_classification"` and trains with
`CrossEntropyLoss`, and `src/vea/stages/emotion_en.py` serves softmax over
argmax.

Item 3 is why the earlier report saw "high convergence deltas (> 0.5)" and
attributed them to too few integration steps. Step count was not the cause.

## Method

Integrated gradients (Captum) over the **word embeddings**, integrating the
raw logit of the predicted class, with a zero-vector baseline and 300 steps.
Confidences are read through softmax so the ablation curves stay in probability
space.

For each of 18 sentences - three per emotion, translated from Russian
documentary speech - tokens are masked cumulatively in two orders: lowest
attribution first, and highest attribution first. Area Between Curves (ABC) is
the summed gap. A positive ABC means the attribution ranking identifies tokens
the prediction actually depends on; a negative ABC means removing the
top-ranked tokens hurt *less* than removing the bottom-ranked ones.

A forward-pass guard runs before the analysis and aborts if the hand-built
attribution path disagrees with the model's own forward pass. It reported a
maximum delta of 0.0000. Defect 2 above could not recur silently.

## Results

| Emotion | Sentence | Confidence | IG delta | ABC |
|---|---|---:|---:|---:|
| Joy | Everything worked out perfectly - we did it! | 0.999 | 2.14 | +1.31 |
| Joy | Come on, it's all fine! | 0.444 | 0.04 | +4.24 |
| Joy | Look, my friends - the border! ... | 0.291 | 2.37 | +3.38 |
| Sadness | We also had to witness death very often. | 0.935 | 0.62 | +2.34 |
| Sadness | He pulled out an axe - and unfortunately ... | 0.462 | 2.55 | +7.89 |
| Sadness | Tabatinga is a very bleak city. | 0.813 | 13.85 | +5.76 |
| Anger | Hide it - quickly, quickly, hide the camera! | 0.388 | 2.44 | **-3.64** |
| Anger | If there's aggression or someone tries ... | 0.994 | 4.67 | +5.05 |
| Anger | This keeps happening all the time! | 0.934 | 10.92 | +0.70 |
| Fear | This feels like some kind of extreme situation ... | 0.974 | 1.21 | +7.29 |
| Fear | What a terrifying place. | 0.553 | 1.14 | +3.58 |
| Fear | Honestly, I've got chills running down my spine. | 0.656 | 0.57 | +4.79 |
| Disgust | But you didn't say it was cocaine. | 0.907 | 11.80 | +3.92 |
| Disgust | I don't need you giving me cocaine! | 0.844 | 1.93 | **-1.14** |
| Disgust | Ugh, what kind of question is that? | 0.543 | 1.92 | +1.17 |
| Surprise | Oh my God, what a question! ... | 0.704 | 2.28 | **-2.49** |
| Surprise | Why did you say it was flour? | 0.767 | 1.25 | +0.34 |
| Surprise | We suddenly sped up like crazy! | 0.714 | 0.83 | **-0.31** |

| Emotion | Mean ABC |
|---|---:|
| Sadness | +5.33 |
| Fear | +5.22 |
| Joy | +2.98 |
| Disgust | +1.32 |
| Anger | +0.70 |
| **Surprise** | **-0.82** |

Per-sentence ablation curves are in `docs/evaluation/figures/xai/`. The full
attribution ranking for every sentence is in `xai_report.json` under
`emotions.<name>.sentences[].attributions`.

## What the results support

**Attribution rankings are stable.** The same run at 50 and at 300 integration
steps produced ABC values agreeing to within 0.5 on every sentence, with no
sign changes. Before the logit fix the same comparison flipped two results.
Whatever these rankings say, they do not depend on the step count.

**Rankings are semantically coherent.** The highest-attribution tokens are the
emotion-bearing content words - `bleak` at 9.99 in the Tabatinga sentence,
`terrifying` at 11.14, `axe` at 5.41, `chills` at 5.15, `sped` at 4.65, `Ugh`
at 4.49, `aggression` at 2.86, `cocaine` at 3.21 - rather than the punctuation
that dominated the previous report. That change is itself the clearest evidence
that defect 2 was real.

**Four of eighteen predictions have negative ABC**, and Surprise is negative on
average. For those inputs no single-token ordering separates the curves:
masking the tokens IG ranks highest degrades the prediction less than masking
the ones it ranks lowest. The honest reading is that these predictions rest on
distributed or interacting features that a per-token attribution cannot
decompose. "Hide it - quickly, quickly, hide the camera!" at -3.64 is the
clearest case, and it is plausibly about repetition and structure rather than
any individual word.

## What the results do not support

**Completeness is not satisfied, and this is the report's main limitation.**
IG's convergence delta should approach zero as steps increase. It does not
here: the range across 18 sentences is 0.04 to 13.85, three sentences exceed 5,
and three - "Tabatinga is a very bleak city" (7.8 to 13.8), "This keeps
happening all the time" (5.2 to 10.9) and "But you didn't say it was cocaine"
(0.9 to 11.8) - got *worse* between 50 and 300 steps. An integral that moves by
an order of magnitude in both directions is not converging slowly; the path is
ill-behaved.

The likely cause is the **zero-vector baseline**, which is not a token and sits
off the embedding manifold, so the straight-line path from it crosses regions
the model never encounters and the accumulated gradients do not telescope to
the output difference. A `[MASK]`-token baseline is the conventional remedy and
would also make the method internally consistent, since the ablation half of
this analysis already removes tokens by replacing them with `[MASK]`. That
change has not been made.

So: the rankings are reproducible and usable, and the negative-ABC findings are
real. Any claim that these attributions sum to the model's output, or that the
magnitudes are calibrated, is not supported by this run.

**Three sentences per emotion is not a sample.** No confidence interval is
computable from n=3, and the per-emotion means above should be read as
descriptions of these specific sentences, not estimates of per-class behaviour.

**These are translated sentences.** They reach the English classifier through
NLLB from Russian documentary speech. Attribution over a translated token
sequence explains the classifier's behaviour on the translation, not on the
original utterance.
