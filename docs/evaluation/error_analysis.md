# Task 9 — End-to-end Error Analysis Report for Emotion Classification

This report summarizes the error analysis conducted on the emotion classification system. It consolidates quantitative findings (accuracy, error rates, confusion patterns), linguistic correlates of model mistakes (length, punctuation and casing features, vocabulary effects), and model confidence behavior. The analysis is grounded in 64,250 labeled samples with an overall accuracy of 89.95% (error rate: 10.05%), as computed from `error_analysis_report.json`. Throughout the report, we reference and interpret the accompanying figures by their titles; open them at full size to ensure the axes remain readable.

- Total samples: 64,250
- Total errors: 6,454
- Overall accuracy: 0.8995
- Overall error rate: 0.1005

## 1. Length effects and information density

Longer inputs are easier for the model. Correct predictions are, on average, longer than incorrect ones both in characters and in words. Specifically, correctly classified texts average 94.60 characters and 18.56 words, while incorrect predictions average 80.05 characters and 15.35 words. A Mann–Whitney U test confirms the difference is highly significant (U ≈ 2.129e8, p ≈ 7.63e−78).

- Interpretation: Short inputs tend to be ambiguous, context-poor, or dominated by non-standard tokens (e.g., slang, interjections). This makes it harder for the model to disambiguate subtle emotions.
- Implication: For production, adding context windows (e.g., preceding/next sentence) or enforcing minimal text length (when possible) may reduce ambiguity-driven errors.

Figure reference: see Figure 1 — Length vs Accuracy: Error Rate by Text Length.

![Figure 1 — Length vs Accuracy: Error Rate by Text Length](./length_analysis.png "Figure 1 — Length vs Accuracy: Error Rate by Text Length")

## 2. Textual features that correlate with errors

We examined surface-level textual features that can make classification difficult. Four indicators stand out: exclamation marks, question marks, ellipses, and ALL-CAPS tokens. Each correlates with noticeably higher error rates relative to the baseline.

- Exclamation mark present: error rate 56.61% (2,194 samples), vs 8.40% without.
- Question mark present: error rate 55.79% (1,217 samples), vs 9.16% without.
- Ellipsis present: error rate 14.29% (7 samples), vs 10.04% without. Note the small support.
- Contains ALL-CAPS words: error rate 56.83% (4,040 samples), vs 6.91% without.

These patterns suggest that highly expressive punctuation and shouting-like casing inflate ambiguity or push the model toward overconfident but wrong decisions (see also Section 5 on confidence). In practice, these markers often occur in short, emphatic posts that omit sentiment-bearing context.

Mitigation ideas:
- Preprocessing: normalize excessive punctuation, map repeated punctuation to a single token, and soften the impact of ALL-CAPS through case normalization while retaining a feature flag to signal emphasis.
- Data augmentation: include more examples with expressive punctuation and casing, balanced across emotion labels.
- Calibration: where expressive markers are present, prefer conservative confidence thresholds or deferral to a fallback classifier.

Figure reference: see Figure 2 — Impact of Textual Features on Error Rate.

![Figure 2 — Impact of Textual Features on Error Rate](./textual_features_analysis.png "Figure 2 — Impact of Textual Features on Error Rate")

## 3. Vocabulary effects and out-of-vocabulary (OOV) risk

The vocabulary analysis highlights tokens that disproportionately appear in errors (“error-biased” words) versus those that appear predominantly in correct predictions. A subset of error-biased tokens is slangy, misspelled, domain-specific, or rare (e.g., “swine”, “macbook”, “twitterland”, “ughh”, “outta”). Some have zero occurrences in correct predictions, which drives their error ratios toward infinity by construction (interpret with care, but the direction is informative).

Conversely, tokens most associated with correct predictions are generally standard and semantically unambiguous (e.g., “optimistic”, “perspective”, “winner”, “engaged”). This aligns with the finding that more formal or descriptive language tends to be easier to classify reliably.

Mitigation ideas:
- Tokenization and normalization: better handle slang and creative spelling (character-level models or robust subword vocabularies), map common colloquialisms to canonical forms, and retain emojis/emoticons as explicit features.
- Domain adaptation: curate in-domain corpora containing the prevalent slang/non-standard tokens seen in the error set; consider targeted fine-tuning.
- Lexical back-off: when encountering many OOV or low-frequency tokens, reduce confidence and defer to alternatives (e.g., larger context or ensemble).

Figure reference: see Figure 3 — Vocabulary Skew in Errors vs Correct Predictions.

![Figure 3 — Vocabulary Skew in Errors vs Correct Predictions](./vocabulary_analysis.png "Figure 3 — Vocabulary Skew in Errors vs Correct Predictions")

## 4. Class-wise error structure and confusion patterns

Class difficulty is not uniform. “Neutral” is hardest (36.77% error), followed by “Fear” (24.87%) and “Surprise” (22.68%). “Joy” (6.26%), “Anger” (6.19%), and “Sadness” (5.73%) are comparatively easier. The most common confusion pairs capture intuitive ambiguities:

- Joy → Neutral: 638 cases. Cheerful but factive statements without explicit affect often look neutral.
- Fear → Sadness: 535 cases; Fear → Neutral: 383; Fear → Disgust: 316; Fear → Surprise: 315. Fear is broadly ambiguous without explicit threat cues.
- Neutral → Joy: 269 and Neutral → Disgust: 239. “Neutral” is a catch-all; low-affect statements sometimes carry subtle positive/negative valence.
- Sadness → Anger: 290 and Sadness → Disgust: 274. Negative-valence emotions overlap linguistically, especially with strong appraisal words.
- Joy → Sadness: 243 and Joy → Disgust: 169. Irony/sarcasm can invert perceived valence.

Per-class error summaries (samples, errors, error rates):
- Anger: 8,695; 538; 6.19%, mostly confused with Disgust (185), Neutral (126), Sadness (125).
- Disgust: 2,147; 181; 8.43%, mostly with Neutral (49), Anger (48), Sadness (36).
- Fear: 8,003; 1,990; 24.87%, commonly with Sadness, Neutral, Disgust, Surprise.
- Joy: 22,181; 1,389; 6.26%, commonly with Neutral (638), Sadness (243), Disgust (169).
- Neutral: 2,010; 739; 36.77%, commonly with Joy (269), Disgust (239), Sadness (134).
- Sadness: 18,842; 1,079; 5.73%, commonly with Anger (290), Disgust (274), Neutral (241).
- Surprise: 2,372; 538; 22.68%, commonly with Neutral (156), Joy (155), Disgust (93).

Mitigation ideas:
- Class-aware training: focal loss or reweighting for “Neutral”, “Fear”, and “Surprise”; targeted augmentation with borderline examples.
- Curriculum and hard-negative mining: prioritize confusable pairs during training and evaluation.
- Context enrichment: where feasible, add neighboring sentences or speaker metadata to disambiguate “Neutral” vs low-intensity “Joy/Disgust”.

Figure reference: see Figure 4 — Class Confusion Heatmap.

![Figure 4 — Class Confusion Heatmap](./class_confusion_analysis.png "Figure 4 — Class Confusion Heatmap")

An alternative rendering is also provided for convenience (same data, different layout). See Figure 5 — Class Confusion Heatmap (Alt View).

![Figure 5 — Class Confusion Heatmap (Alt View)](./class_confusion_analysis%20(1).png "Figure 5 — Class Confusion Heatmap (Alt View)")

## 5. Confidence behavior and calibration

The model’s confidence is well-separated on average: correct predictions have mean confidence ≈ 0.887, while incorrect ones average ≈ 0.428. However, there are 625 high-confidence errors, comprising 9.68% of all errors. These cases are particularly risky for downstream decision-making because they will not be caught by simple confidence thresholds.

Recommendations:
- Temperature scaling or isotonic regression to calibrate probabilities.
- Per-class thresholds: use stricter thresholds for historically ambiguous classes (e.g., “Neutral”, “Fear”) and lenient ones where precision is robust.
- Human-in-the-loop: route high-impact decisions with low margin scores (small difference between top-1 and top-2 logits) to manual review.

Figure reference: see Figure 6 — Confidence Distribution for Correct vs Incorrect Predictions.

![Figure 6 — Confidence Distribution for Correct vs Incorrect Predictions](./confidence_analysis.png "Figure 6 — Confidence Distribution for Correct vs Incorrect Predictions")

## 6. Representative error cases and inspection assets

Two CSV files accompany this analysis for deeper inspection:
- `error_samples.csv`: representative mistakes by class and feature profile (useful for qualitative review and annotation feedback).
- `high_confidence_errors.csv`: the 625 errors with high predicted confidence (prime candidates for calibration, data coverage fixes, or re-labeling checks).

Use these assets to create class- and feature-specific remediation sets. For example, collect “Fear→Sadness” and “Joy→Neutral” candidates for targeted augmentation, and gather samples with ALL-CAPS plus exclamation marks to design robust normalization strategies.

## 7. Actionable next steps

- Data coverage: curate examples rich in punctuation emphasis ("!!!", "???"), ALL-CAPS, and slang/non-standard tokens; ensure balanced label distribution in those slices.
- Preprocessing: implement punctuation normalization, case normalization with an “emphasis” indicator, and better tokenization for slang and emojis.
- Training: class-aware loss reweighting for “Neutral”, “Fear”, and “Surprise”; hard-negative mining for top confusion pairs; consider mild label smoothing.
- Evaluation: add slice-level metrics dashboards for the features above and track per-class calibration error. Re-run this analysis after any model or data change.
- Deployment: add guardrails that lower trust in the presence of risky textual markers and defer borderline cases.

---

Link to model card: see the repository root file `MODEL_CARD.md` for a brief description of how this analysis was addressed and integration guidance.
