# **Model Card: DeBERTa-V2 Emotion Classification Model**

---

## **Model Overview**

The **DeBERTa-V2 Emotion Classifier** is a fine-tuned multilingual Transformer model designed for **single-label emotion classification** in media text.  
It recognizes **seven emotion categories**: *joy, sadness, anger, fear, disgust, surprise,* and *neutral*. 

| Attribute | Description                                                |
|------------|------------------------------------------------------------|
| **Model Type** | Transformer (DeBERTa-V2-Base)                              |
| **Task** | Single-label Emotion Classification                        |
| **Languages** | English                                                    |
| **Domain** | Media transcripts (film, series, and TV dialogue)          |
| **Output** | 7-dimensional sigmoid probabilities for emotion categories |
| **Framework** | PyTorch 2.1 + Transformers 4.57                            |
| **Precision** | Float32                                                    |

---

## **Architecture**

| Component | Specification |
|------------|---------------|
| **Base Architecture** | DeBERTa-V2-Base |
| **Layers** | 12 Transformer encoder layers |
| **Attention Heads** | 12 |
| **Hidden Size** | 768 |
| **Intermediate Size** | 3072 |
| **Feed-Forward Activation** | GELU |
| **Hidden Dropout** | 0.1 |
| **Attention Dropout** | 0.1 |
| **Max Sequence Length** | 512 tokens |
| **Vocabulary Size** | 128,100 |
| **Positional Encoding** | Relative (p2c / c2p) |
| **Classifier Head** | Linear layer (768 → 7) with sigmoid activation |
| **Tokenizer** | SentencePiece (`spm.model`) |
| **Special Tokens** | `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`, `[UNK]` |
| **Case Sensitivity** | Case-preserving (no lower-casing) |
| **Parameters** | ≈ 183 M parameters (confirmed via `tensor_stats.json`) |

---

## **Purpose**

This model was developed for the **Content Intelligence Agency**, a company that builds AI-driven software for **media makers** to analyze their content.  
It identifies and labels emotions expressed in dialogue or transcripts, enabling data-driven insights into **audience perception**, **emotional pacing**, and **storytelling impact**.

**Example use cases**
- Detecting emotional tone in film and TV dialogues  
- Analyzing audience reactions or emotional structure across scenes  
- Supporting emotion-aware search and recommendation systems  

---

## **Development Context**

| Aspect | Details                                                                                                               |
|--------|-----------------------------------------------------------------------------------------------------------------------|
| **Client** | Content Intelligence Agency                                                                                           |
| **Project Goal** | Build the core NLP component of a multimodal pipeline that transcribes, translates, and classifies emotions in videos |
| **Hardware** | NVIDIA RTX A6000 (52 GB VRAM, CUDA 12.8) on BUAS university server                                                    |
| **Training Duration** | ~2-3 hours                                                                                                            |
| **OS / Kernel** | Ubuntu 20.04 LTS                                                                                                      |
| **CPU / RAM** | Intel Xeon Silver 4310 × 2 (20 cores total), 64 GB ECC RAM                                                            |
| **Python / Torch** | Python 3.9.21 · Torch 2.1.0 + cu121                                                                                   |
| **Training Framework** | Hugging Face Transformers                                                                                             |
| **Key Assumptions** | Emotionally salient language patterns generalize across translation                                                   |
| **Constraints** | Limited GPU time, imbalanced data, bilingual diversity, and imperfect transcription quality                           |

---

## **Intended Use**

- Tagging emotions in **translated video transcripts**, **TV subtitles**, or **interview text**.  
- Providing metadata for **emotion-aware analytics**, **scene segmentation**, or **viewer-reaction prediction**.

### **Not Intended For**
- Psychological or medical emotion assessment  
- Diagnosing mood or mental state of individuals  
- Automated moderation or labeling of sensitive personal content  

---

## **Dataset Details**

> **Corrected.** The class table below replaces one whose counts were wrong in
> two independent ways: its class names were misassigned against the descending
> order of its own numbers, and the numbers summed to 433,387 rather than the
> 428,331 this section stated. The figures here come from the build record
> (`docs/dataset_build.md`) and are verified row by row against the delivered
> corpus by `tests/test_emotion_en_training.py`. The evidence that the record
> wins: 15% of each count below equals this card's own held-out validation
> support for that class — exactly in five classes, one row out in Joy and
> Anger, summing to exactly 64,250. Neither number is derivable from the other.
> See `docs/PROVENANCE.md` sections 6 and 7.

**Dataset Name:** `super_emotion_clean.csv`, committed at
`corpora/super_emotion_clean.csv.gz`  
**Published training set:** 428,331 samples  
**Rebuildable today:** 419,180 samples — see *Provenance of the Disgust class*  
**Languages:** English  

| Emotion | Samples | Share | Rebuildable |
|----------|---------:|------:|------------:|
| Joy | 147,869 | 34.52 % | 147,869 |
| Sadness | 125,615 | 29.33 % | 125,615 |
| Anger | 57,963 | 13.53 % | 57,963 |
| Fear | 53,351 | 12.46 % | 53,351 |
| Surprise | 15,816 | 3.69 % | 15,816 |
| Disgust | 14,316 | 3.34 % | **5,165** |
| Neutral | 13,401 | 3.13 % | 13,401 |
| **Total** | **428,331** | | **419,180** |

**Structure (as delivered):**  
`text`, `label`, `source`, `labels_source`, `token_count`

The column list previously given here — `text`, `labels`, `labels_str`,
`labels_source`, `source`, `text_length`, `token_count` — describes a working
frame that was never shipped. `token_count` is a whitespace word count, not a
subword count, and must not be used to set a tokeniser `max_length`.

### **Provenance and register**

**One source:** [`cirimus/super-emotion`](https://huggingface.co/datasets/cirimus/super-emotion)
(CC BY-SA 4.0), itself an aggregation of six corpora. That licence propagates to
this model.

| Source | Rows | Share | Register |
| --- | ---: | ---: | --- |
| `ISEAR` — **actually `dair-ai/emotion` (CARER)** | 349,057 | 83.27 % | Twitter |
| Crowdflower | 34,416 | 8.21 % | Twitter |
| TwitterEmotion | 15,183 | 3.62 % | Twitter |
| MELD | 10,836 | 2.59 % | **television dialogue** |
| SemEval | 8,741 | 2.09 % | Twitter |
| GoEmotions | 947 | 0.23 % | Reddit |

**97.2% of the training data is English Twitter text; 2.59% is television
dialogue.** The block labelled `ISEAR` is not ISEAR: published ISEAR is 7,666
punctuated questionnaire narratives over seven classes, while this block is
349,057 rows with *zero* punctuation, 98.7% of them containing "feel" or "felt",
across only five classes. It is `dair-ai/emotion`, mislabelled upstream; the
evidence is in `docs/PROVENANCE.md` section 7. This bears directly on the
*Domain* field at the top of this card, which reads "Media transcripts (film,
series, and TV dialogue)": the model is *applied* to that domain, and was
essentially not trained on it.

**Preprocessing:** the twelve-step normaliser in `src/vea/text_clean.py`,
vendored verbatim from the dataset build and applied at inference through
`MODEL_REGISTRY["emotion-en-deberta"].preprocess`; then duplicate and
missing-value removal; then a seven-class label collapse whose three rules are
in `docs/dataset_build.md`. Two deliberate artefacts of the cleaner carry into
inference — a URL-masking bug affecting 1,658 of 1,857 URLs, and slang expansion
tuned for social text rather than documentary — both in `docs/PROVENANCE.md`
section 6.

**Provenance of the Disgust class:** 9,151 of its 14,316 rows (64%) were
synthetic examples written for the original project. That file did not survive
and no copy exists, so a rebuild reaches 419,180 rows with 5,165 Disgust and
stops. Every Disgust result below carries that caveat, and metrics from a
retrained checkpoint are not directly comparable to them.

Within the reproducible corpus, Disgust comes from SemEval (3,936), GoEmotions
(914) and MELD (315) — so GoEmotions, which the build record keeps specifically
for this class, is not its main source.

**Integrity**
- No missing values; UTF-8 encoding verified  
- Fully cleaned and ready for fine-tuning or explainability analysis  

---

## **Performance Metrics and Evaluation**

### **Held-Out Validation Set Performance**

| Metric | Overall | Anger | Disgust | Fear | Joy | Neutral | Sadness | Surprise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Precision** | 0.8011* | 0.9122 | 0.6064 | 0.9543 | 0.9590 | 0.4438 | 0.9396 | 0.7922 |
| **Recall** | 0.8415* | 0.9381 | 0.9157 | 0.7513 | 0.9374 | 0.6323 | 0.9427 | 0.7732 |
| **F1-Score** | 0.8127* | 0.9250 | 0.7296 | 0.8407 | 0.9481 | 0.5215 | 0.9412 | 0.7826 |
| **Support** | 64,250 | 8,695 | 2,147 | 8,003 | 22,181 | 2,010 | 18,842 | 2,372 |

- **Macro averages** (equal weight per class)

| Summary Metric | Value |
| --- | --- |
| **Accuracy** | 0.8995 |
| **Macro F1** | 0.8127 |
| **Micro F1** | 0.8995 |
| **Weighted F1** | 0.9028 |

---

### **Reproduction, 2026-09-16**

The original checkpoint is lost. It was retrained from the build record by
`training/emotion_en_deberta/train.py` — `microsoft/deberta-v3-base`, 3 epochs,
batch 16, lr 2e-5, cross-entropy, no class weighting, 15% held out stratified by
class and grouped by text. 130.7 minutes on one RTX A6000. **The results above
reproduce, and are slightly exceeded.**

| Summary Metric | Published | Retrained | Δ |
| --- | ---: | ---: | ---: |
| **Accuracy** | 0.8995 | **0.9202** | +0.0207 |
| **Macro F1** | 0.8127 | **0.8162** | +0.0035 |
| **Weighted F1** | 0.9028 | **0.9193** | +0.0165 |

| Class | F1 published | F1 retrained | Δ | P published → retrained | R published → retrained |
| --- | ---: | ---: | ---: | ---: | ---: |
| anger | 0.9250 | 0.9391 | +0.0141 | 0.9122 → 0.9298 | 0.9381 → 0.9487 |
| **disgust** | 0.7296 | **0.6627** | **−0.0669** | 0.6064 → **0.6770** | 0.9157 → **0.6490** |
| fear | 0.8407 | 0.8531 | +0.0124 | 0.9543 → 0.8426 | 0.7513 → 0.8637 |
| joy | 0.9481 | 0.9619 | +0.0138 | 0.9590 → 0.9513 | 0.9374 → 0.9729 |
| neutral | 0.5215 | 0.5678 | +0.0463 | 0.4438 → 0.5873 | 0.6323 → 0.5496 |
| sadness | 0.9412 | 0.9542 | +0.0130 | 0.9396 → 0.9622 | 0.9427 → 0.9463 |
| surprise | 0.7826 | 0.7746 | −0.0080 | 0.7922 → 0.8553 | 0.7732 → 0.7078 |

**This is a clean natural experiment, by accident.** Six of the seven classes
were trained on byte-identical data; only Disgust changed, losing 9,151
synthetic rows (14,316 → 5,165, −63.9%). Disgust is also the only class that
materially regressed. Across the six unchanged classes macro F1 rises from
0.8265 to 0.8418 (+0.0153); the full seven-class macro gains only +0.0035
because Disgust drags it back.

**The synthetic augmentation was inflating Disgust recall at precision's cost.**
Removing it moved precision *up* 0.6064 → 0.6770 and recall *down* 0.9157 →
0.6490. The published card reads its own Disgust numbers as "overprediction and
overlap with anger"; that overprediction was a property of the augmentation, and
the retrained head is better calibrated on the class and finds less of it. This
is the strongest available evidence about 9,151 rows nobody can inspect, and it
is the reason `--class-weights` defaults to `none` in the trainer: weighting
would reintroduce the same distortion deliberately.

**Neutral improved most** (F1 +0.0463, precision 0.4438 → 0.5873) with no
balancing at all, which further undercuts the card's "manual balancing"
rationale.

**The split protocol matches.** Retrained held-out supports land within five rows
of the published ones on every unchanged class — anger 8,693 vs 8,695, fear
7,998 vs 8,003, joy 22,178 vs 22,181, neutral 2,007 vs 2,010, sadness 18,842 vs
18,842 exactly, surprise 2,372 exactly. Disgust is 772, which is 15% of 5,165.

**Confidence, and a caveat.** Retrained: 0.9765 when correct, 0.6461 when wrong,
with 0.03% of predictions below stage 8's 0.25 floor. Published: 0.8873 and
0.4275. So the retrained model is more confident in both directions and its
*relative* gap is narrower — it is more assertive when wrong, which is the worse
direction for threshold-based filtering. The two are not measured the same way:
this card describes a sigmoid head, the retrained model is cross-entropy, and the
pipeline applies softmax to whatever it loads. Treat the confidence comparison as
indicative only; the F1 comparison is sound.

Contract verification passes (`training/verify_checkpoint.py emotion-en`): seven
lowercase labels, all recognised by `visualize.EMOTION_CONFIG`, six distinct
labels across the probes and all seven above the stage 8 floor. One probe of
seven disagrees with intent — "I am furious about what you did." predicts
*disgust* at 0.881 — which is the anger/disgust confusion this card already
documents, now visible in the opposite direction.

Full metrics and the exact configuration are in `train_config.json` beside the
weights.

---

### **CARER Benchmark Performance (~~Unseen Data~~ — in-distribution)**

> **Retracted as an external benchmark.** CARER (`dair-ai/emotion`) is **83% of
> the training data**: it is the block this corpus labels `ISEAR`, mislabelled
> upstream. These 2,000 samples are the same distribution the model was trained
> on, not unseen data, and the numbers below cannot support a generalisation
> claim. The tell is in the table itself — an "external" benchmark that *beats*
> the held-out set by 2.6 points on accuracy and 7.4 on macro F1, on exactly the
> five classes that the CARER block contains and with the two hardest classes
> (Disgust, Neutral) absent by construction.
>
> The figures are kept, unaltered, as a record of in-distribution performance on
> the corpus's dominant source. No genuinely held-out evaluation exists for this
> model; `docs/PROVENANCE.md` section 7 explains why, and
> `training/emotion_en_deberta/train.py` prints the exposure on every run so the
> claim is not made again.

Evaluation on 2,000 samples from the CARER emotion dataset, containing 5 emotions (*anger, fear, joy, sadness, surprise*).

| Metric | Overall | Anger | Fear | Joy | Sadness | Surprise |
| --- | --- | --- | --- | --- | --- | --- |
| **Precision** | 0.8685* | 0.8318 | 0.9330 | 0.9949 | 0.9235 | 0.6596 |
| **Recall** | 0.9188* | 0.9709 | 0.8080 | 0.9204 | 0.9552 | 0.9394 |
| **F1-Score** | 0.8865* | 0.8960 | 0.8660 | 0.9562 | 0.9391 | 0.7750 |
| **Support** | 2,000 | 275 | 224 | 854 | 581 | 66 |

- **Macro averages** (equal weight per class)

| Summary Metric | Value | Comparison to Test Set |
| --- | --- | --- |
| **Accuracy** | 0.9255 | +0.0260 |
| **Macro F1** | 0.8865 | +0.0738 |

---

### **Qualitative Observations**

**Strong Performance**

- Excellent overall accuracy (**89.95%**) with especially high results for **joy (F1: 0.9481)**, **sadness (F1: 0.9412)**, and **anger (F1: 0.9250)**.  
- ~~Robust generalization to unseen data: **CARER accuracy 92.55%**, indicating strong transfer beyond the training domain.~~ **Withdrawn:** CARER is 83% of the training data, so 92.55% measures in-distribution fit, not transfer. See the note above that table. No evidence of generalisation beyond the training domain is available for this model.

**Class Imbalance Impact**

- **Neutral** underperforms (F1: 0.5215; Precision: 0.4438) due to low representation and subtle distinction from low-emotion text.  
- **Disgust** has high recall (0.9157) but low precision (0.6064), suggesting overprediction and overlap with *anger*.  
- **Surprise** moderately strong (F1: 0.7826) and consistent across datasets.

**Common Confusions**

- *Disgust ↔ Anger*: shared lexical markers (negative tone).  
- *Fear ↔ Sadness*: overlap in emotional semantics (uncertainty, vulnerability).  
- *Neutral misclassifications*: ambiguous or subtle sentences often default to dominant emotions.

**Confidence Analysis**

| Metric | Test Set | CARER |
| --- | --- | --- |
| Avg. Confidence (Correct) | 0.8873 | 0.9176 |
| Avg. Confidence (Incorrect) | 0.4275 | 0.7082 |

A large confidence gap shows the model is well-calibrated and suitable for threshold-based decisions.

> **Caveat on these confidences.** This card describes the classifier head as
> sigmoid ("7-dimensional sigmoid probabilities"), but
> `src/vea/stages/emotion_en.py` applies **softmax** and takes argmax. The
> predicted label is the same either way; the confidence value is not. So the
> numbers above do not describe what the pipeline records — and stage 8 discards
> predictions below 0.25 confidence while stage 9 exports the value, which makes
> the difference visible in the output rather than academic. The retrained
> checkpoint is trained with cross-entropy, matching the softmax the pipeline
> applies, and `train.py` reports the same correct/incorrect confidence gap on
> its own validation set so the comparison is like for like.

**Dataset-Specific Notes**

- ~~CARER performance exceeds internal test performance (+2.6% accuracy, +7.4% macro F1).~~ It does, and that is the contamination signal, not a strength — see the retraction above.
- Cleaner emotion delineation in CARER data improves consistency, especially for *joy* and *sadness*.  
- Absence of *disgust* and *neutral* classes reduces confusion and inflates overall metrics.

---

### **Synthetic Stress Test Performance (Adversarial Evaluation)**

To assess robustness, the model was tested on **5,000 synthetically generated samples** containing sarcasm, negation, mixed emotions, typos/slang, and ambiguous phrasing.

| Metric | Overall | Anger | Disgust | Fear | Joy | Neutral | Sadness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Precision** | 0.4802* | 0.0000 | 0.0763 | 1.0000 | 0.7858 | 0.2938 | 0.7251 |
| **Recall** | 0.3876* | 0.0000 | 1.0000 | 0.0545 | 0.7324 | 0.2776 | 0.2609 |
| **F1-Score** | 0.2787* | 0.0000 | 0.1418 | 0.1033 | 0.7582 | 0.2854 | 0.3837 |
| **Support** | 5,000 | 1,230 | 227 | 863 | 1,252 | 508 | 920 |

- **Macro averages**

| Summary Metric | Value |
| --- | --- |
| **Accuracy** | 0.3144 |
| **Macro F1** | 0.2787 |

---

### **Performance by Outlier Type**

| Outlier Type | Accuracy | F1-Score | Samples |
| --- | --- | --- | --- |
| **None** (control) | 0.3147 | 0.2640 | 3,502 |
| **Emoji-heavy** | 0.7273 | 0.6527 | 187 |
| **Typos/Slang** | 0.7287 | 0.6644 | 188 |
| **Subtle Emotion** | 0.5294 | 0.4286 | 187 |
| **Extreme Intensity** | 0.2674 | 0.2000 | 187 |
| **Ambiguous** | 0.2553 | 0.1356 | 188 |
| **Sarcasm** | 0.0000 | 0.0000 | 187 |
| **Negation** | 0.0000 | 0.0000 | 187 |
| **Mixed Emotions** | 0.0000 | 0.0000 | 187 |

---

### **Critical Stress Test Findings**

**Failures**
- Complete breakdown on **sarcasm**, **negation**, and **mixed emotions** (~0% accuracy).  
- **Disgust overprediction epidemic**: 2,753 incorrect predictions; frequent confusion *anger → disgust* and *fear → disgust*.  
- **Anger recognition collapse**: ~0% recall on adversarial inputs.

**Moderate Weaknesses**
- **Fear detection** nearly fails (5.45% recall).  
- **Domain-specific degradation**: legal (81.6% error), medical (79.8%), and technical (78.2%) texts show severe decline.  
- **Confidence drift**: correct predictions’ confidence falls to 0.5133; incorrect remains 0.5321 → poor calibration under shift.

**Strengths**
- Stable under surface noise: emoji-heavy and typo texts retain ~73% accuracy.  
- *Joy* remains resilient (F1: 0.7582), showing semantic stability.

---

### **Key Takeaways for Deployment**
 
1. **Domain limitations** — avoid deployment in legal, medical, or technical contexts without fine-tuning.  
2. **Sarcasm/Negation detection** — add preprocessing or adversarial retraining to handle these cases.  
3. **Confidence thresholds** — use ≥ 0.7 for clean text; model is well-calibrated in-domain. 

## **Explainability and Transparency**

**Techniques Used:**  
- Gradient × Input (baseline attribution)  
- Integrated Gradients (IG)  
- Conservative Propagation (LRP)  
- Input Perturbation (masking analysis)

### **Summary of Findings**
- Model attends to emotionally meaningful tokens (*scared, bleak, perfectly, ugh*)  
- Punctuation marks contribute strongly to emotional intensity (*!, ?*)  
- IG and CP-LRP show high token-level agreement (4/7 emotions ≥ 90 %)  
- Two reasoning patterns identified:  
  - **Token-dependent** (explicit words → *fear*, *disgust*)  
  - **Distributed-context** (implicit emotions → *joy*, *sadness*)  
- Perturbation tests confirm robustness for explicit emotions  

### **Limitations**
- Diffuse attributions (contextual reasoning) reduce clarity  
- Punctuation bias may reflect dataset artifacts  
- Small XAI sample (18 sentences) limits generalization  

*(Include token-attribution bar charts or attention heatmaps as visuals if available.)*

---

## **Recommendations for Use**

| Aspect | Guidance |
|--------|-----------|
| **Preprocessing** | Input text must be normalized (no timestamps, no markup) |
| **Input Length** | ≤ 512 tokens per instance |
| **Output** | Multi-label probabilities for 7 emotions |
| **Deployment** | Integrate as the final step in the Content Intelligence Agency pipeline |
| **Operational Risks** | Bias toward English phrasing; misclassification of sarcasm or irony |
| **Future Improvements** | Fine-tune with balanced multilingual corpora; add attention regularization; scale XAI dataset |

---

## **Sustainability Considerations**

| Factor | Details                                                                                                                  |
|---------|--------------------------------------------------------------------------------------------------------------------------|
| **Hardware** | NVIDIA RTX A6000 (52 GB VRAM, Ampere)                                                                                    |
| **Training Duration** | ~2-3 hours (single-GPU fine-tuning)                                                                                      |
| **Power Draw** | ≈ 270 W · h average                                                                                                      |
| **Estimated Carbon Footprint** | ≈ 0.26 kg CO₂ eq *(using ML CO₂ Impact Calculator)*                                                       |
| **Optimization** | Reduced epochs (3), small batch size (to limit compute), reused pretrained weights                                       |
| **Recommendations** | Use energy-efficient GPUs (A6000 / L40S), schedule training in off-peak periods, monitor model reuse to avoid retraining |

Overall, the environmental impact of this model is **low** due to the short fine-tuning time, moderate model size, and shared academic hardware resources.

---

## **Limitations and Ethical Considerations**

### **Model Limitations**

1. **Data Imbalance** – The dataset is skewed toward *neutral* and *happiness* labels, which may cause the model to underperform on low-frequency emotions like *fear* and *disgust*.  
2. **Language Bias** – Data is normalized to English. This may lead to weaker performance on translated text, especially for idiomatic or culturally specific expressions.  
3. **Punctuation Overweighting** – XAI analyses showed an overreliance on punctuation (`!`, `?`) as emotional markers, which can distort predictions on text without expressive punctuation.  
4. **Context Dependence** – Distributed contextual reasoning makes explanations diffuse and harder to interpret, reducing transparency for end users.  
5. **Limited Evaluation Scope** – The XAI evaluation used only 18 sentences; results may not fully generalize to broader datasets or real-world video transcripts.  

### **Ethical Considerations**

1. **Misinterpretation Risk** – Emotional labeling is subjective; predictions should not be treated as psychological truth or used to infer user intentions or states.  
2. **Cultural Sensitivity** – Emotional expression differs across cultures and languages. The model may misclassify emotionally neutral statements from certain linguistic or cultural groups.  
3. **Data Provenance and Consent** – While the dataset sources were public (YouTube, TV transcripts), ethical AI development requires transparency about data origin and the absence of personally identifiable content.  
4. **Bias Propagation** – Translation artifacts or imbalance in data sources can embed societal or linguistic bias into predictions. Regular fairness audits are recommended.  
5. **Responsible Deployment** – The model should only be used for analytical or creative media purposes, not for surveillance, profiling, or decision-making about individuals.  

### **Mitigation and Future Work**

- Expand multilingual training and collect culturally diverse emotion samples.  
- Introduce *bias monitoring dashboards* for emotion-by-demographic analysis.  
- Apply *attention regularization* or *rationale supervision* to improve interpretability.  
- Conduct systematic user studies to assess fairness, explainability, and trustworthiness in real-world use.

---



## **References**

- Mitchell, M., Wu, S., Van Hasselt, H., et al. (2019). *Model Cards for Model Reporting*. FAT* Conference.  
- Ali, A., et al. (2022). *XAI for Transformers: Better Explanations through Conservative Propagation*.  
- Ekman, P. & Friesen, W. (1971). *Constants Across Cultures in the Face and Emotion*.
- Data set: "super-emotion" by cirimus. https://huggingface.co/datasets/cirimus/super-emotion



## **References**

- Mitchell, M., Wu, S., Van Hasselt, H., et al. (2019). *Model Cards for Model Reporting*. FAT* Conference.  
- Ali, A., et al. (2022). *XAI for Transformers: Better Explanations through Conservative Propagation*.  
- Ekman, P. & Friesen, W. (1971). *Constants Across Cultures in the Face and Emotion*.  

**Datasets and Models**
- Dataset: [ru-izard-emotions (Djacon)](https://huggingface.co/datasets/Djacon/ru-izard-emotions)  
- Dataset: [super-emotion (cirimus)](https://huggingface.co/datasets/cirimus/super-emotion)  
- Model reference: [emotion-english-distilroberta-base (j-hartmann)](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)  
- Repository: [multilingual_va_prediction (gmendes9)](https://github.com/gmendes9/multilingual_va_prediction)  

**Related Group Deliverables**
- [Task 9 – Error Analysis (GitHub)](https://github.com/BredaUniversityADSAI/fae2-nlpr-group-group-14-1/tree/dev/Task9)  
- [Task 10 – Explainable AI Analysis (GitHub)](https://github.com/BredaUniversityADSAI/fae2-nlpr-group-group-14-1/tree/dev/Task10)
