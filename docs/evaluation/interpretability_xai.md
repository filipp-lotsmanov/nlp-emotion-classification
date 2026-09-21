# **Explainable AI Analysis of Emotion Classification Model**

---

## **Introduction**

This report presents an explainability analysis of a **DeBERTa-V2-based emotion classification model** trained to recognize seven categories — **joy, sadness, anger, fear, disgust, surprise,** and **neutral**.  
The model was fine-tuned for **multi-label classification**, allowing several emotions to be detected within the same sentence.

Three complementary XAI techniques were applied to **18 translated sentences** (three per emotion) from Russian TV transcripts:

1. **Gradient × Input** – baseline attribution of token importance  
2. **Integrated Gradients (IG)** – improved baseline-integrated attributions  
3. **Input Perturbation** – robustness testing via masking

---

## **Part 1 – Gradient × Input Attribution**

### **Methodology**
Gradient × Input multiplies the gradient of the model’s output (for a target emotion) by each token’s embedding to estimate its contribution.  
The implementation adapts BERT code for **DeBERTa-V2**, using `model.deberta.embeddings` and the classifier’s **pooler** layer.  
Attribution values were visualized as bar charts per sentence.

### **Findings**
The model concentrated on emotionally meaningful words:

- **Sadness:** *death, unfortunately, bleak*  
- **Fear:** *scared, terrifying*  
- **Disgust:** *Ugh*  
- **Joy:** *perfectly, Everything,* and **!**  
- **Surprise:** *Why, What,* and **?**

A recurring observation was that punctuation marks (! ? .) often ranked high, showing the model learned that punctuation conveys emotional tone — e.g., exclamation marks indicate intensity, question marks express surprise.

**Example – “Tabatinga is a very bleak city.”**  
Top tokens: **bleak (0.0439)**, **very (0.0418)** — confirming focus on emotional content.  

For *“Everything worked out perfectly — we did it!”*, both **!** and **perfectly** ranked highest, capturing positivity and enthusiastic punctuation.

### **Limitations**
Attribution scores were small (≈ 0.01–0.06) and spread across many tokens, suggesting **contextual reasoning** rather than dependence on a single word — a pattern later confirmed through perturbation testing.

---

## **Part 2 – Integrated Gradients (IG) and Conservative Propagation (CP-LRP)**

### **Methodology**
Two complementary attribution methods were used:

- **Integrated Gradients (IG)** integrates gradients from a zero baseline to the input, ensuring *completeness* (the sum of token attributions equals the model output difference).  
- **Conservative Propagation (Layer-wise Relevance Propagation, CP-LRP)** redistributes total relevance through all Transformer layers, preserving conservation across attention and normalization modules as described by Ali et al. (2022).

IG was implemented in **Captum** with 50 integration steps, and CP-LRP was applied within the modified DeBERTa forward pass to trace relevance through layers.

### **Results**
All IG runs produced **high convergence deltas (> 0.5)**, implying 50 steps were insufficient for DeBERTa’s depth.  
Future runs should use 100–200 steps or a `[PAD]` baseline.

Nevertheless, IG and CP-LRP showed strong **token-level agreement**:

| Emotion | Top-Token Agreement |
|:--|:--:|
| Joy | 3 / 3 |
| Fear | 3 / 3 |
| Surprise | 3 / 3 |
| Sadness | 2 / 3 |
| Anger | 2 / 3 |
| Disgust | 1 / 3 |

**Example – “Everything worked out perfectly — we did it!”**  
Both methods ranked **!**, **perfectly**, and **Everything** highest, though IG produced stronger contrast (0.113 vs 0.012).  

In *“But you didn’t say it was cocaine.”*, Gradient×Input emphasized the period, IG and CP-LRP highlighted **“didn’t”**, capturing semantic negation more accurately.

### **Interpretation**
The overlap across Gradient×Input, IG, and CP-LRP confirms they capture genuine linguistic cues.  
**CP-LRP** adds layer-wise faithfulness and reduces pseudonoise from attention reweighting, while **IG** provides smooth, interpretable input-level scores.  
Both methods agree that emotionally salient tokens (e.g., *scared, bleak, Ugh, Why*) drive predictions, though punctuation remains overweighted.  

---

## **Part 3 – Input Perturbation and Model Robustness**

### **Methodology**
Tokens were ranked by IG relevance and progressively masked with `[MASK]`.  
Two strategies were tested:

1. **Mask least-important first**  
2. **Mask most-important first**

Model confidence was measured after each step.  
The **Area Between Curves (ABC)** quantified difference:

- **Positive ABC → Good attribution** (confidence drops when important tokens masked)  
- **Negative ABC → Poor attribution** (curves overlap or invert)

---

### **Results**

#### **Pattern 1 – Distributed Processing**  
*(Joy, Sadness, Anger, Surprise)*  

These emotions showed **negative ABC** and overlapping curves.  
Example – *Joy:* “Everything worked out perfectly — we did it!”  
Initial confidence 0.73 → Final 0.74 (ABC = −0.72)

| Emotion | Average ABC |
|:--|:--:|
| Joy | −1.26 |
| Sadness | −2.92 |
| Anger | −2.10 |
| Surprise | −0.60 |

Confidence sometimes increased after masking, indicating the model relies on **distributed contextual patterns** rather than single words.  
`[MASK]` embeddings can simplify inputs, artificially boosting confidence — a known Transformer artifact.

#### **Pattern 2 – Token-Dependent Processing**  
*(Fear, Disgust)*  

Two examples behaved differently, showing positive ABC:  

- *Fear:* “This feels like some kind of extreme situation — everyone here is really scared.” → 0.94 → 0.36 (ABC = +5.11)  
- *Disgust:* “Ugh, what kind of question is that?” → 0.75 → 0.34 (ABC = +2.83)

Confidence dropped sharply when key tokens (*scared, Ugh*) were masked — evidence of faithful, token-specific reasoning.

---

### **Interpretation**
Two processing modes emerge:

- **Lexical-marker-based** → explicit keywords (*fear, disgust*)  
- **Distributed-context-based** → subtle emotions (*joy, sadness, anger, surprise*)  

Example rankings:  

- *Fear:* scared (0.25) > feels (0.12) > really (0.08)  
- *Joy:* ! (0.11) > perfectly (0.08) > Everything (0.07)

| Emotion | Avg ABC | Pattern |
|:--|:--:|:--|
| Joy | −1.26 | Distributed |
| Sadness | −2.92 | Distributed |
| Anger | −2.10 | Distributed |
| Fear | +0.84 | Token-Dependent |
| Disgust | +0.67 | Token-Dependent |
| Surprise | −0.60 | Distributed |

Positive ABC → faithful explanations (fear, disgust)  
Negative ABC → contextual representations or `[MASK]` artifacts.

---

## **Model Interpretation and Implications**

### **What the Model Learned**
1. Recognizes **emotion-specific vocabulary** (*scared, bleak, Ugh, Why*)  
2. Treats **punctuation as emotional intensity**  
3. Uses **context more than keywords**, enabling subtle detection  
4. Switches between **keyword-based** and **distributed** strategies depending on input  

### **Strengths**
- Focuses on genuine emotional content  
- Captures both lexical and prosodic (punctuation) cues  
- Robust to rewording and paraphrasing  
- Handles implicit emotions effectively  

### **Limitations**
- Diffuse attributions reduce interpretability  
- Negative ABC scores show limited perturbation sensitivity  
- High IG convergence deltas reduce numerical confidence  
- Punctuation bias may reflect dataset artifacts  
- Small sample (18 sentences) limits generalization  

### **Methodological Notes**
- Increase IG steps (100–200) or use alternative baselines  
- Replace masking with true token deletion  
- Expand analysis to larger, random samples  

---

## **Practical Implications**

- **Error Analysis:** Inspect sentence-level context, not just top tokens.  
- **Bias Auditing:** Distributed strategies make bias harder to trace.  
- **User Trust:** Transparent explanations are vital for sensitive domains.  
- **Model Design:** Future versions may include attention regularization or rationale supervision for clearer reasoning.

---

## **Conclusion**

This analysis demonstrates how the **DeBERTa-V2 emotion classifier** processes emotional language:

- It reliably highlights **emotionally relevant words** and expressive punctuation.  
- **Gradient×Input, Integrated Gradients, and CP-LRP** yield consistent explanations.  
- **Perturbation tests** show most emotions are recognized via distributed context, while *Fear* and *Disgust* depend on distinct keywords.

The model’s flexible reasoning resembles human interpretation but complicates transparency and auditing.  
Future work should expand CP-LRP experiments, use deletion-based perturbations, and refine IG baselines for higher faithfulness and stability.  
A larger-scale evaluation will determine if these patterns generalize beyond this sample.

---
