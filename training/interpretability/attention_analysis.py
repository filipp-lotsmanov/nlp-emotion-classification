"""Integrated-gradients attributions and perturbation curves for the English
emotion classifier.

    uv run python training/interpretability/attention_analysis.py \\
        "$VEA_MODELS_DIR/emotion-en-deberta" \\
        --out-dir docs/evaluation/figures/xai \\
        --out-json docs/evaluation/xai_report.json

Why this exists in this form: the original took its checkpoint from a hardcoded
absolute path on one Windows machine and called plt.show(), so nobody could
rerun it and the figures in docs/evaluation/interpretability_xai.md could not be
traced to a checkpoint. Every figure this writes names the checkpoint it came
from in the JSON beside it.

The activation applied to the logits is a flag, not a constant. See --activation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from captum.attr import IntegratedGradients  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

#: Applied to the classifier logits before attribution and before every
#: confidence read.
#:
#: The original used sigmoid, on the premise that the head is multi-label. It
#: is not: train.py sets problem_type="single_label_classification" and trains
#: with CrossEntropyLoss, and src/vea/stages/emotion_en.py applies softmax and
#: takes argmax at serving time. Attributions computed through sigmoid
#: therefore explain an activation the model was never trained or served with.
#: Softmax is the default here for that reason. Pass --activation sigmoid only
#: to reproduce the pre-2026 figures in docs/evaluation/interpretability_xai.md.
DEFAULT_ACTIVATION = "softmax"

_ACTIVATIONS = {
    "sigmoid": torch.sigmoid,
    "softmax": lambda logits: torch.softmax(logits, dim=-1),
}


def get_attributions_for_perturbation(sentence, model, tokenizer, activation, n_steps):
    model.eval()

    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    embeddings = model.deberta.embeddings(input_ids)
    baseline_embeddings = torch.zeros_like(embeddings)

    def forward_func(embeddings_input):
        outputs = model.deberta(inputs_embeds=embeddings_input, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state
        pooled = model.pooler(hidden_state)
        pooled = model.dropout(pooled)
        logits = model.classifier(pooled)
        return activation(logits)

    with torch.no_grad():
        probs = forward_func(embeddings)
    target_idx = torch.argmax(probs[0]).item()

    ig = IntegratedGradients(forward_func)
    attributions = ig.attribute(
        inputs=embeddings, baselines=baseline_embeddings, target=target_idx, n_steps=n_steps
    )

    attr_scores = attributions.norm(dim=-1).squeeze(0).detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    return attr_scores, tokens, target_idx, input_ids, attention_mask


def perturbation_analysis(sentence, model, tokenizer, activation, n_steps):
    print(f"\nAnalyzing: '{sentence}'")

    attr_scores, tokens, target_emotion, input_ids, attention_mask = (
        get_attributions_for_perturbation(sentence, model, tokenizer, activation, n_steps)
    )

    model.eval()
    with torch.no_grad():
        initial_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        initial_probs = activation(initial_outputs.logits)
        initial_conf = initial_probs[0, target_emotion].item()

    maskable_indices = [i for i, tok in enumerate(tokens) if tok not in ["[CLS]", "[SEP]", "[PAD]"]]

    if len(maskable_indices) == 0:
        print("Warning: No maskable tokens found!")
        return None

    sorted_indices = sorted(maskable_indices, key=lambda i: abs(attr_scores[i]))

    mask_token_id = tokenizer.mask_token_id
    confidences_least_first = [initial_conf]

    for num_masked in range(1, len(sorted_indices) + 1):
        masked_ids = input_ids.clone()
        for idx in sorted_indices[:num_masked]:
            masked_ids[0, idx] = mask_token_id

        with torch.no_grad():
            outputs = model(input_ids=masked_ids, attention_mask=attention_mask)
            probs = activation(outputs.logits)
            conf = probs[0, target_emotion].item()
            confidences_least_first.append(conf)

    confidences_most_first = [initial_conf]

    for num_masked in range(1, len(sorted_indices) + 1):
        masked_ids = input_ids.clone()
        for idx in sorted_indices[-num_masked:]:
            masked_ids[0, idx] = mask_token_id

        with torch.no_grad():
            outputs = model(input_ids=masked_ids, attention_mask=attention_mask)
            probs = activation(outputs.logits)
            conf = probs[0, target_emotion].item()
            confidences_most_first.append(conf)

    return {
        "sentence": sentence,
        "tokens": tokens,
        "sorted_indices": sorted_indices,
        "maskable_indices": maskable_indices,
        "target_emotion": target_emotion,
        "initial_confidence": initial_conf,
        "conf_least_first": confidences_least_first,
        "conf_most_first": confidences_most_first,
        "attr_scores": attr_scores,
    }


def plot_perturbation_curves(results, out_dir):
    sentence = results["sentence"]
    conf_least = results["conf_least_first"]
    conf_most = results["conf_most_first"]
    emotion_idx = results["target_emotion"]

    plt.figure(figsize=(12, 6))
    plt.plot(
        range(len(conf_least)),
        conf_least,
        label="Remove Least Important First",
        marker="o",
        linewidth=2,
        markersize=6,
        color="green",
    )
    plt.plot(
        range(len(conf_most)),
        conf_most,
        label="Remove Most Important First",
        marker="s",
        linewidth=2,
        markersize=6,
        color="red",
    )

    plt.xlabel("Number of Tokens Masked", fontsize=12)
    plt.ylabel("Model Confidence", fontsize=12)
    plt.title(f'Perturbation Analysis - Emotion {emotion_idx}\n"{sentence}"', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    plt.tight_layout()

    safe_filename = (
        sentence[:30]
        .replace(" ", "_")
        .replace("!", "")
        .replace("?", "")
        .replace(",", "")
        .replace("—", "-")
        .replace("…", "")
    )
    out_path = out_dir / f"perturbation_{safe_filename}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    abc_score = float(np.sum(np.array(conf_least) - np.array(conf_most)))

    print(f"Initial confidence: {results['initial_confidence']:.4f}")
    print(f"Final conf least→most: {conf_least[-1]:.4f}")
    print(f"Final conf most→least: {conf_most[-1]:.4f}")
    print(f"Area Between Curves (ABC): {abc_score:.4f}")

    if abc_score > 1.0:
        print("Good attribution quality")
    elif abc_score > 0.5:
        print("Moderate attribution quality")
    else:
        print("Poor attribution quality")

    if len(conf_most) > 1:
        midpoint = len(conf_most) // 2
        drop_most = results["initial_confidence"] - conf_most[min(midpoint, len(conf_most) - 1)]
        drop_least = results["initial_confidence"] - conf_least[min(midpoint, len(conf_least) - 1)]

        if drop_most > 0.3 and drop_least < 0.1:
            print("Sharp drop when removing important tokens")
        else:
            print("Model may rely on distributed features")

    return abc_score


def analyze_token_importance(results):
    tokens = results["tokens"]
    attr_scores = results["attr_scores"]
    maskable = results["maskable_indices"]

    clean_tokens = [tokens[i] for i in maskable]
    clean_attrs = [attr_scores[i] for i in maskable]

    sorted_pairs = sorted(zip(clean_tokens, clean_attrs), key=lambda x: abs(x[1]), reverse=True)
    for i, (tok, score) in enumerate(sorted_pairs[:5], 1):
        print(f"  {i}. '{tok}': {score:.4f}")
    if len(sorted_pairs) > 3:
        for i, (tok, score) in enumerate(sorted_pairs[-3:], 1):
            print(f"  {i}. '{tok}': {score:.4f}")


TEST_SENTENCES = {
    "joy": [
        "Everything worked out perfectly — we did it!",
        "Come on, it's all fine!",
        "Look, my friends — the border! I can see the sign, we're leaving Brazil!",
    ],
    "sadness": [
        "We also had to witness death very often.",
        "He pulled out an axe — and unfortunately, that's how these people end up…",
        "Tabatinga is a very bleak city.",
    ],
    "anger": [
        "Hide it — quickly, quickly, hide the camera!",
        "If there's aggression or someone tries to open the door — we get out immediately!",
        "This keeps happening all the time!",
    ],
    "fear": [
        "This feels like some kind of extreme situation — everyone here is really scared.",
        "What a terrifying place.",
        "Honestly, I've got chills running down my spine.",
    ],
    "disgust": [
        "But you didn't say it was cocaine.",
        "I don't need you giving me cocaine!",
        "Ugh, what kind of question is that?",
    ],
    "surprise": [
        "Oh my God, what a question! Don't joke like that.",
        "Why did you say it was flour?",
        "We suddenly sped up like crazy!",
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="directory holding the model and tokenizer")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/evaluation/figures/xai"))
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--n-steps", type=int, default=50, help="integrated-gradients steps")
    parser.add_argument("--activation", choices=sorted(_ACTIVATIONS), default=DEFAULT_ACTIVATION)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.checkpoint.is_dir():
        parser.error(f"checkpoint directory not found: {args.checkpoint}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    activation = _ACTIVATIONS[args.activation]
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint, local_files_only=True
    ).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model.eval()

    report: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "activation": args.activation,
        "n_steps": args.n_steps,
        "device": args.device,
        "emotions": {},
    }

    for emotion, sentences in TEST_SENTENCES.items():
        print(f"\n{'=' * 70}\nEMOTION: {emotion.upper()}\n{'=' * 70}")
        per_sentence = []

        for i, sentence in enumerate(sentences, 1):
            print(f"\n[{i}/{len(sentences)}] {sentence}")
            results = perturbation_analysis(sentence, model, tokenizer, activation, args.n_steps)
            if results is None:
                continue
            abc = plot_perturbation_curves(results, args.out_dir)
            analyze_token_importance(results)
            per_sentence.append(
                {
                    "sentence": sentence,
                    "predicted_class": int(results["target_emotion"]),
                    "initial_confidence": float(results["initial_confidence"]),
                    "abc": abc,
                }
            )

        abc_scores = [s["abc"] for s in per_sentence]
        report["emotions"][emotion] = {
            "sentences": per_sentence,
            "mean_abc": float(np.mean(abc_scores)) if abc_scores else None,
        }
        if abc_scores:
            print(f"{emotion.capitalize():12s}: mean ABC = {np.mean(abc_scores):.3f}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out_json}")
    print(f"figures in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
