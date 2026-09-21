import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = r"C:\Users\Filip Letmanov\Block A\personal_repository\Task10"

model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model.eval()


class ConservativePropagation:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def compute_attention_rollout(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.model.deberta(
                input_ids=input_ids, attention_mask=attention_mask, output_attentions=True
            )
        attentions = outputs.attentions
        num_layers = len(attentions)

        print(f"  → Extracted attention from {num_layers} transformer layers")

        attention_matrices = []
        for layer_idx, attention in enumerate(attentions):
            avg_attention = attention.mean(dim=1)
            attention_matrices.append(avg_attention.squeeze(0))
        seq_len = attention_matrices[0].shape[0]
        identity = torch.eye(seq_len).to(attention_matrices[0].device)

        attention_with_residual = []
        for attn in attention_matrices:
            attn_res = (attn + identity) / 2.0
            attention_with_residual.append(attn_res)
        rollout = attention_with_residual[-1]

        for i in range(num_layers - 2, -1, -1):
            rollout = torch.matmul(attention_with_residual[i], rollout)

        print(f"Computed attention rollout through all {num_layers} layers")

        rollout = rollout / (rollout.sum(dim=-1, keepdim=True) + 1e-10)

        return rollout

    def compute_gradient_relevance(self, input_ids, attention_mask, target_idx):
        embeddings = self.model.deberta.embeddings(input_ids)
        embeddings.requires_grad_(True)
        embeddings.retain_grad()

        outputs = self.model.deberta(inputs_embeds=embeddings, attention_mask=attention_mask)

        hidden_state = outputs.last_hidden_state
        pooled = self.model.pooler(hidden_state)
        pooled = self.model.dropout(pooled)
        logits = self.model.classifier(pooled)
        probs = torch.sigmoid(logits)

        self.model.zero_grad()
        target_score = probs[0, target_idx]
        target_score.backward()

        grads = embeddings.grad

        relevance = (grads * embeddings).sum(dim=-1).abs()

        print(f"Computed gradient-based relevance for target class {target_idx}")

        return relevance.squeeze(0)

    def apply_conservative_propagation(self, sentence, target_emotion_idx=None):
        inputs = tokenizer(
            sentence, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        print(f"\n  Processing: '{sentence[:50]}...'")

        if target_emotion_idx is None:
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.sigmoid(outputs.logits)
                target_emotion_idx = torch.argmax(probs[0]).item()
            print(f"Predicted emotion: {target_emotion_idx}")

        attention_rollout = self.compute_attention_rollout(input_ids, attention_mask)

        gradient_relevance = self.compute_gradient_relevance(
            input_ids, attention_mask, target_emotion_idx
        )
        cls_attention = attention_rollout[0, :]

        final_relevance = gradient_relevance * cls_attention

        final_relevance = final_relevance / (final_relevance.sum() + 1e-10)

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        return final_relevance.detach().cpu().numpy(), tokens, target_emotion_idx


def gradient_x_input_baseline(sentence, model, tokenizer, target_emotion_idx=None):

    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    if target_emotion_idx is None:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits)
            target_emotion_idx = torch.argmax(probs[0]).item()

    embeddings = model.deberta.embeddings(input_ids)
    embeddings.requires_grad_(True)
    embeddings.retain_grad()

    outputs = model.deberta(inputs_embeds=embeddings, attention_mask=attention_mask)
    hidden_state = outputs.last_hidden_state
    pooled = model.pooler(hidden_state)
    pooled = model.dropout(pooled)
    logits = model.classifier(pooled)
    probs = torch.sigmoid(logits)

    model.zero_grad()
    probs[0, target_emotion_idx].backward()
    grads = embeddings.grad

    attribution = (grads * embeddings).norm(dim=-1)
    return (
        attribution[0].detach().cpu().numpy(),
        tokenizer.convert_ids_to_tokens(input_ids[0]),
        target_emotion_idx,
    )


def visualize_conservative_propagation(cp_scores, tokens, sentence, emotion_idx):
    relevant_indices = [i for i, tok in enumerate(tokens) if tok not in ["[CLS]", "[SEP]", "[PAD]"]]

    clean_tokens = [tokens[i] for i in relevant_indices]
    clean_attr = [cp_scores[i] for i in relevant_indices]

    plt.figure(figsize=(14, 5))
    plt.bar(range(len(clean_tokens)), clean_attr, color="purple", alpha=0.7)
    plt.xticks(range(len(clean_tokens)), clean_tokens, rotation=45, ha="right")
    plt.title(f"Conservative Propagation - Emotion {emotion_idx}: {sentence}")
    plt.xlabel("Tokens")
    plt.ylabel("Relevance Score (Conservative Propagation)")
    plt.grid(axis="y", alpha=0.3)
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
    plt.savefig(f"part2_cp_{safe_filename}.png", dpi=300, bbox_inches="tight")
    plt.show()

    k = 5
    top_indices = np.argsort(clean_attr)[-k:][::-1]
    print(f"\n  Top-{k} most important tokens (Conservative Propagation):")
    for idx in top_indices:
        print(f"    '{clean_tokens[idx]}': {clean_attr[idx]:.4f}")


def compare_methods_detailed(sentence, model, tokenizer, cp_calculator):
    print("\nRunning Gradient × Input...")
    grad_scores, tokens, emotion_idx = gradient_x_input_baseline(sentence, model, tokenizer)

    print("\nRunning Conservative Propagation...")
    cp_scores, tokens, emotion_idx = cp_calculator.apply_conservative_propagation(
        sentence, target_emotion_idx=emotion_idx
    )

    relevant_indices = [i for i, tok in enumerate(tokens) if tok not in ["[CLS]", "[SEP]", "[PAD]"]]

    clean_tokens = [tokens[i] for i in relevant_indices]
    clean_grad = [grad_scores[i] for i in relevant_indices]
    clean_cp = [cp_scores[i] for i in relevant_indices]

    max_grad = max(clean_grad) if max(clean_grad) > 0 else 1
    max_cp = max(clean_cp) if max(clean_cp) > 0 else 1
    clean_grad_norm = [x / max_grad for x in clean_grad]
    clean_cp_norm = [x / max_cp for x in clean_cp]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].bar(range(len(clean_tokens)), clean_grad_norm, color="green", alpha=0.7)
    axes[0].set_xticks(range(len(clean_tokens)))
    axes[0].set_xticklabels(clean_tokens, rotation=45, ha="right")
    axes[0].set_title(f"Part 1: Gradient × Input - Emotion {emotion_idx}")
    axes[0].set_ylabel("Normalized Attribution")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(range(len(clean_tokens)), clean_cp_norm, color="purple", alpha=0.7)
    axes[1].set_xticks(range(len(clean_tokens)))
    axes[1].set_xticklabels(clean_tokens, rotation=45, ha="right")
    axes[1].set_title(f"Part 2: Conservative Propagation - Emotion {emotion_idx}")
    axes[1].set_ylabel("Normalized Relevance")
    axes[1].set_xlabel("Tokens")
    axes[1].grid(axis="y", alpha=0.3)

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
    plt.savefig(f"part2_comparison_{safe_filename}.png", dpi=300, bbox_inches="tight")
    plt.show()

    k = 3
    top_grad_idx = np.argsort(clean_grad)[-k:]
    top_cp_idx = np.argsort(clean_cp)[-k:]
    agreement = len(set(top_grad_idx) & set(top_cp_idx))

    print(f"Top-{k} agreement: {agreement}/{k} tokens")
    print(f"Gradient×Input top tokens: {[clean_tokens[i] for i in top_grad_idx[::-1]]}")
    print(f"Conservative Propagation top tokens: {[clean_tokens[i] for i in top_cp_idx[::-1]]}")

    grad_top_set = set(top_grad_idx)
    cp_top_set = set(top_cp_idx)

    only_in_grad = grad_top_set - cp_top_set
    only_in_cp = cp_top_set - grad_top_set

    if only_in_grad:
        print(f"\nTokens only in Gradient×Input top-{k}: {[clean_tokens[i] for i in only_in_grad]}")
    if only_in_cp:
        print(
            f"Tokens only in Conservative Propagation top-{k}: {[clean_tokens[i] for i in only_in_cp]}"
        )

    print()


if __name__ == "__main__":
    test_sentences = {
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

    cp_calculator = ConservativePropagation(model)

    for emotion, sentences in test_sentences.items():
        print(f"EMOTION: {emotion.upper()}")

        for i, sentence in enumerate(sentences, 1):
            print(f"\n[{i}/3] Analyzing sentence {i} of 3:")

            cp_scores, tokens, emotion_idx = cp_calculator.apply_conservative_propagation(sentence)

            visualize_conservative_propagation(cp_scores, tokens, sentence, emotion_idx)

    for emotion, sentences in test_sentences.items():
        compare_methods_detailed(sentences[0], model, tokenizer, cp_calculator)

    print("  - part2_cp_*.png: Individual Conservative Propagation attributions (18 files)")
    print("  - part2_comparison_*.png: Method comparisons (6 files)")
