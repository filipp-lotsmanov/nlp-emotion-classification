import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = r"C:\Users\Filip Letmanov\Block A\personal_repository\Task10"

model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model.eval()


def gradient_x_input_correct(sentence, model, tokenizer, target_emotion_idx=None):

    model.eval()
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    embeddings = model.deberta.embeddings(input_ids)
    embeddings.requires_grad_()
    embeddings.retain_grad()

    outputs = model.deberta(inputs_embeds=embeddings, attention_mask=attention_mask)

    hidden_state = outputs.last_hidden_state

    pooled = model.pooler(hidden_state)

    pooled = model.dropout(pooled)
    logits = model.classifier(pooled)
    probs = torch.sigmoid(logits)

    if target_emotion_idx is None:
        target_emotion_idx = torch.argmax(probs[0]).item()

    model.zero_grad()
    probs[0, target_emotion_idx].backward()
    grads = embeddings.grad
    attribution = (grads * embeddings).norm(dim=-1)
    attribution = attribution[0].detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    return attribution, tokens, target_emotion_idx


def visualize_attribution(attribution, tokens, sentence, emotion_idx, method_name="Gradient×Input"):
    relevant_indices = [i for i, tok in enumerate(tokens) if tok not in ["[CLS]", "[SEP]", "[PAD]"]]

    clean_tokens = [tokens[i] for i in relevant_indices]
    clean_attr = [attribution[i] for i in relevant_indices]

    plt.figure(figsize=(14, 5))
    colors = ["red" if x < 0 else "green" for x in clean_attr]
    plt.bar(range(len(clean_tokens)), np.abs(clean_attr), color=colors)
    plt.xticks(range(len(clean_tokens)), clean_tokens, rotation=45, ha="right")
    plt.title(f"{method_name} - Emotion {emotion_idx}: {sentence}")
    plt.xlabel("Tokens")
    plt.ylabel("Attribution Score (absolute value)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    safe_filename = (
        sentence[:30]
        .replace(" ", "_")
        .replace("!", "")
        .replace("?", "")
        .replace(",", "")
        .replace("—", "-")
    )
    plt.savefig(f"part1_{safe_filename}.png", dpi=300, bbox_inches="tight")
    plt.show()

    k = 5
    top_indices = np.argsort(np.abs(clean_attr))[-k:][::-1]
    print(f"\nTop-{k} most important tokens:")
    for idx in top_indices:
        print(f"  '{clean_tokens[idx]}': {clean_attr[idx]:.4f}")


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

    print("PART 1: GRADIENT × INPUT ATTRIBUTION")

    results = {}

    for emotion, sentences in test_sentences.items():
        print(f"\n{'=' * 70}")
        print(f"EMOTION: {emotion.upper()}")
        print(f"{'=' * 70}")

        results[emotion] = []

        for i, sentence in enumerate(sentences, 1):
            print(f"\n[{i}/3] Analyzing: '{sentence}'")

            attr, tokens, emotion_idx = gradient_x_input_correct(sentence, model, tokenizer)

            visualize_attribution(attr, tokens, sentence, emotion_idx)

            results[emotion].append(
                {
                    "sentence": sentence,
                    "attribution": attr,
                    "tokens": tokens,
                    "predicted_emotion": emotion_idx,
                }
            )

    print("Part 1 Complete! Check the generated PNG files.")
