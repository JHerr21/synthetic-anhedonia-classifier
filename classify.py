import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import json
import re
import string

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

POSITIVE_WORDS = {
    "love", "loved", "enjoy", "enjoyed", "enjoying", "fun", "happy", "great",
    "excited", "glad", "nice", "good", "like", "liked", "looking", "wonderful",
}


def patient_turns(transcript: str) -> list[str]:
    turns = []
    for line in transcript.splitlines():
        line = line.strip().replace("**", "")  # in case the model uses markdown
        if line.startswith("Patient:"):
            turns.append(line.split(":", 1)[1].strip())
    return turns


def extract_features(transcript: str) -> dict:
    turns = patient_turns(transcript)
    text = " ".join(turns)
    words = [w.lower().strip(string.punctuation) for w in text.split()]
    words = [w for w in words if w]
    n = max(len(words), 1)
    return {
        "words_per_turn": len(words) / max(len(turns), 1),
        "short_pauses": len(re.findall(r"\[pause\]", text)),
        "long_pauses": len(re.findall(r"\[long pause\]", text)),
        "positive_word_rate": sum(w in POSITIVE_WORDS for w in words) / n,
    }


OUT_DIR = "output"
SEED = 42


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open("data/transcripts.jsonl", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    X = pd.DataFrame([extract_features(r["transcript"]) for r in records])
    y = np.array([r["label"] for r in records])
    print(f"{len(y)} transcripts, {y.sum()} high / {len(y) - y.sum()} low")

    X.assign(id=[r["id"] for r in records], label=y).to_csv(
        os.path.join(OUT_DIR, "features.csv"), index=False)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    auc, perm_scores, p_value = permutation_test_score(
        pipeline, X, y, scoring="roc_auc", cv=cv,
        n_permutations=1000, random_state=SEED, n_jobs=-1,
    )
    print(f"AUC = {auc:.3f}, permutation p = {p_value:.4f}")

    results = {
        "auc_cv_mean": round(float(auc), 4),
        "permutation_p": round(float(p_value), 4),
        "n_permutations": 1000,
        "null_auc_mean": round(float(perm_scores.mean()), 4),
        "null_auc_sd": round(float(perm_scores.std()), 4),
        "n_samples": int(len(y)),
        "cv": "StratifiedKFold(5, shuffle=True)",
        "seed": SEED,
        "features": list(X.columns),
    }
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Coefficients from a fit on all data: descriptive only, not an evaluation
    pipeline.fit(X, y)
    coefs = pd.Series(pipeline.named_steps["clf"].coef_[0], index=X.columns)
    coefs.sort_values().to_csv(
        os.path.join(OUT_DIR, "coefficients.csv"), header=["coefficient"])
    print("Standardized coefficients:\n", coefs.sort_values())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(perm_scores, bins=30, color="gray", alpha=0.8, label="Shuffled labels")
    ax.axvline(auc, color="red", linewidth=2, label=f"Observed AUC = {auc:.3f}")
    ax.set_xlabel("Cross-validated AUC")
    ax.set_ylabel("Count")
    ax.set_title(f"Permutation test (n=1000), p = {p_value:.4f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "permutation_null.png"), dpi=150)

    print(f"Saved results to {OUT_DIR}/")


if __name__ == "__main__":
    main()