"""Entraînement + évaluation DistilBERT (6 catégories), Hugging Face Transformers.

Détection automatique CUDA : si un GPU est disponible (`torch.cuda.is_available()`), il est
utilisé ; sinon l'entraînement tourne sur CPU (plus lent, aucune métrique GPU n'est inventée
si aucun GPU n'est présent dans l'environnement d'exécution).

Dépendances optionnelles (voir `requirements-dl.txt`, non installées par défaut car lourdes) :
    torch, transformers, datasets, accelerate

Usage (depuis la racine du projet) :
    python -m llmfw.training.distilbert --epochs 3
    python -m llmfw.training.distilbert --epochs 1 --max-train-samples 500   # run rapide / sanity check
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from llmfw.config import Settings, get_settings
from llmfw.evaluation.metrics import compute_multiclass_metrics
from llmfw.evaluation.plots import plot_confusion_matrix_multiclass

MODEL_NAME = "distilbert-base-uncased"
LABELS = ["benign", "jailbreak", "pii_exfiltration", "prompt_injection",
          "rag_prompt_injection", "secret_exfiltration"]


def _require_torch():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "torch/transformers ne sont pas installés. Installez-les avec :\n"
            "    pip install -r requirements-dl.txt\n"
            "(paquets volumineux, non installés par défaut)."
        ) from exc


def load_split(data_dir: Path, split: str) -> pd.DataFrame:
    return pd.read_csv(data_dir / split / f"{split}.csv")


def train_and_evaluate(settings: Settings, epochs: int = 3, batch_size: int = 16,
                       max_train_samples: int | None = None, learning_rate: float = 2e-5) -> dict:
    _require_torch()
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else None

    seed = settings.random_seed
    train = load_split(settings.data_dir, "train")
    val = load_split(settings.data_dir, "validation")
    test = load_split(settings.data_dir, "test")
    if max_train_samples is not None:
        train = train.groupby("category", group_keys=False).apply(
            lambda g: g.sample(min(len(g), max(1, max_train_samples // len(LABELS))), random_state=seed)
        ).reset_index(drop=True)

    label2id = {lab: i for i, lab in enumerate(LABELS)}
    id2label = {i: lab for lab, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def to_hf(df: pd.DataFrame) -> Dataset:
        ds = Dataset.from_pandas(pd.DataFrame({"text": df["text"], "label": df["category"].map(label2id)}))
        return ds.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=256), batched=True)

    train_ds, val_ds, test_ds = to_hf(train), to_hf(val), to_hf(test)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=id2label, label2id=label2id
    ).to(device)

    out_dir = settings.models_dir / "distilbert"
    out_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        seed=seed,
        report_to=[],
        no_cuda=(device == "cpu"),
    )

    def compute_metrics_cb(eval_pred):
        from sklearn.metrics import accuracy_score, f1_score
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {"accuracy": accuracy_score(labels, preds), "f1_macro": f1_score(labels, preds, average="macro")}

    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer), compute_metrics=compute_metrics_cb,
    )

    t0 = time.perf_counter()
    train_result = trainer.train()
    train_time_s = time.perf_counter() - t0

    val_metrics_hf = trainer.evaluate(val_ds)

    test_logits = trainer.predict(test_ds).predictions
    test_proba = torch.softmax(torch.tensor(test_logits), dim=1).numpy()
    test_pred_ids = np.argmax(test_proba, axis=1)
    test_pred = [id2label[i] for i in test_pred_ids]
    test_metrics = compute_multiclass_metrics(test["category"], test_pred, test_proba, LABELS)

    # Latence : inférence unitaire, CPU ou GPU selon ce que détecte torch dans CET environnement.
    model.eval()
    single_texts = test["text"].tolist()[:100]
    lat = []
    with torch.no_grad():
        for t in single_texts[:5]:
            enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=256).to(device)
            model(**enc)
        for t in single_texts:
            enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=256).to(device)
            s = time.perf_counter()
            model(**enc)
            lat.append((time.perf_counter() - s) * 1000.0)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "labels.json").write_text(json.dumps(LABELS, indent=2), encoding="utf-8")

    res = settings.results_dir
    plot_confusion_matrix_multiclass(test_metrics["confusion_matrix"], LABELS,
                                     "DistilBERT — test", res / "confusion_matrices" / "distilbert.png")

    report = {
        "model": "distilbert",
        "display_name": "DistilBERT (multiclass)",
        "family": "transformer",
        "base_model": MODEL_NAME,
        "dataset_version": settings.dataset_version,
        "seed": seed,
        "labels": LABELS,
        "device": device,
        "gpu_name": gpu_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_train_samples": max_train_samples if max_train_samples else len(train),
        "split_sizes": {"train": len(train), "validation": len(val), "test": len(test)},
        "train_time_s": train_time_s,
        "train_runtime_hf": train_result.metrics,
        "validation_metrics_hf": val_metrics_hf,
        "test_metrics": test_metrics,
        "latency": {
            "latency_single_ms_mean": float(np.mean(lat)),
            "latency_single_ms_median": float(np.median(lat)),
            "latency_single_ms_p95": float(np.percentile(lat, 95)),
        },
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": device, "platform": platform.platform()},
    }
    metrics_dir = res / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "distilbert.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Sous-échantillonne le train (utile pour un run rapide / sanity check).")
    args = parser.parse_args()
    settings = get_settings()
    report = train_and_evaluate(settings, epochs=args.epochs, batch_size=args.batch_size,
                                max_train_samples=args.max_train_samples)
    m = report["test_metrics"]
    print(f"DistilBERT ({report['device']}) acc={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f} "
          f"lat_median={report['latency']['latency_single_ms_median']:.2f}ms", flush=True)


if __name__ == "__main__":
    main()
