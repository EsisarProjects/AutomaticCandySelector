from pathlib import Path
import argparse
import json
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
import yaml

from src.classification.models import create_model
from src.data.dataset import create_dataloaders

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/classification.yaml"
MODELS_DIR = ROOT / "models"
TABLES_DIR = ROOT / "results/tables"
FIGURES_DIR = ROOT / "results/figures"


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponible. Utilisez --device cpu.")
    return torch.device(name)


def compute_metrics(confusion):
    total = confusion.sum().item()
    if total == 0:
        raise ValueError("Aucun échantillon évalué.")

    tp = confusion.diag().float()
    fp = confusion.sum(dim=0).float() - tp
    fn = confusion.sum(dim=1).float() - tp
    denominator = 2 * tp + fp + fn
    f1_per_class = torch.where(
        denominator > 0,
        (2 * tp) / denominator.clamp(min=1),
        torch.zeros_like(tp)
    )
    accuracy = tp.sum().item() / total
    macro_f1 = f1_per_class.mean().item()
    return accuracy, macro_f1


def run_epoch(model, loader, criterion, device, optimizer=None, freeze_backbone=True,
              max_batches=None):
    training = optimizer is not None

    if training:
        model.train()
        if freeze_backbone:
            # Les couches gelées ne doivent pas actualiser BatchNorm.
            model.features.eval()
            model.classifier[:-1].eval()
    else:
        model.eval()

    num_classes = len(loader.dataset.classes)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    total_loss = 0.0
    total_examples = 0

    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        predictions = logits.argmax(dim=1)
        flat_indices = (
            labels.detach().cpu() * num_classes
            + predictions.detach().cpu()
        )
        confusion += torch.bincount(
            flat_indices, minlength=num_classes * num_classes
        ).reshape(num_classes, num_classes)

    if total_examples == 0:
        raise ValueError("Le DataLoader n'a produit aucun batch.")

    accuracy, macro_f1 = compute_metrics(confusion)
    return {
        "loss": total_loss / total_examples,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion": confusion,
    }


def save_curves(history, run_name):
    df = pd.DataFrame(history)
    for metric, ylabel in (
        ("loss", "Cross-entropy loss"),
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro F1-score"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(df["epoch"], df[f"train_{metric}"], marker="o", label="Train")
        ax.plot(df["epoch"], df[f"val_{metric}"], marker="o", label="Validation")
        ax.set_xlabel("Époque")
        ax.set_ylabel(ylabel)
        ax.set_title(f"ACS - {ylabel}")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"{run_name}_{metric}.png", dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="ACS - Entraînement MobileNetV3")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Un seul batch train/val, sans sauvegarde")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--method", choices=["center_crop", "padding"])
    parser.add_argument("--mode", choices=["linear_probe", "fine_tune"])
    parser.add_argument("--run-name", type=str)
    args = parser.parse_args()

    with open(CONFIG_PATH, encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    settings = config.get("training", {})

    seed = int(settings.get("seed", 42))
    batch_size = args.batch_size if args.batch_size is not None else int(settings.get("batch_size", 4))
    workers = int(settings.get("num_workers", 0))
    method = args.method or settings.get("preprocessing_method", "center_crop")
    mode = args.mode or settings.get("mode", "linear_probe")
    epochs = args.epochs if args.epochs is not None else int(settings.get("epochs", 20))
    lr = args.lr if args.lr is not None else float(settings.get("learning_rate", 0.001))
    weight_decay = float(settings.get("weight_decay", 0.0001))
    patience = args.patience if args.patience is not None else int(settings.get("patience", 5))
    device = resolve_device(args.device or settings.get("device", "auto"))
    run_name = args.run_name or f"mobilenetv3_{mode}_v1"

    if not run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_name):
        raise ValueError("Le nom d'expérience doit contenir uniquement lettres, chiffres, _ ou -.")
    if epochs < 1 or batch_size < 1 or patience < 1 or lr <= 0 or workers < 0:
        raise ValueError("Paramètres d'entraînement invalides.")

    set_seed(seed)
    train_loader, val_loader, _test_loader = create_dataloaders(
        batch_size=batch_size,
        num_workers=workers,
        method=method,
        seed=seed,
        enable_photometric=False,
    )
    classes = train_loader.dataset.classes

    model = create_model(num_classes=len(classes), pretrained=True, mode=mode).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=weight_decay,
    )
    print("=== ACS - Entraînement MobileNetV3 Small ===")
    print(f"Device : {device} | Mode : {mode} | Classes : {classes}")
    print(f"Images train/val : {len(train_loader.dataset)}/{len(val_loader.dataset)}")
    print(f"Batch : {batch_size} | LR : {lr} | Époques maximum : {epochs}")

    freeze_backbone = mode == "linear_probe"
    if args.smoke_test:
        train_stats = run_epoch(model, train_loader, criterion, device,
                                optimizer=optimizer, freeze_backbone=freeze_backbone,
                                max_batches=1)
        val_stats = run_epoch(model, val_loader, criterion, device,
                              optimizer=None, max_batches=1)
        print(f"Smoke test train loss : {train_stats['loss']:.4f}")
        print(f"Smoke test val loss   : {val_stats['loss']:.4f}")
        print("Smoke test : OK. Aucun modèle sauvegardé.")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_DIR / f"{run_name}_best.pth"
    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_stats = run_epoch(model, train_loader, criterion, device,
                                optimizer=optimizer, freeze_backbone=freeze_backbone)
        val_stats = run_epoch(model, val_loader, criterion, device)

        row = {"epoch": epoch}
        for name, stats in (("train", train_stats), ("val", val_stats)):
            for metric in ("loss", "accuracy", "macro_f1"):
                row[f"{name}_{metric}"] = stats[metric]
        row["time_seconds"] = time.perf_counter() - start
        history.append(row)

        improved = val_stats["loss"] < best_val_loss
        if improved:
            best_val_loss = val_stats["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "class_to_idx": train_loader.dataset.class_to_idx,
                "architecture": "mobilenet_v3_small",
                "mode": mode,
                "preprocessing_method": method,
                "seed": seed,
                "epoch": epoch,
                "val_loss": best_val_loss,
                "val_macro_f1": val_stats["macro_f1"],
            }, checkpoint_path)
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss={train_stats['loss']:.4f} acc={train_stats['accuracy']:.3f} "
            f"F1={train_stats['macro_f1']:.3f} | "
            f"val loss={val_stats['loss']:.4f} acc={val_stats['accuracy']:.3f} "
            f"F1={val_stats['macro_f1']:.3f} | "
            f"{row['time_seconds']:.1f}s"
            + (" *BEST*" if improved else "")
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping après {epoch} époques.")
            break

    history_path = TABLES_DIR / f"{run_name}_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    save_curves(history, run_name)

    # On recharge le meilleur modèle avant de calculer la matrice de validation.
    best_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    val_best = run_epoch(model, val_loader, criterion, device)

    confusion_path = TABLES_DIR / f"{run_name}_val_confusion.csv"
    pd.DataFrame(
        val_best["confusion"].numpy(), index=classes, columns=classes
    ).to_csv(confusion_path, index_label="classe_reelle")

    summary_path = TABLES_DIR / f"{run_name}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump({
            "run_name": run_name,
            "model": "mobilenet_v3_small",
            "mode": mode,
            "preprocessing_method": method,
            "device": str(device),
            "seed": seed,
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "best_epoch": best_epoch,
            "epochs_executed": len(history),
            "best_val_loss": best_val_loss,
            "best_val_accuracy": val_best["accuracy"],
            "best_val_macro_f1": val_best["macro_f1"],
            "classes": classes,
            "test_set_used": False,
        }, file, indent=2, ensure_ascii=False)

    print("\n=== Résultats ===")
    print(f"Meilleure époque : {best_epoch}")
    print(f"Val loss : {best_val_loss:.4f} | Val accuracy : {val_best['accuracy']:.3f} | Val Macro F1 : {val_best['macro_f1']:.3f}")
    print(f"Modèle : {checkpoint_path}")
    print(f"Historique : {history_path}")
    print(f"Matrice de confusion : {confusion_path}")
    print(f"Résumé : {summary_path}")
    print("Jeu de test non utilisé.")


if __name__ == "__main__":
    main()