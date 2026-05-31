"""
ImpellerVision — Faz 1: Model Eğitimi
=====================================
Döküm pompa çarkı (impeller) yüzey kusuru tespiti — binary classifier.

Model    : MobileNetV2 (ImageNet pretrained) transfer learning
Veri     : data/train/{def_front,ok_front}  (ImageFolder, image-level label)
           ImageFolder sınıfları alfabetik indeksler:
               def_front -> 0  (Defect / kusurlu)
               ok_front  -> 1  (OK / saglam)
Cikti    : models/impeller_model.pt   (state_dict + metadata)
Kanit    : outputs/confusion_matrix.png, training_curves.png,
           metrics.json, classification_report.txt

Kullanim:
    python src/train.py                 # varsayilan ayarlar (GPU varsa GPU)
    python src/train.py --epochs 10 --batch-size 64
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
import matplotlib

matplotlib.use("Agg")  # GUI'siz ortamda dosyaya yaz
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Sabitler
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SEED = 42


def set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms():
    """Train icin augmentation (aydinlatma temasina uygun ColorJitter dahil),
    val/test icin sade resize+normalize."""
    train_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            # Endustriyel aydinlatma degisimini taklit eder (parlaklik/kontrast):
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


def build_model(num_classes: int = 2) -> nn.Module:
    """MobileNetV2 pretrained, classifier head 2 sinifa gore yeniden tanimli."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    torch.set_grad_enabled(train)
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        if train:
            optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    torch.set_grad_enabled(True)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    """Test seti tahminleri — confusion matrix / rapor icin."""
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())
    return np.array(all_labels), np.array(all_preds)


def plot_confusion(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gercek")
    ax.set_title("Confusion Matrix (test seti)")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_curves(history, out_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["train_loss"], "o-", label="train")
    ax1.plot(epochs, history["val_loss"], "o-", label="val")
    ax1.set_title("Loss"); ax1.set_xlabel("epoch"); ax1.legend()
    ax2.plot(epochs, history["train_acc"], "o-", label="train")
    ax2.plot(epochs, history["val_acc"], "o-", label="val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("epoch"); ax2.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="ImpellerVision egitimi")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true", help="GPU varsa bile CPU kullan")
    args = parser.parse_args()

    set_seed()
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    )
    print(f"[cihaz] {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    train_tf, eval_tf = build_transforms()

    # --- Veri setleri -------------------------------------------------------
    full_train = datasets.ImageFolder(DATA_DIR / "train", transform=train_tf)
    class_names = full_train.classes  # ['def_front', 'ok_front']
    print(f"[siniflar] {full_train.class_to_idx}")

    n_val = int(len(full_train) * args.val_split)
    n_train = len(full_train) - n_val
    g = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=g)
    # val seti augmentation gormemeli -> ayri transform'lu kopya uzerinden indeksle
    val_base = datasets.ImageFolder(DATA_DIR / "train", transform=eval_tf)
    val_ds.dataset = val_base

    test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=eval_tf)
    print(f"[veri] train={n_train}  val={n_val}  test={len(test_ds)}")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin)

    # --- Model / loss / optimizer ------------------------------------------
    model = build_model(num_classes=len(class_names)).to(device)

    # Hafif dengesizlik (def>ok) icin sinif agirligi
    counts = np.bincount([y for _, y in full_train.samples], minlength=len(class_names))
    weights = counts.sum() / (len(class_names) * counts)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"[sinif sayilari] {dict(zip(class_names, counts.tolist()))}"
          f"  agirliklar={weights.round(3).tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # --- Egitim dongusu -----------------------------------------------------
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_state = 0.0, None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        vl_loss, vl_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)
        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss); history["val_acc"].append(vl_acc)
        flag = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            flag = "  <- best"
        print(f"[epoch {epoch}/{args.epochs}] "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={vl_loss:.4f} acc={vl_acc:.4f}{flag}")
    print(f"[sure] egitim {time.time() - t0:.1f} sn — en iyi val acc={best_val_acc:.4f}")

    # En iyi modeli yukle
    model.load_state_dict(best_state)

    # --- Test degerlendirmesi ----------------------------------------------
    y_true, y_pred = evaluate(model, test_loader, device)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names))
    )
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    print("\n===== TEST SONUCLARI =====")
    print(f"Accuracy: {acc:.4f}")
    print(report)
    print("Confusion matrix:\n", cm)

    # --- Kayit --------------------------------------------------------------
    plot_confusion(cm, class_names, OUTPUTS_DIR / "confusion_matrix.png")
    plot_curves(history, OUTPUTS_DIR / "training_curves.png")

    metrics = {
        "test_accuracy": float(acc),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "per_class": {
            class_names[i]: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
            }
            for i in range(len(class_names))
        },
        "confusion_matrix": cm.tolist(),
        "classes": class_names,
        "class_to_idx": full_train.class_to_idx,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "best_val_accuracy": float(best_val_acc),
    }
    (OUTPUTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUTPUTS_DIR / "classification_report.txt").write_text(
        f"Test accuracy: {acc:.4f}\n\n{report}\nConfusion matrix:\n{cm}\n",
        encoding="utf-8",
    )

    ckpt = {
        "arch": "mobilenet_v2",
        "img_size": IMG_SIZE,
        "classes": class_names,            # ['def_front', 'ok_front']
        "class_to_idx": full_train.class_to_idx,
        "defect_index": class_names.index("def_front"),
        "normalize": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "test_accuracy": float(acc),
        "state_dict": model.state_dict(),
    }
    torch.save(ckpt, MODELS_DIR / "impeller_model.pt")
    print(f"\n[kayit] model -> {MODELS_DIR / 'impeller_model.pt'}")
    print(f"[kayit] kanitlar -> {OUTPUTS_DIR}/")

    if acc >= 0.98:
        print(f"\n[OK] HEDEF TUTTU: test accuracy {acc*100:.2f}% >= 98%")
    else:
        print(f"\n[!] Hedefin altinda: {acc*100:.2f}% < 98% - epoch/lr ayari gerekebilir")


if __name__ == "__main__":
    main()
