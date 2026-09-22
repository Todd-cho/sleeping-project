import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# 학습 스크립트에서 가져옴
from model import EfficientNetB0Binary
from train import val_tfms

try:
    from sklearn.metrics import (
        average_precision_score,
        classification_report,
        confusion_matrix,
        roc_auc_score,
        roc_curve,
    )

    _SKLEARN = True
except Exception:
    _SKLEARN = False


def plot_confusion_matrix(cm, save_path, title):
    class_names = ["normal", "abnormal"]
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curve(y_true, probs, save_path):
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = roc_auc_score(y_true, probs)
    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def evaluate(
    ckpt_path: str,
    test_dir: str = "data/test",
    batch_size: int = 64,
    num_workers: int = 4,
    threshold: float = 0.5,
    out_csv: str = "predictions_test.csv",
    title: str = "",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 데이터 로더
    test_ds = ImageFolder(test_dir, transform=val_tfms)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    class_names = test_ds.classes  # ['0', '1'] 기대

    # 2) 모델 로드
    ckpt = torch.load(ckpt_path, map_location=device)
    model = EfficientNetB0Binary(pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 3) 추론
    all_logits, all_labels, all_paths = [], [], []
    with torch.no_grad():
        for (imgs, labels), indices in zip(
            test_loader,
            [
                range(
                    i * test_loader.batch_size,
                    min((i + 1) * test_loader.batch_size, len(test_ds)),
                )
                for i in range(len(test_loader))
            ],
        ):
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs).detach().cpu().numpy()
            labels = labels.numpy()
            batch_paths = [test_ds.samples[idx][0] for idx in indices]

            all_logits.append(logits)
            all_labels.append(labels)
            all_paths.extend(batch_paths)

    logits = np.concatenate(all_logits).astype(np.float64).squeeze()
    y_true = np.concatenate(all_labels).astype(np.int64)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(np.int64)

    # 4) 기본 지표
    tp = int(((preds == 1) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())

    acc = (tp + tn) / max(len(y_true), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

    print("=== Test Metrics ===")
    print(f"Samples: {len(y_true)}  |  Threshold: {threshold:.2f}")
    print(f"Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
    print(f"TP={tp}  TN={tn}  FP={fp}  FN={fn}")

    if _SKLEARN:
        cm = confusion_matrix(y_true, preds)
        print("\nConfusion Matrix:\n", cm)
        try:
            roc = roc_auc_score(y_true, probs)
            pr_auc = average_precision_score(y_true, probs)
            print(f"ROC-AUC={roc:.4f}  PR-AUC={pr_auc:.4f}")

            cm_path = Path(out_csv).with_suffix("").as_posix() + "_cm.png"
            roc_path = Path(out_csv).with_suffix("").as_posix() + "_roc.png"
            plot_confusion_matrix(cm, cm_path, title)
            plot_roc_curve(y_true, probs, roc_path)
            print(f"Saved confusion matrix → {cm_path}")
            print(f"Saved ROC curve → {roc_path}")

        except Exception as e:
            print(f"(ROC curve 생성 중 오류 발생: {e})")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true, preds, target_names=["normal", "abnormal"], digits=4
        )
    )

    # 5) CSV 저장
    if out_csv:
        df = pd.DataFrame(
            {
                "path": all_paths,
                "label_true": y_true,
                "prob_class1": probs,
                "pred": preds,
            }
        )
        Path(os.path.dirname(out_csv) or ".").mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"\nSaved predictions → {out_csv}")

    return {
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main():
    TYPE = ["RGB", "IR"]
    COVER = ["uncover", "cover1", "cover2"]

    ap = argparse.ArgumentParser(description="Evaluate IR bedpose model on test set")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    for modality in TYPE:
        for cover in COVER:
            evaluate(
                ckpt_path=f"checkpoints/best-{modality}-{cover}.pt",
                test_dir=f"data/{modality}/{cover}/test",
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                threshold=args.threshold,
                out_csv=f"predictions_test_{modality}_{cover}.csv",
                title=f"{modality}-{cover}",
            )


if __name__ == "__main__":
    main()
