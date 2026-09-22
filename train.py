import os
import time
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import EfficientNetB0Binary

TYPE = ["IR", "RGB"]
COVER = ["uncover", "cover1", "cover2"]  # cover1: 얇은 이불, cover2: 두꺼운 이불

modality = TYPE[1]
cover = COVER[1]
CHECKPOINT_NAME = f"best-{modality}-{cover}.pt"


# ======================
# 1) 하이퍼파라미터
# ======================
@dataclass
class CFG:
    train_dir: str = f"data/{modality}/{cover}/train"
    val_dir: str = f"data/{modality}/{cover}/val"
    img_size: int = 224
    batch_size: int = 64
    num_workers: int = 4
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    # 클래스 불균형 시 1-클래스(정자세 외)에 대한 가중치. 불균형 크면 1.0 -> 2~5 정도로 올리세요.
    pos_weight_for_class1: float = 1.0
    seed: int = 42
    # 데이터 증강 강도
    rotate_deg: int = 10
    # AMP
    use_amp: bool = True


cfg = CFG()

torch.manual_seed(cfg.seed)
torch.cuda.manual_seed_all(cfg.seed)


# ======================
# 2) 변환(Aspect 보존 패딩 -> Resize)
# ======================
class LetterboxToSquare:
    """
    짧은 변 기준으로 여백을 패딩해 정사각형으로 만든 다음, 지정 크기로 리사이즈.
    """

    def __init__(self, size: int, fill: int = 0):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        # img: PIL
        w, h = img.size
        side = max(w, h)  # 정사각형 한 변
        pad_w = (side - w) // 2
        pad_h = (side - h) // 2

        # Pad(left, top, right, bottom)
        padding = (pad_w, pad_h, side - w - pad_w, side - h - pad_h)
        img = transforms.functional.pad(img, padding, fill=self.fill)
        img = transforms.functional.resize(img, (self.size, self.size), antialias=True)
        return img


# IR(1채널) -> 3채널 복제
class GrayTo3:
    def __call__(self, img_tensor):
        # img_tensor: [1, H, W]
        return img_tensor.repeat(3, 1, 1)


train_tfms = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        LetterboxToSquare(cfg.img_size, fill=0),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=cfg.rotate_deg, fill=0),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.ToTensor(),
        GrayTo3(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet 통계 사용
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

val_tfms = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        LetterboxToSquare(cfg.img_size, fill=0),
        transforms.ToTensor(),
        GrayTo3(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# ======================
# 3) 데이터셋 & 로더
# ======================
def build_loaders() -> Tuple[DataLoader, DataLoader, List[str]]:
    train_ds = ImageFolder(cfg.train_dir, transform=train_tfms)
    val_ds = ImageFolder(cfg.val_dir, transform=val_tfms)
    class_names = train_ds.classes  # ['0','1'] 기대
    print("Classes:", class_names)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, class_names


# ======================
# 5) 지표
# ======================
@torch.no_grad()
def compute_metrics(logits: torch.Tensor, targets: torch.Tensor):
    # logits: [B], targets: [B] in {0,1}
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()
    tp = ((preds == 1) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    if (prec + rec) == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return acc, prec, rec, f1


# ======================
# 6) 학습/검증 루프
# ======================
def train_one_epoch(model, loader, optimizer, scaler, criterion, device):
    model.train()
    total_loss = 0.0
    n = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)  # BCE용 float

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=cfg.use_amp
        ):
            logits = model(imgs)
            loss = criterion(logits, labels)

        if cfg.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)

    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    all_logits = []
    all_labels = []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels).long()
    acc, prec, rec, f1 = compute_metrics(logits_cat, labels_cat)
    return total_loss / max(n, 1), acc, prec, rec, f1


# ======================
# 7) 메인
# ======================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, class_names = build_loaders()

    model = EfficientNetB0Binary(pretrained=True).to(device)

    # 불균형 시 pos_weight 조정(>1이면 양성 클래스(정자세 외) 더 강하게)
    pos_weight = torch.tensor([cfg.pos_weight_for_class1], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    # Cosine 스케줄러(워밍업 없이 간단히)
    total_steps = cfg.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    best_f1 = -1.0
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, device
        )

        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(
            model, val_loader, criterion, device
        )

        # 스케줄러 one-step per iteration 기준으로 구성했으므로, 여기서는 epoch 기준 보정
        # 간단히 epoch당 steps 만큼 진행:
        for _ in range(len(train_loader)):
            scheduler.step()
            global_step += 1

        dt = time.time() - t0
        print(
            f"[{epoch:02d}/{cfg.epochs}] "
            f"train_loss={tr_loss:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f} prec={val_prec:.4f} rec={val_rec:.4f} f1={val_f1:.4f} "
            f"| {dt:.1f}s"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_f1": best_f1,
                    "cfg": cfg.__dict__,
                    "classes": class_names,
                },
                f"checkpoints/{CHECKPOINT_NAME}",
            )
            print(f"✅ Saved best model (f1={best_f1:.4f})")

    print("Done. Best F1:", best_f1)


# ======================
# 8) 추론 유틸
# ======================
@torch.no_grad()
def predict_image(model_path: str, image_path: str) -> Tuple[int, float]:
    """
    Returns: (pred_label, prob_of_class1)
      pred_label: 0(정자세) or 1(정자세 외)
      prob_of_class1: sigmoid 확률
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(model_path, map_location=device)
    model = EfficientNetB0Binary(pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    img = Image.open(image_path).convert("L")
    tfm = val_tfms
    x = tfm(img).unsqueeze(0).to(device)
    logit = model(x)
    prob = torch.sigmoid(logit).item()
    pred = 1 if prob >= 0.5 else 0
    return pred, prob


if __name__ == "__main__":
    main()
