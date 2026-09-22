import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


# ======================
# 4) 모델
# ======================
class EfficientNetB0Binary(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = efficientnet_b0(weights=weights)
        else:
            self.backbone = efficientnet_b0(weights=None)

        in_features = self.backbone.classifier[1].in_features  # 1280
        # 이진 분류: 출력 1 (logit). BCEWithLogitsLoss 사용 예정
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True), nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return self.backbone(x).squeeze(1)  # [B]


if __name__ == "__main__":
    model = EfficientNetB0Binary()
    print(model)
