"""
model.py
--------
연예인 분류(celeb)와 표정 분류(smile)를 동시에 수행하는 멀티태스크 ResNet18 모델.
ImageNet으로 사전학습된 ResNet18을 공유 백본으로 쓰고,
공유 특징 위에 태스크별로 독립된 Dropout과 출력층(head)을 둔다.
"""
import torch.nn as nn
from torchvision import models


class MultiTaskResNet(nn.Module):
    def __init__(self, num_celebs=8, dropout_p=0.5):
        super(MultiTaskResNet, self).__init__()

        # 사전학습된 ResNet18을 특징 추출기(backbone)로 사용
        self.backbone = models.resnet18(weights="IMAGENET1K_V1")
        in_features = self.backbone.fc.in_features
        # 원래 ImageNet 1000-class 분류층은 제거
        self.backbone.fc = nn.Identity()

        # 공유 레이어: 두 태스크가 공통으로 사용할 특징을 정제
        self.shared = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
        )

        # 태스크별로 Dropout을 분리해서 각 태스크의 과적합 방지 강도를 독립적으로 조절 가능하게 함
        self.celeb_dropout = nn.Dropout(p=dropout_p)
        self.smile_dropout = nn.Dropout(p=dropout_p)

        # 출력층: 연예인 8명 분류 / 표정(웃음, 무표정) 2분류
        self.celeb_head = nn.Linear(512, num_celebs)
        self.smile_head = nn.Linear(512, 2)

    def forward(self, x):
        features = self.backbone(x)
        shared_feat = self.shared(features)

        celeb_out = self.celeb_head(self.celeb_dropout(shared_feat))
        smile_out = self.smile_head(self.smile_dropout(shared_feat))

        return celeb_out, smile_out
