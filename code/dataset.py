"""
dataset.py
----------
연예인 분류(celeb)와 표정 분류(smile)를 동시에 학습하기 위한
멀티태스크 Dataset/DataLoader 정의.
"""
import os
from collections import Counter

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def get_transforms():
    """train/val/test 각각에 사용할 이미지 전처리(증강 포함) 파이프라인을 반환."""
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    return {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            normalize,
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]),
        "test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ]),
    }


class MultiTaskDataset(Dataset):
    """data/split/{split}/{celeb}/{neutral,smile}/ 폴더 구조를 그대로 라벨로 사용하는 데이터셋."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.celeb_labels = []
        self.smile_labels = []

        self.celebs = sorted(os.listdir(root_dir))
        self.celeb_to_idx = {c: i for i, c in enumerate(self.celebs)}

        self.emotions = ["neutral", "smile"]
        self.emotion_to_idx = {e: i for i, e in enumerate(self.emotions)}

        for celeb in self.celebs:
            celeb_path = os.path.join(root_dir, celeb)
            if not os.path.isdir(celeb_path):
                continue

            for emotion in self.emotions:
                emotion_path = os.path.join(celeb_path, emotion)
                if not os.path.isdir(emotion_path):
                    continue

                for img_name in os.listdir(emotion_path):
                    if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.image_paths.append(os.path.join(emotion_path, img_name))
                        self.celeb_labels.append(self.celeb_to_idx[celeb])
                        self.smile_labels.append(self.emotion_to_idx[emotion])

        self.sample_weights = self._calculate_weights()

    def _calculate_weights(self):
        """(연예인, 표정) 조합별 샘플 수의 역수로 가중치를 계산 (WeightedRandomSampler에 활용 가능)."""
        label_pairs = list(zip(self.celeb_labels, self.smile_labels))
        count = Counter(label_pairs)
        weights = [1.0 / count[(c, s)] for c, s in label_pairs]
        return torch.tensor(weights, dtype=torch.float)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.celeb_labels[idx], self.smile_labels[idx]


def get_dataloaders(data_dir, batch_size=32, num_workers=2):
    """data_dir 하위 train/val/test 폴더로부터 DataLoader와 클래스(연예인) 이름 목록을 만들어 반환."""
    transforms_dict = get_transforms()
    shuffle_by_split = {"train": True, "val": False, "test": False}

    dataloaders = {}
    class_names = None

    for split, shuffle in shuffle_by_split.items():
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue

        dataset = MultiTaskDataset(split_dir, transform=transforms_dict[split])
        dataloaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )

        if split == "train":
            class_names = dataset.celebs

    return dataloaders, class_names


if __name__ == "__main__":
    print("dataset.py 단독 실행: import 및 문법 오류 여부만 확인합니다.")
