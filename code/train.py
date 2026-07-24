"""
train.py
--------
멀티태스크(연예인 분류 + 표정 분류) ResNet18 모델을 학습하는 스크립트.

학습 전략:
1) 처음에는 backbone을 동결하고 head(shared + celeb_head + smile_head)만 학습
2) --unfreeze_epoch 시점부터 backbone도 낮은 학습률로 함께 학습(차등 학습률)
3) 표정 정확도가 --smile_acc_threshold를 넘긴 epoch 중 최고 성능만 체크포인트로 저장
4) 학습 종료 후 저장된 체크포인트를 불러와 test set으로 최종 성능을 확인

epoch별 지표는 CSV로 기록되고, 학습 곡선은 PNG로 저장된다.
"""
import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import CODE_DIR, PROCESSED_DIR
from dataset import get_dataloaders
from model import MultiTaskResNet


def parse_args():
    parser = argparse.ArgumentParser(description="멀티태스크 연예인/표정 분류 모델 학습")
    parser.add_argument("--data_dir", type=str, default=PROCESSED_DIR, help="얼굴 크롭까지 끝난 train/val/test 데이터 경로 (preprocessing.py 결과물)")
    parser.add_argument("--output_dir", type=str, default=CODE_DIR, help="체크포인트/그래프/로그 저장 경로")
    parser.add_argument("--epochs", type=int, default=50, help="최대 학습 epoch 수")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1.0, help="연예인 분류 loss 가중치")
    parser.add_argument("--beta", type=float, default=0.5, help="표정 분류 loss 가중치")
    parser.add_argument("--unfreeze_epoch", type=int, default=10, help="backbone 동결을 해제할 epoch")
    parser.add_argument("--patience", type=int, default=7, help="조기 종료 인내 횟수")
    parser.add_argument("--smile_acc_threshold", type=float, default=0.70, help="체크포인트 저장을 허용할 최소 표정 정확도")
    return parser.parse_args()


def run_epoch(model, loader, criterion, device, alpha, beta, optimizer=None):
    """optimizer가 있으면 학습 모드로, 없으면 평가 모드로 한 epoch을 순회하고 (평균 loss, 이름 정확도, 표정 정확도)를 반환."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, c_correct, s_correct, total = 0.0, 0, 0, 0
    grad_context = torch.enable_grad() if is_train else torch.no_grad()

    with grad_context:
        for images, c_labels, s_labels in loader:
            images, c_labels, s_labels = images.to(device), c_labels.to(device), s_labels.to(device)

            if is_train:
                optimizer.zero_grad()

            c_out, s_out = model(images)
            loss = alpha * criterion(c_out, c_labels) + beta * criterion(s_out, s_labels)

            if is_train:
                loss.backward()
                optimizer.step()
                if hasattr(loader, "set_postfix"):
                    loader.set_postfix(loss=f"{loss.item():.4f}")

            total_loss += loss.item()
            c_correct += (torch.argmax(c_out, 1) == c_labels).sum().item()
            s_correct += (torch.argmax(s_out, 1) == s_labels).sum().item()
            total += c_labels.size(0)

    return total_loss / len(loader), c_correct / total, s_correct / total


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloaders, class_names = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    train_loader, val_loader = dataloaders["train"], dataloaders["val"]
    test_loader = dataloaders.get("test")

    model = MultiTaskResNet(num_celebs=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()

    # 1단계: backbone 동결, head만 학습
    for param in model.backbone.parameters():
        param.requires_grad = False

    head_params = (
        list(model.shared.parameters())
        + list(model.celeb_head.parameters())
        + list(model.smile_head.parameters())
    )
    optimizer = optim.Adam([{"params": head_params, "lr": 3e-4}], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "celeb_acc": [], "smile_acc": []}

    # 체크포인트 저장 기준(best_saved_score)과 조기 종료/스케줄러 판단 기준(best_score)을 분리한다.
    # 하나로 합치면, smile_acc_threshold 미달로 저장되지 않은 epoch의 점수가 best_score를 선점해버려서
    # 이후 조건을 만족하는 더 좋은 epoch이 나와도 "저장할 만큼 좋아졌다"는 판정을 못 받는 버그가 생긴다.
    best_score = 0.0
    best_saved_score = 0.0
    patience_counter = 0

    checkpoint_name = f"best_model_a{args.alpha}_b{args.beta}.pth"
    checkpoint_path = os.path.join(args.output_dir, checkpoint_name)
    csv_path = os.path.join(args.output_dir, f"history_a{args.alpha}_b{args.beta}.csv")

    # 체크포인트만으로는 출력 인덱스가 어떤 연예인인지 알 수 없으므로,
    # demo.py 등에서 그대로 불러다 쓸 수 있게 클래스 순서를 함께 저장해둔다.
    classes_path = os.path.join(args.output_dir, f"classes_a{args.alpha}_b{args.beta}.json")
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "celeb_acc", "smile_acc"])

    print("=" * 60)
    print(f"▶ 학습 시작 (장치: {device})")
    print(f"▶ 가중치: alpha(이름)={args.alpha}, beta(미소)={args.beta}")
    print("=" * 60)

    for epoch in range(args.epochs):
        # 10 epoch 경과 후 backbone도 낮은 학습률로 함께 학습(차등 학습률)
        if epoch == args.unfreeze_epoch:
            print(f"\n▶ {args.unfreeze_epoch} epoch 경과: backbone 동결 해제 (unfreeze)")
            for param in model.backbone.parameters():
                param.requires_grad = True

            optimizer = optim.Adam([
                {"params": model.backbone.parameters(), "lr": 1e-5},
                {"params": head_params, "lr": 3e-4},
            ], weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

        pbar = tqdm(train_loader, desc=f"학습 {epoch + 1}/{args.epochs}")
        train_loss, _, _ = run_epoch(model, pbar, criterion, device, args.alpha, args.beta, optimizer)
        val_loss, c_acc, s_acc = run_epoch(model, val_loader, criterion, device, args.alpha, args.beta)

        current_score = c_acc + s_acc
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["celeb_acc"].append(c_acc)
        history["smile_acc"].append(s_acc)

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch + 1, train_loss, val_loss, c_acc, s_acc])

        print(f"결과: Train loss {train_loss:.4f} | Val loss {val_loss:.4f} | 이름 정확도 {c_acc:.4f} | 표정 정확도 {s_acc:.4f}")
        scheduler.step(current_score)

        # 조기 종료 판단은 smile_acc_threshold와 무관하게 전체 점수 기준으로 진행
        if current_score > best_score:
            best_score = current_score
            patience_counter = 0
        else:
            patience_counter += 1

        # 체크포인트 저장은 표정 정확도 임계값을 넘긴 epoch 중 최고 점수일 때만
        if s_acc >= args.smile_acc_threshold and current_score > best_saved_score:
            best_saved_score = current_score
            torch.save(model.state_dict(), checkpoint_path)
            print(f"▶ 체크포인트 갱신: {checkpoint_name} 저장 완료 (표정 정확도 {s_acc:.4f} >= {args.smile_acc_threshold})")

        if patience_counter >= args.patience:
            print(f"\n▶ {patience_counter}회 연속 점수 향상 없음. 조기 종료(Early Stopping) 발동!")
            break

    # 학습 곡선 저장
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss", linestyle="--")
    plt.title("Loss Trend")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["celeb_acc"], label="Celeb Acc", color="green")
    plt.plot(history["smile_acc"], label="Smile Acc", color="orange")
    plt.title("Accuracy Trend")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()

    plt.suptitle(f"alpha={args.alpha}, beta={args.beta}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    chart_path = os.path.join(args.output_dir, f"result_chart_a{args.alpha}_b{args.beta}.png")
    plt.savefig(chart_path, bbox_inches="tight")
    print(f"\n학습 곡선을 '{chart_path}'에 저장했습니다.")
    print(f"epoch별 지표 로그를 '{csv_path}'에 저장했습니다.")

    # 최종 테스트 평가: 학습 중 저장된 best 체크포인트를 불러와 test set으로 한 번 더 확인
    print("\n" + "=" * 60)
    if test_loader is None:
        print("▶ test 데이터가 없어 최종 평가를 건너뜁니다.")
    elif not os.path.exists(checkpoint_path):
        print("▶ 저장된 체크포인트가 없어 최종 평가를 건너뜁니다. (smile_acc_threshold를 넘긴 epoch이 없었습니다)")
    else:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        _, test_c_acc, test_s_acc = run_epoch(model, test_loader, criterion, device, args.alpha, args.beta)
        print(f"[Test 최종 평가] 이름 정확도 {test_c_acc:.4f} | 표정 정확도 {test_s_acc:.4f}")
    print("=" * 60)

    plt.show()


if __name__ == "__main__":
    main()
