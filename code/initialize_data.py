"""
initialize_data.py
-------------------
수동 분류가 끝난 data/{연예인}/{neutral,smile} 원본 이미지를,
(연예인, 표정) 조합 비율을 유지한 채 8:1:1(train:val:test)로 계층적 분할하여
data/split/{train,val,test} 폴더로 복사한다.
"""
import os
import shutil
from collections import Counter

from sklearn.model_selection import train_test_split

from config import PROJECT_ROOT, SPLIT_DIR

SOURCE_DIR = os.path.join(PROJECT_ROOT, "data")
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def stratified_split():
    print("데이터 8:1:1 계층적 분할 시작")

    # 이전 분할 결과가 남아있으면 섞이지 않도록 삭제 후 재생성
    if os.path.exists(SPLIT_DIR):
        shutil.rmtree(SPLIT_DIR)

    files_info = []

    for celeb in os.listdir(SOURCE_DIR):
        celeb_path = os.path.join(SOURCE_DIR, celeb)
        if not os.path.isdir(celeb_path):
            continue

        for emotion in ["smile", "neutral"]:
            emotion_path = os.path.join(celeb_path, emotion)
            if not os.path.isdir(emotion_path):
                continue

            for img_name in os.listdir(emotion_path):
                if not img_name.lower().endswith(VALID_EXT):
                    continue

                img_path = os.path.join(emotion_path, img_name)
                # 분할 비율을 (연예인, 표정) 조합 단위로 맞추기 위한 복합키
                composite_key = f"{celeb}_{emotion}"
                files_info.append((img_path, composite_key))

    if not files_info:
        print("오류: 원본 데이터 없음")
        return

    X = [info[0] for info in files_info]
    y_stratify = [info[1] for info in files_info]

    print("클래스별 데이터 분포 확인")
    for k, v in sorted(Counter(y_stratify).items()):
        print(f"  {k}: {v}장")

    try:
        # 1차 분할: 전체의 10%를 test로 분리
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_stratify, test_size=0.1, stratify=y_stratify, random_state=42
        )
        # 2차 분할: 남은 90% 중 1/9(전체 기준 약 10%)를 val로, 나머지를 train으로 분리
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=1 / 9, stratify=y_temp, random_state=42
        )
    except ValueError as e:
        print(f"분할 실패: 각 폴더당 최소 10장의 사진 필요. 오류내용: {e}")
        return

    def copy_files(file_list, split_name):
        print(f"{split_name} 폴더 복사 진행 중")
        for file_path in file_list:
            emotion = os.path.basename(os.path.dirname(file_path))
            celeb = os.path.basename(os.path.dirname(os.path.dirname(file_path)))

            target_dir = os.path.join(SPLIT_DIR, split_name, celeb, emotion)
            os.makedirs(target_dir, exist_ok=True)
            shutil.copy2(file_path, os.path.join(target_dir, os.path.basename(file_path)))

    print("실제 파일 복사 시작")
    copy_files(X_train, "train")
    copy_files(X_val, "val")
    copy_files(X_test, "test")

    print("데이터 분할 완료")
    print(f"Train: {len(X_train)}장")
    print(f"Val: {len(X_val)}장")
    print(f"Test: {len(X_test)}장")


if __name__ == "__main__":
    stratified_split()
