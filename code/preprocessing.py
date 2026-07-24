"""
preprocessing.py
----------------
data/split 하위의 원본 이미지에서 MTCNN으로 얼굴을 검출해 크롭한 뒤 224x224로 저장한다.
얼굴을 못 찾았거나 얼굴이 이미지에서 차지하는 비율이 너무 작으면 manual_check 폴더로 분리해
사람이 직접 확인할 수 있게 한다.
"""
import os
import shutil

# facenet-pytorch(torch 의존)를 import하기 전에, OpenMP 런타임 중복 로드로 인한
# "OMP: Error #15" 크래시를 미리 방지해둔다 (demo.py에서 겪은 것과 동일한 환경 이슈).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from facenet_pytorch import MTCNN
from PIL import Image

from config import SPLIT_DIR, PROCESSED_DIR, MANUAL_CHECK_DIR

# 얼굴이 전체 이미지 넓이에서 차지해야 하는 최소 비율 (너무 작은 얼굴/오검출 배제)
MIN_FACE_RATIO = 0.05
# 얼굴 크롭 시 귀나 턱이 잘리지 않도록 추가하는 여백 비율
PADDING = 0.25


def save_fail(img, fail_dir, img_name):
    """검출 실패/기준 미달 이미지를 원본 그대로 수동 검토 폴더에 저장."""
    os.makedirs(fail_dir, exist_ok=True)
    img.save(os.path.join(fail_dir, img_name))


def process_images():
    print("데이터 전처리 및 얼굴 추출 시작")

    # 이전 처리 결과가 남아있으면 섞이지 않도록 삭제 후 재생성
    for d in [PROCESSED_DIR, MANUAL_CHECK_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)

    mtcnn = MTCNN(keep_all=False, select_largest=True, device="cpu")
    stats = {"success": 0, "fail": 0}

    for split in ["train", "val", "test"]:
        split_path = os.path.join(SPLIT_DIR, split)
        if not os.path.exists(split_path):
            continue

        for celeb in os.listdir(split_path):
            for emotion in ["smile", "neutral"]:
                src_dir = os.path.join(split_path, celeb, emotion)
                target_dir = os.path.join(PROCESSED_DIR, split, celeb, emotion)
                fail_dir = os.path.join(MANUAL_CHECK_DIR, split, celeb, emotion)

                if not os.path.exists(src_dir):
                    continue

                for img_name in os.listdir(src_dir):
                    img_path = os.path.join(src_dir, img_name)

                    try:
                        img = Image.open(img_path).convert("RGB")
                        img_w, img_h = img.size
                        img_area = img_w * img_h

                        boxes, _ = mtcnn.detect(img)
                        if boxes is None:
                            save_fail(img, fail_dir, img_name)
                            stats["fail"] += 1
                            continue

                        # 가장 큰 얼굴 하나만 사용 (MTCNN을 select_largest=True로 설정)
                        box = boxes[0]
                        w, h = box[2] - box[0], box[3] - box[1]

                        if (w * h) / img_area < MIN_FACE_RATIO:
                            save_fail(img, fail_dir, img_name)
                            stats["fail"] += 1
                            continue

                        pad_w, pad_h = w * PADDING, h * PADDING
                        nx1 = max(0, box[0] - pad_w)
                        ny1 = max(0, box[1] - pad_h)
                        nx2 = min(img_w, box[2] + pad_w)
                        ny2 = min(img_h, box[3] + pad_h)

                        face_crop = img.crop((nx1, ny1, nx2, ny2)).resize((224, 224), Image.LANCZOS)

                        os.makedirs(target_dir, exist_ok=True)
                        face_crop.save(os.path.join(target_dir, img_name))
                        stats["success"] += 1

                    except Exception as e:
                        print(f"처리 실패 ({img_path}): {e}")
                        save_fail(img, fail_dir, img_name)
                        stats["fail"] += 1

    print(f"처리 완료. 성공: {stats['success']}장, 수동검토: {stats['fail']}장")


if __name__ == "__main__":
    process_images()
