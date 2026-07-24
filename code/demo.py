"""
demo.py
-------
학습된 멀티태스크 모델(best_model_*.pth)로, 업로드한 얼굴 사진에서
연예인 이름과 표정(웃음/무표정)을 함께 예측해 보여주는 Gradio 데모.

별도의 얼굴 검출 없이 업로드된 이미지 전체를 224x224로 리사이즈해 모델에 입력하므로,
사용자가 얼굴이 나온 사진을 직접 업로드해야 한다.

train.py가 체크포인트와 같이 저장한 classes_*.json(클래스 순서)을 그대로 불러와 사용하므로,
학습에 쓰인 checkpoint/classes 파일 쌍이 함께 있어야 한다.
"""
import argparse
import json
import os
import sys

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from torchvision import transforms

# cv2가 자체적으로 가진 config 모듈과 이름이 충돌하지 않도록, 이 파일이 있는 code/ 디렉터리를
# sys.path 맨 앞에 넣어 로컬 config.py를 우선 찾도록 한다.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import HF_CACHE_DIR, HF_CLASSES_FILENAME, HF_MODEL_FILENAME, HF_REPO_ID
from model import MultiTaskResNet

EMOTION_NAMES = ["무표정", "미소"]


def parse_args():
    parser = argparse.ArgumentParser(description="연예인 인물 식별 + 미소 감지 데모")
    parser.add_argument("--model_path", type=str, default=None, help="미지정 시 Hugging Face Hub(%s)에서 자동 다운로드" % HF_REPO_ID)
    parser.add_argument("--classes_path", type=str, default=None, help="미지정 시 model_path와 짝을 이루는 classes_*.json을 자동으로 찾거나(로컬) Hub에서 다운로드")
    parser.add_argument("--share", action="store_true", help="Gradio 공유 링크 생성")
    return parser.parse_args()


def infer_classes_path(model_path):
    """best_model_a{alpha}_b{beta}.pth -> classes_a{alpha}_b{beta}.json 경로 추정."""
    base = os.path.basename(model_path)
    if not base.startswith("best_model_"):
        return None
    suffix = base[len("best_model_"):].replace(".pth", ".json")
    return os.path.join(os.path.dirname(model_path), f"classes_{suffix}")


def build_predict_fn(model, class_names, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def predict(image):
        if image is None:
            return {}, {}

        img = image.convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            c_out, s_out = model(tensor)
            c_prob = torch.softmax(c_out, dim=1)[0].cpu().numpy()
            s_prob = torch.softmax(s_out, dim=1)[0].cpu().numpy()

        celeb_result = {name: float(p) for name, p in zip(class_names, c_prob)}
        smile_result = {name: float(p) for name, p in zip(EMOTION_NAMES, s_prob)}
        return celeb_result, smile_result

    return predict


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_path:
        model_path = args.model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {model_path}")
        classes_path = args.classes_path or infer_classes_path(model_path)
        if not classes_path or not os.path.exists(classes_path):
            raise FileNotFoundError(
                f"클래스 목록 파일을 찾을 수 없습니다: {classes_path}\n"
                f"train.py를 다시 실행하면 체크포인트와 함께 classes_*.json이 생성됩니다."
            )
    else:
        # 로컬 체크포인트를 지정하지 않은 경우, Hugging Face Hub에서 자동으로 내려받는다.
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_MODEL_FILENAME, cache_dir=HF_CACHE_DIR)
        classes_path = args.classes_path or hf_hub_download(
            repo_id=HF_REPO_ID, filename=HF_CLASSES_FILENAME, cache_dir=HF_CACHE_DIR
        )

    with open(classes_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    model = MultiTaskResNet(num_celebs=len(class_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    predict = build_predict_fn(model, class_names, device)

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="얼굴 사진 업로드"),
        outputs=[
            gr.Label(num_top_classes=len(class_names), label="인물 예측"),
            gr.Label(num_top_classes=2, label="표정 예측"),
        ],
        title="연예인 인물 식별 + 미소 감지",
        description="ResNet18 기반 멀티태스크 모델로 얼굴 사진에서 인물과 표정을 함께 예측합니다.",
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
