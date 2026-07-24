"""
demo.py
-------
학습된 멀티태스크 모델(best_model_*.pth)로, 업로드한 얼굴 사진에서
연예인 이름과 표정(웃음/무표정)을 함께 예측해 보여주는 Streamlit 데모.

별도의 얼굴 검출 없이 업로드된 이미지 전체를 224x224로 리사이즈해 모델에 입력하므로,
사용자가 얼굴이 나온 사진을 직접 업로드해야 한다.

모델과 classes_*.json은 Hugging Face Hub(config.HF_REPO_ID)에서 자동으로 다운로드한다.
train.py가 체크포인트와 같이 저장한 classes_*.json(클래스 순서)을 그대로 불러와 사용한다.
"""
import json
import os
import sys

import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

# cv2가 자체적으로 가진 config 모듈과 이름이 충돌하지 않도록, 이 파일이 있는 code/ 디렉터리를
# sys.path 맨 앞에 넣어 로컬 config.py를 우선 찾도록 한다.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import HF_CACHE_DIR, HF_CLASSES_FILENAME, HF_MODEL_FILENAME, HF_REPO_ID
from model import MultiTaskResNet

EMOTION_NAMES = ["무표정", "미소"]
IMAGE_SIZE = (224, 224)

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_MODEL_FILENAME, cache_dir=HF_CACHE_DIR)
    classes_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_CLASSES_FILENAME, cache_dir=HF_CACHE_DIR)

    with open(classes_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    model = MultiTaskResNet(num_celebs=len(class_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    return model, class_names, device


def predict(image, model, class_names, device):
    img = image.convert("RGB").resize(IMAGE_SIZE)
    tensor = TRANSFORM(img).unsqueeze(0).to(device)

    with torch.no_grad():
        c_out, s_out = model(tensor)
        c_prob = torch.softmax(c_out, dim=1)[0].cpu().numpy()
        s_prob = torch.softmax(s_out, dim=1)[0].cpu().numpy()

    celeb_result = {name: float(p) for name, p in zip(class_names, c_prob)}
    smile_result = {name: float(p) for name, p in zip(EMOTION_NAMES, s_prob)}
    return celeb_result, smile_result


def main():
    st.title("연예인 인물 식별 + 미소 감지")
    st.write("ResNet18 기반 멀티태스크 모델로 얼굴 사진에서 인물과 표정을 함께 예측합니다.")

    model, class_names, device = load_model()

    uploaded_file = st.file_uploader("얼굴 사진 업로드", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        return

    image = Image.open(uploaded_file)
    st.image(image, caption="업로드한 이미지", use_container_width=True)

    celeb_result, smile_result = predict(image, model, class_names, device)

    st.write("### 인물 예측")
    for name, prob in sorted(celeb_result.items(), key=lambda x: x[1], reverse=True):
        st.write(f"{name}: {prob:.2%}")

    st.write("### 표정 예측")
    for name, prob in sorted(smile_result.items(), key=lambda x: x[1], reverse=True):
        st.write(f"{name}: {prob:.2%}")


if __name__ == "__main__":
    main()
