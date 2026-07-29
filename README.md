# ResNet18 기반 인물 식별 및 미소 감지 통합 분류 모델

연예인 8명의 얼굴 이미지에서 **① 누구인지(인물 식별)** 와 **② 웃고 있는지(미소 감지)** 를 단일 모델로 동시에 예측하는 멀티태스크 딥러닝 프로젝트입니다.

**Demo** → https://yy-face-recognition.streamlit.app
**발표 자료** → [ResNet18_Portfolio.pdf](docs/ResNet18_Portfolio.pdf)

---

## 왜 두 가지를 동시에 예측하는가

인물 식별과 표정 분류는 다른 문제처럼 보이지만, 둘 다 "같은 얼굴 이미지에서 다른 종류의 정보를 뽑아내는 일"이라는 공통점이 있습니다. 백본(ResNet18)을 공유하고 head만 분리하는 멀티태스크 구조로 학습하면, 각 태스크를 따로 학습할 때보다 적은 파라미터로 두 가지 문제를 함께 풀 수 있는지, 그리고 두 loss의 가중치(α/β)를 어떻게 배분해야 하는지를 실험할 수 있습니다.

---

## 파이프라인

| 단계 | 스크립트 | 설명 |
|---|---|---|
| 1. 크롤링 | `code/crawling.py` | 네이버 이미지 검색 → `data/raw/{키워드}` |
| 2. 수동 분류 | (수작업) | `data/{연예인}/{neutral,smile}` |
| 3. 데이터 분할 | `code/initialize_data.py` | (인물+표정) 비율 유지 8:1:1 계층적 분할 |
| 4. 얼굴 전처리 | `code/preprocessing.py` | MTCNN 얼굴 검출·크롭·리사이즈(224×224) |
| 5. 모델 학습 | `code/train.py` | ResNet18 전이학습 (동결 → 해제 2단계) |
| 6. 평가 | `code/evaluate.py` | AI vs OpenCV Haar Cascade 비교, 혼동행렬 시각화 |
| 7. 데모 | `code/demo.py` | Streamlit 웹앱 |

---

## 모델 구조

```
입력 이미지 (224×224)
        ↓
ResNet18 Backbone (ImageNet 사전학습)
        ↓
Shared Layer (512) · LayerNorm + ReLU
        ↙          ↘
  celeb_head      smile_head
  (8 classes)      (2 classes)
```

Loss = α × celeb_loss + β × smile_loss

---

## 성능 결과

### 가중치 실험 (α/β 비교)

| α / β | 이름 정확도 | 표정 정확도 |
|---|---|---|
| **1.0 / 0.5** | **86.2%** | **82.4% ★ 채택** |
| 1.0 / 1.0 | 85.1% | 79.7% |
| 1.0 / 1.5 | 86.6% | 81.2% |

β를 낮춰 표정에 치우치지 않도록 조정했으나, β를 올릴수록 두 지표가 동시에 오르지 않음 → α:β = 1.0:0.5 채택.

### AI vs OpenCV Haar Cascade (표정 분류 정확도)

| | 정확도 | 정밀도 | 재현율 |
|---|---|---|---|
| **AI 모델** | **0.82** | 0.86 | **0.85** |
| OpenCV Haar | 0.57 | 0.92 | 0.34 |

OpenCV는 정밀도가 높지만 재현율이 0.34로 낮아 실제 웃음을 절반 이상 놓침 → 실용성 부족.

### 연예인별 이름 분류 정확도

최고: 아이유·차은우 0.976 / 최저: 카리나 0.742 (외모 유사성·데이터 품질 차이)

평가 그래프는 `code/results/`에 저장됩니다.

---

## 프로젝트 구조

```
project4/
├── code/
│   ├── config.py            # 경로 설정
│   ├── crawling.py          # 이미지 크롤러
│   ├── initialize_data.py   # 8:1:1 계층적 분할
│   ├── preprocessing.py     # MTCNN 얼굴 검출·크롭
│   ├── dataset.py           # 멀티태스크 Dataset/DataLoader
│   ├── model.py             # MultiTaskResNet (백본 + 공유레이어 + 2개 head)
│   ├── train.py             # 학습 스크립트
│   ├── evaluate.py          # 평가 및 시각화
│   ├── demo.py              # Streamlit 웹앱
│   └── results/             # 평가 그래프 (PNG)
├── docs/                    # 발표 자료 PDF
├── data/                    # 학습 데이터 (초상권 문제로 미포함)
└── requirements.txt
```

---

## 실행 방법

```bash
git clone https://github.com/yong-yeon/face-recognition.git
cd face-recognition
pip install -r requirements.txt
```

**데이터 수집**
```bash
python code/crawling.py --keyword 아이유 --target_count 500
# data/{연예인}/{neutral,smile} 로 수동 분류 후
```

**전처리 및 학습**
```bash
python code/initialize_data.py          # 8:1:1 분할
python code/preprocessing.py           # MTCNN 얼굴 크롭
python code/train.py --alpha 1.0 --beta 0.5 --epochs 50
```

**평가 및 데모**
```bash
python code/evaluate.py                # 그래프 저장
python code/demo.py                    # Streamlit 웹앱 실행
```

---

## 데이터 및 윤리 고지

이 프로젝트의 이미지 데이터는 학습 목적으로 웹에서 수집한 공인(연예인)의 얼굴 사진입니다. 개인 학습·포트폴리오 용도로만 사용하며 데이터셋 자체는 재배포하지 않습니다. 평가 결과 이미지(code/results/All_Correct 등)도 저장소에 포함하지 않았습니다.

---

## 기술 스택

Python · PyTorch(ResNet18 전이학습) · facenet-pytorch(MTCNN) · OpenCV · scikit-learn · Selenium/BeautifulSoup · Streamlit · matplotlib/seaborn

---

## 개발 정보

- **기간**: 2026.05 ~ 2026.07
- **인원**: 1인 (개인 프로젝트)
