# ResNet18 기반 인물 식별 및 미소 감지 통합 분류 모델

연예인 8명(로제, 박보검, 아이유, 임영웅, 장원영, 제니, 차은우, 카리나)의 얼굴 사진에서
**① 누구인지(인물 식별)** 와 **② 웃고 있는지(미소 감지)** 를 하나의 모델로 동시에 예측하는
멀티태스크(Multi-Task) 딥러닝 프로젝트입니다.

크롤링 → 데이터 분할 → 얼굴 검출/전처리 → 모델 학습 → 평가까지 전체 파이프라인을 직접 구현했고,
학습된 딥러닝 모델의 표정 분류 성능을 전통적인 방식(OpenCV Haar Cascade)과 정량적으로 비교했습니다.

## 왜 두 가지를 동시에 예측하는가

인물 식별과 표정 분류는 서로 다른 문제처럼 보이지만, 둘 다 "같은 얼굴 이미지에서 다른 종류의 정보를 뽑아내는 일"이라는 공통점이 있습니다.
백본(ResNet18)을 공유하고 head만 분리하는 멀티태스크 구조로 학습하면, 각 태스크를 따로 학습할 때보다 적은 파라미터로 두 가지 문제를 함께 풀 수 있는지, 그리고 두 loss의 가중치(alpha/beta)를 어떻게 배분해야 하는지를 실험할 수 있습니다.

## 파이프라인

```
1. 크롤링          crawling.py           네이버 이미지 검색 결과 수집 → data/raw/{키워드}
2. 수동 분류        (수작업)              data/{연예인}/{neutral,smile}
3. 데이터 분할      initialize_data.py    (연예인,표정) 비율 유지 8:1:1 분할 → data/split/{train,val,test}
4. 얼굴 전처리      preprocessing.py      MTCNN 얼굴 검출·크롭·리사이즈(224x224) → data/processed
5. 모델 학습        train.py              ResNet18 전이학습 (freeze → unfreeze 2단계) → best_model_*.pth
6. 평가/벤치마크    evaluate.py           AI vs OpenCV Haar Cascade 비교, 혼동행렬/오류분석 시각화
```

## 폴더 구조

```
project4/
├── code/
│   ├── config.py           # 경로 설정 (다른 환경에 clone해도 수정 불필요)
│   ├── crawling.py         # 이미지 크롤러
│   ├── initialize_data.py  # 8:1:1 계층적 분할
│   ├── preprocessing.py    # MTCNN 얼굴 검출/크롭
│   ├── dataset.py          # 멀티태스크 Dataset/DataLoader
│   ├── model.py            # MultiTaskResNet (ResNet18 + 공유 레이어 + 2개 head)
│   ├── train.py            # 학습 스크립트
│   ├── evaluate.py         # 평가 및 시각화
│   └── results/            # 평가 결과 그래프(PNG)
├── data/                   # 원본/전처리 데이터 (용량·초상권 문제로 git 미포함)
└── requirements.txt
```

## 실행 방법

```bash
pip install -r requirements.txt

# 1) 연예인별로 키워드를 바꿔가며 이미지 수집
python code/crawling.py --keyword 아이유 --target_count 500

# 2) (수집한 이미지를 data/{연예인}/{neutral,smile}로 수동 분류했다고 가정하고) 8:1:1 분할
python code/initialize_data.py

# 3) 얼굴 검출/크롭
python code/preprocessing.py

# 4) 학습 (alpha=이름 loss 가중치, beta=미소 loss 가중치)
python code/train.py --alpha 1.0 --beta 0.5 --epochs 50

# 5) 평가 및 시각화
python code/evaluate.py
```

## 결과

`evaluate.py` 실행 시 `code/results/`에 아래 그래프가 저장됩니다. (수치는 실행 환경/데이터에 따라 달라질 수 있어, 최신 결과는 직접 재현해 확인하는 것을 권장합니다.)

| 그래프 | 내용 |
|---|---|
| `01_ppt_benchmark_result.png` | AI(ResNet18) vs OpenCV(Haar Cascade) 정확도/정밀도/재현율 비교 |
| `02_ppt_celeb_accuracy.png` | 연예인별 분류 정확도 |
| `03_ppt_error_donut.png` | 전체 정답 / 이름 오답 / 표정 오답 / 전체 오답 비율 |
| `04_ppt_error_top5.png` | 가장 많이 헷갈린 연예인 조합 TOP 5 |
| `05_ppt_confusion_matrix_celeb.png` | 연예인 분류 혼동행렬 |
| `06_ppt_confusion_matrix_smile.png` | 표정 분류 혼동행렬 |

## 한계와 배운 점

- 데이터가 인물당 148~273장 수준으로 많지 않아, 특정 인물(수집 난이도가 높았던 인물)의 정확도가 상대적으로 낮게 나옴 → 데이터 증강과 클래스 가중치로 일부 보완
- alpha/beta 가중치를 수동으로 바꿔가며 실험(1.0/0.5, 1.0/1.0, 1.0/1.5)한 결과를 비교해 두 태스크 간 균형점을 찾음
- OpenCV Haar Cascade 대비 딥러닝 모델의 우위를 수치로 확인해, "왜 딥러닝을 써야 하는가"에 대한 근거를 직접 검증함

## 데이터 및 윤리 고지

이 프로젝트의 이미지 데이터는 학습 목적으로 웹에서 수집한 공인(연예인)의 얼굴 사진이며,
**개인 학습·포트폴리오 용도로만 사용**하고 데이터셋 자체는 재배포하지 않습니다.
`code/results/All_Correct` 등 개별 인물 사진이 포함된 평가 결과 폴더도 리포지토리에는 포함하지 않았습니다(`.gitignore` 참고).

## 기술 스택

Python, PyTorch/torchvision(ResNet18 전이학습), facenet-pytorch(MTCNN), OpenCV, scikit-learn, Selenium/BeautifulSoup, matplotlib/seaborn
