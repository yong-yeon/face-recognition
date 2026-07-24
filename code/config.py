"""
config.py
---------
프로젝트 전역에서 공통으로 사용하는 경로 설정.
이 파일(config.py)의 위치를 기준으로 상대경로를 계산하기 때문에,
다른 컴퓨터에 그대로 clone해도 경로를 수정할 필요가 없다.
"""
import os

# code/ 폴더
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
# 프로젝트 최상위 폴더
PROJECT_ROOT = os.path.dirname(CODE_DIR)

# 데이터 관련 경로
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
SPLIT_DIR = os.path.join(DATA_DIR, "split")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MANUAL_CHECK_DIR = os.path.join(DATA_DIR, "manual_check")

# 학습 결과물(체크포인트, 그래프, 로그) 저장 경로
RESULTS_DIR = os.path.join(CODE_DIR, "results")

# Hugging Face Hub에 배포된 모델 관련 설정
HF_REPO_ID = "choyongyeon/face-recognition-model"
HF_MODEL_FILENAME = "best_model_a1.0_b0.5.pth"
HF_CLASSES_FILENAME = "classes_a1.0_b0.5.json"
# hf_hub_download가 내려받은 파일을 캐싱해두는 위치
HF_CACHE_DIR = os.path.join(CODE_DIR, "hf_cache")
