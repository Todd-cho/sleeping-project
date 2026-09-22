# SLP Sleeping Posture Classification

RGB 및 적외선(IR) 침상 이미지를 이용해 수면 자세를 **정상 자세(0)**와 **비정상 자세(1)**로 분류하는 딥러닝 프로젝트입니다. 이불이 없는 상태와 얇거나 두꺼운 이불을 덮은 상태를 각각 학습·평가하여 촬영 조건과 가림 정도에 따른 분류 성능을 비교합니다.

## 주요 기능

- RGB/IR 이미지 기반 수면 자세 이진 분류
- `uncover`, `cover1`, `cover2` 조건별 모델 학습
- ImageNet 사전 학습 EfficientNet-B0 사용
- Accuracy, Precision, Recall, F1-score 평가
- Confusion Matrix, ROC Curve, 예측 CSV 생성

## 데이터 구성

데이터는 영상 종류와 이불 조건에 따라 다음과 같이 구성됩니다.

```text
data/
├── RGB/
│   ├── uncover/
│   ├── cover1/
│   └── cover2/
└── IR/
    ├── uncover/
    ├── cover1/
    └── cover2/
```

각 조건 아래에는 `train`, `val`, `test` 데이터와 클래스 폴더 `0`, `1`이 위치합니다. 데이터셋은 용량이 크기 때문에 Git 저장소에는 포함하지 않습니다.

## 프로젝트 구성

```text
├── load_data.py   # 원본 SLP 데이터를 학습 구조로 분리
├── model.py       # EfficientNet-B0 이진 분류 모델
├── train.py       # 학습 및 검증
├── test.py        # 테스트와 성능 지표·그래프 생성
└── pyproject.toml # 프로젝트 및 Python 환경 설정
```

## 실행 흐름

1. `load_data.py`에서 원본 SLP 데이터를 학습·검증·테스트 세트로 구성합니다.
2. `train.py`에서 선택한 modality와 cover 조건으로 모델을 학습합니다.
3. 가장 높은 검증 F1-score를 기록한 모델을 `checkpoints/`에 저장합니다.
4. `test.py`에서 여섯 가지 RGB/IR 및 이불 조건의 체크포인트를 평가합니다.

```bash
python train.py
python test.py
```

학습 조건은 `train.py` 상단의 `modality`, `cover`, `CFG` 설정에서 변경할 수 있습니다.

## 모델

기본 모델은 ImageNet으로 사전 학습된 EfficientNet-B0입니다. 마지막 분류 계층을 단일 logit 출력으로 변경하고 `BCEWithLogitsLoss`를 사용하여 두 클래스를 분류합니다. 입력 이미지는 비율을 유지한 채 정사각형으로 패딩한 후 224×224 크기로 변환합니다.

## 출력 결과

- `checkpoints/best-{modality}-{cover}.pt`: 조건별 최적 모델
- `predictions_test_*.csv`: 이미지별 정답, 예측 확률 및 예측 클래스
- `*_cm.png`: Confusion Matrix
- `*_roc.png`: ROC Curve

## 참고

이 프로젝트는 연구 및 실험 목적으로 작성되었습니다. `load_data.py`의 원본 데이터 경로는 실행 환경에 맞게 수정해야 하며, 데이터셋의 이용 조건과 개인정보 관련 규정을 확인한 뒤 사용해야 합니다.
