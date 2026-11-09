# 베어링 잔여수명(RUL) 예측

KSPHM-KIMM 기계 데이터 챌린지 2026에 팀 포크착기로 참가해 제출한 코드입니다. 47팀 중 17위를 했습니다.

25.6kHz로 측정한 4채널 진동 데이터(TDMS)만으로 베어링이 고장 날 때까지 남은 시간을 초 단위로 예측합니다. 데이터 읽기부터 제출 파일 생성까지 스크립트 하나로 실행됩니다.

## 맡은 부분

진동 신호에서 특징을 새로 뽑아 모델에 추가하고, 실제로 점수가 좋아지는지 검증하는 일을 맡았습니다. 특징 5종을 구현해 교차검증으로 비교했습니다.

## 문제

- 입력: 베어링 4채널 진동 신호(CH1~CH4, 1분 단위 TDMS 파일). 테스트 베어링은 고장 전 일부 구간만 주어집니다.
- 출력: 테스트 베어링 6개의 잔여수명(초)
- 평가: 대회 점수는 실제보다 짧게 예측했을 때 감점이 더 큽니다. 그래서 평균 오차를 줄이는 것보다 짧게 틀리지 않도록 안전 마진을 두는 데 신경을 많이 썼습니다.

## 방법

**1. 특징 추출**

파일마다 70개 넘는 특징을 뽑았습니다.

| 범주 | 특징 |
| --- | --- |
| 시간 영역 | RMS, IQR, Log-RMS, Shape Factor 등 |
| 주파수 영역 | Welch PSD 대역별 에너지 비율 |
| 결함 주파수 | BPFI(내륜)·BPFO(외륜) 1~3차 고조파 에너지 비율 |
| 포락선 | Hilbert 포락선 RMS, 결함 주파수 대역 비율, 첨도 |
| 웨이블릿 | DWT 세부 계수 에너지 비율 |
| 사이드밴드 | BPFI·BPFO 사이드밴드 비율 |
| 채널 간 상관 | 전·후방 채널 사이 상관계수 |
| 시퀀스 | 이동 집계, 초기 6개 파일 대비 상대값, 누적 열화량 |

- 테스트 데이터에는 RPM이 없어서 CH1 FFT의 5~25Hz 피크로 회전수를 추정해 700/950rpm 중 가까운 값으로 맞추고, 그 값으로 결함 주파수를 계산했습니다.
- 노이즈가 커서 채널마다 평균 ± 5σ 밖의 값은 잘라낸 뒤 특징을 뽑았습니다.

**2. 복합 건전성 지표(CHI)**

후방 수직 진동(CH4) 상대 RMS, 포락선 RMS, BPFI·BPFO 고조파 등 7개 특징을 가중합해 열화 정도를 나타내는 값 하나로 만들었습니다. 단조 증가를 억지로 맞추면 말기에 빨라지는 열화가 묻혀서, Savitzky-Golay 스무딩과 이동 중앙값만 적용했습니다.

**3. 예측 앵커 5가지**

| 앵커 | 방법 |
| --- | --- |
| ML 앙상블 | Ridge, BayesianRidge, RandomForest, ExtraTrees, GBM, LightGBM, XGBoost 최대 220개 모델로 log(RUL) 회귀 |
| 지수 외삽 | CHI 말기 구간을 로그 선형으로 맞춰 임계값에 닿는 시간 계산 |
| DTW 유사도 | 학습 베어링 궤적 중 가장 비슷한 3개를 찾아 RUL 가중 평균 |
| 위상 매칭 | CHI 수준과 변화율이 비슷한 학습 시점을 찾아 집계 |
| Library | 관측 길이, CHI, 변화율이 비슷한 학습 사례 표에서 RUL 조회 |

앵커 하나가 크게 튀어도 결과가 흔들리지 않도록 단순 평균 대신 가중 분위수로 섞었습니다.

**4. 상태별 최종 결정**

CHI 분위, 현재 수준, 변화율로 베어링 상태를 6가지(stable, mid, late, transient recovery, collapse, signal reversal)로 나누고, 상태마다 다른 식으로 최종 RUL을 정했습니다. 예를 들어 초기 안정 구간은 짧게 예측하지 않는 쪽으로, 말기는 길게 예측하지 않는 쪽으로 식을 다르게 둡니다.

**5. 검증과 보정**

- 베어링 하나를 빼고 학습한 뒤 빠진 베어링을 예측하는 LOBO(Leave-One-Bearing-Out) 교차검증을 관측 비율 30, 40, 50, 65, 80%에서 각각 돌렸습니다.
- 관측 비율마다 오차를 가장 줄이는 보정 계수를 grid search로 찾고, 최종 예측은 300초~90,000초 범위로 제한했습니다.
- 시드는 42로 고정했습니다.

## 실행 방법

```bash
pip install -r requirements.txt
export KSPHM_BASE=/path/to/data        # merged_features.csv와 Test/ 폴더가 있는 경로
export KSPHM_OUT_DIR=/path/to/output   # 결과 저장 경로 (선택)
python run_rul_bearing.py
```

대회 데이터는 배포할 수 없어서 저장소에 넣지 않았습니다. 실행하면 출력 폴더에 제출 파일 `포크착기_validation.xlsx`(`File`, `RUL_Score`)와 예측 근거를 정리한 진단 파일이 생깁니다.

## 제출 결과

[results/validation_submission.xlsx](results/validation_submission.xlsx)

| Test | 예측 RUL (초) | 예측 RUL (시간) |
| --- | ---: | ---: |
| Test1 | 45,420 | 12.6 |
| Test2 | 44,970 | 12.5 |
| Test3 | 20,508 | 5.7 |
| Test4 | 24,164 | 6.7 |
| Test5 | 33,405 | 9.3 |
| Test6 | 6,298 | 1.7 |

## 문서

- [Technical Report (PDF)](docs/technical_report.pdf): 대회에 제출한 기술 보고서
- [Notion 정리](https://app.notion.com/p/2786d3eae30781779174d6016655525f): 챌린지 진행 기록

## 구조

```
ksphm-2026-bearing-rul/
├── run_rul_bearing.py          # 전체 파이프라인
├── requirements.txt
├── results/
│   └── validation_submission.xlsx
└── docs/
    └── technical_report.pdf
```
