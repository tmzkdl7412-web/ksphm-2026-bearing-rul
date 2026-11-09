# 베어링 잔여수명(RUL) 예측

KSPHM-KIMM 기계 데이터 챌린지 2026에 팀 포크착기로 참가해 제출한 코드입니다. 47팀 중 17위를 했습니다.

25.6kHz로 측정한 4채널 진동 데이터(TDMS)만으로 베어링이 고장 날 때까지 남은 시간을 초 단위로 예측합니다. 데이터 읽기부터 제출 파일 생성까지 `run_rul_bearing.py` 하나로 실행됩니다.

## 맡은 부분

진동 신호에서 특징을 새로 뽑아 모델에 추가하고, 실제로 점수가 좋아지는지 검증하는 일을 맡았습니다. 특징 5종을 구현해 교차검증으로 비교했습니다.

## 문제

- 입력: 베어링 4채널 진동 신호(CH1~CH4, 1분 단위 TDMS 파일). 테스트 베어링은 고장 전 일부 구간만 주어집니다.
- 출력: 테스트 베어링 6개의 잔여수명(초)
- 평가: 실제 RUL 대비 상대오차로 매기는 점수. 실제보다 길게 예측하면 짧게 예측할 때보다 점수가 훨씬 빨리 떨어집니다(상대오차 20%에서 점수가 절반, 짧게 예측하면 50%에서 절반). 그래서 평균 오차보다 어느 쪽으로 틀리는지를 더 신경 썼습니다.

## 방법

**1. 특징 추출**
- 채널별 RMS를 처음 6개 파일 평균으로 나눠 상대값으로 사용
- TKEO(Teager-Kaiser Energy Operator)로 결함 충격 성분 계산
- 테스트 데이터에는 RPM 정보가 없어서 FFT로 축 회전 주파수를 추정하고, 이를 기준으로 BPFI(내륜), BPFO(외륜), BSF(볼), FTF(케이지) 결함 주파수 성분 비율 계산
- 채널 간 상대 RMS 편차(`ch_diversity`)로 결함이 특정 위치에 몰려 있는지 표시

**2. 건전성 지표(HI)**
- 초기 대비 RMS 초과분을 채널별로 가중합해 누적 손상량을 만들고 `log1p`로 변환
- 채널 가중치는 학습한 값이 LOBO 점수를 확실히 올릴 때만 쓰고, 그렇지 않으면 균등 가중치 사용

**3. 두 모델 섞기**
- HI 모델: HI로 RUL을 회귀(선형, 더 잘 맞으면 2차)
- 물리 Ridge 모델: 손상률, 말기 손상 가속, 전·후방 채널 비율, 임펄스 점수 등으로 회귀
- 손상이 진행될수록(HI가 커질수록) Ridge 모델 비중을 높임
- 섞는 비율과 안전 마진은 대회 점수가 가장 높게 나오도록 grid search로 결정

**4. 상태별 보정**
- 테스트 베어링을 같은 관측 길이의 학습 베어링 분포와 비교해 6가지 상태(`active_growth`, `extreme_ood`, `baseline_ood_stable`, `early_mid_stable`, `late_but_not_terminal`, `mid_active`)로 나누고, 상태마다 다른 분위수를 최종 예측값으로 사용
- 보정에 쓰는 파라미터도 여러 관측 길이에서 LOBO로 골라 테스트 조건에 맞춤

**5. 검증**
- 베어링 하나를 빼고 학습한 뒤 빠진 베어링을 예측하는 LOBO(Leave-One-Bearing-Out) 교차검증을 대회 점수로 평가
- `SEED=42`로 고정해 같은 결과가 나오도록 함

## 실행 방법

```bash
pip install -r requirements.txt
export KSPHM_BASE=/path/to/data        # merged_features.csv와 Test/ 폴더가 있는 경로
export KSPHM_OUT_DIR=/path/to/output   # 결과 저장 경로 (선택)
python run_rul_bearing.py
```

대회 데이터는 배포할 수 없어서 저장소에 넣지 않았습니다. 테스트 TDMS를 한 번 읽으면 출력 폴더의 `.tdms_cache/`에 저장돼 다음 실행부터는 빨라집니다.

| 출력 파일 | 내용 |
| --- | --- |
| `포크착기_validation.xlsx` | 제출 파일 (`File`, `RUL_Score`) |
| `bearing_rul_v128_adaptive_all.xlsx` | 예측 근거, LOBO 결과, 학습 궤적, 설정값 |

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
