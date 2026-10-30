# ⚙️ 베어링 잔여수명(RUL) 예측 — KSPHM-KIMM 기계 데이터 챌린지 2026

> 팀 **포크착기** · KSPHM-KIMM 기계 데이터 챌린지 2026 참가 · [📒 Notion 정리](https://app.notion.com/p/2786d3eae30781779174d6016655525f)

25.6 kHz 4채널 진동 데이터(TDMS)만으로 베어링이 고장 나기까지 남은 시간(Remaining Useful Life)을 예측하는 파이프라인입니다.
데이터 로드 → 특징 추출 → 건전성 지표(HI) 생성 → 모델 학습·검증 → 제출 파일 생성까지 **단일 스크립트**로 실행됩니다.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![PHM](https://img.shields.io/badge/Domain-PHM%20%2F%20Predictive%20Maintenance-2E7D32)

## 문제

- **입력**: 베어링 4채널 진동 신호(CH1~CH4, 1분 단위 TDMS 파일), 테스트 베어링은 고장 전 일부 구간만 관측
- **출력**: 테스트 베어링 6개 각각의 잔여수명(초)
- **평가**: 실제 RUL 대비 상대오차 기반의 **비대칭 점수 함수** — 과대예측과 과소예측의 감점 속도가 달라서, 평균 오차보다 "어느 쪽으로 틀리느냐"가 중요

## 접근 방법

```
TDMS 진동 데이터
   ↓  특징 추출 (RMS, TKEO, 결함 주파수 order 분석)
누적 손상 기반 건전성 지표 (Cumulative Damage HI)
   ↓
① HI 회귀 모델 (선형/2차)  +  ② 물리 피처 Ridge 회귀
   ↓  손상 단계별 가변 가중치로 블렌딩
사전 예측 (prior)
   ↓  유사 베어링 분포 비교 + 상태 분류 (6가지)
최종 RUL  →  포크착기_validation.xlsx
```

### 1. 특징 추출
- **채널별 RMS**를 초기 6개 파일 평균 대비 상대값으로 정규화
- **TKEO (Teager-Kaiser Energy Operator)**: 결함 충격의 진폭·주파수 변조를 포착
- **Order 분석**: FFT로 축 회전 주파수를 추정한 뒤 BPFI(내륜)·BPFO(외륜)·BSF(볼)·FTF(케이지) 결함 주파수 성분 비율 계산 → RPM 정보 없이도 결함 신호 추적
- 채널 간 상대 RMS의 편차(`ch_diversity`)로 결함 위치의 국소성 반영

### 2. 누적 손상 건전성 지표 (HI)
- 초기 대비 RMS 초과분을 채널 가중합해 **누적 손상량**으로 쌓고 `log1p`로 변환
- 채널 가중치는 균등 vs 학습 가중치 중 LOBO 교차검증 점수가 확실히 좋을 때만 학습 가중치 채택

### 3. 두 모델 블렌딩
- **HI 모델**: 누적 손상 HI → RUL 회귀 (단조 감소를 보장하는 선형, 더 잘 맞으면 2차)
- **물리 Ridge 모델**: 손상률, 말기 손상 가속, 전·후방 채널 비율, 임펄스 점수 등 물리 피처 회귀
- 손상 단계가 진행될수록(HI↑) 물리 모델 비중을 높이는 **phase-adaptive 가중치**
- 블렌드 가중치와 안전 마진은 대회 점수 함수를 직접 최대화하도록 grid search

### 4. 상태 인식 결정 레이어
- 테스트 베어링과 같은 관측 시점의 학습 베어링 분포(K50 레퍼런스, 거리 기반 이웃)와 비교
- `active_growth`, `extreme_ood`, `baseline_ood_stable`, `early_mid_stable`, `late_but_not_terminal`, `mid_active` 등으로 **상태를 분류**하고, 상태마다 다른 분위수를 목표값으로 선택
- 가드 파라미터도 여러 관측 길이에서 LOBO로 선택해 테스트 조건에 맞춤

### 5. 검증
- **LOBO (Leave-One-Bearing-Out)** 교차검증: 베어링 하나씩 빼고 학습 → 빠진 베어링 예측 → 대회 점수로 평가
- 시드 고정(`SEED=42`)으로 결과 재현 가능

## 실행 방법

```bash
pip install -r requirements.txt
export KSPHM_BASE=/path/to/data        # merged_features.csv 와 Test/ 폴더가 있는 경로
export KSPHM_OUT_DIR=/path/to/output   # (선택) 결과 저장 경로
python run_rul_bearing.py
```

- 대회 데이터는 배포 권한이 없어 저장소에 포함하지 않았습니다.
- 테스트 TDMS 파싱 결과는 출력 폴더의 `.tdms_cache/`에 캐시되어 재실행 시 빨라집니다.

**출력 파일**
| 파일 | 내용 |
|------|------|
| `포크착기_validation.xlsx` | 공식 제출 파일 (`File`, `RUL_Score`) |
| `bearing_rul_v128_adaptive_all.xlsx` | 예측 근거, LOBO 결과, 학습 궤적, 설정값 등 진단 워크북 |

## 제출 결과

[`results/validation_submission.xlsx`](results/validation_submission.xlsx)

| Test | 예측 RUL (초) | 예측 RUL (시간) |
|------|---:|---:|
| Test1 | 45,420 | 12.6 h |
| Test2 | 44,970 | 12.5 h |
| Test3 | 20,508 | 5.7 h |
| Test4 | 24,164 | 6.7 h |
| Test5 | 33,405 | 9.3 h |
| Test6 | 6,298 | 1.7 h |

## 문서

- [Technical Report (PDF)](docs/technical_report.pdf) — 대회 제출용 기술 보고서
- [Notion 정리](https://app.notion.com/p/2786d3eae30781779174d6016655525f) — 챌린지 진행 과정 기록

## 구조

```
ksphm-2026-bearing-rul/
├── run_rul_bearing.py          # 전체 파이프라인 (v128_adaptive)
├── requirements.txt
├── results/
│   └── validation_submission.xlsx
└── docs/
    └── technical_report.pdf
```
