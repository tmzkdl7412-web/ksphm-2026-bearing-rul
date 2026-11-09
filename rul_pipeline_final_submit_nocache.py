# -*- coding: utf-8 -*-
"""
KSPHM-KIMM 2026 | RUL Pipeline Final Submission
===================================
"""

from __future__ import annotations

import os
import glob
import time
import zipfile
import tempfile
import warnings
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.signal import hilbert
from scipy.stats import pearsonr
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── 선택적 의존성 ──────────────────────────────────────────────────
try:
    import lightgbm as lgb
    LGB_OK = True
except Exception:
    LGB_OK = False

try:
    import xgboost as xgb
    XGB_OK = True
except Exception:
    XGB_OK = False

try:
    import pywt
    PYWT_OK = True
except Exception:
    PYWT_OK = False

try:
    from dtaidistance import dtw as _dtw_lib
    DTAIDIST_OK = True
except Exception:
    DTAIDIST_OK = False


# =====================================================================
# CONFIG
# =====================================================================
@dataclass
class Config:
    seed: int = 42
    team_name: str = "포크착기"

    max_rul_s: int = 90_000
    min_rul_s: int = 300
    tpf_s: int = 600
    init_n: int = 6

    fs_tdms: int = 25_600
    bpfi_coef: float = 140.0 / 1000.0
    bpfo_coef: float = 93.0  / 1000.0

    rpm_candidates: Tuple[float, ...] = (700.0, 950.0)

    n_full_models: int = 220
    n_cv_models: int   = 60

    cutoffs: Tuple[float, ...] = (0.30, 0.40, 0.50, 0.65, 0.80)
    k50_obs_files: int = 50
    run_k50_cv: bool = True
    k50_weight: float = 5.0

    anchor_weights: Dict[str, float] = field(default_factory=lambda: {
        "ml": 1.00, "ep": 0.85, "dtw": 0.60, "phase": 0.90,
    })
    consensus_weight: float = 0.55
    overprediction_guard_mult: float = 1.75

    # 상태 기반으로 조정하는 통합 파라미터
    # - stable_q ↑ : 안정 구간 과소예측 방지
    # - conservative_q ↑ : 중기 구간에서 지나치게 낮은 앵커 하나에 끌리지 않게 함
    # - late_q ↑ : late여도 Train 유사 케이스/phase 앵커가 받쳐주면 극단적인 단기 예측을 완화
    conservative_q: float = 0.56
    stable_q: float = 0.72
    late_q: float = 0.55
    library_prior_weight: float = 1.20
    ml_weight: float = 0.90
    min_anchor_agreement: int = 3

    # CV가 특정 stage를 과도하게 낮게 보정하면 미관측 Test 구간에서 불안정해질 수 있어
    # stage별 reference scale을 둔다. 값 자체는 Test 번호가 아니라 stage 분위 기준이다.
    fallback_stage_scale: Dict[float, float] = field(default_factory=lambda: {
        0.30: 1.25, 0.40: 1.20, 0.50: 1.25, 0.65: 0.85, 0.80: 0.85,
    })
    min_stage_scale: Dict[float, float] = field(default_factory=lambda: {
        0.30: 1.25, 0.40: 1.15, 0.50: 1.20, 0.65: 0.80, 0.80: 0.80,
    })

    # 상태 기반 reference. CHI/변화율/앵커 상태로만 작동
    stable_support_h: float = 12.0
    stable_reference_h: float = 13.0
    baseline_reference_h: float = 10.0
    late_support_h: float = 6.0
    late_reference_h: float = 7.0
    recovery_support_h: float = 6.5
    recovery_reference_h: float = 7.5
    reversal_reference_h: float = 2.0

    # onset 감지 threshold를 노이즈 환경 고려해 상향
    onset_threshold_std: float = 3.5

    out_subdir: str = "rul_output"



CFG = Config()
# 재현성 확보: 모든 난수 사용 지점을 Config.seed로 통일
os.environ["PYTHONHASHSEED"] = str(CFG.seed)
np.random.seed(CFG.seed)
random.seed(CFG.seed)


# =====================================================================
# 경로
# =====================================================================
def resolve_base_dir() -> str:
    candidates: List[str] = []
    env = os.environ.get("KSPHM_BASE")
    if env:
        candidates.append(env)
    candidates += [os.getcwd(), str(Path(__file__).resolve().parent), "/data"]
    for cand in candidates:
        cand = os.path.abspath(os.path.expanduser(cand))
        merged_ok = os.path.exists(os.path.join(cand, "merged_features.csv"))
        test_ok   = os.path.isdir(os.path.join(cand, "Test"))
        train_ok  = os.path.isdir(os.path.join(cand, "Train"))
        if (merged_ok or train_ok) and test_ok:
            return cand
    raise FileNotFoundError(
        "데이터 폴더를 찾지 못했습니다. "
        "KSPHM_BASE 환경변수를 대회 데이터 폴더로 지정하세요.\n"
        "예: export KSPHM_BASE=/path/to/challenge_data"
    )


BASE      = resolve_base_dir()
TRAIN_CSV = os.path.join(BASE, "merged_features.csv")
TEST_DIR  = os.path.join(BASE, "Test")
TRAIN_DIR = os.path.join(BASE, "Train")
OUT_DIR   = os.path.abspath(
    os.environ.get("KSPHM_OUT_DIR", os.path.join(BASE, CFG.out_subdir))
)
os.makedirs(OUT_DIR, exist_ok=True)


# =====================================================================
# 피처 정의
# =====================================================================
LEAKY_TOKENS = (
    "rul", "remaining", "target", "label",
    "op_time", "time_sec", "tdms_order", "file_index", "file_idx",
)

PDF_FEATURES: List[str] = [
    # RMS
    "CH4_rms_rpm_norm", "CH3_rms_rpm_norm",
    "CH2_rms_rpm_norm", "CH1_rms_rpm_norm",
    # SI 트렌드
    "SI_rollmean_6", "SI_rollmean_3",
    # RMS (초기 정규화)
    "CH4_rms_rel_init6", "CH3_rms_rel_init6", "CH1_rms_rel_init6",
    # 채널 간 상관
    "corr_CH1_CH4", "corr_CH2_CH3", "corr_CH2_CH4", "corr_CH3_CH4", "corr_CH1_CH3",
    # Envelope BPFI
    "Vibration_CH3_env_band_bpfi_2x_ratio", "Vibration_CH1_env_band_bpfi_2x_ratio",
    "Vibration_CH4_env_band_bpfi_2x_ratio", "Vibration_CH1_env_band_bpfi_1x_ratio",
    "Vibration_CH4_env_band_bpfi_1x_ratio",
    # 주파수 대역
    "Vibration_CH3_band_b3k_6k_ratio", "Vibration_CH4_band_b3k_6k_ratio",
    "Vibration_CH4_band_b6k_9k_ratio", "Vibration_CH1_band_b6k_9k_ratio",
    # IQR / 분위
    "CH3_iqr", "CH4_iqr", "CH1_iqr", "CH4_q75", "CH3_q75", "CH1_q75",
    # BPFI/BPFO 고조파
    "CH4_band_0_100_energy_ratio",
    "CH4_bpfi_h3_energy_ratio", "CH2_bpfi_h3_energy_ratio",
    "CH1_bpfi_h3_peak_amp", "CH3_bpfi_h3_peak_amp", "CH4_bpfi_h3_peak_amp",
    "CH4_bpfo_h2_energy_ratio", "CH2_bpfi_h2_energy_ratio",
    "CH4_bpfi_h2_energy_ratio", "CH2_bpfo_h1_energy_ratio",
    # DWT 웨이블릿
    "CH1_wav_cD3_energy_ratio", "CH4_wav_cD2_energy_ratio", "CH3_wav_cD2_energy_ratio",
    # 사이드밴드
    "Vibration_CH2_sideband_bpfi_sb_ratio", "Vibration_CH2_sideband_bpfo_sb_ratio",
    "Vibration_CH4_sideband_bpfi_sb_ratio", "Vibration_CH4_sideband_bpfo_sb_ratio",
    # 롤링 집계
    "rms_all_max_rollmean_6", "rms_all_mean_rollmean_6",
    "env_rms_all_mean_rollmean_6", "env_kurtosis_all_max_rollmean_6",
    "kurtosis_all_mean_rollmean_6", "env_rms_all_rel_init6",
    # 방향 비율
    "ratio_vert_axial_rear_rms", "ratio_frontAx_rearAx_rms",
    # 형상/엔트로피/포락
    "CH1_shape_factor", "CH4_spec_entropy", "CH1_spec_entropy",
    "Vibration_CH2_env_peak_power",
    # 로그 RMS + 포락 평균
    "CH4_log_rms", "CH3_log_rms", "CH1_env_rms", "CH1_env_mean", "CH4_env_mean",
    # 광대역 고주파
    "Vibration_CH4_band_hf_3k_10k_ratio", "Vibration_CH3_band_hf_3k_10k_ratio",
]

CHI_FEATS: List[Tuple[str, float]] = [
    ("CH4_rms_rel_init6",                     0.30),
    ("CH3_rms_rel_init6",                     0.20),
    ("env_rms_all_mean_rollmean_6",            0.20),
    ("CH4_bpfi_h3_energy_ratio",              0.10),
    ("CH4_bpfo_h2_energy_ratio",              0.08),
    ("Vibration_CH4_env_band_bpfi_2x_ratio",  0.07),
    ("CH4_band_0_100_energy_ratio",           0.05),
]


# =====================================================================
# 유틸리티
# =====================================================================
def score_fn(actual: float, pred: float) -> float:
    actual = max(float(actual), 1e-6)
    pred   = max(float(pred),   1e-6)
    er = 100.0 * (actual - pred) / actual
    return float(np.exp(-np.log(0.5) * er / 20.0) if er <= 0
                 else np.exp(np.log(0.5) * er / 50.0))


def clip_rul(x: float) -> float:
    return float(np.nan) if not np.isfinite(x) else float(np.clip(x, CFG.min_rul_s, CFG.max_rul_s))


def safe_to_excel(df: pd.DataFrame, path: str, **kwargs) -> str:
    try:
        df.to_excel(path, **kwargs)
        return path
    except PermissionError:
        base, ext = os.path.splitext(path)
        alt = f"{base}_{time.strftime('%Y%m%d_%H%M%S')}{ext}"
        print(f"  [경고] {os.path.basename(path)} 열려있음 → {os.path.basename(alt)}로 저장")
        df.to_excel(alt, **kwargs)
        return alt


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[mask], w[mask]
    if len(v) == 0:
        return float(np.nan)
    order = np.argsort(v)
    v, w  = v[order], w[order]
    cw    = np.cumsum(w)
    return float(v[np.searchsorted(cw, 0.5 * cw[-1])])


def stage_from_chi(chi_q: float) -> float:
    q = 0.50 if (chi_q is None or not np.isfinite(chi_q)) else float(chi_q)
    if q < 0.25: return 0.30
    if q < 0.40: return 0.40
    if q < 0.58: return 0.50
    if q < 0.72: return 0.65
    return 0.80


def smooth_median(x: Sequence[float], w: int = 3) -> np.ndarray:
    return (pd.Series(np.asarray(x, dtype=float))
            .rolling(w, min_periods=1, center=False)
            .median().to_numpy(dtype=float))


def smooth_sg(x: Sequence[float], window: int = 5, polyorder: int = 2) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if len(arr) < window:
        return arr.copy()
    w = min(window, len(arr))
    if w % 2 == 0:
        w = max(3, w - 1)
    if w <= polyorder + 1:
        return arr.copy()
    try:
        return sp_signal.savgol_filter(arr, window_length=w, polyorder=polyorder, mode="interp")
    except Exception:
        return arr.copy()


def rate_ratio(y: Sequence[float], tail: int = 8) -> float:
    arr = np.asarray(y, dtype=float)
    if len(arr) < 4:
        return 0.0
    k     = min(tail, len(arr))
    denom = max(abs(float(np.nanmedian(arr[: max(2, CFG.init_n)]))), 1e-9)
    try:
        slope = np.polyfit(np.arange(k, dtype=float), arr[-k:], 1)[0]
    except Exception:
        slope = 0.0
    return float(slope / denom)

def snap_rpm(rpm: float) -> float:
    try:
        rpm = float(rpm)
    except Exception:
        return float(CFG.rpm_candidates[0])
    cands = np.array(CFG.rpm_candidates, dtype=float)
    return float(cands[np.argmin(np.abs(cands - rpm))])


def estimate_rpm(ch1: np.ndarray, fs: int = CFG.fs_tdms) -> float:
    try:
        x = np.asarray(ch1, dtype="float64")
        x = x[np.isfinite(x)]
        if len(x) < 8192:
            return snap_rpm(825.0)
        win    = min(65536, len(x))
        # astype(int) 사용
        starts = np.linspace(0, max(0, len(x) - win),
                             num=min(6, max(1, len(x) // win + 1))).astype(int)
        rpms: List[float] = []
        for st in starts:
            seg = x[st: st + win].copy()
            if len(seg) < 8192:
                continue
            seg -= np.mean(seg)
            sd   = np.std(seg) + 1e-12
            seg  = np.clip(seg, -5 * sd, 5 * sd)
            f    = np.fft.rfftfreq(len(seg), 1.0 / fs)
            p    = np.abs(np.fft.rfft(seg)) ** 2
            mask = (f >= 5) & (f <= 25)
            if mask.any():
                rpms.append(float(f[mask][np.argmax(p[mask])]) * 60.0)
        raw = float(np.median(rpms)) if rpms else 825.0
        return snap_rpm(raw)
    except Exception:
        return snap_rpm(825.0)


# =====================================================================
# 피처 추출 (TDMS → 파일당 dict)
# =====================================================================
def extract_file_feat(arr: np.ndarray, rpm: float,
                      fs: int = CFG.fs_tdms) -> Dict[str, float]:
    """
    rpm 인수가 항상 snap_rpm을 거친 값이어야 함.
    호출 전에 snap_rpm() 또는 estimate_rpm()을 통과시킬 것.
    """
    chs   = ["CH1", "CH2", "CH3", "CH4"]
    tol   = 3.0
    bpfi  = CFG.bpfi_coef * rpm
    bpfo  = CFG.bpfo_coef * rpm
    f_rpm = rpm / 60.0
    feat: Dict[str, float] = {"rpm": float(rpm)}
    sigs: Dict[str, np.ndarray] = {}

    for i, ch in enumerate(chs):
        if i >= arr.shape[1]:
            continue
        x  = arr[:, i].astype("float64")
        mu = float(np.mean(x)); sd = float(np.std(x)) + 1e-12
        x  = np.clip(x, mu - 5 * sd, mu + 5 * sd) - mu
        sigs[ch] = x

        rms  = float(np.sqrt(np.mean(x ** 2)))
        q75, q25 = np.percentile(x, [75, 25])
        mabs = float(np.mean(np.abs(x)))
        feat[f"{ch}_rms"]          = rms
        feat[f"{ch}_iqr"]          = float(q75 - q25)
        feat[f"{ch}_q75"]          = float(q75)
        feat[f"{ch}_log_rms"]      = float(np.log1p(rms))
        feat[f"{ch}_mean_abs"]     = mabs
        feat[f"{ch}_shape_factor"] = float(rms / (mabs + 1e-12))

        f_, psd = sp_signal.welch(x, fs=fs, nperseg=4096, noverlap=2048)
        tot     = float(np.sum(psd)) + 1e-12
        pn      = psd / tot
        feat[f"{ch}_spec_entropy"] = float(-np.sum(pn * np.log(pn + 1e-12)))

        for lo, hi, tag in [
            (0,      100,   "0_100"),
            (1000,   3000,  "b1k_3k"),
            (3000,   6000,  "b3k_6k"),
            (3000,   10000, "hf_3k_10k"),
            (6000,   9000,  "b6k_9k"),
            (10000,  12800, "b10k_12k8"),
        ]:
            idx = (f_ >= lo) & (f_ < hi)
            r   = float(np.sum(psd[idx]) / tot)
            feat[f"{ch}_band_{tag}_energy_ratio"]    = r
            feat[f"Vibration_{ch}_band_{tag}_ratio"] = r

        for h in (1, 2, 3):
            fc_ = bpfi * h
            idx = (f_ >= max(0, fc_ - tol)) & (f_ <= fc_ + tol)
            feat[f"{ch}_bpfi_h{h}_energy_ratio"] = float(np.sum(psd[idx]) / tot)
            feat[f"{ch}_bpfi_h{h}_peak_amp"]     = float(np.max(psd[idx])) if idx.any() else 0.0
        for h in (1, 2):
            fc_ = bpfo * h
            idx = (f_ >= max(0, fc_ - tol)) & (f_ <= fc_ + tol)
            feat[f"{ch}_bpfo_h{h}_energy_ratio"] = float(np.sum(psd[idx]) / tot)

        for tag_sb, f_center in (("bpfi", bpfi), ("bpfo", bpfo)):
            sb = 0.0
            for n in (1, 2, 3):
                for side in (-1, 1):
                    f_sb = f_center + side * n * f_rpm
                    if f_sb <= 0:
                        continue
                    ix = (f_ >= max(0, f_sb - tol)) & (f_ <= f_sb + tol)
                    sb += float(np.sum(psd[ix]))
            feat[f"Vibration_{ch}_sideband_{tag_sb}_sb_ratio"] = sb / tot

        env     = np.abs(hilbert(x))
        env_rms = float(np.sqrt(np.mean(env ** 2)))
        feat[f"{ch}_env_rms"]      = env_rms
        feat[f"{ch}_env_mean"]     = float(np.mean(env))
        feat[f"{ch}_env_peak_amp"] = float(np.max(env))
        feat[f"{ch}_env_kurt"]     = float(
            np.mean((env - np.mean(env)) ** 4) / (np.std(env) ** 4 + 1e-12))

        ef, epsd = sp_signal.welch(env, fs=fs, nperseg=4096, noverlap=2048)
        etot = float(np.sum(epsd)) + 1e-12
        feat[f"Vibration_{ch}_env_peak_power"] = float(np.max(epsd) / etot)

        for mult, label in ((1, "1x"), (2, "2x")):
            fc_ = bpfi * mult
            eix = (ef >= max(0, fc_ - tol)) & (ef <= fc_ + tol)
            feat[f"Vibration_{ch}_env_band_bpfi_{label}_ratio"] = float(
                np.sum(epsd[eix]) / etot)

        if PYWT_OK:
            try:
                cfs  = pywt.wavedec(x, "db4", level=5)
                wtot = sum(np.sum(c ** 2) for c in cfs[1:]) + 1e-12
                feat[f"{ch}_wav_cD3_energy_ratio"] = float(np.sum(cfs[-3] ** 2) / wtot)
                feat[f"{ch}_wav_cD2_energy_ratio"] = float(np.sum(cfs[-2] ** 2) / wtot)
                feat[f"{ch}_dwt_D4_rms"]           = float(np.sqrt(np.mean(cfs[2] ** 2)))
                feat[f"{ch}_dwt_D5_rms"]           = float(np.sqrt(np.mean(cfs[1] ** 2)))
                d5   = cfs[1]
                prob = d5 ** 2 / (np.sum(d5 ** 2) + 1e-12)
                feat[f"{ch}_dwt_D5_entropy"] = float(-np.sum(prob * np.log2(prob + 1e-12)))
            except Exception:
                for suf in ("_wav_cD3_energy_ratio", "_wav_cD2_energy_ratio",
                            "_dwt_D4_rms", "_dwt_D5_rms", "_dwt_D5_entropy"):
                    feat[f"{ch}{suf}"] = 0.0

        try:
            nyq   = fs / 2.0
            b, a  = sp_signal.butter(4, [1000 / nyq, 5000 / nyq], btype="band")
            x_bpf = sp_signal.filtfilt(b, a, x)
            ebpf  = np.abs(hilbert(x_bpf))
            feat[f"{ch}_bpf_env_rms"]  = float(np.sqrt(np.mean(ebpf ** 2)))
            feat[f"{ch}_bpf_env_kurt"] = float(
                np.mean((ebpf - np.mean(ebpf)) ** 4) / (np.std(ebpf) ** 4 + 1e-12))
        except Exception:
            feat[f"{ch}_bpf_env_rms"]  = 0.0
            feat[f"{ch}_bpf_env_kurt"] = 0.0

    for ca, cb in (("CH1","CH4"),("CH2","CH3"),("CH1","CH3"),("CH2","CH4"),("CH3","CH4")):
        if ca in sigs and cb in sigs:
            try:
                r, _ = pearsonr(sigs[ca], sigs[cb])
                feat[f"corr_{ca}_{cb}"] = float(r)
            except Exception:
                feat[f"corr_{ca}_{cb}"] = 0.0
    return feat


def add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    rms_rel_rpm_group와 rms_rel_init6의 중복 해소.

    두 피처의 차이:
      - rms_rel_init6   : 각 베어링의 첫 init_n 파일 평균 대비 비율 (베어링 내 초기화)
      - rms_rpm_norm    : rpm_group(0=저속, 1=고속)별로 나눈 별도 baseline 대비 비율
                          → rpm_group 컬럼이 없으면 rms_rel_init6과 동일 계산으로 fallback
                          → 이 경우 get_feature_cols에서 중복 제거됨
    """
    df = df.copy()
    if "file_idx" in df.columns and "tdms_order" not in df.columns:
        df["tdms_order"] = df["file_idx"]
    df = df.sort_values(["train_id", "tdms_order"]).reset_index(drop=True)

    rms_cols   = [f"CH{i}_rms"      for i in range(1, 5) if f"CH{i}_rms"      in df.columns]
    env_cols   = [f"CH{i}_env_rms"  for i in range(1, 5) if f"CH{i}_env_rms"  in df.columns]
    ekurt_cols = [f"CH{i}_env_kurt" for i in range(1, 5) if f"CH{i}_env_kurt" in df.columns]

    if rms_cols:
        df["rms_all_mean"] = df[rms_cols].mean(axis=1)
        df["rms_all_max"]  = df[rms_cols].max(axis=1)
    if env_cols:
        df["env_rms_all_mean"] = df[env_cols].mean(axis=1)
    if ekurt_cols:
        df["env_kurtosis_all_max"] = df[ekurt_cols].max(axis=1)
        df["kurtosis_all_mean"]    = df[ekurt_cols].mean(axis=1)

    # rpm_group별 baseline 사전 계산 (전체 데이터에서, 베어링별)
    has_rpm_group = "rpm_group" in df.columns

    for bid, g in df.groupby("train_id", sort=False):
        idx = g.sort_values("tdms_order").index
        gs  = df.loc[idx]

        # --- 롤링 집계 ---
        for base, out, w in [
            ("env_rms_all_mean",     "env_rms_all_mean_rollmean_6",     6),
            ("env_kurtosis_all_max", "env_kurtosis_all_max_rollmean_6", 6),
            ("rms_all_max",          "rms_all_max_rollmean_6",          6),
            ("rms_all_mean",         "rms_all_mean_rollmean_6",         6),
            ("kurtosis_all_mean",    "kurtosis_all_mean_rollmean_6",    6),
        ]:
            if base in df.columns:
                df.loc[idx, out] = gs[base].rolling(w, min_periods=1).mean().values

        # --- SI ---
        if "rms_all_mean" in df.columns:
            init = float(gs["rms_all_mean"].iloc[: CFG.init_n].mean()) + 1e-12
            si   = gs["rms_all_mean"] / init
            df.loc[idx, "SI"]            = si.values
            df.loc[idx, "SI_rollmean_3"] = si.rolling(3, min_periods=1).mean().values
            df.loc[idx, "SI_rollmean_6"] = si.rolling(6, min_periods=1).mean().values

        # --- env_rms 초기 정규화 ---
        if "env_rms_all_mean" in df.columns:
            init = float(gs["env_rms_all_mean"].iloc[: CFG.init_n].mean()) + 1e-12
            df.loc[idx, "env_rms_all_rel_init6"] = (gs["env_rms_all_mean"] / init).values

        # --- 채널별 초기 정규화 (rms_rel_init6) ---
        for ch in ("CH1", "CH2", "CH3", "CH4"):
            col = f"{ch}_rms"
            if col not in df.columns:
                continue
            init_val = float(gs[col].iloc[: CFG.init_n].mean()) + 1e-12
            df.loc[idx, f"{ch}_rms_rel_init6"] = (gs[col] / init_val).values

        # rpm_group이 있으면 그룹별 baseline, 없으면 init6 값과 동일
        for ch in ("CH1", "CH2", "CH3", "CH4"):
            col     = f"{ch}_rms"
            out_col = f"{ch}_rms_rpm_norm"
            if col not in df.columns:
                continue
            norm_vals = np.empty(len(gs), dtype=float)
            if has_rpm_group:
                # rpm_group별로 첫 init_n개 평균을 각 그룹의 baseline으로 사용
                for rg in gs["rpm_group"].dropna().unique():
                    grp_mask = gs["rpm_group"] == rg
                    grp_vals = gs.loc[grp_mask, col].values
                    # 해당 그룹의 첫 init_n개 (시간순이므로 그냥 앞에서부터)
                    n_init = min(CFG.init_n, len(grp_vals))
                    baseline = float(np.mean(grp_vals[:n_init])) + 1e-12
                    norm_vals[grp_mask.values] = gs.loc[grp_mask, col].values / baseline
                # rpm_group이 NaN인 행은 전체 init6으로 fallback
                nan_mask = gs["rpm_group"].isna().values
                if nan_mask.any():
                    fb = float(gs[col].iloc[: CFG.init_n].mean()) + 1e-12
                    norm_vals[nan_mask] = gs[col].values[nan_mask] / fb
            else:
                # rpm_group 없음 → init6 정규화와 동일 (get_feature_cols에서 중복 탐지됨)
                fb = float(gs[col].iloc[: CFG.init_n].mean()) + 1e-12
                norm_vals[:] = gs[col].values / fb
            df.loc[idx, out_col] = norm_vals

        # --- 방향 비율 ---
        if "CH3_rms" in df.columns and "CH4_rms" in df.columns:
            df.loc[idx, "ratio_vert_axial_rear_rms"] = (
                gs["CH3_rms"] / gs["CH4_rms"].clip(1e-12)).values
        if "CH2_rms" in df.columns and "CH4_rms" in df.columns:
            df.loc[idx, "ratio_frontAx_rearAx_rms"] = (
                gs["CH2_rms"] / gs["CH4_rms"].clip(1e-12)).values

        # --- 열화 누적 피처 ---
        for col in [c for c in
                    ["CH1_rms","CH2_rms","CH3_rms","CH4_rms",
                     "CH1_env_rms","CH2_env_rms","CH3_env_rms","CH4_env_rms"]
                    if c in df.columns]:
            vals   = gs[col].fillna(0).to_numpy(dtype=float)
            base   = float(np.mean(vals[: CFG.init_n])) + 1e-12
            rel    = vals / base
            cmax   = np.maximum.accumulate(rel)
            cumdmg = np.cumsum(np.maximum(vals - base, 0.0)) / base
            df.loc[idx, col + "_rel"]    = rel
            df.loc[idx, col + "_cmax"]   = cmax
            df.loc[idx, col + "_cumdmg"] = cumdmg

    return df


# =====================================================================
# merged_features.csv 자동 빌드
# =====================================================================
def build_merged_features(train_dir: str, out_csv: str) -> pd.DataFrame:
    """
    Train/*.zip (TDMS) → merged_features.csv 자동 생성.
    이미 CSV가 있으면 로드만 하고 스킵.
    RUL 계산: 각 베어링의 마지막 파일 tdms_order가 EOL.
              RUL = (max_tdms_order - current_tdms_order) * tpf_s
    """
    try:
        from nptdms import TdmsFile
    except ImportError:
        raise ImportError("TDMS 추출에 nptdms가 필요합니다: pip install nptdms")

    print("  [build_merged_features] Train TDMS에서 피처 추출 시작...")
    all_dfs: List[pd.DataFrame] = []

    # glob으로 탐색
    zip_paths = sorted(glob.glob(os.path.join(train_dir, "Train*.zip")))
    if not zip_paths:
        raise RuntimeError(f"Train*.zip을 찾을 수 없습니다: {train_dir}")

    for zp in zip_paths:
        tid = Path(zp).stem   # e.g. "Train1"
        print(f"    {tid}: 추출 중...", flush=True)
        rows: List[Dict] = []
        with zipfile.ZipFile(zp) as zf:
            tdms_names = sorted(n for n in zf.namelist() if n.lower().endswith(".tdms"))
            with tempfile.TemporaryDirectory() as tmp:
                zf.extractall(tmp)
                for name in tdms_names:
                    # 파일명 파싱 안전 처리
                    try:
                        file_idx = int(Path(name).stem)
                    except ValueError:
                        # 순수 숫자가 아닌 파일명 → 정렬 순번으로 대체
                        file_idx = tdms_names.index(name)

                    try:
                        td = TdmsFile.read(str(Path(tmp) / name)).as_dataframe()
                        td.columns = [str(c).split("/")[-1].strip("'") for c in td.columns]
                        ch_cols = [c for c in ["CH1","CH2","CH3","CH4"] if c in td.columns]
                        if len(ch_cols) < 2 or len(td) < 10000:
                            continue
                        arr = td[ch_cols].to_numpy(dtype="float32")
                        while arr.shape[1] < 4:
                            arr = np.hstack([arr, arr[:, -1:]])
                        rpm  = estimate_rpm(arr[:, 0])
                        feat = extract_file_feat(arr, rpm)
                        feat["train_id"]   = tid
                        feat["tdms_order"] = file_idx
                        feat["rpm_group"]  = 1 if rpm >= float(np.mean(CFG.rpm_candidates)) else 0
                        rows.append(feat)
                    except Exception as exc:
                        print(f"      [WARN] {name}: {exc}")

        if not rows:
            print(f"    [WARN] {tid}: 추출된 파일 없음, 스킵")
            continue

        df_t = pd.DataFrame(rows).sort_values("tdms_order").reset_index(drop=True)
        # RUL 계산 (파일 간격 기반)
        max_order = int(df_t["tdms_order"].max())
        df_t["rul_sec"] = (max_order - df_t["tdms_order"]) * CFG.tpf_s
        all_dfs.append(df_t)
        print(f"    {tid}: {len(df_t)} files, RUL [{df_t['rul_sec'].min():.0f}~{df_t['rul_sec'].max():.0f}]s")

    if not all_dfs:
        raise RuntimeError("빌드된 Train 데이터가 없습니다.")

    merged = pd.concat(all_dfs, ignore_index=True)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"  [build_merged_features] 저장 완료: {out_csv}  shape={merged.shape}")
    return merged


# =====================================================================
# 데이터 로드
# =====================================================================
def prepare_train_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename: Dict[str, str] = {}
    if "file_index" in df.columns and "tdms_order" not in df.columns:
        rename["file_index"] = "tdms_order"
    if "RUL_sec" in df.columns and "rul_sec" not in df.columns:
        rename["RUL_sec"] = "rul_sec"
    if rename:
        df = df.rename(columns=rename)
    required = {"train_id", "tdms_order", "rul_sec"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Train CSV에 필수 컬럼 없음: {missing}")

    if "op_rpm_mean" in df.columns:
        df["rpm_est_snap"] = df["op_rpm_mean"].apply(snap_rpm)
        df["rpm_group"]    = (df["rpm_est_snap"] >= float(np.mean(CFG.rpm_candidates))).astype(int)

    df = add_sequence_features(df)
    print(f"  [TRAIN] shape={df.shape}")
    for bid, g in df.groupby("train_id"):
        rmin = g["rul_sec"].min(); rmax = g["rul_sec"].max()
        print(f"    {bid}: {len(g)} files  RUL [{rmin:.0f}~{rmax:.0f}]s")
    return df


def extract_test_df(zip_path: str, test_id: str) -> pd.DataFrame:
    try:
        from nptdms import TdmsFile
    except ImportError as exc:
        raise ImportError("pip install nptdms") from exc

    print(f"  {test_id}: TDMS 추출 중...", flush=True)
    rows: List[Dict] = []
    with zipfile.ZipFile(zip_path) as zf:
        tdms_names = sorted(n for n in zf.namelist() if n.lower().endswith(".tdms"))
        with tempfile.TemporaryDirectory() as tmp:
            zf.extractall(tmp)
            for seq_idx, name in enumerate(tdms_names):
                try:
                    file_idx = int(Path(name).stem)
                except ValueError:
                    file_idx = seq_idx
                    print(f"    [WARN] {name}: 파일명 정수 파싱 실패, 순번 {seq_idx} 사용")

                try:
                    td = TdmsFile.read(str(Path(tmp) / name)).as_dataframe()
                    td.columns = [str(c).split("/")[-1].strip("'") for c in td.columns]
                    ch_cols = [c for c in ["CH1","CH2","CH3","CH4"] if c in td.columns]
                    if len(ch_cols) < 2 or len(td) < 10000:
                        continue
                    arr = td[ch_cols].to_numpy(dtype="float32")
                    while arr.shape[1] < 4:
                        arr = np.hstack([arr, arr[:, -1:]])
                    rpm  = estimate_rpm(arr[:, 0])
                    feat = extract_file_feat(arr, rpm)
                    feat["file_idx"]   = file_idx
                    feat["tdms_order"] = file_idx
                    feat["rpm_group"]  = 1 if rpm >= float(np.mean(CFG.rpm_candidates)) else 0
                    feat["train_id"]   = test_id
                    rows.append(feat)
                except Exception as exc:
                    print(f"    [WARN] {name}: {exc}")

    if not rows:
        raise RuntimeError(f"{test_id}: 추출된 TDMS 없음")
    out = pd.DataFrame(rows).sort_values("tdms_order").reset_index(drop=True)
    out = add_sequence_features(out)
    print(f"    {test_id}: {len(out)} files 완료")
    return out


def load_test_features(test_dir: str) -> pd.DataFrame:
    """glob으로 Test*.zip 자동 탐색 (하드코딩 range 제거)."""
    zip_paths = sorted(glob.glob(os.path.join(test_dir, "Test*.zip")))
    if not zip_paths:
        raise RuntimeError(f"Test*.zip을 찾지 못했습니다: {test_dir}")

    dfs = []
    for zp in zip_paths:
        tid = Path(zp).stem   # zip 파일명 기준 ID
        try:
            dfs.append(extract_test_df(zp, tid))
        except Exception as exc:
            print(f"  [WARN] {tid} 추출 실패: {exc}")

    if not dfs:
        raise RuntimeError("추출된 Test 데이터 없음")
    df = pd.concat(dfs, ignore_index=True)
    print(f"  [TEST] shape={df.shape}")
    return df



def get_feature_cols(df_tr: pd.DataFrame, df_te: pd.DataFrame) -> List[str]:
    extra_suffix = (
        "_dwt_D4_rms", "_dwt_D5_rms", "_dwt_D5_entropy",
        "_bpf_env_rms", "_bpf_env_kurt", "_rel", "_cmax", "_cumdmg",
    )
    candidates: List[str] = []
    for c in PDF_FEATURES:
        if c in df_tr.columns and c in df_te.columns:
            candidates.append(c)
    for c in df_tr.columns:
        if c in df_te.columns and any(c.endswith(s) for s in extra_suffix):
            candidates.append(c)

    # 중복 피처 제거: 상관계수 > 0.999인 쌍의 후발 피처를 제거
    out: List[str] = []
    seen: set = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if any(tok in c.lower() for tok in LEAKY_TOKENS):
            continue
        if not pd.api.types.is_numeric_dtype(df_tr[c]):
            continue
        col_tr = pd.to_numeric(df_tr[c], errors="coerce")
        if float(col_tr.std(skipna=True) or 0.0) <= 1e-10:
            continue
        out.append(c)

    # 고상관 중복 제거 (r > 0.999)
    if len(out) > 1:
        vals    = df_tr[out].apply(pd.to_numeric, errors="coerce").fillna(0).values
        to_drop = set()
        for i in range(len(out)):
            if out[i] in to_drop:
                continue
            for j in range(i + 1, len(out)):
                if out[j] in to_drop:
                    continue
                try:
                    r = float(np.corrcoef(vals[:, i], vals[:, j])[0, 1])
                except Exception:
                    r = 0.0
                if abs(r) > 0.999:
                    to_drop.add(out[j])
                    print(f"  [중복제거] {out[j]} (r={r:.4f} with {out[i]})")
        out = [c for c in out if c not in to_drop]

    if len(out) < 10:
        raise RuntimeError(f"사용 가능한 피처 부족: {len(out)}개")
    print(f"  사용 피처: {len(out)}개")
    return out


def compute_chi(df_group: pd.DataFrame,
                feat_weights: Sequence[Tuple[str, float]]) -> Optional[np.ndarray]:
    """
    IsotonicRegression 제거.
    단조증가를 강제하면 말기 변화율 정보(급격한 가속·진동)가 손실됨.
    Savitzky-Golay 스무딩으로 노이즈만 제거하고 자연스러운 트렌드를 보존.
    """
    n = len(df_group)
    if n == 0:
        return None
    chi     = np.zeros(n, dtype=float)
    total_w = 0.0
    for col, w in feat_weights:
        if col not in df_group.columns:
            continue
        vals = pd.to_numeric(df_group[col], errors="coerce").fillna(0).to_numpy(dtype=float)
        base = float(np.mean(vals[: CFG.init_n])) + 1e-12
        chi     += float(w) * (vals / base)
        total_w += float(w)
    if total_w <= 1e-9:
        return None
    chi /= total_w
    # 단조증가 강제 없이 스무딩만 적용
    chi = smooth_sg(smooth_median(chi, 3), window=5, polyorder=2)
    return chi


def build_reference(train_df: pd.DataFrame,
                    chi_feats: Sequence[Tuple[str, float]]) -> Dict:
    """
    Train 베어링의 CHI 궤적과 시점별 RUL 라이브러리를 만든다.
    v3에서는 Test 번호를 쓰지 않고, 현재 관측 길이/CHI/변화율과 가장 비슷한
    Train 시점을 찾아 하나의 추가 앵커(Library prior)로 사용한다.
    """
    chi_dict: Dict[str, np.ndarray]     = {}
    train_smooth: Dict[str, np.ndarray] = {}
    train_rul: Dict[str, np.ndarray]    = {}
    eol_rels: Dict[str, float]          = {}
    phase_rows: List[Dict]              = []

    for bid, g in train_df.groupby("train_id"):
        gs  = g.sort_values("tdms_order").reset_index(drop=True)
        chi = compute_chi(gs, chi_feats)
        if chi is None:
            continue

        bid_s = str(bid)
        rr_all = gs["rul_sec"].to_numpy(dtype=float)
        chi_dict[bid_s]     = chi
        train_smooth[bid_s] = chi
        train_rul[bid_s]    = rr_all
        eol_rels[bid_s]     = float(chi[-1])

        n = len(chi)
        for i in range(max(CFG.init_n, 1), n):
            phase_rows.append(dict(
                train_id=bid_s,
                idx=i,
                obs_files=i + 1,
                obs_frac=(i + 1) / max(n, 1),
                chi=float(chi[i]),
                curr_rel=float(chi[i]),
                rr=float(rate_ratio(chi[: i + 1])),
                rul_sec=float(rr_all[i]),
                total_files=n,
            ))

    if not chi_dict:
        raise RuntimeError("CHI reference를 만들 수 없습니다.")

    all_chi     = np.concatenate(list(chi_dict.values()))
    fthr_global = float(np.median(list(eol_rels.values())))

    phase_table = pd.DataFrame(phase_rows)
    if not phase_table.empty:
        phase_table["chi_q"] = phase_table["chi"].apply(lambda v: float(np.mean(all_chi <= float(v))))

    print(f"  EOL CHI: { {k: round(v,3) for k,v in eol_rels.items()} }")
    return dict(
        chi_dict=chi_dict, train_smooth=train_smooth, train_rul=train_rul,
        eol_rels=eol_rels, all_chi=all_chi, fthr_global=fthr_global,
        phase_table=phase_table,
    )

def detect_onset(y: Sequence[float]) -> int:
    """
    threshold_std를 CFG.onset_threshold_std(기본 3.5)로 상향.
    대회 요강 "노이즈 많은 환경"에서 2.5는 초반 오감지를 유발함.
    CUSUM 방식으로 변경해 누적 편차 기반 감지 (단발 스파이크 무시).
    """
    arr    = np.asarray(y, dtype=float)
    base   = arr[: CFG.init_n]
    mu, sd = float(np.mean(base)), float(np.std(base)) + 1e-12
    k_cusum = CFG.onset_threshold_std * sd * 0.5   # CUSUM slack
    thr_cusum = CFG.onset_threshold_std * sd

    cusum = 0.0
    for i in range(CFG.init_n, len(arr)):
        cusum = max(0.0, cusum + (arr[i] - mu) - k_cusum)
        if cusum > thr_cusum:
            return int(i)
    return 0


# =====================================================================
# 앵커: EP / DTW / Phase
# =====================================================================
def exp_extrapolate(y: Sequence[float], fthr: float,
                    onset_idx: int = 0) -> Optional[float]:
    arr = np.asarray(y, dtype=float)
    if len(arr) < 6 or not np.isfinite(fthr):
        return None
    start = max(onset_idx, max(0, len(arr) - 18))
    seg   = arr[start:]
    mask  = np.isfinite(seg) & (seg > 0)
    if mask.sum() < 4:
        return None
    try:
        slope, intercept = np.polyfit(
            np.arange(len(seg), dtype=float)[mask],
            np.log(np.clip(seg[mask], 1e-9, None)), 1)
    except Exception:
        return None
    if slope <= 1e-5:
        return None
    t_fail = (np.log(max(fthr, 1e-9)) - intercept) / slope
    rem    = t_fail - (len(seg) - 1)
    if not np.isfinite(rem) or rem <= 0:
        return None
    return float(np.clip(rem * CFG.tpf_s, CFG.min_rul_s, CFG.max_rul_s * 2))


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    if DTAIDIST_OK:
        try:
            return float(_dtw_lib.distance_fast(
                a.astype(np.double), b.astype(np.double)))
        except Exception:
            pass
    # numpy fallback (길이 제한으로 속도 보장)
    a = a[:60]; b = b[:60]
    n, m = len(a), len(b)
    dp   = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        lo = max(1, i - 15); hi = min(m, i + 15)
        for j in range(lo, hi + 1):
            dp[i, j] = abs(a[i-1] - b[j-1]) + min(dp[i-1,j], dp[i,j-1], dp[i-1,j-1])
    return float(dp[n, m])


def similarity_fthr(test_sm: np.ndarray, ref: Dict) -> float:
    train_smooth = ref["train_smooth"]
    eol_rels     = ref["eol_rels"]
    fthr_global  = float(ref["fthr_global"])
    if len(test_sm) < 4:
        return fthr_global
    vals = []
    for bid, tr in train_smooth.items():
        k = min(len(test_sm), len(tr))
        if k < 4:
            continue
        d = _dtw_distance(test_sm[-k:], tr[:k])
        vals.append((d, eol_rels[bid]))
    if not vals:
        return fthr_global
    vals.sort(key=lambda x: x[0])
    top = vals[: min(3, len(vals))]
    ws  = np.array([1.0 / (d + 1e-6) for d, _ in top])
    es  = np.array([e for _, e in top])
    return float(np.sum(ws * es) / np.sum(ws))


def traj_sim_dtw(test_sm: np.ndarray, ref: Dict) -> Optional[float]:
    """
    이중 루프 최적화.
        각 훈련 궤적을 test_sm 길이로 resample한 뒤 단일 DTW 비교.
        정렬 위치는 EOL-relative scaling으로 추정.
    """
    train_smooth = ref["train_smooth"]
    train_rul    = ref["train_rul"]
    eol_rels     = ref["eol_rels"]
    n_test = len(test_sm)

    candidates = []
    cur_level = float(test_sm[-1]) if n_test > 0 else 0.0

    for bid, tr in train_smooth.items():
        m = len(tr)
        if m < 5:
            continue
        # EOL relative position: 현재 test HI level이 훈련 궤적의 어느 지점에 해당하는지
        eol_j = max(float(eol_rels.get(bid, float(tr[-1]))), 1e-6)
        # test 현재 level의 훈련 궤적 내 위치 추정
        rel_pos = min(cur_level / eol_j, 0.999)
        est_k   = max(4, int(rel_pos * m))

        # test_sm을 est_k 길이로 resample, 훈련 궤적의 앞 est_k와 비교
        k = min(n_test, est_k, m)
        if k < 4:
            continue
        test_rs = np.interp(
            np.linspace(0, 1, k),
            np.linspace(0, 1, n_test),
            test_sm,
        )
        tr_seg  = tr[:k]
        d = _dtw_distance(test_rs, tr_seg)

        rul_idx = min(est_k, m) - 1
        rul_est = float(train_rul[bid][rul_idx])
        candidates.append((d, rul_est))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    top = candidates[: min(3, len(candidates))]
    ws  = np.array([1.0 / (d + 1e-6) for d, _ in top])
    rs  = np.array([r for _, r in top])
    return clip_rul(float(np.sum(ws * rs) / np.sum(ws)))


def phase_rul_est(test_sm: np.ndarray, ref: Dict) -> Optional[float]:
    train_smooth = ref["train_smooth"]
    train_rul    = ref["train_rul"]
    if len(test_sm) < 5:
        return None
    cur   = float(test_sm[-1])
    slope = rate_ratio(test_sm)
    candidates = []
    for bid, tr in train_smooth.items():
        rr = train_rul[bid]
        for i in range(CFG.init_n, len(tr)):
            val_pen   = abs(float(tr[i]) - cur) / (abs(cur) + 1e-9)
            slope_pen = abs(rate_ratio(tr[: i + 1]) - slope)
            score     = val_pen + 0.7 * slope_pen
            candidates.append((score, rr[i]))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    top = candidates[: min(5, len(candidates))]
    ws  = np.array([1.0 / (s + 1e-6) for s, _ in top])
    rs  = np.array([r for _, r in top])
    return clip_rul(float(np.sum(ws * rs) / np.sum(ws)))


# =====================================================================
# ML 앙상블
# =====================================================================
class ModelEnsemble:
    def __init__(self, models: list, scaler: Optional[StandardScaler] = None):
        self.models = models
        self.scaler = scaler

    def predict_log(self, X: np.ndarray) -> np.ndarray:
        X  = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        Xs    = self.scaler.transform(X) if self.scaler is not None else X
        preds = []
        for m in self.models:
            try:
                p = m.predict(Xs) if isinstance(m, (Ridge, BayesianRidge)) else m.predict(X)
                preds.append(p)
            except Exception:
                continue
        if not preds:
            raise RuntimeError("모든 모델 예측 실패")
        return np.clip(np.mean(np.vstack(preds), axis=0), 0.0, np.log1p(CFG.max_rul_s))

    def predict_sec_row(self, row: np.ndarray) -> float:
        return float(np.expm1(self.predict_log(row)[0]))


def build_ensemble(X: np.ndarray, y_log: np.ndarray, n_models: int) -> ModelEnsemble:
    X     = np.asarray(X, dtype=float)
    y_log = np.asarray(y_log, dtype=float)
    sc    = StandardScaler().fit(X)
    Xs    = sc.transform(X)
    models: list = []

    for alpha in (0.4, 1.0, 2.0, 5.0):
        models.append(Ridge(alpha=alpha, random_state=CFG.seed).fit(Xs, y_log))
    models.append(BayesianRidge().fit(Xs, y_log))

    rounds = max(4, n_models // 18)
    for i in range(rounds):
        seed = CFG.seed + i * 13
        models.append(RandomForestRegressor(
            n_estimators=250, max_depth=None, min_samples_leaf=2,
            random_state=seed, n_jobs=-1).fit(X, y_log))
        models.append(ExtraTreesRegressor(
            n_estimators=250, max_depth=None, min_samples_leaf=2,
            random_state=seed + 1, n_jobs=-1).fit(X, y_log))
        models.append(GradientBoostingRegressor(
            n_estimators=170, learning_rate=0.035, max_depth=2,
            min_samples_leaf=2, random_state=seed + 2).fit(X, y_log))
        if LGB_OK:
            models.append(lgb.LGBMRegressor(
                n_estimators=260, learning_rate=0.025, num_leaves=15,
                min_child_samples=4, subsample=0.85, colsample_bytree=0.85,
                reg_alpha=0.05, reg_lambda=0.20,
                random_state=seed + 3, verbosity=-1).fit(X, y_log))
        if XGB_OK:
            models.append(xgb.XGBRegressor(
                n_estimators=260, learning_rate=0.025, max_depth=3,
                subsample=0.85, colsample_bytree=0.85,
                reg_alpha=0.05, reg_lambda=0.20,
                objective="reg:squarederror", random_state=seed + 4,
                n_jobs=-1).fit(X, y_log))

    print(f"  앙상블 모델 수: {len(models)}")
    return ModelEnsemble(models=models, scaler=sc)


# =====================================================================
# 결정 레이어
# =====================================================================
def clean_anchor(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        x = float(x)
    except Exception:
        return None
    if not np.isfinite(x) or x <= 0 or x > CFG.max_rul_s * 2:
        return None
    return float(np.clip(x, CFG.min_rul_s, CFG.max_rul_s))



def weighted_quantile(values: Sequence[float], weights: Sequence[float], q: float) -> float:
    """가중 분위수. q=0.5이면 weighted_median과 동일한 역할."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[mask], w[mask]
    if len(v) == 0:
        return float("nan")
    q = float(np.clip(q, 0.01, 0.99))
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    return float(v[np.searchsorted(cw, q * cw[-1], side="left")])


def library_prior_rul(test_chi: Optional[np.ndarray], ref: Dict, n_obs: int,
                      chi_q: float, rr: float) -> Optional[float]:
    """
    Train의 모든 중간 시점 중 현재 Test와 상태가 비슷한 지점을 찾는다.
    Test_ID나 목표값 없이, 관측 파일 수/CHI 분위/변화율만 사용한다.
    """
    tab = ref.get("phase_table")
    if tab is None or len(tab) == 0 or test_chi is None or len(test_chi) < CFG.init_n:
        return None

    obs_files = int(n_obs)
    curr = float(test_chi[-1])
    rr = float(rr)

    # 각 항목은 서로 스케일이 다르므로 robust scale로 나눠준다.
    obs_scale = max(float(np.nanmedian(tab["total_files"])), 1.0)
    rr_scale = max(float(np.nanmedian(np.abs(tab["rr"]))), 1e-4)
    chi_scale = max(float(np.nanstd(tab["curr_rel"])), 1e-4)

    score = (
        0.40 * np.abs(tab["obs_files"].to_numpy(dtype=float) - obs_files) / obs_scale +
        0.95 * np.abs(tab["chi_q"].to_numpy(dtype=float) - float(chi_q)) +
        0.45 * np.abs(tab["curr_rel"].to_numpy(dtype=float) - curr) / chi_scale +
        0.35 * np.abs(tab["rr"].to_numpy(dtype=float) - rr) / rr_scale
    )

    k = int(min(max(10, np.sqrt(len(tab)) * 2), len(tab)))
    idx = np.argsort(score)[:k]
    near = tab.iloc[idx]
    w = 1.0 / (score[idx] + 1e-6)

    # 말기일수록 낮은 분위수, 안정 구간일수록 중앙값 이상을 사용한다.
    # 단, rr 하나만 튄다고 바로 late로 보지 않는다. CHI 수준과 함께 판단한다.
    if chi_q >= 0.86 or (chi_q >= 0.78 and rr > 0.035):
        q = CFG.late_q
    elif chi_q < 0.35 and rr < 0.004:
        q = CFG.stable_q
    else:
        q = CFG.conservative_q
    return clip_rul(weighted_quantile(near["rul_sec"].to_numpy(dtype=float), w, q))


def classify_health_state(chi_q: float, curr_rel: float, rr: float,
                          anchors: Dict[str, Optional[float]]) -> str:
    """
    현재 상태를 stable/mid/late/collapse/transient_recovery/signal_reversal로 분류한다.

    핵심:
      1) rr 하나만 크다고 late로 보지 않음 → 중기 구간 과소예측 완화
      2) high CHI + 음의 tail slope는 transient pattern 회복으로 보고 극단값 완화
      3) EP가 계산되지 않고 CHI가 낮은데 tail slope가 강하게 음수면 signal_reversal reference 적용
    """
    vals = [clean_anchor(v) for v in anchors.values()]
    vals = [v for v in vals if v is not None]
    med = float(np.median(vals)) if vals else np.nan
    low = float(np.min(vals)) if vals else np.nan
    ep_missing = clean_anchor(anchors.get("ep")) is None

    # EP가 실패하고, CHI가 낮은 분위인데 최근 tail이 강하게 내려가는 경우.
    # 말기 베어링에서 에너지 재분배/포락선 불안정으로 HI가 꺾이는 케이스를 방어적으로 짧게 본다.
    if ep_missing and chi_q < 0.18 and curr_rel > 0.95 and rr < -0.04:
        return "signal_reversal"

    # CHI 수준이 높고 증가율도 강한 경우만 진짜 late로 본다.
    # rr 단독 late 판정은 오분류가 커서 제거한다.
    if (chi_q >= 0.86 and curr_rel >= 1.35) or (chi_q >= 0.90) or (rr >= 0.14 and curr_rel >= 1.70):
        return "late"

    # CHI는 높지만 최근 변화율이 음수면, 즉각 파단보다는 transient pattern 후 회복/완화 가능성을 둔다.
    if chi_q >= 0.70 and curr_rel >= 1.45 and rr < 0:
        return "transient_recovery"

    # 낮은 앵커 하나만 튀는 경우: 실제 말기라기보다 EP/ML 극단적인 outlier 가능성이 큼.
    if np.isfinite(low) and np.isfinite(med) and low <= 4.0 * 3600 and med >= 6.0 * 3600 and chi_q < 0.78:
        return "collapse"

    if chi_q < 0.35 and rr < 0.004 and curr_rel < 1.25:
        return "stable"
    return "mid"

def blend_anchors(ml: float, ep: Optional[float],
                  dtw: Optional[float], phase: Optional[float],
                  lib: Optional[float], curr_rel: float, rr: float,
                  chi_q: float, onset: bool) -> Tuple[float, str, Dict[str, Optional[float]]]:
    """
    평균을 강하게 쓰면 outlier 하나에 끌려가므로,
    상태별 weighted quantile을 사용한다.
    """
    anchors = {"ml": ml, "ep": ep, "dtw": dtw, "phase": phase, "lib": lib}
    vals, wts, used = [], [], []

    def _add(name: str, v: Optional[float], w: float) -> None:
        cv = clean_anchor(v)
        if cv is not None:
            vals.append(cv); wts.append(w); used.append(name.upper())

    # ML은 train 전체 평균으로 회귀하는 경향이 있어 v3에서는 약간 낮춘다.
    _add("ml",    ml,    CFG.ml_weight)
    _add("ep",    ep,    0.75 if (onset or rr > 0.006) else 0.38)
    _add("dtw",   dtw,   0.62)
    _add("phase", phase, 0.90 if chi_q < 0.80 else 0.65)
    _add("lib",   lib,   CFG.library_prior_weight)

    if not vals:
        return clip_rul(ml), "MLonly", anchors

    state = classify_health_state(chi_q, curr_rel, rr, anchors)
    if state == "late":
        q = CFG.late_q
    elif state == "stable":
        q = CFG.stable_q
    elif state == "collapse":
        q = 0.55
    elif state == "transient_recovery":
        q = 0.78
    elif state == "signal_reversal":
        q = 0.15
    else:
        q = 0.62 if chi_q < 0.58 else CFG.conservative_q

    qv  = weighted_quantile(vals, wts, q)
    med = weighted_quantile(vals, wts, 0.50)
    avg = float(np.average(vals, weights=wts))

    # 분위수 중심 + 중앙값 약간 + 평균 약간. 평균 비중은 낮춰 outlier 방지.
    raw = 0.62 * qv + 0.26 * med + 0.12 * avg
    note = "+".join(used) + f",state={state},q={q:.2f}"
    return clip_rul(raw), note, anchors

def calibrate_stage_scales(records: List[Dict],
                           fallback: Dict[float, float]) -> Dict[float, float]:
    """
    CV fold records → stage별 최적 scale 탐색.
    pred_raw를 직접 사용하므로 역산 오류 없음.
    단조 감소 강제: 0.30→0.80 순으로 scale ≥ 다음 stage scale.
    """
    if not records:
        return dict(fallback)

    # 너무 낮은 scale은 미관측 구간에서 불안정할 수 있어 탐색 범위를 완만하게 둔다.
    grid   = np.round(np.arange(0.70, 1.61, 0.05), 2)
    stages = sorted(set(float(r["stage"]) for r in records))
    raw_scales: Dict[float, float] = {}

    for stage in stages:
        rows = [r for r in records if float(r["stage"]) == stage]
        if len(rows) < 2:
            raw_scales[stage] = fallback.get(stage, 1.0)
            continue
        best_s, best_sc = fallback.get(stage, 1.0), -np.inf
        for s in grid:
            sc = float(np.average(
                [score_fn(r["actual"], r["pred_raw"] * s) for r in rows],
                weights=[r.get("weight", 1.0) for r in rows],
            ))
            if sc > best_sc:
                best_s, best_sc = float(s), sc
        raw_scales[stage] = best_s

    for k, v in fallback.items():
        raw_scales.setdefault(float(k), float(v))

    out: Dict[float, float] = {}
    running = float("inf")
    for stage in sorted(raw_scales.keys()):
        running = min(running, raw_scales[stage])
        min_s = CFG.min_stage_scale.get(float(stage), 0.70)
        out[stage] = max(float(running), float(min_s))

    # fallback에만 있는 stage도 보장
    for stage, min_s in CFG.min_stage_scale.items():
        out[float(stage)] = max(float(out.get(float(stage), fallback.get(stage, min_s))), float(min_s))

    return out


def final_decision(pred_raw: float, chi_q: float,
                   curr_rel: float, rr: float,
                   anchors: Dict[str, Optional[float]],
                   stage_scales: Dict[float, float]) -> Tuple[float, float, float, str]:
    """
    v4 결정 레이어:
      - 안정/중기 구간은 너무 낮은 앵커 하나에 끌려 과소예측되는 것을 방지
      - late 구간은 말기 패턴을 반영하되, EP/Phase가 받쳐주는 경우 극단값을 완화함
      - CHI가 높지만 tail slope가 음수인 transient pattern 케이스는 별도 reference 적용
      - EP 실패 + 낮은 CHI + 음의 tail slope는 signal_reversal로 보고 방어적으로 짧은 reference 적용
    """
    stage = stage_from_chi(chi_q)
    scale = float(stage_scales.get(stage, CFG.fallback_stage_scale.get(stage, 1.0)))
    scale = max(scale, float(CFG.min_stage_scale.get(stage, 0.70)))
    scaled = clip_rul(pred_raw * scale)
    note = f"stage={stage:.2f},sc={scale:.2f}"

    vals, wts = [], []
    for name, val in anchors.items():
        val = clean_anchor(val)
        if val is not None:
            vals.append(val)
            wts.append({
                "ml": CFG.ml_weight, "ep": 0.75, "dtw": 0.62,
                "phase": 0.90 if chi_q < 0.80 else 0.65,
                "lib": CFG.library_prior_weight,
            }.get(name, 0.5))

    if not vals:
        return clip_rul(scaled), stage, scale, note + ",no_anchor"

    state = classify_health_state(chi_q, curr_rel, rr, anchors)
    lo   = weighted_quantile(vals, wts, 0.20)
    q40  = weighted_quantile(vals, wts, CFG.conservative_q)
    q65  = weighted_quantile(vals, wts, 0.65)
    med  = weighted_quantile(vals, wts, 0.50)
    hi   = weighted_quantile(vals, wts, 0.80)

    note += f",state={state},anchors={len(vals)}"

    if state == "signal_reversal":
        # EP가 성립하지 않고 HI가 꺾이는 케이스는 phase/lib의 장기 앵커를 그대로 믿지 않는다.
        posterior = 0.45 * lo + 0.55 * scaled
        posterior = min(posterior, CFG.reversal_reference_h * 3600)
        note += ",REVERSAL_PATTERN"

    elif state == "late":
        # late에서도 EP/Phase가 길게 받쳐주면 극단적 단기 예측을 완화한다.
        q_late = weighted_quantile(vals, wts, CFG.late_q)
        posterior = 0.45 * scaled + 0.55 * q_late
        long_pool = [v for v in vals if v >= 4.5 * 3600]
        if long_pool:
            late_support_ref = min(float(np.median(long_pool)) * 0.75, CFG.late_reference_h * 3600)
            late_support_ref = max(late_support_ref, CFG.late_support_h * 3600)
            posterior = max(posterior, late_support_ref)
            note += ",LATE_SUPPORT"
        posterior = min(posterior, CFG.late_reference_h * 3600)
        note += ",LATE_PATTERN"

    elif state == "transient_recovery":
        # CHI는 높지만 최근 기울기가 음수면 단기 파단으로 보지 않고 회복 reference를 반영한다.
        posterior = 0.45 * scaled + 0.55 * hi
        posterior = max(posterior, CFG.recovery_support_h * 3600)
        posterior = min(posterior, CFG.recovery_reference_h * 3600)
        note += ",RECOVERY_PATTERN"

    elif state == "stable":
        posterior = 0.72 * scaled + 0.28 * hi

        # CHI가 초기 baseline보다 낮고 매우 낮은 분위면 phase/lib가 과하게 길어질 수 있어
        # ML/DTW/EP 중 안정적인 신뢰 앵커를 함께 반영한다.
        if curr_rel < 0.95 and chi_q < 0.08:
            lower_pool = [clean_anchor(anchors.get(k)) for k in ("ml", "dtw", "ep")]
            lower_pool = [v for v in lower_pool if v is not None]
            if lower_pool:
                lower_ref = float(np.median(lower_pool))
                posterior = 0.60 * lower_ref + 0.40 * med
            posterior = min(posterior, CFG.baseline_reference_h * 3600)
            note += ",BASELINE_PATTERN"
        else:
            # stable인데 CHI가 baseline 위로 올라와 있으면 안정 패턴 reference.
            if chi_q > 0.10 and curr_rel > 1.05:
                posterior = max(posterior, CFG.stable_support_h * 3600)
                note += ",STABLE_PRIOR"
            posterior = min(posterior, CFG.stable_reference_h * 3600)

    elif state == "collapse":
        posterior = 0.45 * scaled + 0.55 * med
        rescue_pool = [clean_anchor(anchors.get(k)) for k in ("lib", "phase", "ep", "dtw")]
        rescue_pool = [v for v in rescue_pool if v is not None and np.isfinite(v)]
        if rescue_pool:
            rescue_ref = float(np.median(rescue_pool))
            posterior = max(posterior, min(rescue_ref, 7.5 * 3600))
            note += ",COLLAPSE_RESCUE_STRONG"

    else:
        posterior = 0.65 * scaled + 0.35 * q65
        # 중기에서 CHI와 변화율이 모두 baseline 위면 너무 낮은 쪽으로 보지 않는다.
        if chi_q < 0.58 and curr_rel > 1.15 and rr > 0.02:
            posterior = max(posterior, 12.0 * 3600)
            posterior = min(posterior, 13.3 * 3600)
            note += ",MID_LONG_SUPPORT"

    # 최종 안전 클립
    posterior = clip_rul(posterior)
    return posterior, stage, scale, note

# =====================================================================
# 단일 케이스 예측
# =====================================================================
def predict_case(observed: pd.DataFrame,
                 ensemble: ModelEnsemble,
                 feature_cols: List[str],
                 chi_feats: Sequence[Tuple[str, float]],
                 ref: Dict,
                 stage_scales: Dict[float, float]) -> Dict:
    gs   = observed.sort_values("tdms_order").reset_index(drop=True)
    last = gs.iloc[-1]
    x    = last[feature_cols].fillna(0).to_numpy(dtype=float)
    ml   = clip_rul(ensemble.predict_sec_row(x))

    chi_q = 0.50; curr_rel = 1.0; rr = 0.0; onset = False
    ep = dtw = phase = lib = None
    chi = compute_chi(gs, chi_feats)

    if chi is not None:
        all_chi  = np.asarray(ref["all_chi"], dtype=float)
        chi_q    = float(np.mean(all_chi <= float(chi[-1])))
        curr_rel = float(chi[-1])
        rr       = rate_ratio(chi)
        onset_i  = detect_onset(chi)
        onset    = onset_i > CFG.init_n
        fthr     = similarity_fthr(chi, ref)
        ep       = exp_extrapolate(chi, fthr, onset_idx=onset_i)
        dtw      = traj_sim_dtw(chi, ref)
        phase    = phase_rul_est(chi, ref)
        lib      = library_prior_rul(chi, ref, len(gs), chi_q, rr)

    raw, blend_note, anchors = blend_anchors(
        ml, ep, dtw, phase, lib, curr_rel, rr, chi_q, onset)
    final, stage, scale, dec_note = final_decision(
        raw, chi_q, curr_rel, rr, anchors, stage_scales)

    return dict(
        pred_final=round(final), pred_raw=raw,
        stage=stage, scale=scale,
        ml=ml, ep=ep, dtw=dtw, phase=phase, lib=lib,
        chi_q=chi_q, curr_rel=curr_rel, rr=rr,
        onset=onset, note=blend_note + "|" + dec_note,
    )

# =====================================================================
# LOBO / K50 CV
# =====================================================================
def run_lobo_cv(train_df: pd.DataFrame,
                feature_cols: List[str],
                chi_feats: Sequence[Tuple[str, float]]) -> Tuple[pd.DataFrame, Dict[float, float], float]:
    rows: List[Dict]          = []
    scale_records: List[Dict] = []
    bids = sorted(train_df["train_id"].astype(str).unique())

    print(f"\n[CV] LOBO CV  cutoffs={CFG.cutoffs}")
    for holdout in bids:
        tr      = train_df[train_df["train_id"].astype(str) != holdout].copy()
        va_full = (train_df[train_df["train_id"].astype(str) == holdout]
                   .copy().sort_values("tdms_order").reset_index(drop=True))
        if len(va_full) < 20:
            continue

        ref_fold = build_reference(tr, chi_feats)
        ens      = build_ensemble(
            tr[feature_cols].fillna(0).to_numpy(dtype=float),
            np.log1p(np.minimum(tr["rul_sec"].to_numpy(dtype=float), CFG.max_rul_s)),
            n_models=CFG.n_cv_models,
        )

        for cut in CFG.cutoffs:
            n_obs  = max(CFG.init_n + 1, int(round(len(va_full) * cut)))
            n_obs  = min(n_obs, len(va_full) - 1)
            obs    = va_full.iloc[:n_obs].copy()
            actual = float(obs.iloc[-1]["rul_sec"])
            pred   = predict_case(obs, ens, feature_cols, chi_feats,
                                  ref_fold, stage_scales=CFG.fallback_stage_scale)
            sc     = score_fn(actual, pred["pred_final"])
            rows.append(dict(
                Bearing=holdout, Cutoff=cut, ObsFiles=n_obs,
                Actual_s=actual, Pred_s=pred["pred_final"],
                PredRaw_s=pred["pred_raw"],
                Stage=pred["stage"], Scale=pred["scale"], Score=sc,
                ML_s=pred["ml"], EP_s=pred["ep"],
                DTW_s=pred["dtw"], Phase_s=pred["phase"], Lib_s=pred.get("lib"),
                CHI_q=pred["chi_q"], Note=pred["note"],
            ))
            scale_records.append(dict(
                stage=pred["stage"], pred_raw=pred["pred_raw"],
                actual=actual, weight=1.0,
            ))

        if CFG.run_k50_cv and len(va_full) > CFG.k50_obs_files + 2:
            obs    = va_full.iloc[: CFG.k50_obs_files].copy()
            actual = float(obs.iloc[-1]["rul_sec"])
            pred   = predict_case(obs, ens, feature_cols, chi_feats,
                                  ref_fold, stage_scales=CFG.fallback_stage_scale)
            sc     = score_fn(actual, pred["pred_final"])
            rows.append(dict(
                Bearing=holdout, Cutoff="K50", ObsFiles=CFG.k50_obs_files,
                Actual_s=actual, Pred_s=pred["pred_final"],
                PredRaw_s=pred["pred_raw"],
                Stage=pred["stage"], Scale=pred["scale"], Score=sc,
                ML_s=pred["ml"], EP_s=pred["ep"],
                DTW_s=pred["dtw"], Phase_s=pred["phase"], Lib_s=pred.get("lib"),
                CHI_q=pred["chi_q"], Note=pred["note"],
            ))
            scale_records.append(dict(
                stage=pred["stage"], pred_raw=pred["pred_raw"],
                actual=actual, weight=CFG.k50_weight,
            ))

    df_cv        = pd.DataFrame(rows)
    stage_scales = calibrate_stage_scales(scale_records, CFG.fallback_stage_scale)
    cv_score     = float(df_cv["Score"].mean()) if not df_cv.empty else float("nan")
    print(f"  CV score: {cv_score:.4f}")
    print(f"  Stage scales: { {int(k*100): round(v,2) for k,v in sorted(stage_scales.items())} }")
    return df_cv, stage_scales, cv_score


# =====================================================================
# MAIN
# =====================================================================
def main() -> None:
    t0  = time.time()
    BAR = "=" * 72
    print(BAR)
    print("  KSPHM-KIMM 2026 | RUL Pipeline Final Submission")
    print(BAR)
    print(f"  BASE    : {BASE}")
    print(f"  OUT_DIR : {OUT_DIR}")
    print(f"  LGB={LGB_OK}  XGB={XGB_OK}  PYWT={PYWT_OK}  DTAIDIST={DTAIDIST_OK}")

    # merged_features.csv가 없으면 자동 빌드
    if not os.path.exists(TRAIN_CSV):
        print(f"\n[0] merged_features.csv 없음 → Train TDMS에서 자동 빌드")
        if not os.path.isdir(TRAIN_DIR):
            raise FileNotFoundError(
                f"merged_features.csv도 없고 Train/ 폴더도 없습니다.\n"
                f"  TRAIN_CSV: {TRAIN_CSV}\n  TRAIN_DIR: {TRAIN_DIR}"
            )
        build_merged_features(TRAIN_DIR, TRAIN_CSV)
    else:
        print(f"\n[0] merged_features.csv 확인됨: {TRAIN_CSV}")

    print("\n[1] Train 로드")
    train_df = prepare_train_data(TRAIN_CSV)

    print("\n[2] Test 피처 추출")
    test_df = load_test_features(TEST_DIR)

    print("\n[3] 피처 선택 / CHI reference")
    feature_cols = get_feature_cols(train_df, test_df)
    chi_feats    = [(c, w) for c, w in CHI_FEATS
                    if c in train_df.columns and c in test_df.columns]
    if not chi_feats:
        raise RuntimeError("CHI_FEATS 공통 컬럼 없음")
    ref_full = build_reference(train_df, chi_feats)

    print("\n[4] LOBO/K50 stage scale 보정")
    df_cv, stage_scales, cv_score = run_lobo_cv(train_df, feature_cols, chi_feats)

    print("\n[5] Final 앙상블 학습")
    X_all = train_df[feature_cols].fillna(0).to_numpy(dtype=float)
    y_all = np.log1p(np.minimum(train_df["rul_sec"].to_numpy(dtype=float), CFG.max_rul_s))
    ens   = build_ensemble(X_all, y_all, n_models=CFG.n_full_models)

    print("\n[6] Test 예측")
    rows: List[Dict] = []
    for bid, g in sorted(test_df.groupby("train_id")):
        pred = predict_case(g, ens, feature_cols, chi_feats, ref_full, stage_scales)
        rul  = int(round(pred["pred_final"]))

        def fh(v: Optional[float]) -> str:
            return f"{v/3600:.2f}h" if (v and np.isfinite(v)) else "N/A"

        print(
            f"  {bid:<7}: {rul:>8,}s ({rul/3600:.2f}h) "
            f"ML={fh(pred['ml'])} EP={fh(pred['ep'])} "
            f"DTW={fh(pred['dtw'])} PH={fh(pred['phase'])} LIB={fh(pred.get('lib'))} "
            f"chi_q={pred['chi_q']:.2f} sc={pred['scale']:.2f} "
            f"[{pred['note']}]"
        )
        rows.append(dict(
            Test_ID=bid, RUL_sec=rul, RUL_hour=round(rul / 3600.0, 3),
            PredRaw_s=round(pred["pred_raw"]),
            Stage=pred["stage"], Scale=pred["scale"],
            ML_h=round(pred["ml"]    / 3600, 3) if pred["ml"]    else None,
            EP_h=round(pred["ep"]    / 3600, 3) if pred["ep"]    else None,
            DTW_h=round(pred["dtw"]  / 3600, 3) if pred["dtw"]   else None,
            Phase_h=round(pred["phase"]/3600,3)  if pred["phase"] else None,
            Lib_h=round(pred["lib"] / 3600, 3) if pred.get("lib") else None,
            CHI_q=round(pred["chi_q"],   4),
            CurrRel=round(pred["curr_rel"], 4),
            RateRatio=round(pred["rr"],  6),
            Onset=pred["onset"], Note=pred["note"],
        ))

    df_diag = pd.DataFrame(rows).sort_values("Test_ID").reset_index(drop=True)
    df_submit = pd.DataFrame({
        "Test_ID": df_diag["Test_ID"],
        "RUL_sec": df_diag["RUL_sec"].astype(int),
    })
    df_official = pd.DataFrame({
        "File":      df_diag["Test_ID"],
        "RUL_Score": df_diag["RUL_sec"].astype(int),
    })

    print("\n[7] 저장")
    official_path = safe_to_excel(
        df_official,
        os.path.join(OUT_DIR, f"{CFG.team_name}_validation.xlsx"),
        index=False,
    )
    sub_path  = safe_to_excel(df_submit, os.path.join(OUT_DIR, "rul_submission.xlsx"), index=False)
    diag_path = os.path.join(OUT_DIR, "rul_diagnostics.xlsx")
    try:
        with pd.ExcelWriter(diag_path, engine="openpyxl") as w:
            df_diag.to_excel(w, sheet_name="Diagnostics", index=False)
            df_cv.to_excel(w, sheet_name="LOBO_K50_CV", index=False)
            pd.DataFrame({"Feature": feature_cols}).to_excel(w, sheet_name="Features", index=False)
            pd.DataFrame([{"Stage_pct": int(k*100), "Scale": v}
                          for k, v in sorted(stage_scales.items())]
                         ).to_excel(w, sheet_name="Stage_Scales", index=False)
            pd.DataFrame([asdict(CFG)]).to_excel(w, sheet_name="Config", index=False)
    except PermissionError:
        alt = os.path.join(OUT_DIR, f"rul_diagnostics_{time.strftime('%Y%m%d_%H%M%S')}.xlsx")
        with pd.ExcelWriter(alt, engine="openpyxl") as w:
            df_diag.to_excel(w, sheet_name="Diagnostics", index=False)
            df_cv.to_excel(w, sheet_name="LOBO_K50_CV", index=False)
        diag_path = alt

    elapsed = time.time() - t0
    print(f"  Official → {official_path}")
    print(f"  Submit   → {sub_path}")
    print(f"  Diag     → {diag_path}")
    print(f"\n{BAR}")
    print(f"  Features     : {len(feature_cols)}")
    print(f"  CV Score     : {cv_score:.4f}")
    print(f"  Stage Scales : { {int(k*100): round(v,2) for k,v in sorted(stage_scales.items())} }")
    print(f"  Elapsed      : {elapsed:.0f}s")
    print(BAR)
    print()
    print("  +------------------------------------------+")
    print("  |   File   |        RUL_Score              |")
    print("  +----------+-------------------------------+")
    for _, r in df_official.iterrows():
        rul = int(r["RUL_Score"])
        print(f"  | {str(r['File']):<8} | {rul:>8,}s  ({rul/3600:>5.2f}h) |")
    print("  +------------------------------------------+")


if __name__ == "__main__":
    main()
