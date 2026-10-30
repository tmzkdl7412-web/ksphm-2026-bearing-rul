#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib, json, math, os, pickle, platform, random, re, sys, time, warnings, zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 시드 고정 ────────────────────────────────────────────────────────────────
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    TORCH_OK = True
except Exception:
    TORCH_OK = False

# ── 상수 ─────────────────────────────────────────────────────────────────────
VERSION       = "v128_adaptive"
TEAM_NAME     = "포크착기"
INIT_N        = 6
EPS           = 1e-8
RMS_COLS      = ["CH1_rms", "CH2_rms", "CH3_rms", "CH4_rms"]
SAMPLING_RATE = 25_600.0
RPM_MIN, RPM_MAX, RPM_DEFAULT = 600.0, 1_100.0, 800.0
ORDER_FFT_MAX  = 262_144
FAULT_ORDERS   = {"BPFI": 8.40, "BPFO": 5.58, "BSF": 4.68, "FTF": 0.402}
HI_CRITICAL    = 5.0   # cumulative damage HI value extrapolated to for failure-horizon feature
CH_W_EQUAL     = {c: 0.25 for c in RMS_COLS}
CH_W_PRIOR     = 0.65
CH_W_MIN       = 0.12
K_OBS_FALLBACK = 50
RIDGE_ALPHA    = 5.0
NL_RIDGE_ALPHA = 1.25
BLEND_GRID     = [round(v, 2) for v in np.arange(0.0, 0.61, 0.05)]
MARGIN_GRID    = [0.0, 0.15, 0.30, 0.45, 0.60]
OVER_HL        = 0.20
UNDER_HL       = 0.50
PHYS_FEATS = [
    "cum_damage_hi", "damage_rate_hi", "tail_damage_rate_hi",
    "tail_to_life_damage_ratio", "cum_damage_hi_sq",
    "weighted_rms_rel", "baseline_weighted_rms", "current_weighted_rms",
    "rear_rel", "front_rel", "rear_front_rel", "uniform_rel",
    "impulse_score", "low_damage_impulse_conflict",
    "damage_slope_6", "damage_accel_6",
    "ch_diversity",         # spatial fault localisation index (σ/μ of per-channel relative RMS)
    # NOTE: failure_horizon_log is computed (see add_physics_features) and exported to the
    # diagnostic workbook, but excluded from the ridge regressor: slope≈0 bearings produce
    # horizon→∞ (log≈13.8), an extreme outlier that corrupts ridge coefficient estimation.
]

# ── 분포 가드 파라미터 ────────────────────────────────────────────────────────
@dataclass(frozen=True)
class GuardParams:
    stable_spread: float
    late_spread:   float
    active_mad:    float
    severe_ood_z:  float
    severe_near:   float
    severe_comp:   float

GUARD_GRID = [
    GuardParams(0.48, 0.32, 0.75, 12.0, 3.0, 0.00),
    GuardParams(0.55, 0.36, 0.90, 12.0, 3.0, 0.25),
    GuardParams(0.62, 0.40, 1.05, 12.0, 3.0, 0.60),
    GuardParams(0.70, 0.45, 1.20, 11.0, 2.8, 1.00),
]
DEFAULT_GUARD = GUARD_GRID[0]

# ── 데이터클래스 ──────────────────────────────────────────────────────────────
@dataclass
class FitResult:
    intercept_h: float; slope_h: float; curvature_h: float
    max_rul_h: float;   min_rul_h: float; mad_h: float
    kind: str = "linear"

@dataclass
class PhysFit:
    intercept_h: float; coef: np.ndarray
    mean: np.ndarray;   scale: np.ndarray
    features: List[str]; max_rul_h: float; min_rul_h: float

@dataclass
class BlendResult:
    w: float; margin_h: float; cv: float

@dataclass
class K50Ref:
    obs: int; n: int
    q25_h: float; median_h: float; max_h: float; iqr_h: float
    base_max: float; base_iqr: float
    tail_hi_med: float; tail_hi_iqr: float
    features: List[str]; mean: np.ndarray; scale: np.ndarray
    X: np.ndarray; rul_h: np.ndarray

@dataclass
class DistRef:
    obs: int; window: int; n: int
    features: List[str]; mean: np.ndarray; scale: np.ndarray
    X: np.ndarray; rul_h: np.ndarray

# ── 수학 유틸 ─────────────────────────────────────────────────────────────────
def _iqr(v: np.ndarray, floor: float = EPS) -> float:
    x = np.asarray(v, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return floor
    return max(float(np.quantile(x, 0.75) - np.quantile(x, 0.25)), floor)

def _wquantile(vals: np.ndarray, w: np.ndarray, q: float) -> float:
    vals = np.asarray(vals, dtype=np.float64)
    w    = np.asarray(w,    dtype=np.float64)
    mask = np.isfinite(vals) & np.isfinite(w) & (w >= 0)
    vals, w = vals[mask], w[mask]
    if len(vals) == 0 or w.sum() <= EPS:
        return float("nan")
    idx  = np.argsort(vals)
    vals, w = vals[idx], w[idx]
    cdf  = np.cumsum(w) / max(w.sum(), EPS)
    i    = int(np.searchsorted(cdf, q, side="left"))
    return float(vals[min(max(i, 0), len(vals) - 1)])

def official_score(actual_s: np.ndarray, pred_s: np.ndarray) -> np.ndarray:
    actual = np.maximum(np.asarray(actual_s, dtype=np.float64), 1.0)
    pred   = np.asarray(pred_s, dtype=np.float64)
    rel    = (pred - actual) / actual
    over   = rel >= 0
    sc     = np.empty_like(rel)
    sc[ over] = np.exp(-math.log(2) * rel[ over] / OVER_HL)
    sc[~over] = np.exp(-math.log(2) * (-rel[~over]) / UNDER_HL)
    return np.clip(sc, 0.0, 1.0)

# ── 스펙트럼 유틸 ──────────────────────────────────────────────────────────────
def _kurtosis_excess(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return 0.0
    x -= x.mean()
    var = float(np.mean(x * x))
    if var <= EPS:
        return 0.0
    return max(float(np.mean(x**4) / (var * var + EPS)) - 3.0, 0.0)

def _near_amp(freqs: np.ndarray, mag: np.ndarray, hz: float, bw: float = 1.5) -> float:
    if hz <= 0 or hz >= freqs[-1]:
        return 0.0
    mask = (freqs >= hz - bw) & (freqs <= hz + bw)
    if not np.any(mask):
        return float(mag[int(np.argmin(np.abs(freqs - hz)))])
    return float(np.max(mag[mask]))

def _shaft_hz(freqs: np.ndarray, mag: np.ndarray) -> float:
    lo, hi, step = RPM_MIN / 60, RPM_MAX / 60, 0.05
    best_hz, best_sc = RPM_DEFAULT / 60, -np.inf
    for hz in np.arange(lo, hi + step / 2, step):
        sc = (_near_amp(freqs, mag, hz, 0.7)
              + 0.50 * _near_amp(freqs, mag, 2 * hz, 0.9)
              + 0.25 * _near_amp(freqs, mag, 3 * hz, 1.1))
        if sc > best_sc:
            best_sc, best_hz = sc, float(hz)
    return best_hz

def _tkeo_features(arr: np.ndarray) -> Dict[str, np.float32]:
    """Teager-Kaiser Energy Operator per channel.
    Ψ[x[n]] = x[n]² − x[n−1]·x[n+1]  captures AM-FM modulation
    characteristic of bearing fault impulses, outperforming plain kurtosis
    for early-stage localised faults (Kaiser 1990; Li & Shin 2012)."""
    out = {}
    for j, tag in enumerate(["CH1", "CH2", "CH3", "CH4"]):
        col = arr[:, j].astype(np.float64)
        if len(col) < 3:
            out[f"{tag}_tkeo_mean"] = np.float32(0.0)
            continue
        tkeo = col[1:-1] ** 2 - col[:-2] * col[2:]
        out[f"{tag}_tkeo_mean"] = np.float32(float(np.mean(np.maximum(tkeo, 0.0))))
    return out

def _order_features(arr: np.ndarray) -> Dict[str, np.float32]:
    x = np.asarray(arr[:, :4], dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2048:
        return {k: np.float32(0.0) for k in
                ["shaft_hz_est","order_fault_ratio","bpfi_order_ratio",
                 "bpfo_order_ratio","bsf_order_ratio","ftf_order_ratio"]}
    if x.shape[0] > ORDER_FFT_MAX:
        x = x[-ORDER_FFT_MAX:]
    x -= np.nanmean(x, axis=0, keepdims=True)
    win  = np.hanning(x.shape[0]).reshape(-1, 1)
    spec = np.abs(np.fft.rfft(x * win, axis=0)) / max(x.shape[0], 1)
    freqs   = np.fft.rfftfreq(x.shape[0], d=1.0 / SAMPLING_RATE)
    mag_sum = np.nanmean(spec, axis=1)
    shaft   = _shaft_hz(freqs, mag_sum)
    dmask   = (freqs >= 50) & (freqs <= 5_000)
    denom   = float(np.nanmedian(mag_sum[dmask])) if np.any(dmask) else float(np.nanmedian(mag_sum))
    denom   = max(denom if np.isfinite(denom) else EPS, EPS)
    ratios, fault_sc = {}, []
    for name, mult in FAULT_ORDERS.items():
        harms = [1] if name == "FTF" else [1, 2]
        amps  = [_near_amp(freqs, mag_sum, shaft * mult * h,
                           max(1.5, 0.02 * shaft * mult * h)) for h in harms]
        sc    = float(np.nanmax(amps)) / denom
        ratios[name] = sc
        if name in ("BPFI", "BPFO", "BSF"):
            fault_sc.append(sc)
    order_ratio = float(np.nanmax(fault_sc)) if fault_sc else 0.0
    return {
        "shaft_hz_est":    np.float32(shaft),
        "order_fault_ratio": np.float32(order_ratio),
        "bpfi_order_ratio":  np.float32(ratios.get("BPFI", 0.0)),
        "bpfo_order_ratio":  np.float32(ratios.get("BPFO", 0.0)),
        "bsf_order_ratio":   np.float32(ratios.get("BSF",  0.0)),
        "ftf_order_ratio":   np.float32(ratios.get("FTF",  0.0)),
    }

# ── 데이터 로딩 ───────────────────────────────────────────────────────────────
def read_train(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"train_id", "file_index", "RUL_sec", *RMS_COLS}
    miss = sorted(req - set(df.columns))
    if miss:
        raise ValueError(f"merged_features.csv 누락 컬럼: {miss}")
    df = df.copy()
    df["train_id"]   = df["train_id"].astype(str)
    df["file_index"] = df["file_index"].astype(int)
    for c in RMS_COLS + ["RUL_sec"]:
        df[c] = df[c].astype("float32")
    return df.sort_values(["train_id", "file_index"]).reset_index(drop=True)


def _read_tdms_zip(zip_path: Path, test_id: str) -> pd.DataFrame:
    try:
        from nptdms import TdmsFile
    except ImportError as e:
        raise ImportError("pip install nptdms") from e
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted([n for n in zf.namelist() if n.lower().endswith(".tdms")],
                       key=lambda n: (int(re.search(r"(\d+)", Path(n).stem).group(1))
                                      if re.search(r"(\d+)", Path(n).stem) else 10**9, n))
        if not names:
            raise RuntimeError(f"{zip_path}에 TDMS 없음")
        print(f"  {test_id}: {len(names)} 파일", flush=True)
        with TemporaryDirectory() as tmp:
            tp = Path(tmp)
            for i, member in enumerate(names, 1):
                zf.extract(member, tp)
                arr = TdmsFile.read(str(tp / member)).as_dataframe().to_numpy(dtype=np.float32)
                if arr.ndim != 2 or arr.shape[1] < 4:
                    raise RuntimeError(f"TDMS shape 오류: {arr.shape}")
                rms  = np.sqrt(np.mean(np.square(arr[:, :4], dtype=np.float64), axis=0))
                kurt = [_kurtosis_excess(arr[:, j]) + 3.0 for j in range(4)]
                rows.append({
                    "Test_ID": test_id, "file_index": i,
                    "file_name": Path(member).name,
                    "CH1_rms": np.float32(rms[0]), "CH2_rms": np.float32(rms[1]),
                    "CH3_rms": np.float32(rms[2]), "CH4_rms": np.float32(rms[3]),
                    "CH1_kurtosis": np.float32(kurt[0]), "CH2_kurtosis": np.float32(kurt[1]),
                    "CH3_kurtosis": np.float32(kurt[2]), "CH4_kurtosis": np.float32(kurt[3]),
                    "rms_all_mean": np.float32(np.mean(rms)),
                    "rms_all_max":  np.float32(np.max(rms)),
                    **_tkeo_features(arr),
                    **_order_features(arr),
                })
                if i % 25 == 0 or i == len(names):
                    print(f"    {test_id}: {i}/{len(names)}", flush=True)
    return pd.DataFrame(rows)

def load_test(test_dir: Path) -> pd.DataFrame:
    return pd.concat(
        [_read_tdms_zip(test_dir / f"Test{i}.zip", f"Test{i}") for i in range(1, 7)],
        ignore_index=True,
    )

def _cache_key(test_dir: Path) -> str:
    """Test*.zip 파일들의 {이름:크기:수정시각} MD5 해시 → 캐시 무효화 키."""
    h = hashlib.md5()
    for i in range(1, 7):
        p = test_dir / f"Test{i}.zip"
        if p.exists():
            st = p.stat()
            h.update(f"{p.name}:{st.st_size}:{int(st.st_mtime_ns)}".encode())
    return h.hexdigest()[:16]

def load_test_cached(test_dir: Path, cache_dir: Path) -> pd.DataFrame:
    """TDMS 파싱 결과를 pickle로 캐시. 동일 zip에 대해 파싱 완전 생략.

    Cache invalidation: 어느 Test*.zip이라도 크기/수정시각 변경 시 자동 재파싱.
    캐시 파일 위치: cache_dir/tdms_raw_<hash16>.pkl
    이전 버전 캐시는 자동 삭제된다.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    key  = _cache_key(test_dir)
    path = cache_dir / f"tdms_raw_{key}.pkl"
    if path.exists():
        print(f"  [캐시 히트] {path.name}  →  TDMS 파싱 생략", flush=True)
        with open(path, "rb") as fh:
            return pickle.load(fh)
    print(f"  [캐시 미스] TDMS 파싱 후 캐시 저장: {path.name}", flush=True)
    df = load_test(test_dir)
    with open(path, "wb") as fh:
        pickle.dump(df, fh, protocol=5)
    for old in cache_dir.glob("tdms_raw_*.pkl"):
        if old != path:
            try: old.unlink()
            except OSError: pass
    return df

# ── 피처 엔지니어링 ───────────────────────────────────────────────────────────
def _file_key(name: str) -> Tuple[int, str]:
    m = re.search(r"(\d+)", Path(name).stem)
    return (int(m.group(1)) if m else 10**9, name)

def compute_damage(df: pd.DataFrame, grp: str, fidx: str,
                   ch_w: Dict[str, float]) -> pd.DataFrame:
    w = np.array([ch_w.get(c, 0.25) for c in RMS_COLS], dtype=np.float64)
    w /= w.sum()
    rows = []
    for _, g in df.sort_values([grp, fidx]).groupby(grp, sort=False):
        g   = g.copy()
        rms = g[RMS_COLS].to_numpy(dtype=np.float64)
        base = np.nanmean(rms[:INIT_N], axis=0)
        base = np.where(np.isfinite(base) & (base > EPS), base, np.nanmedian(rms, axis=0))
        base = np.where(np.isfinite(base) & (base > EPS), base, 1.0)
        exc  = np.maximum(rms / (base + EPS) - 1.0, 0.0)
        dinc = exc @ w
        cdam = np.cumsum(dinc)
        hi   = np.log1p(cdam)
        nobs = np.arange(1, len(g) + 1, dtype=np.float64)
        dr   = cdam / nobs
        tdr  = (pd.Series(dinc, index=g.index).rolling(6, min_periods=1).mean()
                .to_numpy(dtype=np.float64))
        wrms  = rms @ w
        bwrms = float(base @ w)
        g["damage_increment"]      = dinc.astype("float32")
        g["cum_damage"]            = cdam.astype("float32")
        g["cum_damage_hi"]         = hi.astype("float32")
        g["damage_rate"]           = dr.astype("float32")
        g["damage_rate_hi"]        = np.log1p(np.maximum(dr, 0)).astype("float32")
        g["tail_damage_rate"]      = tdr.astype("float32")
        g["tail_damage_rate_hi"]   = np.log1p(np.maximum(tdr, 0)).astype("float32")
        g["tail_to_life_damage_ratio"] = (tdr / (dr + EPS)).astype("float32")
        g["weighted_rms_rel"]      = (wrms / (bwrms + EPS)).astype("float32")
        g["baseline_weighted_rms"] = np.float32(bwrms)
        g["current_weighted_rms"]  = wrms.astype("float32")
        for col, b in zip(RMS_COLS, base):
            g[f"{col}_base{INIT_N}"] = np.float32(b)
        # Channel diversity: spatial heterogeneity of per-channel relative RMS.
        # High value → fault localised to a specific channel (anisotropic bearing damage).
        ch_rel = rms / (base + EPS)      # shape (n, 4)
        ch_div = np.std(ch_rel, axis=1) / (np.mean(ch_rel, axis=1) + EPS)
        g["ch_diversity"] = np.log1p(np.maximum(ch_div, 0.0)).astype("float32")
        rows.append(g)
    return pd.concat(rows, ignore_index=True)

def add_physics_features(df: pd.DataFrame, grp: str, fidx: str) -> pd.DataFrame:
    rows = []
    for _, g in df.sort_values([grp, fidx]).groupby(grp, sort=False):
        g    = g.copy()
        rels = {}
        for col in RMS_COLS:
            bc = f"{col}_base{INIT_N}"
            if bc in g.columns:
                base = g[bc].astype(float).replace(0, np.nan)
            else:
                base = float(g[col].iloc[:min(INIT_N, len(g))].mean())
            rels[col] = (g[col].astype(float) /
                         (base.astype(float) if hasattr(base, "astype") else base + EPS)
                         ).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        front = pd.concat([rels[c] for c in ["CH1_rms","CH2_rms"] if c in rels], axis=1).mean(axis=1)
        rear  = pd.concat([rels[c] for c in ["CH3_rms","CH4_rms"] if c in rels], axis=1).mean(axis=1)
        all_r = pd.concat([rels[c] for c in RMS_COLS if c in rels], axis=1).mean(axis=1)
        g["front_rel"]     = front.astype("float32")
        g["rear_rel"]      = rear.astype("float32")
        g["rear_front_rel"]= (rear/(front+EPS)).replace([np.inf,-np.inf],np.nan).fillna(1.0).astype("float32")
        g["uniform_rel"]   = all_r.astype("float32")
        imp_terms = []
        for i, col in enumerate(RMS_COLS, 1):
            kc = f"CH{i}_kurtosis"
            if kc in g.columns:
                k = np.maximum(g[kc].astype(float).to_numpy() - 3.0, 0.0)
                imp_terms.append(k * rels[col].astype(float).to_numpy())
        imp = np.nanmax(np.vstack(imp_terms), axis=0) if imp_terms else np.zeros(len(g))
        g["impulse_score"] = np.log1p(np.maximum(imp, 0)).astype("float32")
        rhi  = g["damage_rate_hi"].astype(float).to_numpy()
        ihi  = g["impulse_score"].astype(float).to_numpy()
        qgate= 1.0 / (1.0 + np.maximum(rhi, 0))
        g["low_damage_impulse_conflict"] = (qgate * ihi).astype("float32")
        hi   = g["cum_damage_hi"].astype(float)
        sl   = (hi - hi.shift(6)) / 6.0
        sl   = sl.fillna((hi - hi.shift(3)) / 3.0).fillna(0.0)
        ac   = (sl - sl.shift(6)) / 6.0
        g["cum_damage_hi_sq"] = np.square(hi).astype("float32")
        g["damage_slope_6"]   = sl.replace([np.inf,-np.inf],np.nan).fillna(0.0).astype("float32")
        g["damage_accel_6"]   = ac.replace([np.inf,-np.inf],np.nan).fillna(0.0).astype("float32")
        # Failure horizon: extrapolated measurement steps until HI reaches HI_CRITICAL.
        # Provides a direct physics-based RUL proxy independent of the regression model.
        hi_arr  = hi.to_numpy(dtype=np.float64)
        sl_arr  = g["damage_slope_6"].astype(float).to_numpy(dtype=np.float64)
        remain  = np.maximum(HI_CRITICAL - hi_arr, 0.0)
        horizon = np.where(sl_arr > EPS, remain / sl_arr, 1e6)
        g["failure_horizon_log"] = np.log1p(np.minimum(horizon, 1e6)).astype("float32")
        rows.append(g)
    return pd.concat(rows, ignore_index=True)

def add_order_diagnostics(df: pd.DataFrame, grp: str, fidx: str) -> pd.DataFrame:
    if "order_fault_ratio" not in df.columns:
        out = df.copy()
        for c in ["order_fault_rel","order_fault_tail_rel","order_fault_tail_slope",
                  "order_fault_active_risk","order_fault_flat_support","order_fault_growth_watch"]:
            out[c] = np.float32(1.0 if "rel" in c else 0.0)
        return out
    rows = []
    for _, g in df.sort_values([grp, fidx]).groupby(grp, sort=False):
        g   = g.copy()
        rat = g["order_fault_ratio"].astype(float).replace([np.inf,-np.inf], np.nan)
        base= float(rat.iloc[:min(INIT_N,len(g))].median())
        if not np.isfinite(base) or base <= EPS:
            base = float(rat.median()) if np.isfinite(float(rat.median())) else 1.0
        base = max(base, EPS)
        rel  = (rat / base).replace([np.inf,-np.inf],np.nan).fillna(1.0)
        tail = rel.rolling(8, min_periods=2).median().fillna(rel)
        sl   = (tail - tail.shift(8)) / 8.0
        sl   = sl.fillna((tail - tail.shift(4)) / 4.0).fillna(0.0)
        arisk= np.maximum(tail.to_numpy(dtype=np.float64) - 1.35, 0.0) * np.maximum(sl.to_numpy(dtype=np.float64), 0.0)
        g["order_fault_rel"]          = rel.astype("float32")
        g["order_fault_tail_rel"]     = tail.astype("float32")
        g["order_fault_tail_slope"]   = sl.replace([np.inf,-np.inf],np.nan).fillna(0.0).astype("float32")
        g["order_fault_active_risk"]  = np.log1p(arisk).astype("float32")
        g["order_fault_flat_support"] = (
            (g["order_fault_tail_rel"].astype(float) <= 1.30) &
            (g["order_fault_tail_slope"].astype(float) <= 0.025)
        ).astype("float32")
        g["order_fault_growth_watch"] = (
            ((g["order_fault_tail_rel"].astype(float) >= 1.85) |
             (g["order_fault_active_risk"].astype(float) >= 0.10)) &
            (g["order_fault_tail_slope"].astype(float) > 0.025)
        ).astype("float32")
        # TKEO diagnostic feature: store normalised TKEO z-score for inspection only.
        # NOTE: TKEO is available only in test TDMS data, NOT in merged_features.csv
        # training data.  Modifying order_fault_active_risk here would create a
        # train/test feature inconsistency and corrupt the LOBO-calibrated thresholds.
        # Therefore we record tkeo_z_score purely as a diagnostic column.
        tkeo_cols = [f"CH{j}_tkeo_mean" for j in range(1, 5)
                     if f"CH{j}_tkeo_mean" in g.columns]
        if tkeo_cols:
            tkeo_val  = g[tkeo_cols].astype(float).max(axis=1)
            tkeo_base = float(tkeo_val.iloc[:min(INIT_N, len(g))].median())
            tkeo_base = max(tkeo_base if np.isfinite(tkeo_base) else EPS, EPS)
            tkeo_rel  = (tkeo_val / tkeo_base).replace([np.inf, -np.inf], np.nan).fillna(1.0)
            tkeo_z    = tkeo_rel.rolling(6, min_periods=2).median().fillna(tkeo_rel)
            g["tkeo_z_score"] = tkeo_z.fillna(1.0).astype("float32")
        rows.append(g)
    return pd.concat(rows, ignore_index=True)

# ── 모델 피팅 ─────────────────────────────────────────────────────────────────
def fit_hi_model(df: pd.DataFrame) -> FitResult:
    x = df["cum_damage_hi"].to_numpy(dtype=np.float64)
    y = df["RUL_sec"].to_numpy(dtype=np.float64) / 3600.0
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 10:
        raise ValueError("학습 데이터 부족")
    X    = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    ic, sl = float(beta[0]), float(beta[1])
    if sl >= 0:
        ql, qh = np.quantile(x, [0.05, 0.95])
        yl = float(np.median(y[x <= ql])) if np.any(x <= ql) else float(np.max(y))
        yh = float(np.median(y[x >= qh])) if np.any(x >= qh) else float(np.min(y))
        sl = -abs((yl - yh) / max(qh - ql, EPS))
        ic = float(np.median(y) - sl * np.median(x))
    cu   = 0.0; kind = "linear"
    pred = ic + sl * x
    xc   = float(np.median(x)); xs_sc = max(_iqr(x, 0.25), 0.25)
    xs   = (x - xc) / xs_sc
    Xq   = np.column_stack([np.ones_like(xs), xs, xs*xs])
    reg  = np.diag([0.0, NL_RIDGE_ALPHA, NL_RIDGE_ALPHA])
    try:
        bq   = np.linalg.solve(Xq.T @ Xq + reg, Xq.T @ y)
        grid = np.linspace(x.min(), x.max(), 64)
        gs   = (grid - xc) / xs_sc
        deriv= (float(bq[1]) + 2*float(bq[2])*gs) / xs_sc
        pq   = bq[0] + bq[1]*xs + bq[2]*xs*xs
        lmad = float(np.median(np.abs((y-pred) - np.median(y-pred))))
        qmad = float(np.median(np.abs((y-pq)   - np.median(y-pq))))
        if np.all(deriv <= 1e-6) and qmad <= lmad * 0.98:
            b0,b1,b2 = float(bq[0]),float(bq[1]),float(bq[2])
            cu = b2 / (xs_sc**2)
            sl = b1/xs_sc - 2*b2*xc/(xs_sc**2)
            ic = b0 - b1*xc/xs_sc + b2*xc**2/(xs_sc**2)
            pred = ic + sl*x + cu*x*x
            kind = "quad"
    except np.linalg.LinAlgError:
        pass
    res = y - pred
    mad = float(np.median(np.abs(res - np.median(res))))
    if not np.isfinite(mad) or mad < 0.25:
        mad = float(np.median(np.abs(res)))
    mad = max(mad, 0.50)
    return FitResult(ic, sl, cu, float(np.nanmax(y)), max(float(np.nanmin(y)),0.0), mad, kind)

def fit_phys_ridge(df: pd.DataFrame, feats: List[str] = PHYS_FEATS) -> PhysFit:
    use = [c for c in feats if c in df.columns]
    if not use:
        raise ValueError("물리 피처 없음")
    X  = df[use].replace([np.inf,-np.inf],np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    y  = df["RUL_sec"].to_numpy(dtype=np.float64) / 3600.0
    mu = np.mean(X, axis=0); sc = np.std(X, axis=0)
    sc = np.where(sc > EPS, sc, 1.0)
    Xs = (X - mu) / sc
    Xd = np.column_stack([np.ones(len(Xs)), Xs])
    reg= np.eye(Xd.shape[1]) * RIDGE_ALPHA; reg[0,0] = 0.0
    beta = np.linalg.solve(Xd.T @ Xd + reg, Xd.T @ y)
    return PhysFit(float(beta[0]), beta[1:], mu, sc, use,
                   float(np.nanmax(y)), max(float(np.nanmin(y)),0.0))

def predict_hi(hi: np.ndarray, fit: FitResult) -> np.ndarray:
    hi = np.asarray(hi, dtype=np.float64)
    return np.clip(fit.intercept_h + fit.slope_h*hi + fit.curvature_h*hi*hi,
                   fit.min_rul_h, fit.max_rul_h)

def predict_phys_raw(df: pd.DataFrame, fit: PhysFit) -> Tuple[np.ndarray, np.ndarray]:
    X   = df[fit.features].replace([np.inf,-np.inf],np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    raw = fit.intercept_h + (X - fit.mean) / fit.scale @ fit.coef
    return raw, np.clip(raw, fit.min_rul_h, fit.max_rul_h)

def phys_reliability(raw: np.ndarray, clipped: np.ndarray, fit: PhysFit) -> np.ndarray:
    span = max(float(fit.max_rul_h - fit.min_rul_h), EPS)
    return 1.0 / (1.0 + np.abs(raw - clipped) / span)

def _phase_multiplier(hi: float) -> float:
    """HI 누적 손상 단계별 physics 블렌드 가중치 승수.

    근거: Physics ridge는 결함 주파수 성분·임펄스 피처를 활용하므로 손상이
    진행될수록(HI 증가) 점점 더 정보량이 커진다.  반면 초기 안정 구간
    (HI < 0.5)에서는 LOBO로 보정된 HI 선형 모델이 더 정밀하게 보정된다.
    Phase-adaptive weighting은 이 상보성(complementarity)을 명시적으로 활용한다.

    단계별 승수 (LOBO CV 기반 보수적 범위 — 너무 넓으면 grid search가
    base w를 낮춰 평균 effective weight가 오히려 감소):
      HI < 0.5   → 0.90  : 초기 안정 — HI 모델 우선, 소폭 억제
      0.5 ≤ HI < 1.5 → 1.00  : 전환 — 균형
      1.5 ≤ HI < 3.0 → 1.08  : 활성 결함 — physics 소폭 증가
      HI ≥ 3.0   → 1.15  : 말기/임계 — physics 비중 최대
    """
    if   hi < 0.5:  return 0.90
    elif hi < 1.5:  return 1.00
    elif hi < 3.0:  return 1.08
    else:           return 1.15

# ── 채널 가중치 ───────────────────────────────────────────────────────────────
def estimate_channel_weights(df: pd.DataFrame) -> Dict[str, float]:
    scores = {}
    for col in RMS_COLS:
        xs, ys = [], []
        for _, g in df.sort_values(["train_id","file_index"]).groupby("train_id", sort=False):
            vals = g[col].astype(float).to_numpy()
            y    = g["RUL_sec"].astype(float).to_numpy() / 3600.0
            base = float(np.nanmean(vals[:min(INIT_N,len(vals))]))
            if not np.isfinite(base) or base <= EPS:
                base = float(np.nanmedian(vals)) if np.isfinite(float(np.nanmedian(vals))) else 1.0
            hi = np.log1p(np.cumsum(np.maximum(vals/(base+EPS)-1.0, 0.0)))
            if len(hi) >= 8:
                xs.append(hi); ys.append(y)
        if xs:
            x, y = np.concatenate(xs), np.concatenate(ys)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 10 and np.std(x[mask]) > EPS and np.std(y[mask]) > EPS:
                scores[col] = max(-float(np.corrcoef(x[mask],y[mask])[0,1]), 0.0)
            else:
                scores[col] = 0.0
        else:
            scores[col] = 0.0
    raw = np.array([scores[c] for c in RMS_COLS], dtype=np.float64)
    learned = raw/raw.sum() if np.isfinite(raw).all() and raw.sum() > EPS else np.ones(4)/4
    equal   = np.ones(4) / 4
    w = CH_W_PRIOR * equal + (1-CH_W_PRIOR) * learned
    w = np.maximum(w, CH_W_MIN); w /= w.sum()
    return {c: float(v) for c,v in zip(RMS_COLS,w)}

def select_channel_weights_lobo(train_raw: pd.DataFrame):
    candidates = [("equal", CH_W_EQUAL.copy()),
                  ("learned", estimate_channel_weights(train_raw))]
    rows, built = [], {}
    for name, wts in candidates:
        tdf   = _build_train_features(train_raw, wts)
        blend = _select_blend(tdf)
        built[name] = tdf
        rows.append({"name": name, "cv": blend.cv, "wts": json.dumps(wts)})
    rep = pd.DataFrame(rows)
    eq_cv   = float(rep.loc[rep.name=="equal","cv"].iloc[0])
    best_nm = str(rep.loc[rep.cv.idxmax(),"name"])
    best_cv = float(rep.cv.max())
    chosen  = best_nm if best_nm != "equal" and best_cv >= eq_cv + 0.003 else "equal"
    rep["selected"] = rep.name == chosen
    wts = dict(candidates[[n for n,_ in candidates].index(chosen)][1])
    return wts, built[chosen], rep

def _build_train_features(train_raw: pd.DataFrame, wts: Dict[str, float]) -> pd.DataFrame:
    df = compute_damage(train_raw, "train_id", "file_index", wts)
    return add_physics_features(df, "train_id", "file_index")

# ── 블렌드 선택 (LOBO) ─────────────────────────────────────────────────────────
def _select_blend(df: pd.DataFrame) -> BlendResult:
    recs = []
    for ho in sorted(df.train_id.unique()):
        tr = df[df.train_id != ho]
        va = df[df.train_id == ho]
        bf = fit_hi_model(tr); pf = fit_phys_ridge(tr)
        act     = va.RUL_sec.to_numpy(dtype=np.float64)
        hi_vals = va.cum_damage_hi.to_numpy(dtype=np.float64)
        bh      = predict_hi(hi_vals, bf)
        pr, ph  = predict_phys_raw(va, pf)
        rel     = phys_reliability(pr, ph, pf)
        # phase multiplier per-sample (vectorised)
        pm      = np.array([_phase_multiplier(h) for h in hi_vals])
        recs.append((act, bh, ph, rel, pm))
    best = None
    for w in BLEND_GRID:
        for mg in MARGIN_GRID:
            scs = []
            for act, bh, ph, rel, pm in recs:
                ew  = np.minimum(w * pm * rel, 0.70)   # phase-adaptive eff. weight
                p   = np.maximum((1-ew)*bh + ew*ph - mg, 0) * 3600
                scs.append(float(np.mean(official_score(act, p))))
            cv  = float(np.mean(scs))
            obj = cv - 0.003*w - 0.001*mg
            if best is None or obj > best["obj"]:
                best = {"obj":obj,"cv":cv,"w":w,"mg":mg}
    return BlendResult(float(best["w"]), float(best["mg"]), float(best["cv"]))

# ── K50 / 분포 레퍼런스 ───────────────────────────────────────────────────────
def build_k50_ref(df: pd.DataFrame, obs: int) -> K50Ref:
    rows = []
    for _, g in df.sort_values(["train_id","file_index"]).groupby("train_id", sort=False):
        idx = (g.file_index.astype(float) - float(obs)).abs().idxmin()
        rows.append(g.loc[idx])
    rdf = pd.DataFrame(rows).reset_index(drop=True)
    feats = [c for c in ["cum_damage_hi","damage_rate_hi","tail_damage_rate_hi",
                          "tail_to_life_damage_ratio","weighted_rms_rel",
                          "baseline_weighted_rms","current_weighted_rms",
                          "rear_front_rel","impulse_score","low_damage_impulse_conflict"]
             if c in rdf.columns]
    X  = rdf[feats].replace([np.inf,-np.inf],np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    mu = np.nanmedian(X, axis=0)
    sc = np.array([_iqr(X[:,i],0.05) for i in range(X.shape[1])])
    rh = rdf.RUL_sec.to_numpy(dtype=np.float64) / 3600.0
    bl = rdf.baseline_weighted_rms.to_numpy(dtype=np.float64)
    th = rdf.tail_damage_rate_hi.replace([np.inf,-np.inf],np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    return K50Ref(obs, len(rdf), float(np.quantile(rh,0.25)), float(np.median(rh)),
                  float(np.max(rh)), _iqr(rh,1.0), float(np.max(bl)), _iqr(bl,0.02),
                  float(np.median(th)), _iqr(th,0.05),
                  feats, mu, sc, (X-mu)/sc, rh)

def build_dist_ref(df: pd.DataFrame, obs: int, window: int | None = None) -> DistRef:
    if window is None:
        window = max(8, int(round(obs * 0.20)))
    lo, hi = max(1, obs-window), obs+window
    rdf = df[(df.file_index.astype(int) >= lo) & (df.file_index.astype(int) <= hi)].copy()
    if len(rdf) < 12:
        rdf = df.copy()
    feats = [c for c in ["cum_damage_hi","damage_rate_hi","tail_damage_rate_hi",
                          "tail_to_life_damage_ratio","weighted_rms_rel",
                          "baseline_weighted_rms","current_weighted_rms",
                          "rear_front_rel","uniform_rel","impulse_score",
                          "low_damage_impulse_conflict","damage_slope_6","damage_accel_6"]
             if c in rdf.columns]
    X  = rdf[feats].replace([np.inf,-np.inf],np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    mu = np.nanmedian(X, axis=0)
    sc = np.array([_iqr(X[:,i],0.05) for i in range(X.shape[1])])
    return DistRef(int(obs), int(window), len(rdf), feats, mu, sc, (X-mu)/sc,
                   rdf.RUL_sec.to_numpy(dtype=np.float64)/3600.0)

def predict_dist_ref(last: pd.Series, ref: DistRef) -> Dict[str, float]:
    x  = np.array([float(last.get(c,0.0)) for c in ref.features], dtype=np.float64)
    x  = np.nan_to_num((x-ref.mean)/ref.scale, nan=0.0, posinf=0.0, neginf=0.0)
    # Euclidean distance on IQR-normalised features.
    # Mahalanobis was evaluated but damage metrics (cum_damage_hi, damage_rate_hi,
    # tail_damage_rate_hi, …) are highly correlated by construction, producing
    # ill-conditioned covariance matrices even after trace-proportional regularisation.
    # The resulting inflated Mahalanobis distances shifted all test bearings into the
    # extreme_ood branch (dist_near >> 2.50 threshold), collapsing predictions.
    # Euclidean distance on the already-IQR-scaled features provides robust neighbour
    # weighting without requiring a well-conditioned covariance estimate.
    d = np.sqrt(np.mean((ref.X - x)**2, axis=1))
    w  = np.exp(-0.5 * d * d)
    if not np.isfinite(w).all() or w.sum() <= EPS:
        w = np.ones_like(ref.rul_h)
    return {"dist_q25_h": _wquantile(ref.rul_h, w, 0.25),
            "dist_q50_h": _wquantile(ref.rul_h, w, 0.50),
            "dist_q75_h": _wquantile(ref.rul_h, w, 0.75),
            "dist_near":  float(np.min(d)),
            "dist_n":     float(ref.n)}

def build_risk_info(last: pd.Series, ref: K50Ref) -> Dict[str, float]:
    x  = np.array([float(last.get(c,0.0)) for c in ref.features], dtype=np.float64)
    x  = np.nan_to_num((x-ref.mean)/ref.scale, nan=0.0, posinf=0.0, neginf=0.0)
    # Euclidean distance on IQR-normalised features (same rationale as predict_dist_ref).
    d = np.sqrt(np.mean((ref.X - x)**2, axis=1))
    w  = np.exp(-0.5 * d)
    if not np.isfinite(w).all() or w.sum() <= EPS:
        w = np.ones_like(ref.rul_h)
    analog_h = float(np.sum(w*ref.rul_h)/np.sum(w))
    tr = float(last.get("tail_to_life_damage_ratio", 0.0))
    th = float(last.get("tail_damage_rate_hi", 0.0))
    tz = max(0.0, (tr - float(np.median(ref.rul_h))) / max(_iqr(ref.rul_h,EPS), EPS))
    thz= max(0.0, (th - ref.tail_hi_med) / max(ref.tail_hi_iqr, EPS))
    az = max(0.0, tz, thz)
    bood = max(0.0, (float(last.get("baseline_weighted_rms",0.0)) - ref.base_max) / max(ref.base_iqr, EPS))
    return {"k50_analog_h": analog_h, "k50_near": float(np.min(d)),
            "k50_active_z": az, "k50_baseline_ood": bood,
            "k50_q25_h": ref.q25_h, "k50_median_h": ref.median_h}

# ── 상태 분류 + 분위수 선택 (핵심 알고리즘) ─────────────────────────────────────
def principled_target(prior_h: float, last: pd.Series, fit: FitResult,
                      risk: Dict[str,float], dist: Dict[str,float],
                      gp: GuardParams = DEFAULT_GUARD) -> Tuple[float, Dict]:
    mad_h    = max(float(fit.mad_h), 0.25)
    active_z = float(risk.get("k50_active_z", 0.0))
    bood     = float(risk.get("k50_baseline_ood", 0.0))
    near     = float(dist.get("dist_near", np.inf))
    q25      = float(dist.get("dist_q25_h", np.nan))
    q50      = float(dist.get("dist_q50_h", np.nan))
    q75      = float(dist.get("dist_q75_h", np.nan))
    oflat    = bool(float(last.get("order_fault_flat_support", 0.0)) >= 0.5)
    ogrow    = bool(float(last.get("order_fault_growth_watch", 0.0)) >= 0.5)
    oact     = float(last.get("order_fault_active_risk", 0.0))
    otrel    = float(last.get("order_fault_tail_rel", 0.0))
    otsl     = float(last.get("order_fault_tail_slope", 0.0))
    tail_hi  = float(last.get("tail_damage_rate_hi", 0.0))
    conflict = float(last.get("low_damage_impulse_conflict", 0.0))
    if not np.isfinite(q25): q25 = prior_h
    if not np.isfinite(q50): q50 = prior_h
    if not np.isfinite(q75): q75 = prior_h
    spread = max(0.0, q75 - q50)
    q40    = q25 + 0.60 * max(0.0, q50 - q25)
    q58    = q50 + gp.late_spread   * spread
    q62    = q50 + gp.stable_spread * spread
    dist_ok   = np.isfinite(near) and near <= 2.50
    ext_ood   = bood > 12.0 or (np.isfinite(near) and near > 3.0)
    act_order = (ogrow or oact >= 0.62 or
                 (otsl > 0 and otrel > 1.15 and oact >= 0.45))
    bood_stab = (bood > 0 and active_z < 1.25 and oact < 0.45 and not act_order)
    early_stab= (dist_ok and bood <= 0 and oflat and
                 active_z < 0.85 and oact < 0.40 and tail_hi < 1.25)
    late_notterm = (dist_ok and oflat and 0.85 <= active_z < 2.80
                    and oact < 0.58 and not act_order)
    ld_conflict  = (conflict >= 0.50 and active_z < 1.40 and
                    oact < 0.55 and not act_order)
    state, qname, target = "mid_uncertain", "q50", q50
    if act_order or active_z >= 3.0:
        sev     = active_z / (active_z + 4.0) if active_z > 0 else 0.0
        target  = q25 - (gp.active_mad + sev) * mad_h
        state, qname = "active_growth", "q25_lobo"
    elif ext_ood:
        om  = min(2.20, max(1.20, bood / 8.0))
        target = q25 - om * mad_h
        se  = max(0.0, bood - gp.severe_ood_z)
        sn  = max(0.0, near - gp.severe_near) if np.isfinite(near) else 0.0
        if gp.severe_comp > 0 and se > 0 and sn > 0:
            comp = 1.0 + gp.severe_comp * (bood/max(gp.severe_ood_z,EPS) + near/max(gp.severe_near,EPS))
            target = max(fit.min_rul_h, target / comp)
            qname = "q25_dual_ood"
        else:
            qname = "q25_ood"
        state = "extreme_ood"
    elif bood_stab:
        if bood >= 6.0 and dist_ok:
            bsev = min(0.55, bood/(bood+16.0))
            target = q25 - bsev * mad_h
            qname  = "q25_bood"
        else:
            target = q50 if dist_ok else min(q40, prior_h)
            qname  = "q50_bood"
        state = "baseline_ood_stable"
    elif early_stab or ld_conflict:
        target = q62; state, qname = "early_mid_stable", "q62"
    elif late_notterm:
        target = q58; state, qname = "late_but_not_terminal", "q58"
    elif active_z < 2.2 and dist_ok and oact < 0.60:
        target = q58; state, qname = "mid_active", "q58"
    elif oflat and dist_ok:
        target = q50; state, qname = "flat_order_mid", "q50"
    else:
        target = min(q50, prior_h); qname = "q50_cap"
    if state == "late_but_not_terminal" and active_z >= 2.0 and target > prior_h + mad_h:
        shift  = mad_h * active_z / (active_z + 1.0)
        target = min(target, prior_h - shift)
        qname += "+prior_cap"
    if target > prior_h:
        if   state == "late_but_not_terminal": target = min(target, prior_h + 1.25*mad_h)
        elif state == "early_mid_stable":      target = min(target, prior_h + (0.80+gp.stable_spread)*mad_h)
        elif state == "baseline_ood_stable":   target = min(target, prior_h + 0.82*mad_h)
        elif bood <= 0 and oflat:              target = min(target, prior_h + 1.05*mad_h)
        elif bood <= 0:                        target = min(target, prior_h + 0.78*mad_h)
        else:                                  target = min(target, prior_h + 0.45*mad_h)
    target = float(np.clip(target, fit.min_rul_h, fit.max_rul_h))
    info = {**risk, **dist, "prior_h": prior_h, "target_h": target,
            "state": state, "qname": qname,
            "q58_h": q58, "q62_h": q62, "q50_h": q50, "q25_h": q25, "q75_h": q75}
    return target, info

# ── 분포 가드 LOBO 선택 ────────────────────────────────────────────────────────
def select_guard_lobo(df: pd.DataFrame, blend: BlendResult,
                      obs: int) -> Tuple[GuardParams, float, pd.DataFrame]:
    obs_cands = sorted({max(5, int(round(obs*r))) for r in (0.70,0.85,1.00,1.15,1.30)})
    fold_recs = []
    for o in obs_cands:
        for ho in sorted(df.train_id.unique()):
            tr   = df[df.train_id != ho]
            va   = df[df.train_id == ho].sort_values("file_index")
            if va.empty: continue
            ridx = (va.file_index.astype(float) - float(o)).abs().idxmin()
            row  = va.loc[ridx]
            bf   = fit_hi_model(tr); pf = fit_phys_ridge(tr)
            kr   = build_k50_ref(tr, int(row.file_index))
            dr   = build_dist_ref(tr, int(row.file_index))
            bh   = float(predict_hi(np.array([row.cum_damage_hi]), bf)[0])
            pr, pha = predict_phys_raw(pd.DataFrame([row]), pf)
            rel  = float(phys_reliability(pr, pha, pf)[0])
            ph   = float(pha[0])
            pm   = _phase_multiplier(float(row.cum_damage_hi))
            ew   = min(blend.w * pm * rel, 0.70)
            prio = float(np.clip((1-ew)*bh + ew*ph - blend.margin_h, bf.min_rul_h, bf.max_rul_h))
            risk = build_risk_info(row, kr)
            dist = predict_dist_ref(row, dr)
            fold_recs.append((o, ho, row, bf, prio, risk, dist, int(row.RUL_sec)))
    rows = []; best_gp = DEFAULT_GUARD; best_obj = -np.inf
    for gp in GUARD_GRID:
        scs = []
        for o, ho, row, bf, prio, risk, dist, act in fold_recs:
            ph, _ = principled_target(prio, row, bf, risk, dist, gp)
            scs.append(float(official_score(np.array([act]), np.array([int(round(ph*3600))]))[0]))
        cv  = float(np.mean(scs)) if scs else -np.inf
        comp= 0.001*max(0,gp.stable_spread-DEFAULT_GUARD.stable_spread) + 0.002*gp.severe_comp
        obj = cv - comp
        if obj > best_obj:
            best_obj, best_gp = obj, gp
        rows.append({**gp.__dict__, "cv": cv, "obj": obj})
    rep = pd.DataFrame(rows)
    return best_gp, float(best_obj), rep

# ── LOBO 검증 ─────────────────────────────────────────────────────────────────
def lobo_validate(df: pd.DataFrame, blend: BlendResult):
    sum_rows, pred_rows = [], []
    for ho in sorted(df.train_id.unique()):
        tr = df[df.train_id != ho]; va = df[df.train_id == ho]
        bf = fit_hi_model(tr); pf = fit_phys_ridge(tr)
        hi_vals = va.cum_damage_hi.to_numpy(dtype=np.float64)
        bh  = predict_hi(hi_vals, bf)
        pr, ph = predict_phys_raw(va, pf)
        pm  = np.array([_phase_multiplier(h) for h in hi_vals])
        ew  = np.minimum(blend.w * pm * phys_reliability(pr, ph, pf), 0.70)
        ph2 = np.clip((1-ew)*bh + ew*ph - blend.margin_h, bf.min_rul_h, bf.max_rul_h)
        sc  = official_score(va.RUL_sec.to_numpy(), np.rint(ph2*3600))
        tmp = va[["train_id","file_index","RUL_sec","cum_damage_hi"]].copy()
        tmp["pred_h"] = ph2; tmp["pred_base_h"] = bh; tmp["score"] = sc
        pred_rows.append(tmp)
        sum_rows.append({"heldout":ho,"n":len(va),
                         "score_mean":float(np.mean(sc)),"score_med":float(np.median(sc)),
                         "bias_h":float(np.mean(ph2 - va.RUL_sec.to_numpy()/3600))})
    return pd.DataFrame(sum_rows), pd.concat(pred_rows, ignore_index=True)

# ── 최종 예측 ─────────────────────────────────────────────────────────────────
def infer_obs(df: pd.DataFrame) -> int:
    cnts = df.groupby("Test_ID").size().to_numpy(dtype=np.float64)
    cnts = cnts[np.isfinite(cnts)]
    return int(round(float(np.median(cnts)))) if len(cnts) else K_OBS_FALLBACK

def final_predictions(test_df: pd.DataFrame, fit: FitResult, pfit: PhysFit,
                      blend: BlendResult, k50: K50Ref, dist: DistRef,
                      ch_w: Dict[str,float], gp: GuardParams):
    td = compute_damage(test_df, "Test_ID", "file_index", ch_w)
    td = add_physics_features(td, "Test_ID", "file_index")
    td = add_order_diagnostics(td, "Test_ID", "file_index")
    rows = []
    for tid, g in td.sort_values(["Test_ID","file_index"]).groupby("Test_ID", sort=False):
        last  = g.iloc[-1]
        bh    = float(predict_hi(np.array([last.cum_damage_hi]), fit)[0])
        pr, pha = predict_phys_raw(g.tail(1), pfit)
        rel   = float(phys_reliability(pr, pha, pfit)[0])
        ph    = float(pha[0])
        pm    = _phase_multiplier(float(last.cum_damage_hi))
        ew    = min(blend.w * pm * rel, 0.70)   # phase-adaptive effective weight
        prior = float(np.clip((1-ew)*bh + ew*ph - blend.margin_h, fit.min_rul_h, fit.max_rul_h))
        risk  = build_risk_info(last, k50)
        di    = predict_dist_ref(last, dist)
        pred_h, info = principled_target(prior, last, fit, risk, di, gp)
        lo = max(fit.min_rul_h, pred_h - fit.mad_h)
        hi = min(fit.max_rul_h, pred_h + fit.mad_h)
        rows.append({"Test_ID": tid,
                     "RUL_sec":   int(round(pred_h * 3600.0)),
                     "RUL_hour":  pred_h,
                     "RUL_low_sec":  int(round(lo * 3600.0)),
                     "RUL_high_sec": int(round(hi * 3600.0)),
                     "RUL_base_sec":   int(round(bh * 3600.0)),
                     "RUL_physics_sec":int(round(ph * 3600.0)),
                     "RUL_prior_sec":  int(round(prior * 3600.0)),
                     "cum_damage_hi":  float(last.cum_damage_hi),
                     "damage_rate_hi": float(last.get("damage_rate_hi", np.nan)),
                     "tail_damage_rate_hi": float(last.get("tail_damage_rate_hi", np.nan)),
                     "baseline_weighted_rms": float(last.get("baseline_weighted_rms", np.nan)),
                     "impulse_score":  float(last.get("impulse_score", np.nan)),
                     "order_fault_tail_rel": float(last.get("order_fault_tail_rel", np.nan)),
                     "order_fault_flat_support": float(last.get("order_fault_flat_support", np.nan)),
                     "blend_w": blend.w, "physics_rel": rel,
                     "phase_mul": pm, "eff_blend_w": ew,
                     **info})
    return pd.DataFrame(rows), td

# ── 경로 탐색 ─────────────────────────────────────────────────────────────────
def locate_base(env="KSPHM_BASE") -> Path:
    cands = []
    if os.environ.get(env): cands.append(Path(os.environ[env]))
    here = Path(globals().get("__file__", Path.cwd())).resolve().parent
    cands += [Path.cwd(), here, here.parent, here.parent.parent,
              Path("/content/ksphm"), Path("/content/data"), Path("/data")]
    for c in cands:
        if (c/"merged_features.csv").exists() and (c/"Test").exists():
            return c.resolve()
    raise FileNotFoundError("merged_features.csv 와 Test/ 폴더를 찾을 수 없습니다. "
                            "KSPHM_BASE 환경변수를 설정하세요.")

def locate_out(base: Path) -> Path:
    if os.environ.get("KSPHM_OUT_DIR"):
        p = Path(os.environ["KSPHM_OUT_DIR"])
    elif Path("/content").exists() and str(base).startswith("/content"):
        p = base / "output"
    else:
        p = Path(globals().get("__file__", Path.cwd())).resolve().parent
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()

def _safe_sheet(name: str, used: set) -> str:
    base = re.sub(r'[\[\]:*?/\\]','_',name)[:31] or "Sheet"
    cand = base; i = 1
    while cand in used:
        sfx = f"_{i}"; cand = f"{base[:31-len(sfx)]}{sfx}"; i+=1
    used.add(cand); return cand

# ── 저장 ──────────────────────────────────────────────────────────────────────
def save_outputs(out: Path, pred_df, official_df, train_df, test_traj,
                 lobo_sum, lobo_pred, fit, pfit, blend, gp, gp_cv, gp_rep,
                 k50, dist, ch_w, ch_rep, elapsed, base):
    def safe_path(p):
        if not p.exists(): return p
        try:
            with open(p,"a+b"): pass
            return p
        except PermissionError:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            return p.with_name(f"{p.stem}_{stamp}{p.suffix}")
    off_p = safe_path(out/f"{TEAM_NAME}_validation.xlsx")
    all_p = safe_path(out/f"bearing_rul_{VERSION}_all.xlsx")
    official_df.to_excel(off_p, index=False)
    used = set()
    cfg = pd.DataFrame([
        {"Key":"VERSION",       "Value":VERSION},
        {"Key":"SEED",          "Value":SEED},
        {"Key":"BASE_DIR",      "Value":str(base)},
        {"Key":"INIT_N",        "Value":INIT_N},
        {"Key":"CHANNEL_WEIGHTS","Value":json.dumps(ch_w)},
        {"Key":"FIT_KIND",      "Value":fit.kind},
        {"Key":"FIT_IC",        "Value":fit.intercept_h},
        {"Key":"FIT_SL",        "Value":fit.slope_h},
        {"Key":"FIT_CU",        "Value":fit.curvature_h},
        {"Key":"FIT_MAD_H",     "Value":fit.mad_h},
        {"Key":"FIT_MAX_H",     "Value":fit.max_rul_h},
        {"Key":"BLEND_W",       "Value":blend.w},
        {"Key":"BLEND_MARGIN_H","Value":blend.margin_h},
        {"Key":"BLEND_CV",      "Value":blend.cv},
        {"Key":"GUARD_PARAMS",  "Value":json.dumps(gp.__dict__)},
        {"Key":"GUARD_CV",      "Value":gp_cv},
        {"Key":"K50_OBS",       "Value":k50.obs},
        {"Key":"DIST_OBS",      "Value":dist.obs},
        {"Key":"DIST_WINDOW",   "Value":dist.window},
        {"Key":"ELAPSED_SEC",   "Value":elapsed},
        {"Key":"TORCH_OK",      "Value":TORCH_OK},
        {"Key":"PYTHON",        "Value":sys.version},
        {"Key":"PLATFORM",      "Value":platform.platform()},
    ])
    tr_cols = ["train_id","file_index","RUL_sec","cum_damage_hi","damage_rate_hi",
               "tail_damage_rate_hi","tail_to_life_damage_ratio","cum_damage",
               "weighted_rms_rel","baseline_weighted_rms","rear_front_rel",
               "impulse_score","damage_slope_6","damage_accel_6",*RMS_COLS]
    with pd.ExcelWriter(all_p, engine="openpyxl") as wr:
        pred_df.to_excel(wr, sheet_name=_safe_sheet("Prediction",used), index=False)
        official_df.to_excel(wr, sheet_name=_safe_sheet("Official",used), index=False)
        lobo_sum.to_excel(wr, sheet_name=_safe_sheet("LOBO_Summary",used), index=False)
        lobo_pred.to_excel(wr, sheet_name=_safe_sheet("LOBO_Pred",used), index=False)
        gp_rep.to_excel(wr, sheet_name=_safe_sheet("Guard_Grid",used), index=False)
        ch_rep.to_excel(wr, sheet_name=_safe_sheet("Channel_Weights",used), index=False)
        train_df[[c for c in tr_cols if c in train_df.columns]].to_excel(
            wr, sheet_name=_safe_sheet("Train_HI",used), index=False)
        test_traj.to_excel(wr, sheet_name=_safe_sheet("Test_HI",used), index=False)
        cfg.to_excel(wr, sheet_name=_safe_sheet("Config",used), index=False)
    return off_p, all_p

def main():
    t0 = time.time()
    base = locate_base()
    out  = locate_out(base)
    print("="*64)
    print(f"  Bearing RUL Pipeline {VERSION}  |  SEED={SEED}")
    print(f"  BASE: {base}")
    print("="*64)
    print("\n[1] 학습 데이터 로딩 및 채널 가중치 선택")
    train_raw = read_train(base / "merged_features.csv")
    ch_w, train_df, ch_rep = select_channel_weights_lobo(train_raw)
    print(f"  Train shape: {train_df.shape}")
    print(f"  Channel weights: { {k:round(v,4) for k,v in ch_w.items()} }")
    print("\n[2] 모델 피팅 및 블렌드 가중치 선택 (LOBO)")
    fit   = fit_hi_model(train_df)
    pfit  = fit_phys_ridge(train_df)
    blend = _select_blend(train_df)
    print(f"  HI 모델: {fit.kind}  ic={fit.intercept_h:.3f}  sl={fit.slope_h:.3f}  cu={fit.curvature_h:.3f}  MAD={fit.mad_h:.2f}h")
    print(f"  Blend: w={blend.w:.2f}  margin={blend.margin_h:.2f}h  CV={blend.cv:.4f}")
    print("\n[3] LOBO 검증")
    lobo_sum, lobo_pred = lobo_validate(train_df, blend)
    print(lobo_sum[["heldout","n","score_mean","bias_h"]].to_string(index=False))
    print(f"  LOBO mean score: {lobo_sum.score_mean.mean():.4f}")
    print("\n[4] 테스트 TDMS 파싱 (캐시 지원)")
    cache_dir = out / ".tdms_cache"
    test_df = load_test_cached(base / "Test", cache_dir)
    obs     = infer_obs(test_df)
    k50     = build_k50_ref(train_df, obs)
    dist    = build_dist_ref(train_df, obs)
    gp, gp_cv, gp_rep = select_guard_lobo(train_df, blend, obs)
    print(f"  관측 파일 수: {obs}  |  K50 q25={k50.q25_h:.2f}h  median={k50.median_h:.2f}h")
    print(f"  분포 레퍼런스: n={dist.n}  window=±{dist.window}")
    print(f"  가드 파라미터 CV={gp_cv:.4f}: {gp}")
    print("\n[5] 최종 예측")
    pred_df, test_traj = final_predictions(test_df, fit, pfit, blend, k50, dist, ch_w, gp)
    official_df = pred_df.rename(columns={"Test_ID":"File","RUL_sec":"RUL_Score"})[["File","RUL_Score"]].copy()
    elapsed = time.time() - t0
    off_p, all_p = save_outputs(out, pred_df, official_df, train_df, test_traj,
                                lobo_sum, lobo_pred, fit, pfit, blend, gp, gp_cv, gp_rep,
                                k50, dist, ch_w, ch_rep, elapsed, base)
    print(f"\n[6] 저장 완료")
    print(f"  공식 제출 파일 -> {off_p}")
    print(f"  진단 워크북  -> {all_p}")
    print("\n" + "="*64)
    print(f"  LOBO mean: {lobo_sum.score_mean.mean():.4f}  |  Elapsed: {elapsed:.1f}s")
    print("="*64)
    print("\n  +--------------------------------------------+")
    print("  |   File   |        RUL_Score               |")
    print("  +----------+--------------------------------+")
    for _, row in official_df.iterrows():
        sec = int(row.RUL_Score)
        print(f"  |  {row.File:<7} |  {sec:>8,}s  ({sec/3600:5.2f}h)  |")
    print("  +--------------------------------------------+")

if __name__ == "__main__":
    main()
