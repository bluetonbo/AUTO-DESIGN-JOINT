# -*- coding: utf-8 -*-
"""
================================================================================
Generative Design - Phase 1 (v2, Excel 원본 데이터 기반)
VOLVO_SPA12_CABJ_TRAIN_DATA_260706_r1.xlsx 데이터 사용

이전 CSV(AUTO-DESIGN-JOINT-INPUT-VOLVO.csv) 대비 변경점:
  - 변수/KPI가 약어(BD, CID...) 대신 이 엑셀의 정식 명칭(ballstud_diameter_mm 등) 사용
  - CSV 1~3번 샘플의 bearing_Height_mm / case_inner_height_mm 값이
    스펙 범위를 크게 벗어나 있었는데(예: BH가 15.31로 스펙 24.9~25.1 밖),
    엑셀 원본 데이터로 대조한 결과 CSV 쪽 데이터 오류로 확인됨.
    -> 본 스크립트는 엑셀의 6개 샘플(정정된 값)을 그대로 사용.
  - 스펙 텍스트("Ø35±0.02", "0~8 Nm", "25 Min" 등)를 정규식으로 직접 파싱해서
    설계변수 탐색범위 / KPI 스펙 판정에 반영 (스펙 자동 파싱, 하단 parse_spec 참고)
  - 수식으로 파생되는 치수(예: case_inner_height_mm = (12.5)+(6.13)=18.63)나
    공차가 없는 참고치수(예: seat_R_mm = R18.8)는 스펙 파싱이 불가능하므로
    실측 데이터 범위(min/max)로 대체 -> 콘솔에 "fallback"으로 표시됨

*** 데이터 규모 주의 ***
  실측 샘플이 6개뿐입니다. 34개 변수 대비 매우 적어서, 이 스크립트가 만드는
  대리모델(Ridge Regression)은 파이프라인 동작 검증용입니다.
  실제 설계 의사결정에는 데이터 추가 확보 후 재학습이 필요합니다.

실행:
  python generative_design_volvo_v2.py
================================================================================
"""

from __future__ import annotations

import re
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_score

sys.path.insert(0, "/home/claude")
from generative_design_core import DesignVariable, DesignObjective, GenerativeDesignProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

warnings.filterwarnings("ignore")

XLSX_PATH = "/mnt/user-data/uploads/VOLVO_SPA12_CABJ_TRAIN_DATA_260706_r1.xlsx"

# ------------------------------------------------------------------------------
# KPI(성능지표) 10개 - 엑셀 정식 명칭 기준
# 방향(direction)은 이번 엑셀에서 확인된 스펙 문구를 근거로 확정:
#   - breakaway/running_torque, gap 계열: "0~X" 형태 -> 작을수록 좋음 (min)
#   - stiffness(강성 유지율): "25 Min" -> 클수록 좋음, 최소 25% 필요 (max)
# ------------------------------------------------------------------------------
KPI_COLUMNS = [
    "breakaway_torque_Nm",          # 최초 기동(브레이크어웨이) 토크
    "running_torque_Nm",            # 회전 중 토크
    "axial_gap_before_mm",          # 내구시험 전 축방향 갭
    "radial_gap_before_mm",         # 내구시험 전 반경방향 갭
    "axial_gap_after_mm",           # 내구시험 후 축방향 갭
    "radial_gap_after_mm",          # 내구시험 후 반경방향 갭
    "axial_gap_increase_mm",        # 축방향 갭 증가량 (내구성 지표)
    "radial_gap_increase_mm",       # 반경방향 갭 증가량
    "axial_before_after_min_stiffness_%",   # 축방향 강성 유지율(%)
    "radial_before_after_min_stiffness_%",  # 반경방향 강성 유지율(%)
]

KPI_DIRECTIONS: dict[str, str] = {
    "breakaway_torque_Nm": "min",
    "running_torque_Nm": "min",
    "axial_gap_before_mm": "min",
    "radial_gap_before_mm": "min",
    "axial_gap_after_mm": "min",
    "radial_gap_after_mm": "min",
    "axial_gap_increase_mm": "min",
    "radial_gap_increase_mm": "min",
    "axial_before_after_min_stiffness_%": "max",
    "radial_before_after_min_stiffness_%": "max",
}

# 한글 설명 (콘솔 출력/리포트용)
KOR_LABEL: dict[str, str] = {
    "ballstud_diameter_mm": "볼스터드 직경",
    "case_inner_diameter_mm": "케이스 내경",
    "bearing_Height_mm": "베어링(시트) 높이",
    "case_inner_height_mm": "케이스 내부 높이",
    "case_inner_taper_height_mm": "케이스 내부 테이퍼 높이",
    "case_outer_height_before_mm": "스웨이징 전 케이스 외부 높이",
    "case_outer_height_after_mm": "스웨이징 후 케이스 외부 높이",
    "case_d1_mm": "케이스 직경1", "case_d2_mm": "케이스 직경2",
    "case_d3_mm": "케이스 직경3", "case_d4_mm": "케이스 직경4", "case_d5_mm": "케이스 직경5",
    "case_h1_mm": "케이스 높이1", "case_h2_mm": "케이스 높이2",
    "case_h3_mm": "케이스 높이3", "case_h4_mm": "케이스 높이4",
    "case_groove_width_mm": "케이스 홈 폭", "case_groove_depth_mm": "케이스 홈 깊이",
    "case_d6_mm": "케이스 직경6",
    "case_roundness_mm": "케이스 진원도", "case_flatness_mm": "케이스 평면도",
    "seat_h1_mm": "시트 높이1", "seat_R_mm": "시트 반경",
    "seat_h3_mm": "시트 높이3", "seat_h4_mm": "시트 높이4",
    "seat_inner_d_mm": "시트 내경", "seat_outer_d_mm": "시트 외경",
    "seat_h7_mm": "시트 높이7", "seat_h8_mm": "시트 높이8", "seat_R2_mm": "시트 반경2",
    "breakaway_torque_Nm": "기동 토크", "running_torque_Nm": "회전 중 토크",
    "axial_gap_before_mm": "축방향 갭(시험전)", "radial_gap_before_mm": "반경방향 갭(시험전)",
    "axial_gap_after_mm": "축방향 갭(시험후)", "radial_gap_after_mm": "반경방향 갭(시험후)",
    "axial_gap_increase_mm": "축방향 갭 증가량", "radial_gap_increase_mm": "반경방향 갭 증가량",
    "axial_before_after_min_stiffness_%": "축방향 강성 유지율(%)",
    "radial_before_after_min_stiffness_%": "반경방향 강성 유지율(%)",
}


# ------------------------------------------------------------------------------
# 스펙 문자열 파서
# ------------------------------------------------------------------------------
def parse_spec(s) -> tuple[str, object]:
    """
    엑셀 Spec 행의 다양한 표기를 해석.
    반환: (종류, 값)
      'range'     : (lo, hi)               - 정상 파싱된 양측 공차/범위
      'min_only'  : lo                     - "X Min" 형태 (하한만 존재)
      'zero_to_x' : hi                     - 공차 없는 단일 양수 (형상공차류, 0~X로 간주)
      'fallback'  : 원문 문자열             - 수식/무공차 참고치수 (파싱 불가 -> 데이터 기반 대체)
      'none'      : None                   - 스펙 없음 (NaN)
    """
    if pd.isna(s):
        return ("none", None)
    s = str(s).strip()

    m = re.match(r"^([\d.]+)\s*~\s*([\d.]+)\s*\w*$", s)
    if m:
        return ("range", (float(m.group(1)), float(m.group(2))))

    m = re.match(r"^([\d.]+)\s*Min$", s, re.IGNORECASE)
    if m:
        return ("min_only", float(m.group(1)))

    m = re.match(r"^(?:SPH\.)?[ØR]?\s*([\d.]+)\s*±\s*([\d.]+)$", s)
    if m:
        base, tol = float(m.group(1)), float(m.group(2))
        return ("range", (base - tol, base + tol))

    m = re.match(r"^(?:SPH\.)?[ØR]?\s*([\d.]+)\s+([+-]?[\d.]+)\s*/\s*([+-]?[\d.]+)$", s)
    if m:
        base = float(m.group(1))
        off1, off2 = float(m.group(2)), float(m.group(3))
        return ("range", (base + min(off1, off2), base + max(off1, off2)))

    m = re.match(r"^([\d.]+)$", s)
    if m:
        return ("zero_to_x", float(m.group(1)))

    return ("fallback", s)


# ------------------------------------------------------------------------------
# 데이터 로드
# ------------------------------------------------------------------------------
def load_volvo_excel(xlsx_path: str = XLSX_PATH):
    raw = pd.read_excel(xlsx_path, sheet_name=0, header=None)

    all_names = raw.iloc[1, 1:].tolist()
    spec_texts = raw.iloc[8, 1:].tolist()
    data = raw.iloc[2:8, 1:].reset_index(drop=True)
    data.columns = all_names
    data = data.apply(pd.to_numeric, errors="coerce")

    var_cols = [c for c in all_names if c not in KPI_COLUMNS]

    # 데이터가 전부 결측인 변수는 제외 (seat_h2/h5/h6/h9)
    all_nan_cols = [c for c in var_cols if data[c].isna().all()]
    var_cols = [c for c in var_cols if c not in all_nan_cols]

    # 결측치는 평균 대체 (샘플이 6개뿐이라 행 삭제는 손실이 큼)
    used_cols = var_cols + KPI_COLUMNS
    data[used_cols] = data[used_cols].fillna(data[used_cols].mean())

    spec_map = {n: parse_spec(s) for n, s in zip(all_names, spec_texts)}

    # 설계변수 메타 생성
    variables: list[DesignVariable] = []
    parse_report = []
    for c in var_cols:
        kind, val = spec_map[c]
        if kind == "range":
            lo, hi = val
            parse_report.append((c, "스펙 파싱", f"{lo}~{hi}"))
        elif kind == "zero_to_x":
            lo, hi = 0.0, val
            parse_report.append((c, "형상공차(0~X)", f"0~{val}"))
        else:
            # fallback / none -> 데이터 기반 범위 (여유 5%)
            dmin, dmax = data[c].min(), data[c].max()
            margin = (dmax - dmin) * 0.05 if dmax > dmin else (abs(dmax) * 0.02 or 0.05)
            lo, hi = dmin - margin, dmax + margin
            reason = "수식/무공차 참고치수 -> 데이터기반" if kind == "fallback" else "스펙없음 -> 데이터기반"
            parse_report.append((c, reason, f"{lo:.3f}~{hi:.3f}"))
        variables.append(DesignVariable(name=c, lower=float(lo), upper=float(hi)))

    # KPI 메타 생성
    objectives: list[DesignObjective] = []
    kpi_spec_ranges: dict[str, tuple] = {}
    for c in KPI_COLUMNS:
        kind, val = spec_map[c]
        direction = KPI_DIRECTIONS[c]
        if kind == "range":
            lo, hi = val
            kpi_spec_ranges[c] = (lo, hi)
            limit = hi if direction == "min" else lo
        elif kind == "min_only":
            kpi_spec_ranges[c] = (val, None)
            limit = val
        else:
            kpi_spec_ranges[c] = None
            limit = None
        objectives.append(DesignObjective(name=c, direction=direction, spec_limit=limit))

    return variables, objectives, var_cols, data, kpi_spec_ranges, parse_report


# ------------------------------------------------------------------------------
# 대리모델 학습
# ------------------------------------------------------------------------------
def train_surrogate_models(var_cols: list[str], data: pd.DataFrame):
    X = data[var_cols].values
    models, cv_scores = {}, {}
    for kpi in KPI_COLUMNS:
        y = data[kpi].values
        model = Ridge(alpha=5.0)
        model.fit(X, y)
        models[kpi] = model
        try:
            scores = cross_val_score(model, X, y, cv=LeaveOneOut(), scoring="neg_mean_absolute_error")
            cv_scores[kpi] = float(-scores.mean())
        except Exception:
            cv_scores[kpi] = float("nan")
    return models, cv_scores


def make_predictor(models):
    def predictor(x: np.ndarray, variables: list[DesignVariable]) -> dict[str, float]:
        x_arr = x.reshape(1, -1)
        return {kpi: float(model.predict(x_arr)[0]) for kpi, model in models.items()}
    return predictor


def run_nsga2(variables, objectives, predictor, pop_size=120, n_generations=60, seed=1):
    problem = GenerativeDesignProblem(variables, objectives, predictor)
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )
    result = minimize(problem, algorithm, get_termination("n_gen", n_generations), seed=seed, verbose=False)

    active_vars = problem.variables
    rows = []
    for x, f in zip(result.X, result.F):
        row = {v.name: round(float(val), 4) for v, val in zip(active_vars, x)}
        kpi = predictor(x, active_vars)
        for obj in objectives:
            row[obj.name] = round(float(kpi[obj.name]), 4)
        rows.append(row)
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def filter_within_spec(df: pd.DataFrame, objectives: list[DesignObjective], spec_ranges: dict):
    filtered = df.copy()
    for obj in objectives:
        rng = spec_ranges.get(obj.name)
        if rng is None:
            continue
        lo, hi = rng
        if lo is not None:
            filtered = filtered[filtered[obj.name] >= lo]
        if hi is not None:
            filtered = filtered[filtered[obj.name] <= hi]
    return filtered.reset_index(drop=True)


if __name__ == "__main__":
    print("=" * 78)
    print("1) 엑셀 데이터 로드 + 스펙 파싱")
    print("=" * 78)
    variables, objectives, var_cols, data, kpi_spec_ranges, parse_report = load_volvo_excel()
    print(f"설계변수: {len(variables)}개 / KPI: {len(objectives)}개 / 샘플: {len(data)}행\n")
    for name, reason, rng in parse_report:
        kor = KOR_LABEL.get(name, "")
        print(f"  {name:32s} {kor:16s} [{reason:22s}] {rng}")

    print()
    print("=" * 78)
    print("2) 대리모델(Ridge) 학습")
    print("=" * 78)
    models, cv_scores = train_surrogate_models(var_cols, data)
    for kpi, mae in cv_scores.items():
        print(f"  {kpi:38s} {KOR_LABEL.get(kpi,''):16s} LOO-MAE(참고용): {mae:.4f}")

    print()
    print("=" * 78)
    print("3) NSGA-II 다목적 최적화")
    print("=" * 78)
    predictor = make_predictor(models)
    candidates = run_nsga2(variables, objectives, predictor)
    print(f"파레토 후보안: {len(candidates)}개")

    print()
    print("=" * 78)
    print("4) 스펙 필터링")
    print("=" * 78)
    valid = filter_within_spec(candidates, objectives, kpi_spec_ranges)
    print(f"스펙 만족 후보: {len(valid)}개 / {len(candidates)}개")

    out_cols = [o.name for o in objectives]
    print()
    if len(valid) > 0:
        print("--- 강성 유지율(축방향) 높은 순 상위 10개 ---")
        print(valid.sort_values("axial_before_after_min_stiffness_%", ascending=False)[out_cols]
              .head(10).to_string(index=False))
    else:
        print("스펙 만족 후보 없음 - 데이터 6개뿐이라 대리모델 정확도 한계. 데이터 보강 필요.")

    candidates.to_csv("/home/claude/volvo_v2_candidates_all.csv", index=False)
    valid.to_csv("/home/claude/volvo_v2_candidates_valid.csv", index=False)
    print()
    print("전체 후보 -> volvo_v2_candidates_all.csv")
    print("스펙 통과 후보 -> volvo_v2_candidates_valid.csv")
