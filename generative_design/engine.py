# -*- coding: utf-8 -*-
"""
================================================================================
Generative Design 엔진 (부품 무관, 재사용 가능)
================================================================================

이 파일은 특정 부품(볼조인트/에어스프링/암 등)에 종속되지 않는 공용 로직만 담습니다.
부품별 정보(변수/KPI 목록, 방향, 파생 수식, 고정값 등)는 configs/ 폴더의
설정 파일에서 정의하고, 이 엔진에 주입해서 사용합니다.

새 부품을 추가하려면:
  1. configs/ 폴더에 새 설정 파일 작성 (configs/ball_joint_volvo.py 참고)
  2. run_pipeline.py에서 설정 파일만 바꿔서 실행

구성 요소:
  - PartConfig            : 부품별 설정을 담는 데이터클래스
  - parse_spec()          : 스펙 텍스트 파서 (범용)
  - DesignVariable / DesignObjective : pymoo 최적화용 메타 정의
  - load_part_data()      : 엑셀 로드 + 스펙 파싱 + 파생치수 계산 + 고정값 처리
  - train_surrogate_models() / run_nsga2() / filter_within_spec()
================================================================================
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_score

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------------------
# 데이터 구조
# ------------------------------------------------------------------------------

@dataclass
class DesignVariable:
    name: str
    lower: float
    upper: float
    unit: str = ""
    fixed: bool = False
    fixed_value: float | None = None


@dataclass
class DesignObjective:
    name: str
    direction: str  # "min" or "max"
    spec_limit: float | None = None


@dataclass
class PartConfig:
    """부품 하나를 정의하는 설정 묶음. configs/*.py 에서 이 클래스의 인스턴스를 만든다."""
    part_name: str                                          # 예: "VOLVO SPA12 볼조인트"
    kpi_columns: list[str] = field(default_factory=list)     # KPI(목표) 컬럼명 리스트
    kpi_directions: dict[str, str] = field(default_factory=dict)  # KPI별 "min"/"max"

    # 데이터 소스: "excel" 또는 "gsheet"
    data_source: str = "excel"
    xlsx_path: str = ""                          # data_source="excel"일 때 사용
    sheet_id: str = ""                           # data_source="gsheet"일 때 사용 (Google Sheets 문서 ID)
    worksheet_name: str = ""                     # data_source="gsheet"일 때 사용 (탭 이름, 비우면 첫 탭)

    # 시트/엑셀 내 행 위치 (0-index) - 엑셀/구글시트 모두 동일 레이아웃이라고 가정
    header_row: int = 1                          # 변수명이 있는 행
    spec_row: int = 8                            # 스펙 텍스트가 있는 행
    data_row_start: int = 2                      # 실측 데이터 시작 행
    data_row_end: int = 8                        # 실측 데이터 끝 행 (exclusive)

    kor_labels: dict[str, str] = field(default_factory=dict)   # 한글 설명 (선택)
    derived_formulas: dict[str, Callable[[dict], float]] = field(default_factory=dict)
    # 파생(종속) 치수: 다른 설계변수로 계산되므로 독립변수에서 제외되고,
    # 후보 생성 후 수식으로 값이 채워짐
    fixed_vars: dict[str, float] = field(default_factory=dict)
    # 실질적으로 고정된 치수 (스펙에 공차 없이 상수로 명시됨): 탐색 대상에서 제외, 항상 해당 값 사용
    shape_tolerance_vars: list[str] = field(default_factory=list)
    # 형상공차류(진원도/평면도 등): 스펙이 단일 숫자여도 "0~X"로 해석


# ------------------------------------------------------------------------------
# 스펙 문자열 파서 (범용)
# ------------------------------------------------------------------------------

def parse_spec(s) -> tuple[str, object]:
    """
    스펙 텍스트를 해석. 반환: (종류, 값)
      'range'     : (lo, hi)   - 양측 공차/범위
      'min_only'  : lo         - "X Min" 형태 (하한만)
      'zero_to_x' : hi         - 공차 없는 단일 양수 (형상공차 후보, 호출측에서 판단)
      'fallback'  : 원문 문자열 - 수식/무공차 참고치수 (파싱 불가)
      'none'      : None       - 스펙 없음
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
# 원시 테이블 로드 (엑셀 또는 Google Sheets 공용)
# ------------------------------------------------------------------------------

def load_raw_table(cfg: PartConfig, gcp_credentials: dict | None = None) -> pd.DataFrame:
    """
    cfg.data_source에 따라 엑셀 또는 Google Sheets에서 원시 표(헤더 없음, 0-index 행)를 읽어온다.
    반환되는 DataFrame은 pd.read_excel(..., header=None)과 동일한 모양이어야 한다
    (즉, 실제 헤더/스펙/데이터 행 위치는 cfg.header_row 등으로 이후 단계에서 지정).

    gsheet 소스는 서비스 계정 자격증명(dict, st.secrets["gcp_service_account"] 형식)이 필요하다.
    """
    if cfg.data_source == "gsheet":
        if gcp_credentials is None:
            raise ValueError(
                "data_source='gsheet' 사용 시 gcp_credentials(서비스 계정 dict)가 필요합니다."
            )
        return _load_raw_table_from_gsheet(cfg, gcp_credentials)
    return pd.read_excel(cfg.xlsx_path, sheet_name=0, header=None)


def _load_raw_table_from_gsheet(cfg: PartConfig, gcp_credentials: dict) -> pd.DataFrame:
    import gspread
    from google.oauth2.service_account import Credentials

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_info(dict(gcp_credentials), scopes=scopes)
    gc = gspread.authorize(creds)

    sh = gc.open_by_key(cfg.sheet_id)
    ws = sh.worksheet(cfg.worksheet_name) if cfg.worksheet_name else sh.get_worksheet(0)

    values = ws.get_all_values()
    raw = pd.DataFrame(values)
    # 빈 문자열 -> NaN (엑셀 로드 시 빈 셀이 NaN으로 오는 것과 동일하게 맞춤)
    raw = raw.replace("", np.nan)
    # 숫자로 보이는 셀은 문자열로 남아있으므로, 이후 pd.to_numeric(errors="coerce")에서 자동 변환됨
    return raw


# ------------------------------------------------------------------------------
# 데이터 로드 (원시 표 -> 변수/KPI 메타 + 정제된 데이터)
# ------------------------------------------------------------------------------

def load_part_data(cfg: PartConfig, gcp_credentials: dict | None = None, raw: pd.DataFrame | None = None):
    """
    엑셀 또는 Google Sheets에서 데이터를 읽어:
      - 독립 설계변수 목록 (파생/고정 변수 제외)
      - KPI 목표 목록
      - 정제된 데이터(dict/DataFrame, 파생치수는 수식으로 재계산됨)
      - 파싱 리포트 (어떤 변수가 스펙파싱/데이터기반/파생/고정 인지)
    를 반환.

    raw를 직접 넘기면 로딩을 건너뛰고 그 표를 바로 사용한다 (테스트/캐싱 용도).
    """
    if raw is None:
        raw = load_raw_table(cfg, gcp_credentials)
    all_names = raw.iloc[cfg.header_row, 1:].tolist()
    spec_texts = raw.iloc[cfg.spec_row, 1:].tolist()
    data = raw.iloc[cfg.data_row_start:cfg.data_row_end, 1:].reset_index(drop=True)
    data.columns = all_names
    data = data.apply(pd.to_numeric, errors="coerce")

    spec_map = {n: parse_spec(s) for n, s in zip(all_names, spec_texts)}

    all_var_cols = [c for c in all_names if c not in cfg.kpi_columns]

    # 완전 결측 변수 제외
    all_nan_cols = [c for c in all_var_cols if data[c].isna().all()]

    # 파생(종속) 변수, 고정 변수는 독립 탐색변수에서 제외
    derived_cols = list(cfg.derived_formulas.keys())
    fixed_cols = list(cfg.fixed_vars.keys())
    excluded = set(all_nan_cols) | set(derived_cols) | set(fixed_cols)

    independent_var_cols = [c for c in all_var_cols if c not in excluded]

    # 결측치 평균 대체 (독립변수 + KPI만)
    used_cols = independent_var_cols + cfg.kpi_columns
    data[used_cols] = data[used_cols].fillna(data[used_cols].mean())

    # 파생 변수는 수식으로 재계산 (원본 데이터 대신 항상 일관되게)
    for name, formula in cfg.derived_formulas.items():
        data[name] = data.apply(lambda row: formula(row.to_dict()), axis=1)

    # 고정 변수는 상수로 채움
    for name, val in cfg.fixed_vars.items():
        data[name] = val

    # 설계변수 메타 생성 (독립변수만 - pymoo 탐색 대상)
    variables: list[DesignVariable] = []
    parse_report: list[tuple] = []

    for c in independent_var_cols:
        kind, val = spec_map.get(c, ("none", None))
        if kind == "range":
            lo, hi = val
            parse_report.append((c, "스펙 파싱", f"{lo:.4g}~{hi:.4g}"))
        elif kind == "zero_to_x" and c in cfg.shape_tolerance_vars:
            lo, hi = 0.0, val
            parse_report.append((c, "형상공차(0~X)", f"0~{val}"))
        else:
            dmin, dmax = data[c].min(), data[c].max()
            margin = (dmax - dmin) * 0.05 if dmax > dmin else (abs(dmax) * 0.02 or 0.05)
            lo, hi = dmin - margin, dmax + margin
            reason = "수식/무공차 참고치수 -> 데이터기반" if kind == "fallback" else "스펙없음 -> 데이터기반"
            parse_report.append((c, reason, f"{lo:.3f}~{hi:.3f}"))
        variables.append(DesignVariable(name=c, lower=float(lo), upper=float(hi)))

    # 파생/고정 변수도 리포트에 표시 (탐색 대상은 아님)
    for c in derived_cols:
        parse_report.append((c, "파생(수식 계산)", "다른 변수로부터 계산됨"))
    for c, v in cfg.fixed_vars.items():
        parse_report.append((c, "고정값", f"={v}"))

    # KPI 메타 생성
    objectives: list[DesignObjective] = []
    kpi_spec_ranges: dict[str, tuple] = {}
    for c in cfg.kpi_columns:
        kind, val = spec_map.get(c, ("none", None))
        direction = cfg.kpi_directions[c]
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

    return variables, objectives, independent_var_cols, data, kpi_spec_ranges, parse_report


# ------------------------------------------------------------------------------
# 대리모델
# ------------------------------------------------------------------------------

def train_surrogate_models(var_cols: list[str], data: pd.DataFrame, kpi_columns: list[str],
                            progress_callback=None):
    """
    progress_callback(idx, total, kpi_name, status, extra) 형태로 매 KPI마다 호출됨.
      status: "start" | "done"
      extra : status="done"일 때 해당 KPI의 LOO-MAE (float)
    """
    X = data[var_cols].values
    models, cv_scores = {}, {}
    total = len(kpi_columns)
    for idx, kpi in enumerate(kpi_columns):
        if progress_callback:
            progress_callback(idx, total, kpi, "start", None)
        y = data[kpi].values
        model = Ridge(alpha=5.0)
        model.fit(X, y)
        models[kpi] = model
        try:
            scores = cross_val_score(model, X, y, cv=LeaveOneOut(), scoring="neg_mean_absolute_error")
            cv_scores[kpi] = float(-scores.mean())
        except Exception:
            cv_scores[kpi] = float("nan")
        if progress_callback:
            progress_callback(idx, total, kpi, "done", cv_scores[kpi])
    return models, cv_scores


def make_predictor(models: dict, cfg: PartConfig):
    """
    독립변수 -> KPI 예측 + 파생변수/고정변수를 채운 '전체 변수 딕셔너리'까지 함께 반환하는 predictor.
    """
    def predictor(x: np.ndarray, variables: list[DesignVariable]) -> dict[str, float]:
        x_arr = x.reshape(1, -1)
        kpi_pred = {kpi: float(model.predict(x_arr)[0]) for kpi, model in models.items()}
        return kpi_pred

    return predictor


def expand_with_derived(row_dict: dict, cfg: PartConfig) -> dict:
    """독립변수 값(dict)에 파생변수 + 고정변수를 계산해서 채워넣은 전체 딕셔너리 반환."""
    full = dict(row_dict)
    for name, formula in cfg.derived_formulas.items():
        full[name] = formula(full)
    for name, val in cfg.fixed_vars.items():
        full[name] = val
    return full


# ------------------------------------------------------------------------------
# pymoo Problem
# ------------------------------------------------------------------------------

class GenerativeDesignProblem(Problem):
    def __init__(self, variables: list[DesignVariable], objectives: list[DesignObjective], predictor):
        self.variables = [v for v in variables if not v.fixed]
        self.objectives = objectives
        self.predictor = predictor

        n_var = len(self.variables)
        n_obj = len(objectives)
        xl = np.array([v.lower for v in self.variables])
        xu = np.array([v.upper for v in self.variables])
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu)

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs):
        F = np.zeros((X.shape[0], self.n_obj))
        for i, x in enumerate(X):
            kpi = self.predictor(x, self.variables)
            for j, obj in enumerate(self.objectives):
                val = kpi[obj.name]
                F[i, j] = val if obj.direction == "min" else -val
        out["F"] = F


def run_nsga2(variables, objectives, predictor, cfg: PartConfig,
              pop_size=120, n_generations=60, seed=1, progress_callback=None) -> pd.DataFrame:
    """
    progress_callback(gen, n_generations)로 매 세대 종료 시 호출됨 (1-indexed gen).
    """
    problem = GenerativeDesignProblem(variables, objectives, predictor)
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )
    algorithm.setup(problem, termination=get_termination("n_gen", n_generations), seed=seed, verbose=False)

    gen = 0
    while algorithm.has_next():
        algorithm.next()
        gen += 1
        if progress_callback:
            progress_callback(gen, n_generations)

    result = algorithm.result()

    active_vars = problem.variables
    rows = []
    for x, f in zip(result.X, result.F):
        indep = {v.name: float(val) for v, val in zip(active_vars, x)}
        full_vars = expand_with_derived(indep, cfg)
        kpi = predictor(x, active_vars)
        row = {k: round(v, 4) for k, v in full_vars.items()}
        for obj in objectives:
            row[obj.name] = round(float(kpi[obj.name]), 4)
        rows.append(row)

    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def filter_within_spec(df: pd.DataFrame, objectives: list[DesignObjective], spec_ranges: dict) -> pd.DataFrame:
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
