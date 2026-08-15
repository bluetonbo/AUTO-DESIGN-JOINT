# -*- coding: utf-8 -*-
"""
================================================================================
Generative Design 엔진 (부품 무관, 재사용 가능)
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
from pymoo.optimize import minimize
from pymoo.termination import get_termination

warnings.filterwarnings("ignore")


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
    direction: str
    spec_limit: float | None = None


@dataclass
class PartConfig:
    part_name: str
    kpi_columns: list[str] = field(default_factory=list)
    kpi_directions: dict[str, str] = field(default_factory=dict)

    data_source: str = "excel"
    xlsx_path: str = ""
    sheet_id: str = ""
    worksheet_name: str = ""

    header_row: int = 1
    spec_row: int = 8
    data_row_start: int = 2
    data_row_end: int = 8

    kor_labels: dict[str, str] = field(default_factory=dict)
    derived_formulas: dict[str, Callable[[dict], float]] = field(default_factory=dict)
    fixed_vars: dict[str, float] = field(default_factory=dict)
    shape_tolerance_vars: list[str] = field(default_factory=list)


def parse_spec(s) -> tuple[str, object]:
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


def load_raw_table(cfg: PartConfig, gcp_credentials: dict | None = None) -> pd.DataFrame:
    if cfg.data_source == "gsheet":
        if gcp_credentials is None:
            raise ValueError("data_source='gsheet' 사용 시 gcp_credentials(서비스 계정 dict)가 필요합니다.")
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
    raw = raw.replace("", np.nan)
    return raw


def load_part_data(cfg: PartConfig, gcp_credentials: dict | None = None, raw: pd.DataFrame | None = None):
    if raw is None:
        raw = load_raw_table(cfg, gcp_credentials)
    all_names = raw.iloc[cfg.header_row, 1:].tolist()
    spec_texts = raw.iloc[cfg.spec_row, 1:].tolist()
    data = raw.iloc[cfg.data_row_start:cfg.data_row_end, 1:].reset_index(drop=True)
    data.columns = all_names
    data = data.apply(pd.to_numeric, errors="coerce")

    spec_map = {n: parse_spec(s) for n, s in zip(all_names, spec_texts)}

    all_var_cols = [c for c in all_names if c not in cfg.kpi_columns]

    all_nan_cols = [c for c in all_var_cols if data[c].isna().all()]

    derived_cols = list(cfg.derived_formulas.keys())
    fixed_cols = list(cfg.fixed_vars.keys())
    excluded = set(all_nan_cols) | set(derived_cols) | set(fixed_cols)

    independent_var_cols = [c for c in all_var_cols if c not in excluded]

    used_cols = independent_var_cols + cfg.kpi_columns
    data[used_cols] = data[used_cols].fillna(data[used_cols].mean())

    for name, formula in cfg.derived_formulas.items():
        data[name] = data.apply(lambda row: formula(row.to_dict()), axis=1)

    for name, val in cfg.fixed_vars.items():
        data[name] = val

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

    for c in derived_cols:
        parse_report.append((c, "파생(수식 계산)", "다른 변수로부터 계산됨"))
    for c, v in cfg.fixed_vars.items():
        parse_report.append((c, "고정값", f"={v}"))

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


def train_surrogate_models(var_cols: list[str], data: pd.DataFrame, kpi_columns: list[str]):
    X = data[var_cols].values
    models, cv_scores = {}, {}
    for kpi in kpi_columns:
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


def make_predictor(models: dict, cfg: PartConfig):
    def predictor(x: np.ndarray, variables: list[DesignVariable]) -> dict[str, float]:
        x_arr = x.reshape(1, -1)
        kpi_pred = {kpi: float(model.predict(x_arr)[0]) for kpi, model in models.items()}
        return kpi_pred
    return predictor


def expand_with_derived(row_dict: dict, cfg: PartConfig) -> dict:
    full = dict(row_dict)
    for name, formula in cfg.derived_formulas.items():
        full[name] = formula(full)
    for name, val in cfg.fixed_vars.items():
        full[name] = val
    return full


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
              pop_size=120, n_generations=60, seed=1) -> pd.DataFrame:
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
