# -*- coding: utf-8 -*-
"""
Generative Design 대시보드 (Streamlit)
UI/레이아웃: JOINT-AI-APP-6.py와 동일한 다크 콘솔 테마 적용
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataclasses
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import plotly.express as px
import gspread
from google.oauth2.service_account import Credentials

from generative_design.configs.ball_joint_volvo import CONFIG as BASE_CONFIG
from generative_design.engine import (
    load_part_data, train_surrogate_models, make_predictor,
    run_nsga2, filter_within_spec,
)

st.set_page_config(
    layout="wide",
    page_title="Generative Design - Process Optimization Suite",
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        background-color: #0f0f0f !important;
        color: #ececec !important;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stAppViewContainer"] .main .block-container {
        max-width: 100% !important;
        width: 100% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    [data-testid="stSidebar"] {
        background-color: #161616 !important;
        border-right: 1px solid #262626;
        min-width: 360px !important;
    }

    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
        color: #f2f2f2 !important;
    }

    .glass-card {
        background: #1a1a1a;
        border: 1px solid #2e2e2e;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }

    .glass-card-title {
        color: #ff9f1c;
        font-size: 0.9rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid #262626;
    }

    .stat-box {
        background: #1a1a1a;
        border: 1px solid #2e2e2e;
        border-radius: 6px;
        padding: 14px 18px;
        text-align: center;
    }
    .stat-label {
        color: #8a8a8a;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .stat-value {
        color: #ff9f1c;
        font-size: 1.7rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    .stButton>button, .stDownloadButton>button {
        height: 2.8rem !important;
        font-size: 0.9rem !important;
        border-radius: 4px !important;
        background: #10b981 !important;
        color: #ffffff !important;
        font-weight: 600;
        border: none !important;
        transition: all 0.2s ease;
        width: 100%;
    }

    label, .stTextInput label, .stSelectbox label, .stSlider label,
    .stNumberInput label, .stRadio label, .stFileUploader label,
    [data-testid="stWidgetLabel"] p {
        color: #b8b8b8 !important;
    }
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #aaaaaa !important;
    }

    [data-testid="stAlert"] p, [data-testid="stAlert"] span,
    [data-testid="stAlert"] div, [data-testid="stAlert"] *,
    [data-testid="stSidebar"] [data-testid="stAlert"] * {
        color: #efefef !important;
    }
    [data-testid="stSidebar"] h3 { color: #ffb300 !important; }

    .stTabs [data-baseweb="tab-list"] { border-bottom: 2px solid #3a3a3a; gap: 8px; }
    .stTabs [data-baseweb="tab"], .stTabs button[data-baseweb="tab"], .stTabs [role="tab"] {
        background-color: #171717 !important; border: 1px solid #3a3a3a !important;
        border-bottom: none !important; border-radius: 8px 8px 0 0 !important;
        color: #ececec !important; font-weight: 700 !important; opacity: 1 !important;
        padding: 10px 22px !important;
    }
    .stTabs [data-baseweb="tab"] * { color: #ececec !important; opacity: 1 !important; }
    .stTabs [aria-selected="true"], .stTabs button[aria-selected="true"] {
        background-color: #3d2a0f44 !important; border-color: #ff9f1c !important; color: #ff9f1c !important;
    }
    .stTabs [aria-selected="true"] * { color: #ff9f1c !important; }

    [data-testid="stExpander"] { border: 1px solid #3a3a3a !important; border-radius: 8px !important; background: #1c1c1c !important; margin-bottom: 6px !important; overflow: hidden !important; }
    .streamlit-expanderHeader, [data-testid="stExpander"] details summary { background: #1c1c1c !important; color: #d4d4d4 !important; font-weight: 600 !important; border: none !important; border-radius: 8px !important; padding: 12px 16px !important; }
    [data-testid="stExpander"] details[open] summary { background: #2a1f0f !important; color: #ff9f1c !important; border-bottom: 1px solid #3a3a3a !important; border-radius: 8px 8px 0 0 !important; }
    .streamlit-expanderContent, [data-testid="stExpander"] details > div { background: #131313 !important; border-top: 1px solid #3a3a3a !important; border-radius: 0 0 8px 8px !important; }

    [data-testid="stDataFrame"] { border: 1px solid #2e2e2e !important; border-radius: 6px !important; }
    </style>
""", unsafe_allow_html=True)

try:
    GCP_CREDENTIALS = dict(st.secrets["gcp_service_account"])
    SHEET_ID = st.secrets["sheet_id"]
    WORKSHEET_NAME = st.secrets.get("worksheet_name", "")
    CONFIG = dataclasses.replace(
        BASE_CONFIG,
        data_source="gsheet",
        sheet_id=SHEET_ID,
        worksheet_name=WORKSHEET_NAME,
    )
except KeyError:
    st.error(
        "Google Sheets 연동 정보(secrets)가 없습니다. "
        "Streamlit Cloud의 App settings → Secrets에 "
        "`gcp_service_account`, `sheet_id`를 등록해주세요."
    )
    st.stop()

# ──────────────────────────────────────────────────────────────
# 인증 시스템 (JOINT-AI-APP-6.py와 동일한 방식)
# 같은 Google Sheets 문서 안에 "temp_pwd_store" 탭을 두고 임시 비번을 관리합니다.
# ──────────────────────────────────────────────────────────────
_TEMP_PWD_WORKSHEET = "temp_pwd_store"
OWNER_PWD = "nt1234"  # 소유자 비번 (항상 유효)


@st.cache_resource(show_spinner=False)
def _get_temp_pwd_worksheet():
    """Google Sheets 워크시트 연결 객체를 세션 내에서 재사용(캐싱). 쓰기 권한 필요."""
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(GCP_CREDENTIALS, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(SHEET_ID)
    try:
        ws = sh.worksheet(_TEMP_PWD_WORKSHEET)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=_TEMP_PWD_WORKSHEET, rows=200, cols=3)
        ws.update([["password", "expires", "created"]])
    return ws


def _load_temp_pwds():
    """Google Sheets에서 임시 비번 목록 로드 — Streamlit Reboot 후에도 유지"""
    try:
        ws = _get_temp_pwd_worksheet()
        records = ws.get_all_records()
        st.session_state["_sheets_last_error"] = None
        result = {}
        for row in records:
            pwd = str(row.get("password", "")).strip()
            if not pwd:
                continue
            exp = row.get("expires")
            cre = row.get("created")
            result[pwd] = {
                "expires": datetime.fromisoformat(exp) if exp else None,
                "created": datetime.fromisoformat(cre) if cre else datetime.now(),
            }
        return result
    except Exception as e:
        st.session_state["_sheets_last_error"] = f"[로드 실패] {type(e).__name__}: {e}"
        return {}


def _save_temp_pwds(pwd_dict):
    """임시 비번 목록을 Google Sheets에 저장 (전체 덮어쓰기)"""
    try:
        ws = _get_temp_pwd_worksheet()
        rows = [["password", "expires", "created"]]
        for pwd, info in pwd_dict.items():
            exp = info.get("expires")
            cre = info.get("created")
            rows.append([
                pwd,
                exp.isoformat() if isinstance(exp, datetime) else (exp if isinstance(exp, str) else ""),
                cre.isoformat() if isinstance(cre, datetime) else (cre if isinstance(cre, str) else str(datetime.now())),
            ])
        ws.clear()
        ws.update(rows)
        st.session_state["_sheets_last_error"] = None
        return True
    except Exception as e:
        st.session_state["_sheets_last_error"] = f"[저장 실패] {type(e).__name__}: {e}"
        return False


def _check_temp_pwd(p):
    """임시 비번 유효성 검사 — 시트에서 항상 최신 목록 확인"""
    fresh = _load_temp_pwds()
    st.session_state.temp_pwd_list = fresh
    info = fresh.get(p)
    if info is None:
        return False
    if info["expires"] is None:  # 만료일 없음 = 무기한
        return True
    return datetime.now() < info["expires"]


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "is_owner" not in st.session_state:
    st.session_state.is_owner = False
if "temp_pwd_list" not in st.session_state:
    st.session_state.temp_pwd_list = _load_temp_pwds()

if not st.session_state.authenticated:
    _, center, _ = st.columns([1, 1.8, 1])
    with center:
        st.markdown("<div style='height:60px;'></div>", unsafe_allow_html=True)
        st.markdown(
            """<div class='glass-card' style='text-align:center; padding:22px 36px; margin-top:12px;'>
                <div style='color:#ff9f1c; font-size:0.78rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:8px;'>GENERATIVE DESIGN SYSTEM</div>
                <h2 style='color:#f2f2f2; font-size:1.35rem; font-weight:600; margin:0 0 4px 0;'>🔧 설계안 자동 생성 로그인</h2>
                <div style='width:56px; height:2px; background:#10b981; margin:12px auto 0 auto;'></div>
            </div>""",
            unsafe_allow_html=True,
        )
        pw_col, btn_col = st.columns([4, 1])
        with pw_col:
            pwd = st.text_input(
                "비밀번호", type="password", label_visibility="collapsed", placeholder="비밀번호 입력"
            )
        with btn_col:
            login_btn = st.button("접속", type="primary", use_container_width=True)
        if login_btn:
            if pwd == OWNER_PWD:
                st.session_state.authenticated = True
                st.session_state.is_owner = True
                st.rerun()
            elif _check_temp_pwd(pwd):
                st.session_state.authenticated = True
                st.session_state.is_owner = False
                st.session_state.logged_temp_pwd = pwd
                st.rerun()
            else:
                st.error("비밀번호가 올바르지 않습니다.")
    st.stop()

st.markdown(
    f"""<div class='glass-card' style='text-align:center; padding:22px 36px; margin-top:12px;'>
        <div style='color:#ff9f1c; font-size:0.78rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:8px;'>GENERATIVE DESIGN SYSTEM</div>
        <div style='font-size:1.7rem; font-weight:700; color:#f2f2f2;'>🔧 설계안 자동 생성 &nbsp; V1.0</div>
        <div style='color:#8a8a8a; font-size:0.85rem; margin-top:6px;'>부품: {CONFIG.part_name} · NSGA-II 다목적 최적화 기반 파레토 후보 탐색</div>
        <div style='width:56px; height:2px; background:#10b981; margin:12px auto 0 auto;'></div>
    </div>""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def compute_candidates(pop_size: int, n_generations: int, seed: int):
    variables, objectives, var_cols, data, kpi_spec_ranges, parse_report = load_part_data(
        CONFIG, gcp_credentials=GCP_CREDENTIALS
    )
    models, cv_scores = train_surrogate_models(var_cols, data, CONFIG.kpi_columns)
    predictor = make_predictor(models, CONFIG)
    candidates = run_nsga2(variables, objectives, predictor, CONFIG,
                            pop_size=pop_size, n_generations=n_generations, seed=seed)
    valid = filter_within_spec(candidates, objectives, kpi_spec_ranges)
    return candidates, valid, objectives, kpi_spec_ranges, cv_scores, parse_report, len(data)


with st.sidebar:
    st.markdown("### ⚙️ 데이터 컨트롤")
    pop_size = st.slider("Population Size", 40, 300, 120, step=20)
    n_generations = st.slider("세대 수 (Generations)", 20, 200, 60, step=10)
    seed = st.number_input("Random Seed", value=1, step=1)
    run_btn = st.button("🚀 학습 초기화 및 재계산 실행", use_container_width=True)

    st.divider()
    st.caption(
        "⚠️ 실측 데이터가 6개 샘플뿐입니다. 이 대시보드의 예측치는 "
        "파이프라인 검증용 참고 자료이며, 실제 설계 확정에는 데이터 추가 확보가 필요합니다."
    )

    # ── 소유자 전용: 임시 비번 관리 패널 ──────────────────────────
    if st.session_state.get("is_owner", False):
        st.divider()
        st.markdown("### 🔐 임시 비밀번호 관리")

        sheets_err = st.session_state.get("_sheets_last_error")
        if sheets_err:
            st.error(f"⚠️ Google Sheets 오류\n\n{sheets_err}")
        else:
            st.caption("🟢 Google Sheets 연결 정상")

        if st.button("🔄 Sheets 연결 테스트", key="sb_test_sheets", use_container_width=True):
            try:
                test_ws = _get_temp_pwd_worksheet()
                test_ws.get_all_records()
                st.session_state["_sheets_last_error"] = None
                st.success("✅ Sheets 연결 성공")
            except Exception as e_test:
                st.session_state["_sheets_last_error"] = f"[테스트 실패] {type(e_test).__name__}: {e_test}"
                st.error(f"⚠️ {st.session_state['_sheets_last_error']}")

        new_tp = st.text_input("새 임시 비밀번호", key="sb_new_tp")
        exp_opt = st.selectbox("유효 기간", ["1일", "3일", "7일", "30일", "무제한"], key="sb_exp_sel")
        day_map = {"1일": 1, "3일": 3, "7일": 7, "30일": 30, "무제한": None}

        if st.button("➕ 추가", key="sb_add_tp", use_container_width=True):
            if new_tp and new_tp != OWNER_PWD:
                days = day_map.get(exp_opt)
                exp_dt = (datetime.now() + timedelta(days=days)) if days else None
                st.session_state.temp_pwd_list[new_tp] = {"expires": exp_dt, "created": datetime.now()}
                saved_ok = _save_temp_pwds(st.session_state.temp_pwd_list)
                if saved_ok:
                    st.success(f"추가됨: {new_tp}")
                else:
                    st.error("⚠️ Sheets 저장 실패 — 위 오류 메시지를 확인하세요 (재시작 시 사라질 수 있습니다)")
                st.rerun()
            elif new_tp == OWNER_PWD:
                st.error("소유자 비번은 사용할 수 없습니다.")
            else:
                st.warning("비밀번호를 입력하세요.")

        if st.session_state.temp_pwd_list:
            st.caption("등록된 임시 비밀번호")
            for tp_k, tp_v in list(st.session_state.temp_pwd_list.items()):
                exp_v = tp_v["expires"]
                if exp_v is None:
                    icon, txt = "🟢", "무제한"
                elif datetime.now() < exp_v:
                    hrs = int((exp_v - datetime.now()).total_seconds() // 3600)
                    icon, txt = "🟡", f"남음: {hrs}시간"
                else:
                    icon, txt = "🔴", "만료됨"
                rc1, rc2 = st.columns([3, 1])
                rc1.markdown(
                    f"<span style='font-size:0.8rem;'>{icon} <code>{tp_k}</code><br>"
                    f"<span style='color:#8a8a8a;font-size:0.72rem;'>{txt}</span></span>",
                    unsafe_allow_html=True,
                )
                if rc2.button("🗑️", key=f"sb_del_{tp_k}"):
                    del st.session_state.temp_pwd_list[tp_k]
                    _save_temp_pwds(st.session_state.temp_pwd_list)
                    st.rerun()
        else:
            st.caption("등록된 임시 비밀번호가 없습니다.")

if "computed" not in st.session_state or run_btn:
    with st.spinner("NSGA-II 다목적 최적화 실행 중..."):
        st.session_state["computed"] = compute_candidates(pop_size, n_generations, seed)

candidates, valid, objectives, kpi_spec_ranges, cv_scores, parse_report, n_samples = st.session_state["computed"]
kpi_names = [o.name for o in objectives]
n_indep_vars = len([p for p in parse_report if p[1] not in ("파생(수식 계산)", "고정값")])

stat_cols = st.columns(4)
stats = [
    ("실측 샘플 수", n_samples),
    ("생성된 후보안", len(candidates)),
    ("스펙 통과 후보", len(valid)),
    ("설계변수 / KPI", f"{n_indep_vars} / {len(kpi_names)}"),
]
for col, (label, value) in zip(stat_cols, stats):
    col.markdown(
        f"""<div class='stat-box'>
            <div class='stat-label'>{label}</div>
            <div class='stat-value'>{value}</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 파레토 프론트", "📋 후보 목록", "🧬 변수 구성"])

with tab1:
    st.markdown(
        "<div class='glass-card'><div class='glass-card-title'>파레토 프론트 시각화</div>",
        unsafe_allow_html=True,
    )

    show_df = valid if len(valid) > 0 else candidates
    if len(valid) == 0:
        st.warning("스펙을 만족하는 후보가 없어 전체 후보를 표시합니다.")

    c1, c2, c3 = st.columns(3)
    x_kpi = c1.selectbox("X축 KPI", kpi_names, index=kpi_names.index("axial_before_after_min_stiffness_%"))
    y_kpi = c2.selectbox("Y축 KPI", kpi_names, index=kpi_names.index("breakaway_torque_Nm"))
    color_kpi = c3.selectbox("색상(3번째 KPI)", kpi_names, index=kpi_names.index("radial_before_after_min_stiffness_%"))

    fig = px.scatter(
        show_df, x=x_kpi, y=y_kpi, color=color_kpi,
        color_continuous_scale=["#f87171", "#ff9f1c", "#10b981"],
        hover_data=kpi_names,
    )
    fig.update_traces(marker=dict(size=11, line=dict(width=1, color="#0f0f0f")))
    fig.update_layout(
        height=520,
        paper_bgcolor="#1a1a1a",
        plot_bgcolor="#131313",
        font=dict(color="#ececec", family="Inter"),
        xaxis=dict(gridcolor="#2e2e2e", zerolinecolor="#2e2e2e"),
        yaxis=dict(gridcolor="#2e2e2e", zerolinecolor="#2e2e2e"),
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "각 점은 하나의 설계 후보(치수 조합)입니다. 두 KPI 사이 trade-off 곡선(파레토 프론트) 위에 "
        "위치할수록 '이 두 지표 사이에서는 더 개선할 여지가 없는' 효율적인 후보입니다."
    )
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown(
        "<div class='glass-card'><div class='glass-card-title'>스펙 통과 후보 (Recommended Candidates)</div>",
        unsafe_allow_html=True,
    )

    if len(valid) > 0:
        sort_kpi = st.selectbox("정렬 기준 KPI", kpi_names,
                                 index=kpi_names.index("axial_before_after_min_stiffness_%"))
        sort_dir = next(o.direction for o in objectives if o.name == sort_kpi)
        sorted_valid = valid.sort_values(sort_kpi, ascending=(sort_dir == "min")).reset_index(drop=True)

        top_n = min(10, len(sorted_valid))
        rows_html = ""
        for i in range(top_n):
            row = sorted_valid.iloc[i]
            cells = ""
            for kpi in kpi_names:
                rng = kpi_spec_ranges.get(kpi)
                val = row[kpi]
                if rng is not None:
                    lo, hi = rng
                    ok = (lo is None or val >= lo) and (hi is None or val <= hi)
                    color = "#10b981" if ok else "#f87171"
                else:
                    color = "#9c9c9c"
                cells += (
                    f"<td style='padding:6px 10px;text-align:center;font-family:JetBrains Mono,monospace;"
                    f"color:{color};font-weight:700;'>{val:.3f}</td>"
                )
            badge = (
                "<span style='background:#0f2410;color:#10b981;font-size:0.62rem;padding:1px 6px;"
                "border-radius:3px;margin-left:6px;'>✅ PASS</span>"
            )
            rows_html += (
                f"<tr style='border-bottom:1px solid #262626;'>"
                f"<td style='padding:6px 10px;color:#ececec;font-weight:700;'>#{i+1}{badge}</td>"
                f"{cells}</tr>"
            )

        header_cells = "".join(
            f"<th style='padding:5px 10px;font-size:0.68rem;color:#8a8a8a;text-align:center;"
            f"text-transform:uppercase;'>{k}</th>"
            for k in kpi_names
        )
        st.markdown(
            f"<div style='background:#131313;border:1px solid #2e2e2e;border-radius:8px;padding:12px 14px;"
            f"overflow-x:auto;'>"
            f"<table style='width:100%;border-collapse:collapse;'>"
            f"<thead><tr style='border-bottom:1px solid #3a3a3a;'>"
            f"<th style='padding:5px 10px;font-size:0.68rem;color:#8a8a8a;text-align:left;'>후보</th>"
            f"{header_cells}</tr></thead><tbody>{rows_html}</tbody></table>"
            f"<div style='color:#8a8a8a;font-size:0.72rem;margin-top:8px;'>"
            f"★ 초록 = 스펙 통과 값 · 빨강 = 스펙 이탈 값 (참고용, 현재는 스펙 통과 후보만 표시됨)</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
        csv = sorted_valid.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ 전체 후보 CSV 다운로드 (전체 치수 포함)", csv,
                            file_name="ball_joint_generative_candidates.csv", mime="text/csv")

        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.85rem;font-weight:700;color:#ff9f1c;'>▣ 후보 상세 치수 보기</p>", unsafe_allow_html=True)
        idx = st.number_input("후보 번호 (표의 행 인덱스)", min_value=0,
                               max_value=len(sorted_valid) - 1, value=0, step=1)
        detail = sorted_valid.iloc[int(idx)]
        dim_cols = [c for c in sorted_valid.columns if c not in kpi_names]

        detail_rows_html = ""
        for c in dim_cols:
            kor = CONFIG.kor_labels.get(c, "")
            detail_rows_html += (
                f"<tr style='border-bottom:1px solid #262626;'>"
                f"<td style='padding:5px 10px;color:#ececec;font-weight:700;'>{c}</td>"
                f"<td style='padding:5px 10px;color:#9c9c9c;'>{kor}</td>"
                f"<td style='padding:5px 10px;text-align:center;font-family:JetBrains Mono,monospace;"
                f"color:#ff9f1c;font-weight:700;'>{detail[c]:.4f}</td></tr>"
            )
        st.markdown(
            f"<div style='background:#131313;border:1px solid #2e2e2e;border-radius:8px;padding:12px 14px;"
            f"max-height:420px;overflow-y:auto;'>"
            f"<table style='width:100%;border-collapse:collapse;'>"
            f"<thead><tr style='border-bottom:1px solid #3a3a3a;'>"
            f"<th style='padding:5px 10px;font-size:0.68rem;color:#8a8a8a;text-align:left;'>변수</th>"
            f"<th style='padding:5px 10px;font-size:0.68rem;color:#8a8a8a;text-align:left;'>한글명</th>"
            f"<th style='padding:5px 10px;font-size:0.68rem;color:#8a8a8a;text-align:center;'>값(mm)</th>"
            f"</tr></thead><tbody>{detail_rows_html}</tbody></table></div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("스펙 통과 후보가 없습니다.")
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown(
        "<div class='glass-card'><div class='glass-card-title'>설계변수 구성 리포트</div>",
        unsafe_allow_html=True,
    )

    reason_color = {
        "스펙 파싱": "#10b981",
        "형상공차(0~X)": "#10b981",
        "파생(수식 계산)": "#ff9f1c",
        "고정값": "#9c9c9c",
    }
    rows_html = ""
    for p in parse_report:
        name = p[0]
        reason = p[1]
        rng = p[2] if len(p) > 2 else ""
        kor = CONFIG.kor_labels.get(name, "")
        color = next((v for k, v in reason_color.items() if k in reason), "#f87171")
        rows_html += (
            f"<tr style='border-bottom:1px solid #262626;'>"
            f"<td style='padding:5px 10px;color:#ececec;font-weight:700;'>{name}</td>"
            f"<td style='padding:5px 10px;color:#9c9c9c;'>{kor}</td>"
            f"<td style='padding:5px 10px;'><span style='color:{color};font-weight:700;'>{reason}</span></td>"
            f"<td style='padding:5px 10px;color:#9c9c9c;font-family:JetBrains Mono,monospace;'>{rng}</td>"
            f"</tr>"
        )
    st.markdown(
        f"<div style='background:#131313;border:1px solid #2e2e2e;border-radius:8px;padding:12px 14px;"
        f"max-height:460px;overflow-y:auto;'>"
        f"<table style='width:100%;border-collapse:collapse;'>"
        f"<thead><tr style='border-bottom:1px solid #3a3a3a;'>"
        f"<th style='padding:5px 10px;font-size:0.68rem;color:#8a8a8a;text-align:left;'>변수명</th>"
        f"<th style='padding:5px 10px;font-size:0.68rem;color:#8a8a8a;text-align:left;'>한글명</th>"
        f"<th style='padding:5px 10px;font-size:0.68rem;color:#8a8a8a;text-align:left;'>구분</th>"
        f"<th style='padding:5px 10px;font-size:0.68rem;color:#8a8a8a;text-align:left;'>범위/비고</th>"
        f"</tr></thead><tbody>{rows_html}</tbody></table>"
        f"<div style='color:#8a8a8a;font-size:0.72rem;margin-top:8px;'>"
        f"● 스펙 파싱/형상공차(초록) · 파생(오렌지) · 고정값(회색) · 데이터기반(빨강)</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='glass-card'><div class='glass-card-title'>대리모델(Ridge) 참고 정확도 (LOO-MAE)</div>",
        unsafe_allow_html=True,
    )
    cv_rows_html = ""
    for k, v in cv_scores.items():
        kor = CONFIG.kor_labels.get(k, "")
        cv_rows_html += (
            f"<tr style='border-bottom:1px solid #262626;'>"
            f"<td style='padding:5px 10px;color:#ececec;font-weight:700;'>{k}</td>"
            f"<td style='padding:5px 10px;color:#9c9c9c;'>{kor}</td>"
            f"<td style='padding:5px 10px;text-align:center;font-family:JetBrains Mono,monospace;"
            f"color:#ff9f1c;font-weight:700;'>{v:.4f}</td></tr>"
        )
    st.markdown(
        f"<div style='background:#131313;border:1px solid #2e2e2e;border-radius:8px;padding:12px 14px;'>"
        f"<table style='width:100%;border-collapse:collapse;'>"
        f"<thead><tr style='border-bottom:1px solid #3a3a3a;'>"
        f"<th style='padding:5px 10px;font-size:0.68rem;color:#8a8a8a;text-align:left;'>KPI</th>"
        f"<th style='padding:5px 10px;font-size:0.68rem;color:#8a8a8a;text-align:left;'>한글명</th>"
        f"<th style='padding:5px 10px;font-size:0.68rem;color:#8a8a8a;text-align:center;'>LOO-MAE</th>"
        f"</tr></thead><tbody>{cv_rows_html}</tbody></table></div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
