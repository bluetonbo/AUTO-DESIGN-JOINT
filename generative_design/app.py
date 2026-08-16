# -*- coding: utf-8 -*-
"""
Generative Design 대시보드 (Streamlit)
UI/레이아웃: JOINT-AI-APP-6.py와 동일한 다크 콘솔 테마 적용
"""

import sys
import os
import io
import sqlite3
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

    html {
        font-size: 90% !important;
    }

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

# ──────────────────────────────────────────────────────────────
# 다국어 사전 (JOINT-AI-APP-6.py의 LANG_DICT 방식과 동일)
# ──────────────────────────────────────────────────────────────
LANG_DICT = {
    "KO": {
        "login_badge": "GENERATIVE DESIGN SYSTEM",
        "login_title": "설계안 자동 생성 로그인",
        "pwd_label": "비밀번호 입력",
        "auth_btn": "접속",
        "invalid_pwd": "비밀번호가 올바르지 않습니다.",
        "owner_only": "소유자 비번은 사용할 수 없습니다.",
        "enter_pwd_warn": "비밀번호를 입력하세요.",

        "header_title": "설계안 자동 생성 &nbsp; V1.0",
        "header_subtitle": "부품: {part} · NSGA-II 다목적 최적화 기반 파레토 후보 탐색",

        "input_data_title": "입력 데이터",
        "upload_label": "설계/실측 데이터 파일 업로드 (XLSX, DB)",
        "upload_help": "VOLVO_SPA12_CABJ_TRAIN_DATA 형식과 동일한 레이아웃의 엑셀 또는 SQLite DB 파일",
        "data_control_title": "데이터 컨트롤",
        "run_btn": "학습 초기화 및 재계산 실행",
        "data_caution": "[주의] 실측 데이터가 6개 샘플뿐입니다. 이 대시보드의 예측치는 "
                         "파이프라인 검증용 참고 자료이며, 실제 설계 확정에는 데이터 추가 확보가 필요합니다.",

        "temp_pwd_title": "임시 비밀번호 관리",
        "sheets_error_prefix": "[오류] Google Sheets 연결에 문제가 있습니다.",
        "sheets_ok": "Google Sheets 연결 정상",
        "sheets_test_btn": "Sheets 연결 테스트",
        "sheets_test_ok": "Sheets 연결 성공",
        "sheets_test_fail_prefix": "[오류]",
        "new_temp_pwd_label": "새 임시 비밀번호",
        "expiry_label": "유효 기간",
        "expiry_opts": ["1일", "3일", "7일", "30일", "무제한"],
        "add_btn": "추가",
        "added_msg": "추가됨: ",
        "save_fail_msg": "[오류] Sheets 저장 실패 — 위 오류 메시지를 확인하세요 (재시작 시 사라질 수 있습니다)",
        "registered_pwds": "등록된 임시 비밀번호",
        "no_registered": "등록된 임시 비밀번호가 없습니다.",
        "delete_btn": "삭제",
        "unlimited": "[무제한]", "unlimited_txt": "무제한",
        "active": "[사용중]", "remaining_txt": "남음: {h}시간",
        "expired": "[만료]", "expired_txt": "만료됨",

        "engine_inactive": "학습 비활성화: 왼쪽 사이드바에서 입력 데이터 파일(XLSX 또는 DB)을 업로드해주세요.\n\n"
                            "VOLVO_SPA12_CABJ_TRAIN_DATA 원본과 동일한 레이아웃(2행: 변수명, 9행: 스펙, 3~8행: 실측 데이터)이어야 합니다.",
        "ready_to_run": "파일 업로드가 완료되었습니다. 사이드바의 \"학습 초기화 및 재계산 실행\" 버튼을 눌러 계산을 시작하세요.",

        "stat_samples": "실측 샘플 수",
        "stat_candidates": "생성된 후보안",
        "stat_valid": "스펙 통과 후보",
        "stat_vars_kpi": "설계변수 / KPI",

        "tab1": "파레토 프론트", "tab2": "후보 목록", "tab3": "변수 구성",

        "pareto_exp_title": "파레토 프론트 시각화",
        "no_valid_warning": "스펙을 만족하는 후보가 없어 전체 후보를 표시합니다.",
        "x_kpi": "X축 KPI", "y_kpi": "Y축 KPI", "color_kpi": "색상(3번째 KPI)",
        "pareto_caption": "각 점은 하나의 설계 후보(치수 조합)입니다. 두 KPI 사이 trade-off 곡선(파레토 프론트) 위에 "
                           "위치할수록 '이 두 지표 사이에서는 더 개선할 여지가 없는' 효율적인 후보입니다.",

        "candidates_exp_title": "스펙 통과 후보 (Recommended Candidates)",
        "sort_kpi": "정렬 기준 KPI",
        "candidate_col": "후보",
        "legend_note": "※ 초록 = 스펙 통과 값 · 빨강 = 스펙 이탈 값 (참고용, 현재는 스펙 통과 후보만 표시됨)",
        "dl_format": "내보내기 파일 포맷 선택",
        "dl_btn": "전체 후보 다운로드",
        "detail_exp_title": "후보 상세 치수 보기",
        "candidate_idx_label": "후보 번호 (표의 행 인덱스)",
        "col_var": "변수", "col_desc": "한글명", "col_val_mm": "값(mm)",
        "no_valid_info": "스펙 통과 후보가 없습니다.",

        "var_report_exp_title": "설계변수 구성 리포트",
        "col_var_name": "변수명", "col_reason": "구분", "col_range": "범위/비고",
        "var_report_legend": "● 스펙 파싱/형상공차(초록) · 파생(오렌지) · 고정값(회색) · 데이터기반(빨강)",
        "cv_exp_title": "대리모델(Ridge) 참고 정확도 (LOO-MAE)",
        "col_kpi": "KPI",

        "prog_train_prep": "대리모델 학습 준비 중... (0%)",
        "prog_train_step": "({i}/{n}) {kpi} 대리모델 학습 중... ({pct}%)",
        "prog_train_done": "대리모델 학습 완료 (100%)",
        "status_training": "학습 중 -> ",
        "status_done": "완료: {kpi} 학습 완료 (LOO-MAE={mae:.4f})",
        "prog_opt_prep": "NSGA-II 최적화 준비 중... (0%)",
        "prog_opt_step": "세대 진행 중 ({g}/{n}) ({pct}%)",
        "prog_opt_done": "NSGA-II 최적화 완료 (100%)",
    },
    "EN": {
        "login_badge": "GENERATIVE DESIGN SYSTEM",
        "login_title": "Generative Design Login",
        "pwd_label": "Enter Password",
        "auth_btn": "Login",
        "invalid_pwd": "Incorrect password.",
        "owner_only": "Owner password cannot be used here.",
        "enter_pwd_warn": "Please enter a password.",

        "header_title": "Generative Design &nbsp; V1.0",
        "header_subtitle": "Part: {part} · Pareto candidate search via NSGA-II multi-objective optimization",

        "input_data_title": "Input Data",
        "upload_label": "Upload design/measurement data file (XLSX, DB)",
        "upload_help": "Excel or SQLite DB file with the same layout as VOLVO_SPA12_CABJ_TRAIN_DATA",
        "data_control_title": "Data Control",
        "run_btn": "Run Initialization & Recalculation",
        "data_caution": "[Note] Only 6 measured samples are available. Predictions on this dashboard are "
                         "for pipeline validation reference only; more data is needed before finalizing designs.",

        "temp_pwd_title": "Temporary Password Management",
        "sheets_error_prefix": "[Error] Problem connecting to Google Sheets.",
        "sheets_ok": "Google Sheets connection OK",
        "sheets_test_btn": "Test Sheets Connection",
        "sheets_test_ok": "Sheets connection successful",
        "sheets_test_fail_prefix": "[Error]",
        "new_temp_pwd_label": "New temporary password",
        "expiry_label": "Validity Period",
        "expiry_opts": ["1 day", "3 days", "7 days", "30 days", "Unlimited"],
        "add_btn": "Add",
        "added_msg": "Added: ",
        "save_fail_msg": "[Error] Failed to save to Sheets — check the error above (may reset on reboot)",
        "registered_pwds": "Registered temporary passwords",
        "no_registered": "No temporary passwords registered.",
        "delete_btn": "Delete",
        "unlimited": "[UNLIMITED]", "unlimited_txt": "Unlimited",
        "active": "[ACTIVE]", "remaining_txt": "{h}h remaining",
        "expired": "[EXPIRED]", "expired_txt": "Expired",

        "engine_inactive": "ENGINE INACTIVE: Please upload an input data file (XLSX or DB) via the left sidebar.\n\n"
                            "It must use the same layout as VOLVO_SPA12_CABJ_TRAIN_DATA "
                            "(row 2: variable names, row 9: spec, rows 3-8: measured data).",
        "ready_to_run": "File uploaded successfully. Click \"Run Initialization & Recalculation\" in the sidebar to start.",

        "stat_samples": "Measured Samples",
        "stat_candidates": "Generated Candidates",
        "stat_valid": "Spec-Passing Candidates",
        "stat_vars_kpi": "Design Vars / KPIs",

        "tab1": "Pareto Front", "tab2": "Candidates", "tab3": "Variables",

        "pareto_exp_title": "Pareto Front Visualization",
        "no_valid_warning": "No candidates satisfy the spec — showing all candidates instead.",
        "x_kpi": "X-axis KPI", "y_kpi": "Y-axis KPI", "color_kpi": "Color (3rd KPI)",
        "pareto_caption": "Each point is one design candidate (dimension combination). Points on the trade-off "
                           "curve (Pareto front) between two KPIs cannot be improved further on both at once.",

        "candidates_exp_title": "Spec-Passing Candidates (Recommended Candidates)",
        "sort_kpi": "Sort by KPI",
        "candidate_col": "Candidate",
        "legend_note": "* Green = within spec · Red = out of spec (reference only; only spec-passing candidates are shown)",
        "dl_format": "Select export file format",
        "dl_btn": "Download all candidates",
        "detail_exp_title": "Candidate Detail Dimensions",
        "candidate_idx_label": "Candidate number (table row index)",
        "col_var": "Variable", "col_desc": "Description", "col_val_mm": "Value (mm)",
        "no_valid_info": "No spec-passing candidates.",

        "var_report_exp_title": "Design Variable Report",
        "col_var_name": "Variable", "col_reason": "Category", "col_range": "Range / Note",
        "var_report_legend": "* Green = spec-parsed/shape tolerance · Orange = derived · Gray = fixed · Red = data-based",
        "cv_exp_title": "Surrogate Model (Ridge) Reference Accuracy (LOO-MAE)",
        "col_kpi": "KPI",

        "prog_train_prep": "Preparing surrogate model training... (0%)",
        "prog_train_step": "({i}/{n}) Training surrogate model for {kpi}... ({pct}%)",
        "prog_train_done": "Surrogate model training complete (100%)",
        "status_training": "Training -> ",
        "status_done": "Done: {kpi} training complete (LOO-MAE={mae:.4f})",
        "prog_opt_prep": "Preparing NSGA-II optimization... (0%)",
        "prog_opt_step": "Generation in progress ({g}/{n}) ({pct}%)",
        "prog_opt_done": "NSGA-II optimization complete (100%)",
    },
}

if "lang" not in st.session_state:
    st.session_state["lang"] = "KO"

try:
    GCP_CREDENTIALS = dict(st.secrets["gcp_service_account"])
    SHEET_ID = st.secrets["sheet_id"]  # 로그인/임시비번 저장용 (설계 데이터와는 무관)
except KeyError:
    st.error(
        "Google Sheets 연동 정보(secrets)가 없습니다. "
        "Streamlit Cloud의 App settings → Secrets에 "
        "`gcp_service_account`, `sheet_id`를 등록해주세요. "
        "(로그인/임시비번 저장에 사용됩니다)"
    )
    st.stop()

CONFIG = BASE_CONFIG  # 설계 데이터는 파일 업로드로 받으므로 sheet 관련 설정 불필요

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

        _, lang_col = st.columns([5, 1.1])
        with lang_col:
            lang_choice_login = st.selectbox(
                "Language", ["KO", "EN"],
                index=["KO", "EN"].index(st.session_state["lang"]),
                label_visibility="collapsed",
                key="login_lang_select",
            )
            if lang_choice_login != st.session_state["lang"]:
                st.session_state["lang"] = lang_choice_login
                st.rerun()

        LT = LANG_DICT[st.session_state["lang"]]

        st.markdown(
            f"""<div class='glass-card' style='text-align:center; padding:22px 36px; margin-top:12px;'>
                <div style='color:#ff9f1c; font-size:0.78rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:8px;'>{LT['login_badge']}</div>
                <h2 style='color:#f2f2f2; font-size:1.35rem; font-weight:600; margin:0 0 4px 0;'>{LT['login_title']}</h2>
                <div style='width:56px; height:2px; background:#10b981; margin:12px auto 0 auto;'></div>
            </div>""",
            unsafe_allow_html=True,
        )
        pw_col, btn_col = st.columns([4, 1])
        with pw_col:
            pwd = st.text_input(
                LT["pwd_label"], type="password", label_visibility="collapsed", placeholder=LT["pwd_label"]
            )
        with btn_col:
            login_btn = st.button(LT["auth_btn"], type="primary", use_container_width=True)
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
                st.error(LT["invalid_pwd"])
    st.stop()

L = LANG_DICT[st.session_state["lang"]]

col_title, col_lang = st.columns([8, 1])
with col_lang:
    lang_choice = st.selectbox(
        "Language", ["KO", "EN"],
        index=["KO", "EN"].index(st.session_state["lang"]),
        label_visibility="collapsed",
        key="main_lang_select",
    )
    if lang_choice != st.session_state["lang"]:
        st.session_state["lang"] = lang_choice
        st.rerun()

st.markdown(
    f"""<div class='glass-card' style='text-align:center; padding:22px 36px; margin-top:12px;'>
        <div style='color:#ff9f1c; font-size:0.78rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:8px;'>{L['login_badge']}</div>
        <div style='font-size:1.7rem; font-weight:700; color:#f2f2f2;'>{L['header_title']}</div>
        <div style='color:#8a8a8a; font-size:0.85rem; margin-top:6px;'>{L['header_subtitle'].format(part=CONFIG.part_name)}</div>
        <div style='width:56px; height:2px; background:#10b981; margin:12px auto 0 auto;'></div>
    </div>""",
    unsafe_allow_html=True,
)


def _load_raw_from_bytes(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """xlsx 또는 db(SQLite) 업로드 파일을 원본 시트와 동일한 위치기반(raw) 표로 변환.

    - xlsx: pd.read_excel(header=None)과 동일하게 위치 그대로 읽음
    - db: 첫 번째 테이블을 그대로 읽어옴. 컬럼 "이름"은 무관하며(엔진은 위치(iloc)로만
      접근), 테이블의 행/열 순서가 원본 엑셀 시트의 셀 배치와 동일해야 함
      (예: A열=라벨, B열~=값, 2행=변수명, 9행=스펙, 3~8행=데이터)
    """
    if file_name.lower().endswith(".db"):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            conn = sqlite3.connect(tmp_path)
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table'", conn
            )
            if len(tables) == 0:
                raise ValueError("DB 파일에 테이블이 없습니다.")
            table_name = tables.iloc[0]["name"]
            df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
            conn.close()
        finally:
            os.remove(tmp_path)
        # 위치 기반 접근을 위해 컬럼명을 정수 인덱스로 재설정 (이름은 무관, 순서만 중요)
        df.columns = range(df.shape[1])
        return df
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=0, header=None)


def run_pipeline(file_bytes: bytes, file_name: str, pop_size: int, n_generations: int, seed: int):
    """대리모델 학습 + NSGA-II 최적화를 진행상황을 표시하며 실행 (JOINT-AI-APP-6.py와 동일한 진행바 방식)."""
    raw_df = _load_raw_from_bytes(file_bytes, file_name)
    variables, objectives, var_cols, data, kpi_spec_ranges, parse_report = load_part_data(
        CONFIG, raw=raw_df
    )

    # 1) 대리모델 학습 진행상황
    train_prog = st.progress(0, text=L["prog_train_prep"])
    algo_status = st.empty()
    total_kpi = len(CONFIG.kpi_columns)

    def kpi_progress(idx, total, kpi_name, status, extra):
        pct = idx / total
        kor = CONFIG.kor_labels.get(kpi_name, "")
        if status == "start":
            train_prog.progress(
                pct, text=L["prog_train_step"].format(i=idx + 1, n=total, kpi=kpi_name, pct=int(pct * 100))
            )
            algo_status.markdown(
                f"<div style='background:#131313;border-left:3px solid #ff9f1c;border-radius:5px;padding:5px 10px;"
                f"font-size:0.75rem;color:#d4d4d4;'>{L['status_training']}<b style='color:#ff9f1c;'>{kpi_name}</b>"
                f" ({kor})</div>",
                unsafe_allow_html=True,
            )
        else:
            algo_status.markdown(
                f"<div style='background:#131313;border-left:3px solid #10b981;border-radius:5px;padding:5px 10px;"
                f"font-size:0.75rem;color:#a3e635;'>{L['status_done'].format(kpi=kpi_name, mae=extra)}</div>",
                unsafe_allow_html=True,
            )

    models, cv_scores = train_surrogate_models(
        var_cols, data, CONFIG.kpi_columns, progress_callback=kpi_progress
    )
    train_prog.progress(1.0, text=L["prog_train_done"])
    algo_status.empty()

    predictor = make_predictor(models, CONFIG)

    # 2) NSGA-II 최적화 진행상황
    opt_prog = st.progress(0, text=L["prog_opt_prep"])

    def gen_progress(gen, total):
        pct = gen / total
        opt_prog.progress(pct, text=L["prog_opt_step"].format(g=gen, n=total, pct=int(pct * 100)))

    candidates = run_nsga2(
        variables, objectives, predictor, CONFIG,
        pop_size=pop_size, n_generations=n_generations, seed=seed,
        progress_callback=gen_progress,
    )
    opt_prog.progress(1.0, text=L["prog_opt_done"])

    valid = filter_within_spec(candidates, objectives, kpi_spec_ranges)
    return candidates, valid, objectives, kpi_spec_ranges, cv_scores, parse_report, len(data)


with st.sidebar:
    with st.expander(L["input_data_title"], expanded=True):
        uploaded_file = st.file_uploader(
            L["upload_label"],
            type=["xlsx", "db"],
            help=L["upload_help"],
        )

    with st.expander(L["data_control_title"], expanded=True):
        pop_size = st.slider("Population Size", 40, 300, 120, step=20)
        n_generations = st.slider("Generations", 20, 200, 60, step=10)
        seed = st.number_input("Random Seed", value=1, step=1)
        run_btn = st.button(L["run_btn"], use_container_width=True)
        st.caption(L["data_caution"])

    # ── 소유자 전용: 임시 비번 관리 패널 ──────────────────────────
    if st.session_state.get("is_owner", False):
        with st.expander(L["temp_pwd_title"], expanded=False):
            sheets_err = st.session_state.get("_sheets_last_error")
            if sheets_err:
                st.error(f"{L['sheets_error_prefix']}\n\n{sheets_err}")
            else:
                st.caption(L["sheets_ok"])

            if st.button(L["sheets_test_btn"], key="sb_test_sheets", use_container_width=True):
                try:
                    test_ws = _get_temp_pwd_worksheet()
                    test_ws.get_all_records()
                    st.session_state["_sheets_last_error"] = None
                    st.success(L["sheets_test_ok"])
                except Exception as e_test:
                    st.session_state["_sheets_last_error"] = f"[테스트 실패] {type(e_test).__name__}: {e_test}"
                    st.error(f"{L['sheets_test_fail_prefix']} {st.session_state['_sheets_last_error']}")

            new_tp = st.text_input(L["new_temp_pwd_label"], key="sb_new_tp")
            exp_opt = st.selectbox(L["expiry_label"], L["expiry_opts"], key="sb_exp_sel")
            day_map = dict(zip(L["expiry_opts"], [1, 3, 7, 30, None]))

            if st.button(L["add_btn"], key="sb_add_tp", use_container_width=True):
                if new_tp and new_tp != OWNER_PWD:
                    days = day_map.get(exp_opt)
                    exp_dt = (datetime.now() + timedelta(days=days)) if days else None
                    st.session_state.temp_pwd_list[new_tp] = {"expires": exp_dt, "created": datetime.now()}
                    saved_ok = _save_temp_pwds(st.session_state.temp_pwd_list)
                    if saved_ok:
                        st.success(f"{L['added_msg']}{new_tp}")
                    else:
                        st.error(L["save_fail_msg"])
                    st.rerun()
                elif new_tp == OWNER_PWD:
                    st.error(L["owner_only"])
                else:
                    st.warning(L["enter_pwd_warn"])

            if st.session_state.temp_pwd_list:
                st.caption(L["registered_pwds"])
                for tp_k, tp_v in list(st.session_state.temp_pwd_list.items()):
                    exp_v = tp_v["expires"]
                    if exp_v is None:
                        icon, txt = L["unlimited"], L["unlimited_txt"]
                    elif datetime.now() < exp_v:
                        hrs = int((exp_v - datetime.now()).total_seconds() // 3600)
                        icon, txt = L["active"], L["remaining_txt"].format(h=hrs)
                    else:
                        icon, txt = L["expired"], L["expired_txt"]
                    rc1, rc2 = st.columns([3, 1])
                    rc1.markdown(
                        f"<span style='font-size:0.8rem;'>{icon} <code>{tp_k}</code><br>"
                        f"<span style='color:#8a8a8a;font-size:0.72rem;'>{txt}</span></span>",
                        unsafe_allow_html=True,
                    )
                    if rc2.button(L["delete_btn"], key=f"sb_del_{tp_k}"):
                        del st.session_state.temp_pwd_list[tp_k]
                        _save_temp_pwds(st.session_state.temp_pwd_list)
                        st.rerun()
            else:
                st.caption(L["no_registered"])

if uploaded_file is None:
    st.info(L["engine_inactive"])
    st.stop()

if run_btn:
    st.session_state["computed"] = run_pipeline(
        uploaded_file.getvalue(), uploaded_file.name, pop_size, n_generations, seed
    )

if "computed" not in st.session_state:
    st.info(L["ready_to_run"])
    st.stop()

candidates, valid, objectives, kpi_spec_ranges, cv_scores, parse_report, n_samples = st.session_state["computed"]
kpi_names = [o.name for o in objectives]
n_indep_vars = len([p for p in parse_report if p[1] not in ("파생(수식 계산)", "고정값")])

stat_cols = st.columns(4)
stats = [
    (L["stat_samples"], n_samples),
    (L["stat_candidates"], len(candidates)),
    (L["stat_valid"], len(valid)),
    (L["stat_vars_kpi"], f"{n_indep_vars} / {len(kpi_names)}"),
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

tab1, tab2, tab3 = st.tabs([L["tab1"], L["tab2"], L["tab3"]])

with tab1:
    with st.expander(L["pareto_exp_title"], expanded=True):
        show_df = valid if len(valid) > 0 else candidates
        if len(valid) == 0:
            st.warning(L["no_valid_warning"])

        c1, c2, c3 = st.columns(3)
        x_kpi = c1.selectbox(L["x_kpi"], kpi_names, index=kpi_names.index("axial_before_after_min_stiffness_%"))
        y_kpi = c2.selectbox(L["y_kpi"], kpi_names, index=kpi_names.index("breakaway_torque_Nm"))
        color_kpi = c3.selectbox(L["color_kpi"], kpi_names, index=kpi_names.index("radial_before_after_min_stiffness_%"))

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

        st.caption(L["pareto_caption"])

with tab2:
    with st.expander(L["candidates_exp_title"], expanded=True):
        if len(valid) > 0:
            sort_kpi = st.selectbox(L["sort_kpi"], kpi_names,
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
                    "border-radius:3px;margin-left:6px;'>PASS</span>"
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
                f"<th style='padding:5px 10px;font-size:0.68rem;color:#8a8a8a;text-align:left;'>{L['candidate_col']}</th>"
                f"{header_cells}</tr></thead><tbody>{rows_html}</tbody></table>"
                f"<div style='color:#8a8a8a;font-size:0.72rem;margin-top:8px;'>"
                f"{L['legend_note']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
            dl_col1, dl_col2 = st.columns([1, 1])
            with dl_col1:
                dl_fmt = st.selectbox(L["dl_format"], ["Excel/CSV (.csv)", "Database (.db)"],
                                       key="candidates_dl_fmt", label_visibility="collapsed")
            with dl_col2:
                if "CSV" in dl_fmt:
                    csv = sorted_valid.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(L["dl_btn"], csv,
                                        file_name="ball_joint_generative_candidates.csv", mime="text/csv")
                else:
                    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_out:
                        tmp_out_path = tmp_out.name
                    conn_out = sqlite3.connect(tmp_out_path)
                    sorted_valid.to_sql("candidates", conn_out, index=False, if_exists="replace")
                    conn_out.close()
                    with open(tmp_out_path, "rb") as f:
                        db_bytes = f.read()
                    os.remove(tmp_out_path)
                    st.download_button(L["dl_btn"], db_bytes,
                                        file_name="ball_joint_generative_candidates.db", mime="application/x-sqlite3")

            st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
            with st.expander(L["detail_exp_title"], expanded=False):
                idx = st.number_input(L["candidate_idx_label"], min_value=0,
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
                    f"<th style='padding:5px 10px;font-size:0.68rem;color:#8a8a8a;text-align:left;'>{L['col_var']}</th>"
                    f"<th style='padding:5px 10px;font-size:0.68rem;color:#8a8a8a;text-align:left;'>{L['col_desc']}</th>"
                    f"<th style='padding:5px 10px;font-size:0.68rem;color:#8a8a8a;text-align:center;'>{L['col_val_mm']}</th>"
                    f"</tr></thead><tbody>{detail_rows_html}</tbody></table></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info(L["no_valid_info"])

with tab3:
    with st.expander(L["var_report_exp_title"], expanded=True):
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
            f"<th style='padding:5px 10px;font-size:0.68rem;color:#8a8a8a;text-align:left;'>{L['col_var_name']}</th>"
            f"<th style='padding:5px 10px;font-size:0.68rem;color:#8a8a8a;text-align:left;'>{L['col_desc']}</th>"
            f"<th style='padding:5px 10px;font-size:0.68rem;color:#8a8a8a;text-align:left;'>{L['col_reason']}</th>"
            f"<th style='padding:5px 10px;font-size:0.68rem;color:#8a8a8a;text-align:left;'>{L['col_range']}</th>"
            f"</tr></thead><tbody>{rows_html}</tbody></table>"
            f"<div style='color:#8a8a8a;font-size:0.72rem;margin-top:8px;'>"
            f"{L['var_report_legend']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with st.expander(L["cv_exp_title"], expanded=False):
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
            f"<th style='padding:5px 10px;font-size:0.68rem;color:#8a8a8a;text-align:left;'>{L['col_kpi']}</th>"
            f"<th style='padding:5px 10px;font-size:0.68rem;color:#8a8a8a;text-align:left;'>{L['col_desc']}</th>"
            f"<th style='padding:5px 10px;font-size:0.68rem;color:#8a8a8a;text-align:center;'>LOO-MAE</th>"
            f"</tr></thead><tbody>{cv_rows_html}</tbody></table></div>",
            unsafe_allow_html=True,
        )
