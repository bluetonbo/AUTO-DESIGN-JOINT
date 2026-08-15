# -*- coding: utf-8 -*-
"""
Generative Design 대시보드 (Streamlit)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataclasses

import streamlit as st
import pandas as pd
import plotly.express as px

from generative_design.configs.ball_joint_volvo import CONFIG as BASE_CONFIG
from generative_design.engine import (
    load_part_data, train_surrogate_models, make_predictor,
    run_nsga2, filter_within_spec,
)

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

st.set_page_config(page_title="Generative Design - 볼조인트", layout="wide")

PRIMARY = "#0054A6"

st.markdown(
    f"""
    <style>
    h1, h2, h3 {{ color: {PRIMARY}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔧 Generative Design - 설계안 자동 생성")
st.caption(f"부품: **{CONFIG.part_name}**  ·  NSGA-II 다목적 최적화 기반 파레토 후보 탐색")


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
    st.header("⚙️ 최적화 설정")
    pop_size = st.slider("population size", 40, 300, 120, step=20)
    n_generations = st.slider("세대 수", 20, 200, 60, step=10)
    seed = st.number_input("random seed", value=1, step=1)
    run_btn = st.button("🚀 재계산", use_container_width=True)

    st.divider()
    st.caption(
        "⚠️ 실측 데이터가 6개 샘플뿐입니다. 이 대시보드의 예측치는 "
        "파이프라인 검증용 참고 자료이며, 실제 설계 확정에는 데이터 추가 확보가 필요합니다."
    )

if "computed" not in st.session_state or run_btn:
    with st.spinner("NSGA-II 최적화 실행 중..."):
        st.session_state["computed"] = compute_candidates(pop_size, n_generations, seed)

candidates, valid, objectives, kpi_spec_ranges, cv_scores, parse_report, n_samples = st.session_state["computed"]
kpi_names = [o.name for o in objectives]

col1, col2, col3, col4 = st.columns(4)
col1.metric("실측 샘플 수", n_samples)
col2.metric("생성된 후보안", len(candidates))
col3.metric("스펙 통과 후보", len(valid))
col4.metric("설계변수 / KPI", f"{len([p for p in parse_report if p[1] not in ('파생(수식 계산)','고정값')])} / {len(kpi_names)}")

tab1, tab2, tab3 = st.tabs(["📊 파레토 프론트", "📋 후보 목록", "🧬 변수 구성"])

with tab1:
    show_df = valid if len(valid) > 0 else candidates
    if len(valid) == 0:
        st.warning("스펙을 만족하는 후보가 없어 전체 후보를 표시합니다.")

    c1, c2, c3 = st.columns(3)
    x_kpi = c1.selectbox("X축 KPI", kpi_names, index=kpi_names.index("axial_before_after_min_stiffness_%"))
    y_kpi = c2.selectbox("Y축 KPI", kpi_names, index=kpi_names.index("breakaway_torque_Nm"))
    color_kpi = c3.selectbox("색상(3번째 KPI)", kpi_names, index=kpi_names.index("radial_before_after_min_stiffness_%"))

    fig = px.scatter(
        show_df, x=x_kpi, y=y_kpi, color=color_kpi,
        color_continuous_scale="Blues",
        hover_data=kpi_names,
        title=f"파레토 후보: {x_kpi} vs {y_kpi} (색상 = {color_kpi})",
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    fig.update_layout(height=520, plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "각 점은 하나의 설계 후보(치수 조합)입니다. 두 KPI 사이 trade-off 곡선(파레토 프론트) 위에 "
        "위치할수록 '이 두 지표 사이에서는 더 개선할 여지가 없는' 효율적인 후보입니다."
    )

with tab2:
    st.subheader("스펙 통과 후보")
    if len(valid) > 0:
        sort_kpi = st.selectbox("정렬 기준 KPI", kpi_names,
                                 index=kpi_names.index("axial_before_after_min_stiffness_%"))
        sort_dir = next(o.direction for o in objectives if o.name == sort_kpi)
        sorted_valid = valid.sort_values(sort_kpi, ascending=(sort_dir == "min"))

        st.dataframe(sorted_valid[kpi_names].head(30), use_container_width=True)

        csv = sorted_valid.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ 전체 후보 CSV 다운로드 (전체 치수 포함)", csv,
                            file_name="ball_joint_generative_candidates.csv", mime="text/csv")

        st.divider()
        st.subheader("후보 상세 치수 보기")
        idx = st.number_input("후보 번호 (표의 행 인덱스)", min_value=0,
                               max_value=len(sorted_valid) - 1, value=0, step=1)
        detail = sorted_valid.iloc[int(idx)]
        dim_cols = [c for c in sorted_valid.columns if c not in kpi_names]
        detail_df = pd.DataFrame({
            "변수": dim_cols,
            "한글명": [CONFIG.kor_labels.get(c, "") for c in dim_cols],
            "값(mm)": [detail[c] for c in dim_cols],
        })
        st.dataframe(detail_df, use_container_width=True, hide_index=True)
    else:
        st.info("스펙 통과 후보가 없습니다.")

with tab3:
    st.subheader("설계변수 구성 리포트")
    report_df = pd.DataFrame(
        [{"변수명": p[0], "한글명": CONFIG.kor_labels.get(p[0], ""),
          "구분": p[1], "범위/비고": p[2] if len(p) > 2 else ""} for p in parse_report]
    )
    st.dataframe(report_df, use_container_width=True, hide_index=True)

    st.caption(
        "**스펙 파싱**: 엑셀 스펙 문자열에서 직접 추출한 공차 범위 · "
        "**파생(수식 계산)**: 다른 설계변수로부터 계산되어 탐색 대상에서 제외 · "
        "**고정값**: 전 샘플 동일값이라 상수로 고정 · "
        "**데이터기반**: 스펙 파싱이 어려워 실측 데이터 범위로 대체"
    )

    st.subheader("대리모델(Ridge) 참고 정확도 (LOO-MAE)")
    cv_df = pd.DataFrame(
        [{"KPI": k, "한글명": CONFIG.kor_labels.get(k, ""), "LOO-MAE": round(v, 4)} for k, v in cv_scores.items()]
    )
    st.dataframe(cv_df, use_container_width=True, hide_index=True)
