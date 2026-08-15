# -*- coding: utf-8 -*-
"""
VOLVO SPA12 CABJ 볼조인트 부품 설정.

수식 파생/고정값 근거 (엑셀 Spec 행 원문 대조):
  - case_inner_height_mm       spec "(12.5)+(6.13)=18.63"       -> case_h4_mm + case_h1_mm
  - case_inner_taper_height_mm spec "(12.5)+(6.13)-(0.81)=17.82" -> case_h4_mm + case_h1_mm - case_groove_depth_mm
  - case_h2_mm                 spec "(6.13±0.02)-(0.81 0/-0.04)" -> case_h1_mm - case_groove_depth_mm
  - case_h3_mm                 spec "(12.5±0.02)+(6.13±0.02)"    -> case_h4_mm + case_h1_mm  (※ case_inner_height_mm과 동일 수식)
  - seat_h8_mm                 spec "(36.06)"  -> 전 샘플에서 36.06으로 고정 -> 고정값 처리
  - seat_R2_mm                 spec "R0.4"     -> 전 샘플에서 0.4로 고정   -> 고정값 처리
"""

from generative_design.engine import PartConfig

KPI_COLUMNS = [
    "breakaway_torque_Nm",
    "running_torque_Nm",
    "axial_gap_before_mm",
    "radial_gap_before_mm",
    "axial_gap_after_mm",
    "radial_gap_after_mm",
    "axial_gap_increase_mm",
    "radial_gap_increase_mm",
    "axial_before_after_min_stiffness_%",
    "radial_before_after_min_stiffness_%",
]

KPI_DIRECTIONS = {
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

KOR_LABEL = {
    "ballstud_diameter_mm": "볼스터드 직경", "case_inner_diameter_mm": "케이스 내경",
    "bearing_Height_mm": "베어링(시트) 높이", "case_inner_height_mm": "케이스 내부 높이(파생)",
    "case_inner_taper_height_mm": "케이스 내부 테이퍼 높이(파생)",
    "case_outer_height_before_mm": "스웨이징 전 케이스 외부 높이",
    "case_outer_height_after_mm": "스웨이징 후 케이스 외부 높이",
    "case_d1_mm": "케이스 직경1", "case_d2_mm": "케이스 직경2", "case_d3_mm": "케이스 직경3",
    "case_d4_mm": "케이스 직경4", "case_d5_mm": "케이스 직경5",
    "case_h1_mm": "케이스 높이1", "case_h2_mm": "케이스 높이2(파생)",
    "case_h3_mm": "케이스 높이3(파생)", "case_h4_mm": "케이스 높이4",
    "case_groove_width_mm": "케이스 홈 폭", "case_groove_depth_mm": "케이스 홈 깊이",
    "case_d6_mm": "케이스 직경6", "case_roundness_mm": "케이스 진원도", "case_flatness_mm": "케이스 평면도",
    "seat_h1_mm": "시트 높이1", "seat_R_mm": "시트 반경", "seat_h3_mm": "시트 높이3", "seat_h4_mm": "시트 높이4",
    "seat_inner_d_mm": "시트 내경", "seat_outer_d_mm": "시트 외경", "seat_h7_mm": "시트 높이7",
    "seat_h8_mm": "시트 높이8(고정)", "seat_R2_mm": "시트 반경2(고정)",
    "breakaway_torque_Nm": "기동 토크", "running_torque_Nm": "회전 중 토크",
    "axial_gap_before_mm": "축방향 갭(시험전)", "radial_gap_before_mm": "반경방향 갭(시험전)",
    "axial_gap_after_mm": "축방향 갭(시험후)", "radial_gap_after_mm": "반경방향 갭(시험후)",
    "axial_gap_increase_mm": "축방향 갭 증가량", "radial_gap_increase_mm": "반경방향 갭 증가량",
    "axial_before_after_min_stiffness_%": "축방향 강성 유지율(%)",
    "radial_before_after_min_stiffness_%": "반경방향 강성 유지율(%)",
}

DERIVED_FORMULAS = {
    "case_inner_height_mm": lambda d: d["case_h4_mm"] + d["case_h1_mm"],
    "case_inner_taper_height_mm": lambda d: d["case_h4_mm"] + d["case_h1_mm"] - d["case_groove_depth_mm"],
    "case_h2_mm": lambda d: d["case_h1_mm"] - d["case_groove_depth_mm"],
    "case_h3_mm": lambda d: d["case_h4_mm"] + d["case_h1_mm"],
}

FIXED_VARS = {
    "seat_h8_mm": 36.06,
    "seat_R2_mm": 0.4,
}

SHAPE_TOLERANCE_VARS = ["case_roundness_mm", "case_flatness_mm"]

CONFIG = PartConfig(
    part_name="VOLVO SPA12 CABJ 볼조인트",

    data_source="excel",
    xlsx_path="",

    sheet_id="1eoZtecLXXIDYK-euWtk72HqzEANgCwYkMhn5dKlAihE",
    worksheet_name="",

    header_row=1,
    spec_row=8,
    data_row_start=2,
    data_row_end=8,
    kpi_columns=KPI_COLUMNS,
    kpi_directions=KPI_DIRECTIONS,
    kor_labels=KOR_LABEL,
    derived_formulas=DERIVED_FORMULAS,
    fixed_vars=FIXED_VARS,
    shape_tolerance_vars=SHAPE_TOLERANCE_VARS,
)
