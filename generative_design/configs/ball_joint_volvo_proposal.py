# -*- coding: utf-8 -*-
"""
VOLVO SPA12 CABJ 볼조인트 부품 설정 - "제안용(Proposal)" 버전.

원본(ball_joint_volvo.py) 대비 유일한 차이점:
  기존에 "다른 변수의 합/차 수식"으로 결정되던 파생변수 4개를
  각자 독립적인 공차를 갖는 독립 설계변수로 전환했습니다.

  - case_inner_height_mm       (기존 수식: case_h4_mm + case_h1_mm)
  - case_inner_taper_height_mm (기존 수식: case_h4_mm + case_h1_mm - case_groove_depth_mm)
  - case_h2_mm                 (기존 수식: case_h1_mm - case_groove_depth_mm)
  - case_h3_mm                 (기존 수식: case_h4_mm + case_h1_mm)

전환 이유: 합산 수식으로 값을 강제로 묶어버리면, 실제로는 조립 과정에서
독립적으로 관리 가능한 치수까지 서로 종속시켜버려 탐색 폭이 부당하게 좁아짐.
설계 "제안" 단계에서는 이 4개도 자체 공차를 갖는 독립변수로 열어두는 것이 합리적.

-> DERIVED_FORMULAS를 비워서 이 4개가 자동으로 독립 탐색 대상에 포함되도록 함
   (engine.py의 load_part_data는 derived_formulas에 없는 변수를 전부 독립변수로 처리)

고정값(seat_h8_mm, seat_R2_mm)은 "합산 수식"이 아니라 "실측상 항상 동일한 상수"라는
별개의 성격이라 이번 전환 대상에서 제외하고 그대로 유지했습니다.
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
    "bearing_Height_mm": "베어링(시트) 높이",
    "case_inner_height_mm": "케이스 내부 높이(독립 전환)",
    "case_inner_taper_height_mm": "케이스 내부 테이퍼 높이(독립 전환)",
    "case_outer_height_before_mm": "스웨이징 전 케이스 외부 높이",
    "case_outer_height_after_mm": "스웨이징 후 케이스 외부 높이",
    "case_d1_mm": "케이스 직경1", "case_d2_mm": "케이스 직경2", "case_d3_mm": "케이스 직경3",
    "case_d4_mm": "케이스 직경4", "case_d5_mm": "케이스 직경5",
    "case_h1_mm": "케이스 높이1", "case_h2_mm": "케이스 높이2(독립 전환)",
    "case_h3_mm": "케이스 높이3(독립 전환)", "case_h4_mm": "케이스 높이4",
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

# 파생 변수 없음 - 전부 독립 탐색 대상 (핵심 변경점)
DERIVED_FORMULAS = {}

# 고정값은 원본과 동일하게 유지 (합산수식이 아니라 실측상 상수이므로 전환 대상 아님)
FIXED_VARS = {
    "seat_h8_mm": 36.06,
    "seat_R2_mm": 0.4,
}

SHAPE_TOLERANCE_VARS = ["case_roundness_mm", "case_flatness_mm"]

CONFIG = PartConfig(
    part_name="VOLVO SPA12 CABJ 볼조인트 (제안용 - 독립변수 버전)",

    data_source="excel",
    xlsx_path="/home/claude/sample_input/AUTO_DESIGN_JOINT_SAMPLE_INPUT_PROPOSAL.xlsx",

    sheet_id="",
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
