# -*- coding: utf-8 -*-
"""
Generative Design 파이프라인 실행 - VOLVO 볼조인트

실행:
  cd /home/claude && python -m generative_design.run_ball_joint
"""
from generative_design.configs.ball_joint_volvo import CONFIG
from generative_design.engine import (
    load_part_data, train_surrogate_models, make_predictor,
    run_nsga2, filter_within_spec,
)


def run(cfg=CONFIG, pop_size=120, n_generations=60, out_prefix="/home/claude/ball_joint", gcp_credentials=None):
    print("=" * 78)
    print(f"부품: {cfg.part_name}")
    print("=" * 78)
    print("\n1) 데이터 로드 + 스펙 파싱 + 파생/고정 변수 처리")
    variables, objectives, var_cols, data, kpi_spec_ranges, parse_report = load_part_data(
        cfg, gcp_credentials=gcp_credentials
    )
    print(f"독립 설계변수: {len(variables)}개 / KPI: {len(objectives)}개 / 샘플: {len(data)}행\n")
    for item in parse_report:
        name, reason = item[0], item[1]
        rng = item[2] if len(item) > 2 else ""
        kor = cfg.kor_labels.get(name, "")
        print(f"  {name:32s} {kor:18s} [{reason:22s}] {rng}")

    print("\n" + "=" * 78)
    print("2) 대리모델(Ridge) 학습")
    print("=" * 78)
    models, cv_scores = train_surrogate_models(var_cols, data, cfg.kpi_columns)
    for kpi, mae in cv_scores.items():
        print(f"  {kpi:38s} {cfg.kor_labels.get(kpi,''):16s} LOO-MAE(참고용): {mae:.4f}")

    print("\n" + "=" * 78)
    print("3) NSGA-II 다목적 최적화")
    print("=" * 78)
    predictor = make_predictor(models, cfg)
    candidates = run_nsga2(variables, objectives, predictor, cfg,
                            pop_size=pop_size, n_generations=n_generations)
    print(f"파레토 후보안: {len(candidates)}개")

    print("\n" + "=" * 78)
    print("4) 스펙 필터링")
    print("=" * 78)
    valid = filter_within_spec(candidates, objectives, kpi_spec_ranges)
    print(f"스펙 만족 후보: {len(valid)}개 / {len(candidates)}개")

    candidates.to_csv(f"{out_prefix}_candidates_all.csv", index=False)
    valid.to_csv(f"{out_prefix}_candidates_valid.csv", index=False)
    print(f"\n전체 후보 -> {out_prefix}_candidates_all.csv")
    print(f"스펙 통과 후보 -> {out_prefix}_candidates_valid.csv")

    return candidates, valid, objectives, kpi_spec_ranges, cfg


if __name__ == "__main__":
    run()
