"""Enumerate the full factorial design space of the 'merging into freeway' scenario (Chinese version)
and save it to data/1_scenario_configurations/S1_merging_into_freeway/all_scenarios_cn.csv.
Every row is one unique combination of the contextual factors listed in Table 1 of the paper."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # replication-package root

import itertools
import pandas as pd
from pathlib import Path

def build_scenarios() -> pd.DataFrame:
    # A. Environmental conditions
    weather        = ["晴天", "雨天"]
    time_of_day    = ["白天", "夜晚"]
    road_condition = ["干燥", "湿滑"]

    # B. Main-lane traffic
    front_space = ["畅通", "100米", "50米", "20米"]

    speed_gap_dict = {
        35:  ["10米", "50米", "100米"],
        60:  ["50米", "100米", "150米"],
        100: ["100米", "150米", "200米"],
    }
    speed_gap_pairs = [
        (speed, gap) for speed, gaps in speed_gap_dict.items() for gap in gaps
    ]

    main_vehicle_type = ["普通轿车", "卡车", "应急车辆"]

    # C. Ego vehicle and passenger information
    occupant_req = ["紧急（送医）", "赶时间（通勤/赶飞机）", "无特别需求"]
    ego_role      = ["普通轿车", "应急车辆", "卡车"]

    following_options = ["无", "普通轿车", "应急车辆"]

    current_states = [("静止", "0米", "0公里/小时")]
    for dist in ["100米", "50米", "10米"]:
        for vel in ["80公里/小时", "50公里/小时", "30公里/小时"]:
            current_states.append(("接近中", dist, vel))

    rows = []
    scenario_id = 1
    for (w, t, rc,
         fs,
         (spd, gap),
         mv_type,
         occ_req,
         role,
         follow_type,
         (state, dist, vel)) in itertools.product(
             weather, time_of_day, road_condition,
             front_space,
             speed_gap_pairs,
             main_vehicle_type,
             occupant_req,
             ego_role,
             following_options,
             current_states,
    ):
        rows.append({
            "id":                     scenario_id,
            "天气":                    w,
            "时间":                    t,
            "路况":                    rc,
            "主车道前方空隙":          fs,
            "主车道车速":              f"{spd}公里/小时",
            "主车道车距":              gap,
            "主车道车辆类型":          mv_type,
            "乘客需求":                occ_req,
            "本车角色":                role,
            "后车类型":                follow_type,
            "本车状态":                state,
            "本车距合流点距离":        dist,
            "本车速度":                vel,
        })
        scenario_id += 1

    return pd.DataFrame(rows)

def main(output_file: str | Path =
         ROOT / "data/1_scenario_configurations/S1_merging_into_freeway/all_scenarios_cn.csv"
) -> None:
    df = build_scenarios()
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Saved {len(df):,} Chinese scenario configurations to {output_file}")

if __name__ == "__main__":
    main()
