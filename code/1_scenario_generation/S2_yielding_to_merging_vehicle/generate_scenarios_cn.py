"""Enumerate the full factorial design space of the 'yielding to a merging vehicle' scenario (Chinese version)
and save it to data/1_scenario_configurations/S2_yielding_to_merging_vehicle/all_scenarios_cn.csv.
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

    merging_vehicle_type = ["普通轿车", "卡车", "应急车辆"]

    # C. Ego vehicle and passenger information
    occupant_req = ["紧急（送医）", "赶时间（通勤/赶飞机）", "无特殊需求"]
    ego_role      = ["普通轿车", "应急车辆", "卡车"]

    # Following vehicle
    following_options = ["无", "普通轿车", "应急车辆"]

    # Ego current state (approaching or stopped)
    current_states = []
    for dist in ["100米", "50米", "10米"]:
        for vel in ["80公里/小时", "50公里/小时", "30公里/小时"]:
            current_states.append(("接近中", dist, vel))

    # Cartesian product of all dimensions
    rows = []
    scenario_id = 1
    for (w, t, rc,
         fs,
         (speed, gap),
         mv_type,
         occ_req,
         role,
         follow_type,
         (state, dist, vel)) in itertools.product(
             weather, time_of_day, road_condition,
             front_space,
             speed_gap_pairs,
             merging_vehicle_type,
             occupant_req,
             ego_role,
             following_options,
             current_states,
    ):
        rows.append({
            "id":                 scenario_id,
            "天气":                 w,
            "时间":                 t,
            "路况":                 rc,
            "并车车速":             f"{speed}公里/小时",
            "并车与本车距":          gap,
            "并车车辆类型":          mv_type,
            "主车道前方空隙":        fs,
            "乘客需求":             occ_req,
            "本车角色":             role,
            "本车状态":             state,
            "本车距合流点距离":      dist,
            "本车速度":             vel,
            "后车类型":             follow_type,
        })
        scenario_id += 1

    return pd.DataFrame(rows)

def main(output_file: str | Path =
         ROOT / "data/1_scenario_configurations/S2_yielding_to_merging_vehicle/all_scenarios_cn.csv"
) -> None:
    """
    Generate the full scenario table and save it as a single CSV file.

    参数
    ----------
    output_file : str | Path
        Full path (including file name) of the CSV to be written.
    """
    # 1. Build the full DataFrame
    df = build_scenarios()

    # 2. Make sure the parent directory exists
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 3. Write the CSV once, no chunking
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df):,} scenarios to {output_file}")

if __name__ == "__main__":
    main()

