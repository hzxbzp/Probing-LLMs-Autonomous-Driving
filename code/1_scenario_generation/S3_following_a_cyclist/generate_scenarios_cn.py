"""Enumerate the full factorial design space of the 'following a cyclist' scenario (Chinese version)
and save it to data/1_scenario_configurations/S3_following_a_cyclist/all_scenarios_cn.csv.
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
    road_condition = ["干燥", "潮湿"]

    # B. Main-lane traffic
    front_space = ["无车辆", "100米", "50米", "20米"]
    bicycle_rider = ["成年人", "儿童"]
    bicycle_position = ["车道中间", "车道右侧"]

    speed_gap_dict = {
        25:  ["10米", "50米", "100米"],
        50:  ["10米", "50米", "100米"],
    }
    speed_gap_pairs = [
        (speed, gap) for speed, gaps in speed_gap_dict.items() for gap in gaps
    ]

    # C. Ego vehicle and passenger information
    occupant_req = ["紧急（去医院）", "赶时间（通勤或赶飞机）", "无特殊需求"]
    ego_role      = ["普通车辆", "应急车辆", "卡车"]

    # Following-vehicle presence/type
    following_options = ["无", "普通车辆", "应急车辆"]


    # Cartesian product of all dimensions
    rows = []
    scenario_id = 1
    for (w, t, rc,
         fs,
         (spd, gap),
         occ_req,
         role,
         follow_type,
         rider,
         bi_pos) in itertools.product(
             weather, time_of_day, road_condition,
             front_space,
             speed_gap_pairs,
             occupant_req,
             ego_role,
             following_options,
            bicycle_rider,
            bicycle_position,
    ):
        rows.append({
            "id":                  scenario_id,
            "天气":              w,
            "时间":              t,
            "路况":              rc,
            "自行车前方空隙":     fs,
            "骑车人类型":        rider,
            "骑车人位置":        bi_pos,
            "与骑车人距离":      gap,
            "本车速度":          f"{spd}km/h",
            "本车角色":          role,
            "乘客需求":          occ_req,
            "后车类型":          follow_type,
        })

        scenario_id += 1

    return pd.DataFrame(rows)

def main(output_file: str | Path =
         ROOT / "data/1_scenario_configurations/S3_following_a_cyclist/all_scenarios_cn.csv"
) -> None:
    """
    Generate the full merge-scenario table and save it to a single CSV.

    Parameters
    ----------
    output_file : str | Path
        Full path (including filename) of the CSV to be written.
    """
    # 1  build the full DataFrame
    df = build_scenarios()          # ← assumes your build_scenarios() is defined elsewhere

    # 2  make sure the parent directory exists
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 3  write once, no chunking
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df):,} scenarios to {output_file}")

if __name__ == "__main__":
    main()
