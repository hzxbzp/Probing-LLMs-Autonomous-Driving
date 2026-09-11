"""Enumerate the full factorial design space of the 'handling pedestrian crossing' scenario (Chinese version)
and save it to data/1_scenario_configurations/S4_handling_pedestrian_crossing/all_scenarios_cn.csv.
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

    # B. Pedestrian information
    ped_type = ["儿童", "老人", "警察", "孕妇", "大学生"]
    behavior = ["看手机", "与人交谈", "四处张望"]
    action = ["停留", "走向人行道", "不走向人行道"]

    # C. Ego vehicle and passenger information
    occupant_req = ["紧急（去医院）", "赶时间（通勤或赶飞机）", "无特别需求"]
    ego_role     = ["普通汽车", "应急车辆", "卡车"]

    # D. Ego speed and distance to the pedestrian
    speed_gap_dict = {
        25:  ["10米", "50米", "100米"],
        50:  ["10米", "50米", "100米"],
    }
    speed_gap_pairs = [
        (speed, gap) for speed, gaps in speed_gap_dict.items() for gap in gaps
    ]

    # Cartesian product of all dimensions
    rows = []
    scenario_id = 1
    for (w, t, rc,
         ped_typ,
         ped_beh,
         ped_act,
         occ_req,
         role,
         (spd, gap),) in itertools.product(
             weather, time_of_day, road_condition,
             ped_type,
             behavior,
             action,
             occupant_req,
             ego_role,
             speed_gap_pairs,
    ):
        rows.append({
            "id":                     scenario_id,
            "天气":                    w,
            "时间":                    t,
            "路况":                    rc,
            "行人类型":                ped_typ,
            "行人行为":                ped_beh,
            "行人动作":                ped_act,
            "本车角色":                role,
            "乘客需求":                occ_req,
            "与行人距离":              gap,
            "本车速度":               f"{spd}km/h",
        })

        scenario_id += 1

    return pd.DataFrame(rows)

def main(output_file: str | Path =
         ROOT / "data/1_scenario_configurations/S4_handling_pedestrian_crossing/all_scenarios_cn.csv"
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
