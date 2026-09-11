"""Enumerate the full factorial design space of the 'handling pedestrian crossing' scenario (English version)
and save it to data/1_scenario_configurations/S4_handling_pedestrian_crossing/all_scenarios_en.csv.
Every row is one unique combination of the contextual factors listed in Table 1 of the paper."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # replication-package root

import itertools
import pandas as pd
from pathlib import Path

def build_scenarios() -> pd.DataFrame:
    # A. Environmental conditions
    weather        = ["sunny", "rainy"]
    time_of_day    = ["day", "night"]
    road_condition = ["dry", "wet"]

    # B. Main-lane traffic
    ped_type = ["children", "elderly", "police", "pregnant", "college student"]
    behavior = ["watching phone","talking to someone", "looking around"]
    action = ["stop","moving orientation towards crosswalk", "moving orientation not towards crosswalk"]

    # C. Ego & passenger info
    occupant_req = ["urgent (to hospital)", "hurry (commute or flight)", "none"]
    ego_role      = ["normal car", "emergency vehicle", "truck"]

    speed_gap_dict = {
        25:  ["10m", "50m", "100m"],
        50:  ["10m", "50m", "100m"],
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
            "weather":                w,
            "time":                   t,
            "road_condition":         rc,
            "pedestrian_type":        ped_typ,
            "pedestrian_behavior":    ped_beh,
            "pedestiran_action":    ped_act,
            "ego_role":               role,
            "occupant_requirement":   occ_req,
            "distance to pedestrian": gap,
            "ego_vehicle_speed":f"{spd}km/h",
        })
        scenario_id += 1

    return pd.DataFrame(rows)

def main(output_file: str | Path =
         ROOT / "data/1_scenario_configurations/S4_handling_pedestrian_crossing/all_scenarios_en.csv"
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
