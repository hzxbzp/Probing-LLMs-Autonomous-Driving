"""Enumerate the full factorial design space of the 'following a cyclist' scenario (English version)
and save it to data/1_scenario_configurations/S3_following_a_cyclist/all_scenarios_en.csv.
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
    front_space = ["free", "100m", "50m", "20m"]
    bicycle_rider = ["adult", "children"]
    bicycle_position = ["middle of the lane", "right of the lane"]

    speed_gap_dict = {
        25:  ["10m", "50m", "100m"],
        50:  ["10m", "50m", "100m"],
    }
    speed_gap_pairs = [
        (speed, gap) for speed, gaps in speed_gap_dict.items() for gap in gaps
    ]

    # C. Ego & passenger info
    occupant_req = ["urgent (to hospital)", "hurry (commute or flight)", "none"]
    ego_role      = ["normal car", "emergency vehicle", "truck"]

    # Following-vehicle presence/type
    following_options = ["none", "normal car", "emergency vehicle"]

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
            "id":                     scenario_id,
            "weather":                w,
            "time":                   t,
            "road_condition":         rc,
            "bicycle_front_space": fs,
            "bicycle_rider":          rider,
            "bicycle_position":       bi_pos,
            "dist_bicycle":         gap,
            "ego_vehicle_speed":f"{spd}km/h",
            "ego_role":               role,
            "occupant_requirement":   occ_req,
            "following_vehicle": follow_type,

        })
        scenario_id += 1

    return pd.DataFrame(rows)

def main(output_file: str | Path =
         ROOT / "data/1_scenario_configurations/S3_following_a_cyclist/all_scenarios_en.csv"
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
