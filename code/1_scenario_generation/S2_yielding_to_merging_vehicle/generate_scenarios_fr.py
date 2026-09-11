"""Enumerate the full factorial design space of the 'yielding to a merging vehicle' scenario (French version)
and save it to data/1_scenario_configurations/S2_yielding_to_merging_vehicle/all_scenarios_fr.csv.
Every row is one unique combination of the contextual factors listed in Table 1 of the paper."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # replication-package root

import itertools
import pandas as pd
from pathlib import Path

def build_scenarios() -> pd.DataFrame:
    # A. Environmental conditions
    weather        = ["ensoleillé", "pluvieux"]
    time_of_day    = ["jour", "nuit"]
    road_condition = ["sec", "mouillé"]

    # B. Circulation sur la voie principale
    front_space = ["libre", "100 m", "50 m", "20 m"]

    speed_gap_dict = {
        35:  ["10 m", "50 m", "100 m"],
        60:  ["50 m", "100 m", "150 m"],
        100: ["100 m", "150 m", "200 m"],
    }
    speed_gap_pairs = [
        (speed, gap) for speed, gaps in speed_gap_dict.items() for gap in gaps
    ]

    merging_vehicle_type = ["voiture", "camion", "véhicule prioritaire"]

    # C. Ego vehicle and passenger information
    occupant_req = ["urgence (vers l'hôpital)", "pressé (travail ou vol)", "aucune"]
    ego_role     = ["voiture", "véhicule prioritaire", "camion"]

    # Following-vehicle presence/type
    following_options = ["aucun", "voiture", "véhicule prioritaire"]

    merging_options = ["aucun", "voiture", "véhicule prioritaire"]

    # Ego vehicle state (approaching)
    current_states = []
    for dist in ["100 m", "50 m", "10 m"]:
        for vel in ["80 km/h", "50 km/h", "30 km/h"]:
            current_states.append(("en approche", dist, vel))

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
            "id":                         scenario_id,
            "Météo":                      w,
            "Moment de la journée":       t,
            "État de la route":           rc,
            "Vitesse du véhicule en insertion": f"{speed} km/h",
            "Distance du véhicule en insertion": gap,
            "Type de véhicule en insertion":     mv_type,
            "Espace devant sur la voie principale": fs,
            "Besoin du passager":         occ_req,
            "Rôle du véhicule ego":       role,
            "État du véhicule ego":       state,
            "Distance du véhicule ego":   dist,
            "Vitesse du véhicule ego":    vel,
            "Véhicule suiveur":           follow_type,
        })
        scenario_id += 1

    return pd.DataFrame(rows)

def main(output_file: str | Path =
         ROOT / "data/1_scenario_configurations/S2_yielding_to_merging_vehicle/all_scenarios_fr.csv"
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
