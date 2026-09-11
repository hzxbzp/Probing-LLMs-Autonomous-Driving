"""Enumerate the full factorial design space of the 'merging into freeway' scenario (French version)
and save it to data/1_scenario_configurations/S1_merging_into_freeway/all_scenarios_fr.csv.
Every row is one unique combination of the contextual factors listed in Table 1 of the paper."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # replication-package root

import itertools
import pandas as pd
from pathlib import Path

def build_scenarios() -> pd.DataFrame:
    # A. Environmental conditions
    weather        = ["Ensoleillé", "Pluvieux"]
    time_of_day    = ["Jour", "Nuit"]
    road_condition = ["Sec", "Mouillé"]

    # B. Main-lane traffic
    front_space = ["Libre", "100 mètres", "50 mètres", "20 mètres"]

    speed_gap_dict = {
        35:  ["10 mètres", "50 mètres", "100 mètres"],
        60:  ["50 mètres", "100 mètres", "150 mètres"],
        100: ["100 mètres", "150 mètres", "200 mètres"],
    }
    speed_gap_pairs = [
        (speed, gap) for speed, gaps in speed_gap_dict.items() for gap in gaps
    ]

    main_vehicle_type = ["Voiture", "Camion", "Véhicule prioritaire"]

    # C. Ego & passenger info
    occupant_req = ["Urgent (transport médical)", "Pressé (travail/vol)", "Pas de besoin particulier"]
    ego_role      = ["Voiture", "Véhicule prioritaire", "Camion"]

    following_options = ["Aucun", "Voiture", "Véhicule prioritaire"]

    current_states = [("À l'arrêt", "0 mètre", "0 km/h")]
    for dist in ["100 mètres", "50 mètres", "10 mètres"]:
        for vel in ["80 km/h", "50 km/h", "30 km/h"]:
            current_states.append(("En approche", dist, vel))

    # Cartesian product of all dimensions
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
            "Météo":                  w,
            "Moment de la journée":   t,
            "État de la route":       rc,
            "Espace devant sur voie principale": fs,
            "Vitesse sur voie principale":       f"{spd} km/h",
            "Distance sur voie principale":      gap,
            "Type de véhicule sur voie principale": mv_type,
            "Besoin du passager":     occ_req,
            "Rôle du véhicule autonome": role,
            "Type de véhicule suivant": follow_type,
            "État du véhicule":       state,
            "Distance jusqu'au point de fusion": dist,
            "Vitesse du véhicule":    vel,
        })
        scenario_id += 1

    return pd.DataFrame(rows)

def main(output_file: str | Path =
         ROOT / "data/1_scenario_configurations/S1_merging_into_freeway/all_scenarios_fr.csv"
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
