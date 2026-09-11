"""Enumerate the full factorial design space of the 'handling pedestrian crossing' scenario (French version)
and save it to data/1_scenario_configurations/S4_handling_pedestrian_crossing/all_scenarios_fr.csv.
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

    # B. Pedestrian information
    ped_type = ["enfant", "personne âgée", "policier", "femme enceinte", "étudiant universitaire"]
    behavior = ["regarde son téléphone", "parle à quelqu'un", "regarde autour"]
    action = ["s’arrêter", "se déplacer en direction du passage piéton", "se déplacer dans une direction autre que celle du passage piéton"]

    # C. Ego vehicle and passenger information
    occupant_req = ["urgence (vers hôpital)", "pressé (travail ou vol)", "aucun"]
    ego_role     = ["voiture normale", "véhicule d'urgence", "camion"]

    # D. Ego speed and distance to the pedestrian
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
            "id":                         scenario_id,
            "Météo":                      w,
            "Moment de la journée":       t,
            "État de la route":           rc,
            "Type de piéton":             ped_typ,
            "Comportement du piéton":     ped_beh,
            "Action du piéton":           ped_act,
            "Rôle du véhicule ego":       role,
            "Besoin du passager":         occ_req,
            "Distance au piéton":         gap,
            "Vitesse du véhicule ego":    f"{spd}km/h",
        })

        scenario_id += 1

    return pd.DataFrame(rows)

def main(output_file: str | Path =
         ROOT / "data/1_scenario_configurations/S4_handling_pedestrian_crossing/all_scenarios_fr.csv"
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
