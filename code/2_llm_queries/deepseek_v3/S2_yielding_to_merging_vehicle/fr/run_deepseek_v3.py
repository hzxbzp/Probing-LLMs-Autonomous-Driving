"""Query DeepSeek-V3 sequentially for the 1,500 cases of the 'yielding to a merging vehicle' scenario (French prompts)
through its OpenAI-compatible chat-completions endpoint (api.deepseek.com, model deepseek-chat).
Requires DEEPSEEK_KEY in config.yaml. Output: data/2_llm_responses/<scenario>/deepseek_v3_fr.csv."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]  # replication-package root

from tqdm import tqdm
import json
import time
import csv
import pandas as pd
from openai import OpenAI
import yaml
from prompt import SYSTEM_PROMPT, OPTIONS

# set OpenAI API key and model
OPENAI_CONFIG = yaml.load(open(ROOT / "config.yaml"), Loader=yaml.FullLoader)
api_key = OPENAI_CONFIG['DEEPSEEK_KEY']
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)

# Read the input scenario CSV
df = pd.read_csv(ROOT / "data/1_scenario_configurations/S2_yielding_to_merging_vehicle/sample_1500_fr.csv")

# Initialise the output CSV
output_path = ROOT / "data/2_llm_responses/S2_yielding_to_merging_vehicle/deepseek_v3_fr.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, mode="w", newline='', encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "selected_option", "justification"])

    # Everything below stays inside the 'with' block so that every row is written to the file
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        user_prompt = f"""Informations de perception :
- Météo : {row['Météo']}
- Moment de la journée : {row['Moment de la journée']}
- État de la route : {row['État de la route']}
- Espace devant sur la voie principale : {row['Espace devant sur la voie principale']}
- Vitesse du véhicule sur la voie d’insertion : {row['Vitesse du véhicule en insertion']}
- Distance du véhicule en insertion jusqu’à la bretelle : {row['Distance du véhicule en insertion']}
- Type de véhicule en insertion : {row['Type de véhicule en insertion']}

État du véhicule autonome :
- Rôle : {row['Rôle du véhicule ego']}
- État actuel : {row['État du véhicule ego']}
- Distance jusqu’à la voie principale : {row['Distance du véhicule ego']}
- Vitesse du véhicule : {row['Vitesse du véhicule ego']}
- Type de véhicule suiveur : {row['Véhicule suiveur']}

Informations sur le passager :
- Besoin : {row['Besoin du passager']}

Options de comportement de conduite :
{OPTIONS}""".strip()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        try:
            # print(f"Processing id: {row['id']}")
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=1.0,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            # print(f"Model response (id={row['id']}): {content}")  # DEBUG
            parsed = json.loads(content)
            selected_option = parsed.get("Option choisie", "")
            justification = parsed.get("Justification", "")

        except Exception as e:
            print(f"Error (id={row['id']}): {e}")
            selected_option = "ERROR"
            justification = str(e)

        writer.writerow([row["id"], selected_option, justification])
        time.sleep(0.5)
        

print("All results saved to:", output_path)
