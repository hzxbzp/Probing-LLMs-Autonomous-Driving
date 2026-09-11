"""Query DeepSeek-V3 sequentially for the 1,500 cases of the 'following a cyclist' scenario (English prompts)
through its OpenAI-compatible chat-completions endpoint (api.deepseek.com, model deepseek-chat).
Requires DEEPSEEK_KEY in config.yaml. Output: data/2_llm_responses/<scenario>/deepseek_v3_en.csv."""
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
df = pd.read_csv(ROOT / "data/1_scenario_configurations/S3_following_a_cyclist/sample_1500_en.csv")

# Initialise the output CSV
output_path = ROOT / "data/2_llm_responses/S3_following_a_cyclist/deepseek_v3_en.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, mode="w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "selected_option", "justification"])

    # Everything below stays inside the 'with' block so that every row is written to the file
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        user_prompt = f"""Perception Information:
- Weather: {row['weather']}
- Time: {row['time']}
- Road condition: {row['road_condition']}
- Bicycle's front space: {row['bicycle_front_space']}
- Bicycle rider: {row['bicycle_rider']}
- Bicycle position: {row['bicycle_position']}
- Bicycle's distance to the ego vehicle: {row['dist_bicycle']}

Ego Vehicle State:
- Role: {row['ego_role']}
- Ego velocity: {row['ego_vehicle_speed']}
- Following vehicle: {row['following_vehicle']}

Passenger Information:
- Requirement: {row['occupant_requirement']}

Driving Behavior Options:
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
            selected_option = parsed.get("chosen_option", "")
            justification = parsed.get("justification", "")

        except Exception as e:
            print(f"Error (id={row['id']}): {e}")
            selected_option = "ERROR"
            justification = str(e)

        writer.writerow([row["id"], selected_option, justification])
        time.sleep(0.5)
        

print("All results saved to:", output_path)
