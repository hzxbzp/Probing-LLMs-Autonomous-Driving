"""Build the 1,500 GPT-4o requests of the 'following a cyclist' scenario (English prompts) and submit them
to the OpenAI Batch API. Requires OPENAI_KEY in config.yaml (copy config.example.yaml).
Usage: python submit_batch.py [--dry-run]   (--dry-run only writes the JSONL request file)
The request file is written to data/2_llm_responses/<scenario>/gpt4o_requests_<lang>.jsonl."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]  # replication-package root

import os
import json
import time
import csv
import pandas as pd
from openai import OpenAI
import yaml
from prompt import SYSTEM_PROMPT, OPTIONS

# set OpenAI API key and model
OPENAI_CONFIG = yaml.load(open(ROOT / "config.yaml"), Loader=yaml.FullLoader)
if OPENAI_CONFIG['OPENAI_API_TYPE'] == 'openai':
    api_key = OPENAI_CONFIG['OPENAI_KEY']
    client = OpenAI(api_key = api_key)

df = pd.read_csv(ROOT / "data/1_scenario_configurations/S3_following_a_cyclist/sample_1500_en.csv")

TEMPERATURE = 1.0  # sampling temperature

tasks = []

for index, row in df.iterrows():
    # Dynamically generate user content from row
    user_content = f"""Perception Information:
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

    task = {
        "custom_id": str(row["id"]),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-2024-11-20",
            "temperature": TEMPERATURE,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
        }
    }
    tasks.append(task)

# Creating the jsonl file
file_name = ROOT / "data/2_llm_responses/S3_following_a_cyclist/gpt4o_requests_en.jsonl"

file_name.parent.mkdir(parents=True, exist_ok=True)
with open(file_name, 'w') as file:
    for obj in tasks:
        file.write(json.dumps(obj) + '\n')

if "--dry-run" in sys.argv:
    print(f"Dry run: wrote {len(tasks)} requests to {file_name}; nothing was uploaded.")
    sys.exit(0)

batch_file = client.files.create(
    file=open(file_name, "rb"),
    purpose="batch"
)

# Start batch job
batch_job = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)

# Print the batch job ID
print(f"Batch job ID: {batch_job.id}")
print(f"Batch job status: {batch_job.status}")
