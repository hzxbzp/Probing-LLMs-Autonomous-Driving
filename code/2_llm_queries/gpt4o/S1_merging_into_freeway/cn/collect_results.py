"""Download a finished GPT-4o batch of the 'merging into freeway' scenario (Chinese) and parse every reply into
id, selected_option, justification. Usage: python collect_results.py <batch_id>
Output: data/2_llm_responses/<scenario>/gpt4o_<lang>.csv."""
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
import re

# set OpenAI API key and model
OPENAI_CONFIG = yaml.load(open(ROOT / "config.yaml"), Loader=yaml.FullLoader)
if OPENAI_CONFIG['OPENAI_API_TYPE'] == 'openai':
    api_key = OPENAI_CONFIG['OPENAI_KEY']
    client = OpenAI(api_key = api_key)

# Check whether the batch job has finished
if len(sys.argv) < 2:
    sys.exit("Usage: python collect_results.py <batch_id>   (the id printed by submit_batch.py)")
BATCH_ID = sys.argv[1]
batch_job = client.batches.retrieve(BATCH_ID)
if batch_job.status != "completed":
    print(f"Job {batch_job.id} is not done yet. Status: {batch_job.status}")
    exit(0)
else:
    print(f"Job {batch_job.id} is done.")
    print(batch_job)

# Download the result file
results_list = []
result_file_id = batch_job.output_file_id
result = client.files.content(result_file_id).content
result = result.decode('utf-8')
result_entries = result.strip().split("\n")

# Create the CSV file that stores the parsed results
out_csv = ROOT / "data/2_llm_responses/S1_merging_into_freeway/gpt4o_cn.csv"
out_csv.parent.mkdir(parents=True, exist_ok=True)
with open(out_csv, mode="w", newline='', encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "selected_option", "justification"])  # CSV header

    for r in result_entries:
        data = json.loads(r)

        custom_id = data.get("custom_id", "unknown")

        # Extract the model output (a single message containing both the option and the justification)
        completion = data["response"]["body"]["choices"][0]["message"]["content"]

        # Simple extraction (assumes the reply follows this format):
        # Chosen Option: Assertive Merge
        # Justification: Because...

        selected_option = ""
        justification_lines = []
        in_justification = False

        for line in completion.split("\n"):
            line = line.strip()

            if "选择的选项" in line:
                parts = line.split("：", 1)
                if len(parts) == 2:
                    selected_option = parts[1].strip()

            elif "选择理由" in line:
                in_justification = True  # start collecting the justification paragraph
                continue  # this line itself is usually empty

            elif in_justification:
                # Stop at an empty line or a new heading
                if line.startswith("- ") or line == "":
                    break
                justification_lines.append(line)

        justification = " ".join(justification_lines).strip()



        # Write one row per case
        writer.writerow([custom_id, selected_option, justification])



