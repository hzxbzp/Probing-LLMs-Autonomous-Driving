# Experimental Prompts

Complete prompt set used in the experiments: **4 scenarios × 3 languages = 12 prompt sets**.

These files are extracted from the batch-task records that were actually submitted to
the model APIs, rather than regenerated from the scenario-generation scripts, so they
are the exact strings the models received.

## Files

| Scenario | English | Chinese | French |
|---|---|---|---|
| Merging into freeway | [`merging_en.txt`](merging_en.txt) | [`merging_cn.txt`](merging_cn.txt) | [`merging_fr.txt`](merging_fr.txt) |
| Yielding to a merging vehicle | [`mainstream_en.txt`](mainstream_en.txt) | [`mainstream_cn.txt`](mainstream_cn.txt) | [`mainstream_fr.txt`](mainstream_fr.txt) |
| Following a cyclist | [`bicycle_en.txt`](bicycle_en.txt) | [`bicycle_cn.txt`](bicycle_cn.txt) | [`bicycle_fr.txt`](bicycle_fr.txt) |
| Handling pedestrian crossings | [`pedestrian_en.txt`](pedestrian_en.txt) | [`pedestrian_cn.txt`](pedestrian_cn.txt) | [`pedestrian_fr.txt`](pedestrian_fr.txt) |

## Structure of each file

Every file contains three blocks:

1. **`[SYSTEM PROMPT]`** — role definition, decision criteria, scenario description,
   task instructions and required output format. Fixed within a scenario–language cell.
2. **`[BEHAVIORAL OPTIONS]`** — the ordered action space presented to the model.
   Appended to every user message in that cell and identical across all 1,500 cases.
3. **`[USER CONTEXT BLOCK]`** — the case-specific context (weather, lighting, road
   surface, surrounding traffic, ego-vehicle state, occupant urgency). One
   representative case is shown; the field values vary across the 1,500 cases of each
   cell according to the factor levels documented in Table 1 of the paper.

At run time the model receives the system prompt as the `system` message, and the
context block followed by the behavioral options as the `user` message.

## Differences across models

The same prompts were used for GPT-4o, DeepSeek-V3 and Llama 3.1-405B, with one
deliberate exception:

- **Output format.** GPT-4o was instructed to return the chosen option and its
  justification as two labelled lines of plain text. DeepSeek-V3 and Llama 3.1-405B
  were instructed to return the same two fields as a JSON object, which was required
  for reliable automated parsing of their API responses. The files here show the
  plain-text variant.

Only the output-format block differs. The role description, decision criteria,
scenario description, task instructions, context block and behavioral option
descriptions are identical across the three models; the option texts, which define the
ordered action space, are byte-identical within each scenario–language cell.

## Prompt length

Token counts for every prompt, computed under the native tokenizer of each evaluated
model (`o200k_base` for GPT-4o, and the DeepSeek-V3 and Llama 3.1 tokenizers), are
reported in the supplementary appendix. Mean total prompt length per case ranges from
roughly 410 to 1,030 tokens depending on scenario, language and tokenizer. Because only
the context values vary across cases, the within-cell standard deviation is below 5
tokens.
