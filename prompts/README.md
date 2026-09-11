# Experimental Prompts

Complete prompt set used in the experiments: **4 scenarios × 3 languages = 12 prompt sets**.

Each file lists the system prompt, the behavioural options and one example context block
of the corresponding scenario and language.

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

## Note on the cyclist scenario

In the original experiment, the scenario-description sentence of the system prompt in the cyclist scenario
(`bicycle_*.txt`) was carried over from the "yielding to a merging vehicle" scenario: *"You are driving on the main
road (mainstream), and a vehicle is attempting to merge from a ramp."* (Chinese and French: its translation). The
context block and the behavioural options describe the cyclist scenario. The files here and the prompt modules in
`code/2_llm_queries/*/S3_following_a_cyclist/` keep this sentence exactly as it was sent to the three models, so
that re-running the code reproduces the experiment.
