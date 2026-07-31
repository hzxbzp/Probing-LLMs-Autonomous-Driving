# Probing Large Language Models for Autonomous Driving Behavior

This repository provides the supplementary materials, code, and data for the paper:

> **Probing Large Language Models for Autonomous Driving Behavior**  
> Zhipeng Bao, Wenjie Zhao, Qianwen Li  
> *Communications in Transportation Research*

## Abstract

As large language models (LLMs) are increasingly integrated into autonomous vehicles (AV), understanding their reasoning and behavioral tendencies becomes essential. Trained on vast datasets, LLMs carry behavioral priors and social biases that may shape their driving decisions. To address this, we probe LLM behavior in representative driving scenarios, each with 1,500 contextual variants. Three widely used LLMs (GPT-4o, DeepSeek-V3, Llama 3.1-405B) are evaluated with multilingual prompts (English, Chinese, French) to select from predefined behavioral options ordered by aggressiveness. An Ordered Logit Model quantifies how contextual factors influence decisions, complemented by thematic analysis to reveal underlying reasoning tendencies.

<p align="center">
  <img src="assets/pipeline.png" alt="Pipeline for probing LLM driving styles and decision logic" width="100%">
</p>
<p align="center"><i>Fig. 1. Pipeline for probing LLM driving styles and decision logic.</i></p>

**Key findings:**
- GPT-4o is more conservative, while DeepSeek and LLaMA act more assertively, especially in vehicle interactions.
- Chinese and French prompts yield more assertive behavior than English, with French strongest.
- All models shift toward more protective behavior when vulnerable road users (cyclists, pedestrians) appear.

## Repository Structure

```
.
├── README.md
├── LICENSE
├── prompts/                   # Complete prompt set (4 scenarios × 3 languages)
│   ├── README.md
│   ├── merging_{en,cn,fr}.txt
│   ├── mainstream_{en,cn,fr}.txt
│   ├── bicycle_{en,cn,fr}.txt
│   └── pedestrian_{en,cn,fr}.txt
├── paper/
│   └── Appendix.pdf          # Supplementary appendices (A1–A4)
├── code/                      # Source code (to be released)
│   └── TODO.md
└── data/                      # Experimental data (to be released)
    └── TODO.md
```

## Scenarios

The study evaluates LLM driving decisions across four representative traffic scenarios:

1. **Merging into Freeway** — The AV attempts to merge onto a freeway, choosing between forceful merge, cooperative merge, or waiting.
2. **Yielding to a Merging Vehicle** — The AV is on the main lane and decides whether to yield to, resist, or proactively accommodate a merging vehicle.
3. **Following a Cyclist** — The AV follows a cyclist in a shared lane, deciding whether to overtake, negotiate, or follow patiently.
4. **Handling Pedestrian Crossing** — The AV approaches a crosswalk and decides whether to proceed, probe cautiously, or yield proactively.

## Methods

- **Ordered Logit Model (OLM):** Quantifies the influence of contextual factors (weather, traffic, ego state, occupant urgency) on driving style selection.
- **Thematic Analysis:** Extracts reasoning patterns from LLM-generated justifications using zero-shot Natural Language Inference.
- **Multilingual Evaluation:** Prompts in English, Chinese, and French to assess cross-language behavioral consistency.

## Prompts

The complete set of experimental prompts is available in [`prompts/`](prompts/): the
system prompt, the ordered behavioral options and a representative context block for
each of the four scenarios in English, Chinese and French. See
[`prompts/README.md`](prompts/README.md) for the structure of each file and for the one
deliberate difference in prompt across models (output format).

## TODO

- [x] Release prompt set for all scenarios and languages
- [ ] Release scenario generation scripts
- [ ] Release LLM API calling and response collection code
- [ ] Release data encoding and preprocessing scripts
- [ ] Release Ordered Logit Model estimation code
- [ ] Release thematic analysis (NLI-based) code
- [ ] Release raw and processed experimental datasets
- [ ] Release back-translation validation code

> **Note:** The prompt set is available now. The remaining code and data will be made
> publicly available upon paper acceptance.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
