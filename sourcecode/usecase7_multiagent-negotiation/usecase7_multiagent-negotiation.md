# Use Case 7: LLM-Based Multi-Agent Negotiation Simulation

## Purpose
Simulate a three-party natural-language negotiation — EV driver, charging-station operator, and grid operator — negotiating charging schedules, in order to study (1) agent role behavior under different prompt designs, (2) the reliability of LLM-as-judge evaluation, and (3) the effect of fine-tuning (specializing) a single agent.

## Data
A synthetic multi-agent negotiation dataset with `agent_prompt_template`s and 120 transcripts carrying ground-truth outcome/satisfaction labels. The dataset is perfectly balanced with a fixed, deterministic scenario→outcome mapping — an important caveat for interpreting Step 2 below.

## Experiment Design
Three steps:

- **Step 1 — Zero-shot negotiation**, three sub-experiments:
  - 1a: minimal prompt, each agent sees only the last message.
  - 1b: rich prompt, each agent sees the full conversation history.
  - 1c: Station and Grid roles fixed, EV-driver persona varied across 3 personas.
- **Step 2 — LLM-as-Judge**, scoring the 120 dataset transcripts plus the Step 1 outputs on negotiation outcome and satisfaction, compared against ground truth.
- **Step 3 — LoRA SFT of the Grid agent only**, using 240 training samples, comparing zero-shot vs. fine-tuned Grid agent responses.

## Results
- **Step 1**: Minimal prompts (last-message-only context) caused role collapse and circular deadlock between agents — they lost track of what had already been agreed. Rich prompts (full history) restored clear role differentiation, but exposed that the Grid agent could not adapt its strategy after being rejected — it kept repeating the same offer.
- **Step 2**: The LLM-as-Judge achieved 100% accuracy, but this was identified as **misleading**: because the dataset's outcomes are deterministically tied to scenario type, the judge could score correctly by pattern-matching the scenario description alone, without needing to actually read or evaluate the transcript content.
- **Step 3**: Severe overfitting (eval loss near zero by epoch 2), attributable to highly templated training targets. A data-labeling bug was also found: fixed turn-index extraction for the training targets misaligned in V2G scenarios where the Grid agent speaks first instead of last.

## Key Conceptual Takeaway
This use case's clearest lesson is methodological rather than architectural: synthetic, heavily templated datasets can make evaluation metrics (LLM-judge accuracy, SFT training loss) look artificially strong while masking that nothing meaningful was actually learned or evaluated. A follow-on research idea — mining social media for empirically grounded, more realistic agent personas — was discussed but explicitly set aside for future work.
