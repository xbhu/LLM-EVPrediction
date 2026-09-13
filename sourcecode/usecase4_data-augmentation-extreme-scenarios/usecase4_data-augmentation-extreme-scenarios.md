# Use Case 4: LLM as Data Augmenter for Rare/Extreme Scenarios

## Purpose
Test whether an LLM can synthesize realistic joint (feature, demand) data for rare extreme scenarios that have no real historical examples — cold snap, grid outage, EV fleet event, electricity price spike — and evaluate whether the synthetic data actually improves downstream forecasting, rather than just looking plausible on its own.

## Data
Pre-built dataset: 2,160 real rows + 8,640 LLM-generated synthetic rows, spanning the 4 extreme scenario types.

## Experiment Design
Four parts:

1. **EDA / quality assessment** of the synthetic data — checked physical plausibility (e.g., whether cold_snap rows correctly show demand rising as temperature falls; an initial bug showed the wrong slope direction).
2. **LLM generation pipeline** itself — zero-shot generation without scale constraints produced physically absurd values (e.g., 125,000 kWh); adding explicit scale constraints and few-shot examples partially fixed magnitude but could not fix deeper temporal-structure errors.
3. **XGBoost augmentation experiments**: Model A (real data only) vs. Model B (real + all synthetic, indiscriminate) vs. Model C (real + one targeted scenario at a time).
4. **Flan-T5 fine-tuning** on the same augmented data, to test whether a text-conditioned model handles mixed-scenario augmentation better than a purely numeric model (XGBoost).

## Results
- **Model A** (no augmentation): strong on real-data test (MAE 3.33) but **failed catastrophically on extreme scenarios** — e.g., R² = −2.455 on EV fleet events, worse than simply predicting the mean.
- **Model B** (indiscriminate full augmentation): hurt normal-case performance (real-test MAE rose to 17.62) while only partially helping extreme cases — mixing all scenario types together confuses the model on ordinary days.
- **Model C** (scenario-targeted augmentation): clearly the best strategy — near-real accuracy preserved on the real test set, and strong performance on the matching extreme scenario. Example: cold_snap → MAE 4.20 (real test) / 5.50 (cold_snap test); price_spike → MAE 7.92 (real test) / 8.26 (price_spike test).
- **Flan-T5** avoided Model B's side effects because the scenario description in its text input lets it condition its prediction appropriately per scenario — unlike XGBoost, which only sees undifferentiated numeric features and cannot distinguish scenario context.

## Key Conceptual Finding
Evaluating LLM-generated synthetic data is fundamentally a **joint-distribution plausibility problem**, not a prediction problem (data generation produces X and Y jointly from a prompt; prediction maps a known X to an unknown Y). In the absence of real extreme-event data to benchmark against, only indirect proxies are available: physical-consistency checks, discriminability tests (can a classifier tell real from synthetic?), and downstream task utility (TSTR — train on synthetic, test on real). Targeted, scenario-specific augmentation clearly outperforms indiscriminate augmentation.
