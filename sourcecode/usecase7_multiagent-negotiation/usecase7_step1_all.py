"""
Use Case 7 - Step 1: Zero-shot Multi-Agent EV Charging Negotiation
==================================================================
Step 1a: use the dataset's minimal prompt; each agent sees only the last message
Step 1b: use a designed rich prompt; each agent sees the full conversation history
Step 1c: fix Station/Grid, swap in three EV Driver personas to observe behavioral differences

Usage: python sourcecode/uc7_step1_all.py
Output: outputs/usecase7_zeroshot/
"""

import json
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =============================================================================
# CONFIG
# =============================================================================
MODEL_NAME        = "Qwen/Qwen3-4B"
DATA_PATH         = "/home/xzh5180/Research/llm-evprediction/datasets/dataset7_multiagent.csv"
OUTPUT_DIR        = "/home/xzh5180/Research/llm-evprediction/outputs/usecase7_zeroshot"
TARGET_SESSION_ID = 1       # peak_hour_conflict; use this session for all comparisons
MAX_NEW_TOKENS    = 150     # maximum tokens each agent generates per turn
NUM_ROUNDS        = 6       # total number of negotiation rounds
TURN_ORDER        = ["EV_User", "Station", "Grid", "EV_User", "Station", "Grid"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# STEP 1a: original minimal prompt from the dataset
# Design rationale: each agent sees only the last message ({incoming_message}),
#                  simulating the simplest reactive agent — responds only to the current stimulus
# =============================================================================
def build_minimal_system_prompt(agent_role: str, scenario_description: str) -> str:
    """
    Directly use the dataset's agent_prompt_template, filling in agent_role and scenario.
    Note: this template does not differentiate goal/constraint among the three roles;
         all agents share the same objective: "maximize your own objective"
    """
    return (
        f"You are the {agent_role} agent in an EV charging negotiation.\n"
        f"Scenario: {scenario_description}\n"
        f"Your goal: maximize your own objective while reaching a feasible agreement.\n"
        f"Respond concisely and propose a concrete action or counteroffer."
    )

# =============================================================================
# STEP 1b: designed rich role prompts
# Design rationale: four elements (persona / goal / constraint / strategy) clearly differentiate the three roles;
#                  each agent sees the full conversation history (full history)
# =============================================================================
RICH_PROMPTS = {
    "EV_User": """You are an EV driver in a real-time charging negotiation.
Persona: You have an urgent long-distance trip and need sufficient charge before departure.
Goal: Secure DC fast charging to at least 80% SOC before your 6 PM departure.
Constraint: You cannot delay your trip — waiting until after 7 PM is completely unacceptable.
Strategy: Assert your time constraint clearly upfront. Accept a cost premium if necessary.
  Consider Level 2 charging only as a last resort if DC Fast is unavailable within 30 minutes.
Keep your response to 2-3 sentences. Propose or respond to concrete actions.""",

    "Station": """You are a charging station operator managing a public EV facility.
Persona: You balance service quality, charger throughput, and revenue across multiple customers.
Goal: Maximize charger utilization and revenue while maintaining acceptable customer satisfaction.
Constraint: DC fast charger slots are limited and currently in high demand.
  You must serve multiple customers fairly — you cannot give one user indefinite priority.
Strategy: Offer tiered pricing (Level 2 vs DC Fast). Use staggered reservation scheduling.
  Propose concrete slot times with associated costs.
Keep your response to 2-3 sentences. Propose or respond to concrete actions.""",

    "Grid": """You are the regional grid operator responsible for power grid stability.
Persona: You manage demand response programs and use financial incentives to shift load.
Goal: Reduce peak demand during the 4–7 PM critical window by at least 30%.
Constraint: You cannot mandate curtailment — incentives only, no forced load reduction.
  Your rebate budget is limited; escalate only if the initial offer is refused.
Strategy: Lead with a specific dollar rebate tied to a clear delay condition.
  Acknowledge user urgency, but emphasize the grid emergency and the compensation value.
Keep your response to 2-3 sentences. Propose or respond to concrete actions.""",
}

# =============================================================================
# STEP 1c: three EV Driver personas (Station and Grid reuse RICH_PROMPTS)
# Design rationale: same scenario, only EV Driver's persona and strategy change,
#                  to observe how user heterogeneity affects negotiation outcomes
# =============================================================================
EV_PERSONAS = {
    "Flexible_PriceSensitive": """You are an EV driver in a real-time charging negotiation.
Persona: Your schedule today is flexible — you don't have a fixed departure time.
Goal: Charge to at least 70% SOC at the lowest possible cost.
Constraint: You prefer to avoid peak-hour pricing if a cheaper option is available soon.
Strategy: Actively explore off-peak pricing options and rebate offers.
  You are willing to wait 1–2 hours if the cost saving is meaningful (more than $3).
Keep your response to 2-3 sentences. Propose or respond to concrete actions.""",

    "Urgent_CostInsensitive": """You are an EV driver in a real-time charging negotiation.
Persona: You are running late and have an inflexible 6 PM departure for a 4-hour drive.
Goal: Get DC fast charging immediately — cost is secondary to speed.
Constraint: Any solution that does not start charging within 20 minutes is unacceptable.
  Delaying to off-peak hours is completely out of the question.
Strategy: Reject any delay-based offers firmly. Demand the next available DC fast slot.
  Express willingness to pay a premium if it means immediate service.
Keep your response to 2-3 sentences. Propose or respond to concrete actions.""",

    "V2G_Willing": """You are an EV driver in a real-time charging negotiation.
Persona: You are parked for 3 hours and currently at 85% SOC — no urgent charging need.
Goal: Earn compensation by participating in V2G discharge if the terms are acceptable.
Constraint: You will not discharge below 60% SOC. Minimum acceptable compensation: $0.25/kWh.
Strategy: Signal your V2G availability early. Negotiate the compensation rate and SOC floor.
  Accept if the grid's offer meets your minimum; counter if it falls short.
Keep your response to 2-3 sentences. Propose or respond to concrete actions.""",
}

# =============================================================================
# MODEL LOADING
# =============================================================================
def load_model(model_name: str):
    print(f"\n[Loading] {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"[Loaded]  dtype=bfloat16 | device={next(model.parameters()).device}")
    return tokenizer, model

# =============================================================================
# INFERENCE: generate one round of response for a single agent
# =============================================================================
def agent_speak(
    system_prompt: str,
    user_content: str,
    tokenizer,
    model,
) -> str:
    messages = [
        {"role": "user", "content": f"{system_prompt}\n\n{user_content}"},
    ]
    # Qwen3 concatenates into user role when independent system role is unsupported; enable_thinking=False
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# =============================================================================
# STEP 1a negotiation: each agent sees only the last message
# =============================================================================
def run_1a(session_row: pd.Series, tokenizer, model) -> list:
    print("\n" + "="*60)
    print("STEP 1a | Minimal Prompt | Last-message-only context")
    print("="*60)

    history = []
    for agent_role in TURN_ORDER:
        sys_prompt = build_minimal_system_prompt(
            agent_role, session_row["scenario_description"]
        )
        # Pass only the last message (or empty)
        if history:
            last = history[-1]
            user_content = f"Current message from other agent: {last['message']}"
        else:
            user_content = "You are starting the negotiation. Begin with your opening statement."

        response = agent_speak(sys_prompt, user_content, tokenizer, model)
        history.append({"agent": agent_role, "message": response})
        print(f"  [{agent_role}]: {response}")

    return history

# =============================================================================
# STEP 1b negotiation: each agent sees the full conversation history
# =============================================================================
def run_1b(session_row: pd.Series, tokenizer, model) -> list:
    print("\n" + "="*60)
    print("STEP 1b | Rich Prompt | Full history context")
    print("="*60)

    history = []
    for agent_role in TURN_ORDER:
        sys_prompt = RICH_PROMPTS[agent_role]

        # Build full history text
        if history:
            history_text = "\n".join(
                f"[{t['agent']}]: {t['message']}" for t in history
            )
            user_content = (
                f"Scenario: {session_row['scenario_description']}\n\n"
                f"Full conversation so far:\n{history_text}\n\n"
                f"It is your turn as {agent_role}. Respond to the conversation above."
            )
        else:
            user_content = (
                f"Scenario: {session_row['scenario_description']}\n\n"
                f"No messages yet. You ({agent_role}) speak first."
            )

        response = agent_speak(sys_prompt, user_content, tokenizer, model)
        history.append({"agent": agent_role, "message": response})
        print(f"  [{agent_role}]: {response}")

    return history

# =============================================================================
# STEP 1c negotiation: fix Station/Grid, swap EV Driver persona
# =============================================================================
def run_1c(session_row: pd.Series, persona_name: str, tokenizer, model) -> list:
    print("\n" + "="*60)
    print(f"STEP 1c | EV Persona: {persona_name}")
    print("="*60)

    # 1c reuses the full history design from the rich prompt approach
    prompts_1c = {**RICH_PROMPTS, "EV_User": EV_PERSONAS[persona_name]}

    history = []
    for agent_role in TURN_ORDER:
        sys_prompt = prompts_1c[agent_role]

        if history:
            history_text = "\n".join(
                f"[{t['agent']}]: {t['message']}" for t in history
            )
            user_content = (
                f"Scenario: {session_row['scenario_description']}\n\n"
                f"Full conversation so far:\n{history_text}\n\n"
                f"It is your turn as {agent_role}. Respond to the conversation above."
            )
        else:
            user_content = (
                f"Scenario: {session_row['scenario_description']}\n\n"
                f"No messages yet. You ({agent_role}) speak first."
            )

        response = agent_speak(sys_prompt, user_content, tokenizer, model)
        history.append({"agent": agent_role, "message": response})
        print(f"  [{agent_role}]: {response}")

    return history

# =============================================================================
# Print Ground Truth (for comparison)
# =============================================================================
def print_ground_truth(session_row: pd.Series):
    print("\n" + "="*60)
    print("GROUND TRUTH (original dataset conversation)")
    print("="*60)
    gt = json.loads(session_row["negotiation_transcript"])
    for turn in gt:
        print(f"  [{turn['agent']}]: {turn['message']}")
    print(f"\n  Outcome:      {session_row['outcome']}")
    print(f"  Satisfaction: {session_row['user_satisfaction_score']:.3f}")
    print(f"  Grid stress reduction: {session_row['grid_stress_reduction_pct']:.1f}%")

# =============================================================================
# MAIN
# =============================================================================
def main():
    df = pd.read_csv(DATA_PATH)
    session_row = df[df["session_id"] == TARGET_SESSION_ID].iloc[0]

    print(f"\n{'='*60}")
    print(f"Session {TARGET_SESSION_ID} | {session_row['scenario']}")
    print(f"User need   : {session_row['user_need']}")
    print(f"Station     : {session_row['station_status']}")
    print(f"Grid signal : {session_row['grid_signal']}")
    print(f"{'='*60}")

    # Print ground truth first for easy comparison after running
    print_ground_truth(session_row)

    # Load model once; shared across all three steps
    tokenizer, model = load_model(MODEL_NAME)

    results = {
        "session_id": int(session_row["session_id"]),
        "scenario": session_row["scenario"],
        "ground_truth": {
            "transcript": json.loads(session_row["negotiation_transcript"]),
            "outcome": session_row["outcome"],
            "user_satisfaction_score": session_row["user_satisfaction_score"],
            "grid_stress_reduction_pct": session_row["grid_stress_reduction_pct"],
        }
    }

    # ── Step 1a ──────────────────────────────────────────────────
    hist_1a = run_1a(session_row, tokenizer, model)
    results["step1a"] = {
        "design": "minimal prompt, last-message-only context",
        "transcript": hist_1a,
    }

    # ── Step 1b ──────────────────────────────────────────────────
    hist_1b = run_1b(session_row, tokenizer, model)
    results["step1b"] = {
        "design": "rich prompt (persona/goal/constraint/strategy), full history",
        "transcript": hist_1b,
    }

    # ── Step 1c: three EV personas ─────────────────────────────────────
    results["step1c"] = {}
    for persona_name in EV_PERSONAS:
        hist = run_1c(session_row, persona_name, tokenizer, model)
        results["step1c"][persona_name] = {
            "ev_persona": EV_PERSONAS[persona_name],
            "transcript": hist,
        }

    # Save complete results
    out_path = os.path.join(OUTPUT_DIR, "step1_all_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n\n[Saved] {out_path}")
    print("[Done]  Step 1 complete. Please analyze the output before proceeding to Step 2.")


if __name__ == "__main__":
    main()
