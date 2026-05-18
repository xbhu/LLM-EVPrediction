"""
Use Case 7 - Step 1: Zero-shot Multi-Agent EV Charging Negotiation
==================================================================
Step 1a: 用数据集的 minimal prompt，每个 agent 只看上一条消息
Step 1b: 用设计过的 rich prompt，每个 agent 看完整对话历史
Step 1c: 固定 Station/Grid，给 EV Driver 换三种 persona，观察行为差异

运行：python sourcecode/uc7_step1_all.py
输出：outputs/usecase7_zeroshot/
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
TARGET_SESSION_ID = 1       # peak_hour_conflict，用这一条做所有对比
MAX_NEW_TOKENS    = 150     # 每个 agent 每轮最多生成的 token 数
NUM_ROUNDS        = 6       # 谈判总轮数
TURN_ORDER        = ["EV_User", "Station", "Grid", "EV_User", "Station", "Grid"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# STEP 1a：数据集原始 minimal prompt
# 设计逻辑：每个 agent 只看上一条消息（{incoming_message}），
#           模拟最简单的 reactive agent——只响应当前刺激
# =============================================================================
def build_minimal_system_prompt(agent_role: str, scenario_description: str) -> str:
    """
    直接用数据集的 agent_prompt_template，填入 agent_role 和 scenario。
    注意：这个模板没有区分三个角色的 goal/constraint，
         所有 agent 的目标都是同一句话："maximize your own objective"
    """
    return (
        f"You are the {agent_role} agent in an EV charging negotiation.\n"
        f"Scenario: {scenario_description}\n"
        f"Your goal: maximize your own objective while reaching a feasible agreement.\n"
        f"Respond concisely and propose a concrete action or counteroffer."
    )

# =============================================================================
# STEP 1b：设计过的 rich role prompts
# 设计逻辑：四要素 persona / goal / constraint / strategy 明确区分三个角色，
#           每个 agent 看完整对话历史（full history）
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
# STEP 1c：三种 EV Driver persona（Station 和 Grid 沿用 RICH_PROMPTS）
# 设计逻辑：同一场景，只改 EV Driver 的 persona 和 strategy，
#           观察用户异质性如何影响谈判走向
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
# INFERENCE：单个 agent 生成一轮回应
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
    # Qwen3 不支持独立 system role 时用 user 拼接；enable_thinking=False
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
# STEP 1a 谈判：每个 agent 只看上一条消息
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
        # 只传入上一条消息（或空）
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
# STEP 1b 谈判：每个 agent 看完整对话历史
# =============================================================================
def run_1b(session_row: pd.Series, tokenizer, model) -> list:
    print("\n" + "="*60)
    print("STEP 1b | Rich Prompt | Full history context")
    print("="*60)

    history = []
    for agent_role in TURN_ORDER:
        sys_prompt = RICH_PROMPTS[agent_role]

        # 构建完整历史文本
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
# STEP 1c 谈判：固定 Station/Grid，换 EV Driver persona
# =============================================================================
def run_1c(session_row: pd.Series, persona_name: str, tokenizer, model) -> list:
    print("\n" + "="*60)
    print(f"STEP 1c | EV Persona: {persona_name}")
    print("="*60)

    # 1c 沿用 rich prompt 的完整历史设计
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
# 打印 Ground Truth（便于对比）
# =============================================================================
def print_ground_truth(session_row: pd.Series):
    print("\n" + "="*60)
    print("GROUND TRUTH（数据集原始对话）")
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

    # Ground truth 先打出来，方便跑完后对比
    print_ground_truth(session_row)

    # 加载模型（只加载一次，三个 step 共用）
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

    # ── Step 1c：三种 EV persona ──────────────────────────────────
    results["step1c"] = {}
    for persona_name in EV_PERSONAS:
        hist = run_1c(session_row, persona_name, tokenizer, model)
        results["step1c"][persona_name] = {
            "ev_persona": EV_PERSONAS[persona_name],
            "transcript": hist,
        }

    # 保存完整结果
    out_path = os.path.join(OUTPUT_DIR, "step1_all_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n\n[Saved] {out_path}")
    print("[Done]  Step 1 complete. 请分析输出后我们进入 Step 2。")


if __name__ == "__main__":
    main()
