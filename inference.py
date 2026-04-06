"""
inference.py
Baseline inference script for JanSevaEnv.

Runs an LLM agent against all three tasks using the OpenAI-compatible client.
Emits structured stdout logs in [START] / [STEP] / [END] format for automated scoring.

Environment variables required:
  API_BASE_URL   The base URL of the LLM API (e.g. https://api-inference.huggingface.co/v1)
  MODEL_NAME     The model identifier (e.g. meta-llama/Llama-3.1-8B-Instruct)
  HF_TOKEN       Your Hugging Face API token (used as the API key)

Usage:
  python inference.py
  python inference.py --tasks task1 task2 task3
  python inference.py --cases-per-task 1   # faster, 1 case per task
"""

import json
import os
import sys
import argparse
import traceback
from datetime import datetime, timezone

from openai import OpenAI

# ---------------------------------------------------------------------------
# Import environment directly (no HTTP server needed for inference)
# ---------------------------------------------------------------------------
from app.environment import JanSevaEnv
from app.models import AskQuestionAction, SubmitDiagnosisAction

# ---------------------------------------------------------------------------
# Client setup — reads from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ---------------------------------------------------------------------------
# Logging helpers — strict [START] / [STEP] / [END] format
# ---------------------------------------------------------------------------

def log_start(task_id: str, case_id: str, episode: int):
    record = {
        "task_id": task_id,
        "case_id": case_id,
        "episode": episode,
        "model": MODEL_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    print(f"[START] {json.dumps(record)}", flush=True)


def log_step(step: int, action: dict, reward: float, cumulative_reward: float, done: bool, info: dict):
    record = {
        "step": step,
        "action": action,
        "reward": round(reward, 4),
        "cumulative_reward": round(cumulative_reward, 4),
        "done": done,
        "info": {k: v for k, v in info.items() if k in ("answer", "cause_correct", "resolution_correct", "episode_score", "error")},
    }
    print(f"[STEP] {json.dumps(record)}", flush=True)


def log_end(task_id: str, case_id: str, episode_score: float, steps_used: int, cause_correct: bool, resolution_correct: bool):
    record = {
        "task_id": task_id,
        "case_id": case_id,
        "episode_score": round(episode_score, 4),
        "steps_used": steps_used,
        "cause_correct": cause_correct,
        "resolution_correct": resolution_correct,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    print(f"[END] {json.dumps(record)}", flush=True)


# ---------------------------------------------------------------------------
# Agent prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a welfare grievance resolution expert for Indian government schemes.
Your job is to identify the root cause of a citizen's grievance by asking targeted diagnostic questions, then submit the correct resolution.

Rules:
1. Ask ONE question per turn by choosing a question_id from the available_questions list.
2. When you have enough information, submit your diagnosis with a cause_id and resolution_id.
3. Be efficient — fewer questions is better.
4. Respond with valid JSON only. No extra text, no markdown, no explanations outside the JSON.

Response format for asking a question:
{"action_type": "ask_question", "question_id": "Q06"}

Response format for submitting diagnosis:
{"action_type": "submit_diagnosis", "cause_id": "aadhaar_not_seeded", "resolution_id": "seed_aadhaar", "reasoning": "Aadhaar not seeded confirmed by Q06 and Q44"}
"""


def build_user_prompt(obs) -> str:
    """Build the user message from the current observation."""
    qa_lines = ""
    if obs.qa_history:
        qa_lines = "\n".join(
            f"  Q [{pair.question_id}] {pair.question_text}\n  A: {pair.answer}"
            for pair in obs.qa_history
        )
    else:
        qa_lines = "  (No questions asked yet)"

    questions_text = "\n".join(
        f"  {qid}: {text}" for qid, text in list(obs.available_questions.items())[:30]
    )

    causes_text = "\n".join(
        f"  {c['id']}: {c['label']}" for c in obs.available_causes
    )

    resolutions_text = "\n".join(
        f"  {r['id']}: {r['label']}" for r in obs.available_resolutions
    )

    return f"""GRIEVANCE CASE: {obs.case_id}
Scheme: {obs.scheme}
Grievance: {obs.grievance_text}

Steps used: {obs.step_number} / {obs.max_steps} (steps remaining: {obs.max_steps - obs.step_number})

--- INVESTIGATION HISTORY ---
{qa_lines}

--- AVAILABLE QUESTIONS (choose one question_id) ---
{questions_text}

--- AVAILABLE CAUSES (choose one cause_id for diagnosis) ---
{causes_text}

--- AVAILABLE RESOLUTIONS (choose one resolution_id for diagnosis) ---
{resolutions_text}

Respond with JSON only."""


# ---------------------------------------------------------------------------
# LLM call with JSON parsing and fallback
# ---------------------------------------------------------------------------

def call_llm(messages: list[dict], max_retries: int = 3) -> dict:
    """
    Call the LLM and parse the JSON response.
    Retries up to max_retries times on parse failure.
    Returns a fallback ask_question action if all retries fail.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=256,
                temperature=0.0,
            )
            content = response.choices[0].message.content.strip()

            # Strip markdown code fences if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            parsed = json.loads(content)
            return parsed

        except json.JSONDecodeError as e:
            last_error = e
            # Add a correction message on retry
            if attempt < max_retries - 1:
                messages = messages + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": "Your response was not valid JSON. Respond with valid JSON only, no markdown."},
                ]
        except Exception as e:
            last_error = e
            break

    # Fallback: ask the first available question
    print(f"[WARN] LLM parse failed after {max_retries} attempts: {last_error}", file=sys.stderr)
    return {"action_type": "ask_question", "question_id": "Q34"}


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_episode(env: JanSevaEnv, task_id: str, case_id: str, episode_num: int) -> dict:
    """
    Run one full episode and return result summary.
    """
    log_start(task_id, case_id, episode_num)

    obs = env.reset(task_id=task_id, case_id=case_id)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    episode_score = 0.0
    steps_used = 0
    cause_correct = False
    resolution_correct = False

    while not obs.done:
        # Force diagnosis if only 2 steps remain
        force_diagnosis = (obs.max_steps - obs.step_number) <= 2

        user_msg = build_user_prompt(obs)
        if force_diagnosis:
            user_msg += "\n\nIMPORTANT: You must submit your diagnosis now (action_type: submit_diagnosis). Steps are almost exhausted."

        messages.append({"role": "user", "content": user_msg})
        action_dict = call_llm(messages)

        # Parse and execute action
        action_type = action_dict.get("action_type", "ask_question")
        try:
            if action_type == "ask_question":
                question_id = action_dict.get("question_id", "Q34")
                # Validate question is available, fallback to Q34 if not
                if question_id not in obs.available_questions:
                    question_id = next(iter(obs.available_questions))
                action = AskQuestionAction(question_id=question_id)
                action_log = {"action_type": "ask_question", "question_id": question_id}

            elif action_type == "submit_diagnosis":
                cause_id = action_dict.get("cause_id", "")
                resolution_id = action_dict.get("resolution_id", "")
                action = SubmitDiagnosisAction(
                    cause_id=cause_id,
                    resolution_id=resolution_id,
                    reasoning=action_dict.get("reasoning"),
                )
                action_log = {"action_type": "submit_diagnosis", "cause_id": cause_id, "resolution_id": resolution_id}

            else:
                # Unknown action type — ask a safe fallback question
                action = AskQuestionAction(question_id="Q34")
                action_log = {"action_type": "ask_question", "question_id": "Q34"}

            result = env.step(action)

        except Exception as e:
            print(f"[WARN] Action execution error: {e}", file=sys.stderr)
            result = env.step(AskQuestionAction(question_id="Q34"))
            action_log = {"action_type": "ask_question", "question_id": "Q34"}

        # Add assistant turn to message history
        messages.append({"role": "assistant", "content": json.dumps(action_log)})

        log_step(
            step=result.observation.step_number,
            action=action_log,
            reward=result.reward.step_reward,
            cumulative_reward=result.reward.cumulative_reward,
            done=result.done,
            info=result.info,
        )

        obs = result.observation
        steps_used = obs.step_number

        if result.done:
            episode_score = result.reward.episode_score or 0.0
            cause_correct = result.info.get("cause_correct", False)
            resolution_correct = result.info.get("resolution_correct", False)
            break

    log_end(
        task_id=task_id,
        case_id=case_id,
        episode_score=episode_score,
        steps_used=steps_used,
        cause_correct=cause_correct,
        resolution_correct=resolution_correct,
    )

    return {
        "task_id": task_id,
        "case_id": case_id,
        "episode_score": episode_score,
        "steps_used": steps_used,
        "cause_correct": cause_correct,
        "resolution_correct": resolution_correct,
    }


# ---------------------------------------------------------------------------
# Case bank — maps task_id to available case IDs
# ---------------------------------------------------------------------------

TASK_CASES = {
    "task1": ["T1_001", "T1_002", "T1_003", "T1_004", "T1_005"],
    "task2": ["T2_001", "T2_002", "T2_003", "T2_004", "T2_005"],
    "task3": ["T3_001", "T3_002", "T3_003", "T3_004", "T3_005"],
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="JanSevaEnv baseline inference script")
    parser.add_argument(
        "--tasks", nargs="+",
        default=["task1", "task2", "task3"],
        choices=["task1", "task2", "task3"],
        help="Which tasks to run (default: all three)",
    )
    parser.add_argument(
        "--cases-per-task", type=int, default=5,
        help="Number of cases to run per task (default: 5 = all cases)",
    )
    args = parser.parse_args()

    env = JanSevaEnv()
    all_results = []
    episode_counter = 0

    for task_id in args.tasks:
        case_ids = TASK_CASES[task_id][: args.cases_per_task]
        task_scores = []

        for case_id in case_ids:
            episode_counter += 1
            try:
                result = run_episode(env, task_id, case_id, episode_counter)
                all_results.append(result)
                task_scores.append(result["episode_score"])
            except Exception as e:
                print(f"[ERROR] Episode failed for {task_id}/{case_id}: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                all_results.append({
                    "task_id": task_id,
                    "case_id": case_id,
                    "episode_score": 0.0,
                    "steps_used": 0,
                    "cause_correct": False,
                    "resolution_correct": False,
                })

    # Final summary
    summary = {}
    for task_id in args.tasks:
        task_results = [r for r in all_results if r["task_id"] == task_id]
        if task_results:
            scores = [r["episode_score"] for r in task_results]
            summary[task_id] = {
                "avg_score": round(sum(scores) / len(scores), 4),
                "min_score": round(min(scores), 4),
                "max_score": round(max(scores), 4),
                "cases_run": len(scores),
                "cause_accuracy": round(sum(r["cause_correct"] for r in task_results) / len(task_results), 4),
            }

    overall_scores = [r["episode_score"] for r in all_results]
    overall_avg = round(sum(overall_scores) / len(overall_scores), 4) if overall_scores else 0.0

    print(f"[SUMMARY] {json.dumps({'tasks': summary, 'overall_avg_score': overall_avg, 'total_episodes': len(all_results)})}", flush=True)


if __name__ == "__main__":
    main()
