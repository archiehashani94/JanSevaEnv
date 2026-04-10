"""
inference.py — JanSevaEnv OpenEnv inference script.

Environment variables (set these before running):
──────────────────────────────────────────────────
  ENV_BASE_URL   URL of your JanSevaEnv server
                 e.g. https://your-username-janseva.hf.space
                 (default: http://localhost:8084)

  API_BASE_URL   LLM API endpoint
                 e.g. https://router.huggingface.co/v1
                 (default: https://router.huggingface.co/v1)

  MODEL_NAME     LLM model identifier
                 e.g. Qwen/Qwen2.5-72B-Instruct
                 (default: Qwen/Qwen2.5-72B-Instruct)

  HF_TOKEN       Your Hugging Face token  ← PUT YOUR TOKEN HERE
                 Get it from: https://huggingface.co/settings/tokens

  TASK_ID        task1 / task2 / task3   (default: task1)
  CASE_ID        Specific case for reproducible runs (optional)
"""

import json
import os
import subprocess
import sys
import textwrap
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

# Auto-install openai if not available (evaluator may run in a clean env)
try:
    from openai import OpenAI
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai>=1.0.0", "-q"])
    from openai import OpenAI

import uuid

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  ← edit the defaults below OR set environment variables
# ─────────────────────────────────────────────────────────────────────────────

# Your JanSevaEnv server (HF Space URL or localhost)
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://apurvaadeshpande-JanSevaEnv.hf.space").rstrip("/")

# LLM endpoint + model — evaluators will override these via env vars
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")

# HF token — injected by evaluators via HF_TOKEN env var (no default, per spec)
HF_TOKEN = os.getenv("HF_TOKEN")

TASK_ID  = os.getenv("TASK_ID",  "task1")
CASE_ID  = os.getenv("CASE_ID",  "")       # leave blank for random case
BENCHMARK = "janseva-env"
MAX_STEPS = 10
SESSION_ID = os.getenv("SESSION_ID", str(uuid.uuid4()))

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING  (exact hackathon format)
# ─────────────────────────────────────────────────────────────────────────────

def log_start(task, env, model):
    # type: (str, str, str) -> None
    print("[START] task={} env={} model={}".format(task, env, model), flush=True)


def log_step(step, action, reward, done, error):
    # type: (int, str, float, bool, Optional[str]) -> None
    print(
        "[STEP] step={} action={} reward={:.2f} done={} error={}".format(
            step, action, reward, str(done).lower(), error if error else "null"
        ),
        flush=True,
    )


def log_end(success, steps, score, rewards):
    # type: (bool, int, float, List[float]) -> None
    print(
        "[END] success={} steps={} score={:.2f} rewards={}".format(
            str(success).lower(),
            steps,
            score,
            ",".join("{:.2f}".format(r) for r in rewards),
        ),
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENV HTTP HELPERS  (calls JanSevaEnv server, not the LLM)
# ─────────────────────────────────────────────────────────────────────────────

def _env_headers():
    # type: () -> Dict[str, str]
    return {
        "Content-Type": "application/json", 
        "Accept": "application/json",
        "X-Session-ID": SESSION_ID
    }


def env_get(path):
    # type: (str) -> Any
    url = ENV_BASE_URL + path
    req = urllib.request.Request(url, headers=_env_headers())
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def env_post(path, payload):
    # type: (str, dict) -> Any
    url = ENV_BASE_URL + path
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=_env_headers(), method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# TAXONOMY
# ─────────────────────────────────────────────────────────────────────────────

def load_cause_map():
    # type: () -> Dict[str, dict]
    data = env_get("/taxonomy/causes")
    result = {}
    for c in data.get("causes", []):
        result[c["id"]] = {
            "label":                c.get("label", c["id"]),
            "resolution_id":        c.get("resolution_id", ""),
            "signal_questions":     c.get("signal_questions", []),
            "diagnostic_questions": c.get("diagnostic_questions", []),
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# LLM AGENT
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert welfare grievance investigator for Indian government schemes
    (PM-KISAN, OAP, WAP, DAP, MGNREGA, NFSA-PDS).

    Your task: identify the ROOT CAUSE of a beneficiary's grievance by asking
    diagnostic questions, then submit a diagnosis.

    At each step you will receive:
    - The grievance description
    - Available questions (as JSON: {question_id: question_text})
    - Available causes (as JSON list)
    - Q&A history so far

    Rules:
    1. If you want to ask a question, reply with EXACTLY:
       ask_question:<question_id>
       Example: ask_question:Q06

    2. If you are confident of the root cause, reply with EXACTLY:
       submit_diagnosis:<cause_id>:<resolution_id>
       Example: submit_diagnosis:aadhaar_not_seeded:seed_aadhaar

    3. Ask the most informative question first. Do not ask redundant questions.
    4. After 3-5 questions you should have enough information to diagnose.
    5. Reply with ONLY the action string — no explanation, no extra text.
""").strip()


def build_user_prompt(obs, cause_map):
    # type: (dict, Dict[str, dict]) -> str
    available_questions = obs.get("available_questions", {})
    available_causes = obs.get("available_causes", [])
    qa_history = obs.get("qa_history", [])

    history_lines = []
    for qa in qa_history:
        history_lines.append("  {}: {}".format(qa["question_id"], qa["answer"]))
    history_str = "\n".join(history_lines) if history_lines else "  (none yet)"

    causes_str = "\n".join(
        "  {} -> resolution: {}".format(c["id"], c.get("resolution_id", ""))
        for c in available_causes[:15]  # top 15 to stay within token budget
    )

    questions_str = "\n".join(
        "  {}: {}".format(qid, qtxt)
        for qid, qtxt in list(available_questions.items())[:20]
    )

    return textwrap.dedent("""
        GRIEVANCE: {grievance}
        SCHEME: {scheme}
        STEP: {step}/{max_steps}

        Q&A HISTORY:
        {history}

        AVAILABLE QUESTIONS:
        {questions}

        AVAILABLE CAUSES (cause_id -> resolution_id):
        {causes}

        What is your next action?
    """).format(
        grievance=obs.get("grievance_text", ""),
        scheme=obs.get("scheme", ""),
        step=obs.get("step_number", 0),
        max_steps=obs.get("max_steps", 10),
        history=history_str,
        questions=questions_str,
        causes=causes_str,
    ).strip()


def get_llm_action(client, obs, cause_map, step):
    # type: (OpenAI, dict, Dict[str, dict], int) -> str
    """Ask the LLM what to do next. Returns raw action string."""
    user_prompt = build_user_prompt(obs, cause_map)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=60,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # strip markdown code fences if model wraps output
        text = text.strip("`").strip()
        return text if text else _fallback_action(obs, cause_map)
    except Exception as exc:
        print("[DEBUG] LLM call failed: {}".format(exc), file=sys.stderr)
        return _fallback_action(obs, cause_map)


def _fallback_action(obs, cause_map):
    # type: (dict, Dict[str, dict]) -> str
    """
    Signal-greedy fallback used when LLM is unavailable.
    Picks the question with highest coverage across cause signal sets.
    """
    available_q = list(obs.get("available_questions", {}).keys())
    asked = set(qa["question_id"] for qa in obs.get("qa_history", []))
    remaining = [q for q in available_q if q not in asked]

    if not remaining:
        # diagnose with best signal-scored cause
        available_causes = obs.get("available_causes", [])
        asked_list = list(asked)
        for _, cid in _score_causes(asked_list, cause_map):
            for c in available_causes:
                if c["id"] == cid:
                    return "submit_diagnosis:{}:{}".format(cid, c.get("resolution_id", ""))
        if available_causes:
            c = available_causes[0]
            return "submit_diagnosis:{}:{}".format(c["id"], c.get("resolution_id", ""))

    coverage = {}
    for qid in remaining:
        score = 0
        for info in cause_map.values():
            if qid in info["signal_questions"]:
                score += 2
            elif qid in info["diagnostic_questions"]:
                score += 1
        coverage[qid] = score

    best_q = max(remaining, key=lambda q: (coverage.get(q, 0), q))
    return "ask_question:{}".format(best_q)


def _score_causes(asked_qids, cause_map):
    # type: (List[str], Dict[str, dict]) -> List[tuple]
    asked = set(asked_qids)
    scores = []
    for cid, info in cause_map.items():
        s = len(set(info["signal_questions"]) & asked) * 2.0
        s += len(set(info["diagnostic_questions"]) & asked) * 1.0
        scores.append((s, cid))
    scores.sort(key=lambda x: -x[0])
    return scores


def parse_action(action_str, obs):
    # type: (str, dict) -> Optional[dict]
    """
    Parse LLM output into an API payload.
    Returns None if the string is unrecognisable.
    """
    s = action_str.strip()

    if s.startswith("ask_question:"):
        qid = s.split(":", 1)[1].strip()
        if qid in obs.get("available_questions", {}):
            return {"action_type": "ask_question", "question_id": qid}

    if s.startswith("submit_diagnosis:"):
        parts = s.split(":")
        if len(parts) >= 3:
            cause_id      = parts[1].strip()
            resolution_id = parts[2].strip()
            return {
                "action_type":   "submit_diagnosis",
                "cause_id":      cause_id,
                "resolution_id": resolution_id,
            }

    return None


# ─────────────────────────────────────────────────────────────────────────────
# EPISODE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(client, cause_map):
    # type: (OpenAI, Dict[str, dict]) -> None

    rewards    = []   # type: List[float]
    steps_taken = 0
    score      = 0.0
    success    = False
    info       = {}   # type: dict

    # ── reset ─────────────────────────────────────────────────────────────────
    reset_payload = {"task_id": TASK_ID}
    if CASE_ID:
        reset_payload["case_id"] = CASE_ID

    obs = env_post("/reset", reset_payload)
    case_id = obs["case_id"]

    log_start(task="{}-{}".format(TASK_ID, case_id), env=BENCHMARK, model=MODEL_NAME)

    done = obs.get("done", False)

    try:
        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            # get action from LLM (or fallback)
            action_str = get_llm_action(client, obs, cause_map, step)

            # if model hasn't diagnosed yet but we're at last step, force diagnose
            if step == MAX_STEPS and not action_str.startswith("submit_diagnosis:"):
                action_str = _fallback_action(
                    dict(obs, available_questions={}),  # empty questions forces diagnose
                    cause_map,
                )

            payload = parse_action(action_str, obs)
            error_msg = None

            if payload is None:
                # LLM gave garbage — use fallback
                error_msg = "unparseable_action: {}".format(action_str[:60])
                action_str = _fallback_action(obs, cause_map)
                payload = parse_action(action_str, obs)

            # still None → hard fallback: ask first available question
            if payload is None:
                available_q = list(obs.get("available_questions", {}).keys())
                asked = set(qa["question_id"] for qa in obs.get("qa_history", []))
                remaining = [q for q in available_q if q not in asked]
                if remaining:
                    payload = {"action_type": "ask_question", "question_id": remaining[0]}
                    action_str = "ask_question:{}".format(remaining[0])
                else:
                    c = obs.get("available_causes", [{}])[0]
                    payload = {
                        "action_type":   "submit_diagnosis",
                        "cause_id":      c.get("id", ""),
                        "resolution_id": c.get("resolution_id", ""),
                    }
                    action_str = "submit_diagnosis:{}:{}".format(
                        c.get("id", ""), c.get("resolution_id", "")
                    )

            # call env
            result = env_post("/step", payload)

            reward_info = result.get("reward", {})
            step_reward = float(reward_info.get("step_reward", 0.0))
            done        = result.get("done", False)
            obs         = result["observation"]
            info        = result.get("info", {})

            rewards.append(step_reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_str,
                reward=step_reward,
                done=done,
                error=error_msg,
            )

            if done:
                break

    finally:
        episode_score = info.get("episode_score") if info.get("episode_score") is not None else 0.0
        score   = float(episode_score)
        success = bool(info.get("cause_correct", False))
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # type: () -> None

    if not HF_TOKEN:
        print(
            "WARNING: HF_TOKEN is not set. LLM calls will fail. "
            "Set it with: export HF_TOKEN=hf_...",
            file=sys.stderr,
        )

    # health check
    try:
        health = env_get("/health")
        assert health.get("status") == "healthy"
    except Exception as e:
        print("ERROR: JanSevaEnv server unreachable at {}: {}".format(ENV_BASE_URL, e),
              file=sys.stderr)
        sys.exit(1)

    # OpenAI client pointed at HuggingFace router (or any compatible endpoint)
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN or "no-token",
    )

    cause_map = load_cause_map()
    run_episode(client, cause_map)


if __name__ == "__main__":
    main()
