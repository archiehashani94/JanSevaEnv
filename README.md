# JanSevaEnv

An OpenEnv-compliant reinforcement learning environment that simulates **Indian welfare and pension grievance resolution**.

An AI agent learns to identify the correct root cause of a citizen's welfare complaint — from 35 predefined issues — by asking targeted diagnostic questions in a multi-turn conversation, then suggests the correct resolution. Evaluation is fully rule-based with no LLM judges.

---

## Why This Environment

Millions of Indians fail to receive entitled welfare benefits (PM-KISAN, MGNREGA wages, pensions) due to administrative, banking, and KYC issues. Frontline workers and helpline agents must navigate ~35 distinct root causes across multiple schemes to resolve grievances. This environment trains and evaluates AI agents on that exact task — a genuine, high-stakes real-world problem that no existing OpenEnv covers.

---

## Environment Description

| Property | Value |
|---|---|
| Interaction mode | Multi-turn text (ask questions → submit diagnosis) |
| Tasks | 3 (easy → medium → hard) |
| Schemes covered | PM-KISAN, MGNREGA, OAP, WAP, DAP, PDS, EPFO, NPS, and more |
| Root causes | 35 across 8 categories |
| Question bank | 50 diagnostic questions |
| Evaluation | Fully deterministic, rule-based (0.0–1.0 score) |
| Episode termination | Agent submits diagnosis OR max steps exhausted |

### What the agent does

1. Receives a grievance description from a beneficiary (e.g. "I haven't received PM-KISAN for 6 months")
2. Asks diagnostic questions from a predefined question bank
3. Receives case-specific answers from the environment
4. Submits a root cause + resolution when confident

---

## Action Space

Two action types:

### `ask_question`
```json
{
  "action_type": "ask_question",
  "question_id": "Q06"
}
```
Selects one question from the scheme-filtered question bank (Q01–Q50). The environment returns the case-specific answer.

### `submit_diagnosis`
```json
{
  "action_type": "submit_diagnosis",
  "cause_id": "aadhaar_not_seeded",
  "resolution_id": "seed_aadhaar",
  "reasoning": "optional explanation"
}
```
Ends the episode. The grader scores the submission (0.0–1.0).

---

## Observation Space

Each step returns:

| Field | Type | Description |
|---|---|---|
| `case_id` | string | Unique case identifier |
| `grievance_text` | string | Beneficiary's original complaint |
| `scheme` | string | Welfare scheme (PM-KISAN, MGNREGA, OAP…) |
| `step_number` | int | Current step |
| `max_steps` | int | Step budget (10 / 15 / 20) |
| `qa_history` | list | All Q&A pairs so far |
| `available_questions` | dict | Scheme-filtered question bank `{id: text}` |
| `available_causes` | list | All 35 causes `{id, label}` |
| `available_resolutions` | list | All 35 resolutions `{id, label}` |
| `done` | bool | Episode ended flag |

---

## Reward Function

### Step reward (on `ask_question`)
| Condition | Reward |
|---|---|
| Signal question for true cause | +0.05 |
| Diagnostic question for true cause | +0.02 |
| Irrelevant question | -0.01 |
| Repeated question | 0.00 |
| Invalid question ID | -0.02 |

### Episode score (on `submit_diagnosis`)
| Outcome | Score |
|---|---|
| Correct cause + resolution + efficient + asked signal question | up to **1.00** |
| Correct cause + resolution (any efficiency) | **0.70** |
| Correct cause only + asked signal question | **0.55** |
| Correct cause only | **0.50** |
| Same category, wrong specific cause | **0.20** |
| Asked signal question but wrong cause | **0.10** |
| Completely wrong | **0.00** |

Efficiency bonus scales linearly: `0.20 × (1 − steps_used / max_steps)` when both cause and resolution are correct.

---

## Tasks

### Task 1 — Easy: PM-KISAN Payment Not Received
- **Scheme:** PM-KISAN (Pradhan Mantri Kisan Samman Nidhi)
- **Max steps:** 10
- **Cases:** 5
- **Root causes:** `aadhaar_not_seeded`, `bank_account_not_linked`, `incorrect_bank_details`, `land_records_mismatch`, `npci_mapping_error`
- **Strategy:** 1–3 targeted questions are sufficient. Each cause has a clear confirming signal question.
- **Expected baseline score:** ~0.75

### Task 2 — Medium: Old Age / Widow Pension Stopped
- **Scheme:** OAP / WAP
- **Max steps:** 15
- **Cases:** 5
- **Root causes:** `life_certificate_pending`, `pensioner_marked_deceased`, `aadhaar_mismatch`, `bank_account_frozen`, `district_level_pending`
- **Strategy:** 4–7 questions needed. Some cases have misleading signals (e.g. life certificate submitted but pensioner marked deceased anyway).
- **Expected baseline score:** ~0.60

### Task 3 — Hard: MGNREGA Wages Not Received
- **Scheme:** MGNREGA
- **Max steps:** 20
- **Cases:** 5
- **Root causes:** `fto_generation_pending`, `muster_roll_error`, `work_not_measured`, `aadhaar_inactive`, `rejection_by_bank`
- **Strategy:** Systematic pipeline traversal required (Work Approval → Measurement → Muster Roll → FTO → Bank). Some cases include deliberate misleading signals (FTO generated but Aadhaar inactive; bank appears active but IFSC changed).
- **Expected baseline score:** ~0.45

---

## Root Cause Taxonomy

35 causes across 8 categories. Full definitions in `app/data/taxonomy.json`.

| Category | Causes |
|---|---|
| banking | `bank_account_not_linked`, `incorrect_bank_details`, `bank_account_frozen`, `npci_mapping_error`, `duplicate_account`, `dbt_not_enabled` |
| kyc | `aadhaar_not_seeded`, `aadhaar_mismatch`, `aadhaar_inactive`, `kyc_pending`, `mobile_not_linked` |
| eligibility | `income_above_threshold`, `land_records_mismatch`, `age_not_eligible`, `not_registered`, `scheme_exclusion_criteria` |
| administrative | `application_under_review`, `data_entry_error`, `block_level_pending`, `district_level_pending`, `state_portal_sync_error` |
| pension | `pensioner_marked_deceased`, `dfc_not_submitted`, `life_certificate_pending`, `pension_amount_revised`, `co_pensioner_issue` |
| pds | `ration_card_not_updated`, `dealer_diversion`, `biometric_failure`, `quota_exhausted` |
| mgnrega | `work_not_measured`, `muster_roll_error`, `fto_generation_pending`, `rejection_by_bank` |
| misc | `system_technical_error` |

---

## Setup

### Local development

```bash
# Clone and install
git clone <repo-url>
cd JanSevaEnv
pip install -r requirements.txt

# Start the API server
uvicorn app.main:app --reload --port 7860

# Docs available at
open http://localhost:7860/docs
```

### Docker

```bash
docker build -t janseva-env .
docker run -p 7860:7860 janseva-env
```

### Run inference baseline

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your_hf_token_here"

python inference.py
# Run only 1 case per task for a quick check:
python inference.py --cases-per-task 1
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Environment info and endpoint index |
| GET | `/health` | Liveness check |
| GET | `/tasks` | List all task metadata |
| GET | `/tasks/{task_id}` | Single task metadata |
| POST | `/reset` | Start a new episode |
| POST | `/step` | Take one action |
| GET | `/state` | Current episode state |
| GET | `/taxonomy/causes` | All 35 root causes |
| GET | `/taxonomy/resolutions` | All 35 resolutions |
| GET | `/taxonomy/questions` | Full question bank (Q01–Q50) |
| GET | `/taxonomy/schemes` | Scheme definitions |

### Example: full episode via curl

```bash
# Start episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1"}'

# Ask a question
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "ask_question", "question_id": "Q06"}'

# Submit diagnosis
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "submit_diagnosis", "cause_id": "aadhaar_not_seeded", "resolution_id": "seed_aadhaar"}'
```

---

## Project Structure

```
JanSevaEnv/
├── app/
│   ├── main.py                  # FastAPI entry point
│   ├── environment.py           # JanSevaEnv: step() / reset() / state()
│   ├── models.py                # Pydantic models (Observation, Action, Reward, State)
│   ├── data/
│   │   ├── taxonomy.json        # 35 causes, 35 resolutions, 50 questions (source of truth)
│   │   └── cases.json           # 15 grievance cases (5 per task) with ground truth
│   ├── schemes/
│   │   ├── root_causes.py       # Load/query causes from taxonomy.json
│   │   ├── resolutions.py       # Load/query resolutions from taxonomy.json
│   │   └── policies.py          # Question bank, scheme definitions
│   ├── rewards/
│   │   └── reward_fn.py         # Step reward + episode grading (0.0–1.0)
│   ├── tasks/
│   │   ├── task1.py             # Easy task (PM-KISAN, 10 steps)
│   │   ├── task2.py             # Medium task (OAP/WAP, 15 steps)
│   │   └── task3.py             # Hard task (MGNREGA, 20 steps)
│   └── routers/
│       └── api.py               # FastAPI route definitions
├── inference.py                 # Baseline inference script
├── openenv.yaml                 # OpenEnv metadata
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Baseline Scores

Scores from the baseline inference script using `meta-llama/Llama-3.1-8B-Instruct`:

| Task | Difficulty | Avg Score | Cause Accuracy |
|---|---|---|---|
| Task 1 (PM-KISAN) | Easy | ~0.75 | ~85% |
| Task 2 (OAP/WAP) | Medium | ~0.60 | ~70% |
| Task 3 (MGNREGA) | Hard | ~0.45 | ~55% |
| **Overall** | — | **~0.60** | **~70%** |

Random agent baseline (no questions, random cause/resolution): ~0.02–0.05.

---

## Design Notes

- **No LLM judges:** Every score is computed deterministically from `taxonomy.json` + `cases.json`. Graders are reproducible across runs.
- **Single source of truth:** `taxonomy.json` owns all cause/resolution/question definitions. Task files, graders, and reward functions all read from it — nothing is hardcoded.
- **Partial progress signals:** The agent receives reward on every step, not just at episode end, making it suitable for RL training.
- **Scalable case bank:** New cases can be added to `cases.json` without changing any logic.
