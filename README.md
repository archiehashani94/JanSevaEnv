---

title: JanSevaEnv
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
-------------

# 🏛️ JanSevaEnv — OpenEnv Environment

JanSevaEnv is a real-world simulation environment designed for training and evaluating AI agents on public grievance resolution tasks related to government welfare schemes.

---

## 🎯 Objective

The agent must:

* Understand a grievance
* Ask relevant diagnostic questions
* Identify the correct root cause
* Suggest the correct resolution

---

## 🧠 Environment Design

This environment follows the **OpenEnv specification**.

---

## 🔁 Core Endpoints

### 1. `/reset`

Initializes a new episode.

**Request:**

```json
{
  "task_id": "task1",
  "case_id": "optional"
}
```

**Response:**
Returns initial observation.

---

### 2. `/step`

Agent takes an action.

**Request:**

```json
{
  "action_type": "ask_question",
  "question_id": "Q01"
}
```

OR

```json
{
  "action_type": "submit_diagnosis",
  "cause_id": "C01",
  "resolution_id": "R01"
}
```

**Response:**

* observation
* reward
* done flag
* info

---

### 3. `/state`

Returns full internal state of the environment.

---

## 👁️ Observation Space

Each step returns:

* `case_id`
* `grievance_text`
* `scheme`
* `step_number`
* `max_steps`
* `qa_history`
* `available_questions`
* `available_causes`
* `available_resolutions`
* `done`

---

## 🎮 Action Space

### 1. Ask Question

```json
{
  "action_type": "ask_question",
  "question_id": "QXX"
}
```

### 2. Submit Diagnosis

```json
{
  "action_type": "submit_diagnosis",
  "cause_id": "CXX",
  "resolution_id": "RXX"
}
```

---

## 🏆 Tasks

The environment includes **3 difficulty levels**:

| Task  | Difficulty | Description                  |
| ----- | ---------- | ---------------------------- |
| task1 | Easy       | Simple grievance cases       |
| task2 | Medium     | Moderate complexity          |
| task3 | Hard       | Complex multi-step reasoning |

---

## 🎯 Reward Function

* Partial rewards for useful questions
* Final reward based on:

  * Correct cause
  * Correct resolution

Reward range:

```
0.0 → 1.0
```

---

## ⚙️ Setup Instructions (Local)

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

---

## 🐳 Docker Support

This project is deployed using Docker on Hugging Face Spaces.

---

## 📊 Evaluation

* Agents interact using `/reset` and `/step`
* Performance is measured using cumulative reward
* Final score reflects diagnostic accuracy

---

## 🚀 Deployment

Deployed on Hugging Face Spaces using Docker SDK.

---

## 📌 Notes

* Fully OpenEnv-compliant
* Uses typed Pydantic models
* Designed for real-world AI evaluation

---
