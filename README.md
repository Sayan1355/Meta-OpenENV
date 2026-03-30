# 🏥 Smart Hospital Resource Management Environment — v2

A **production-quality, OpenEnv-compatible** reinforcement learning environment that simulates real-world hospital resource management with typed data models, emergency patient handling, and improved reward shaping.

---

## 📁 Project Structure

```
hospital-rl-env/
├── env/
│   ├── __init__.py
│   ├── models.py              # Pydantic typed models (Patient, Doctor, State, Action)
│   └── hospital_env.py        # Core environment (HospitalEnv v2)
├── graders/
│   ├── __init__.py
│   └── grader.py              # Easy / Medium / Hard graders
├── run.py                     # Baseline smart agent + grader evaluation
├── openenv.yaml               # OpenEnv configuration
├── requirements.txt           # Dependencies (pydantic)
└── README.md                  # This file
```

---

## 🧠 System Design

The environment models a hospital emergency department over discrete time steps. An AI agent observes the current queue of patients and doctor availability, then decides which doctor to assign to which patient.

### Core components

| Component | Description |
|---|---|
| `Patient` | Pydantic model: id, severity (0–1), wait_time, is_emergency, treated, lost |
| `Doctor` | Pydantic model: id, available, cooldown |
| `HospitalState` | Full observable state: time, patients[], doctors[], stats{} |
| `AssignAction` | Structured action: `{type, patient_id, doctor_id}` |

### Episode parameters

| Parameter | Default | Description |
|---|---|---|
| `max_steps` | 50 | Steps per episode |
| `num_doctors` | 3 | Number of doctors |
| `initial_patients` | 5 | Patients at start |
| `arrival_rate` | 0.6 | P(new patient each step) |
| `emergency_rate` | 0.2 | P(new patient is emergency) |
| `treatment_cooldown` | 2 | Doctor busy steps after treating |
| `max_wait_before_lost` | 15 | Steps before patient is lost |

---

## 📊 Observation Space

Each step the agent receives a dict:

```python
{
    "time": 12,
    "patients": [
        {"id": 3, "severity": 0.85, "wait_time": 4, "is_emergency": True, "treated": False, "lost": False},
        {"id": 7, "severity": 0.42, "wait_time": 1, "is_emergency": False, "treated": False, "lost": False},
    ],
    "doctors": [
        {"id": 1, "available": True,  "cooldown": 0},
        {"id": 2, "available": False, "cooldown": 1},
        {"id": 3, "available": True,  "cooldown": 0},
    ],
    "stats": {
        "total_patients_seen": 15,
        "total_treated": 10,
        "total_emergency_treated": 3,
        "untreated_count": 2,
        "avg_wait_time": 2.5,
        "total_severity_treated": 6.42,
        "total_patients_lost": 0,
        "total_emergency_delay_penalties": 1,
        "total_invalid_actions": 0,
    }
}
```

---

## 🎮 Action Space

Actions are **structured dicts**:

| Action | Format |
|---|---|
| Assign a doctor to a patient | `{"type": "assign", "patient_id": 3, "doctor_id": 1}` |
| Do nothing | `{"type": "wait"}` |

Invalid actions (wrong patient/doctor ID, unavailable doctor) result in a **−2.0 penalty**.

---

## 🚨 Emergency Handling

- ~20% of arriving patients are flagged as **emergencies**
- Emergencies have higher base severity (≥ 0.6)
- Treating an emergency yields **double reward** (+10 × severity vs +5)
- If an emergency waits more than **5 steps**, a **−5.0 penalty** is applied every step
- If any patient waits more than **15 steps**, they are **lost** (−20.0 penalty)

---

## 🎯 Reward Strategy

| Event | Reward |
|---|---|
| Treat normal patient | `+5 × severity` |
| Treat emergency patient | `+10 × severity` |
| Cumulative wait penalty | `−0.2 × sum(wait_times)` |
| Emergency delay (wait > 5) | `−5.0` per emergency per step |
| Idle doctors (with patients waiting) | `−1.0` per idle doctor |
| Invalid action | `−2.0` |
| Patient lost (wait > 15) | `−20.0` |
| Queue fully cleared | `+3.0` bonus |

---

## 🧪 Graders

| Grader | Metric | Weight | Passing |
|---|---|---|---|
| `grade_easy` | % of patients treated | — | ≥ 0.6 |
| `grade_medium` | Treated % − emergency delay penalty | — | ≥ 0.5 |
| `grade_hard` | 40% efficiency + 30% wait score + 30% emergency handling | Composite | ≥ 0.4 |

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python3 run.py
```

This runs a **smart baseline agent** that prioritises emergencies → highest severity, assigns to the first available doctor, and prints step-by-step results with grader scores.

### Run graders standalone

```bash
python3 -m graders.grader
```

---

## 🤖 Writing Your Own Agent

```python
from env.hospital_env import HospitalEnv

env = HospitalEnv(seed=42)
state = env.reset()
done = False
total_reward = 0.0

while not done:
    patients = state["patients"]
    doctors = state["doctors"]

    # Your policy here
    if patients:
        best = max(patients, key=lambda p: (p["is_emergency"], p["severity"]))
        avail = [d for d in doctors if d["available"]]
        if avail:
            action = {"type": "assign", "patient_id": best["id"], "doctor_id": avail[0]["id"]}
        else:
            action = {"type": "wait"}
    else:
        action = {"type": "wait"}

    state, reward, done, info = env.step(action)
    total_reward += reward

print(f"Total reward: {total_reward}")
```

---

## ⚙️ Configuration (openenv.yaml)

The `openenv.yaml` file describes the environment entry point, typed model definitions, available tasks (easy / medium / hard), and maps each task to its grading function.

---

## 📜 License

This project is provided as-is for educational and research purposes.
