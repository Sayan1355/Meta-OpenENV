#!/usr/bin/env python3
"""
run.py — Smart High-Performance Agent for Hospital Resource Management v2.

Strategy: Multi-factor scoring with death prevention overrides,
          severity maximisation, and emergency-first triage.

Usage:
    python3 run.py
"""

from env.hospital_env import HospitalEnv
from graders.grader import grade_easy, grade_medium, grade_hard


# ------------------------------------------------------------------
# Thresholds (must match env constants)
# ------------------------------------------------------------------

DEATH_THRESHOLD = 15
EMERGENCY_DELAY_THRESHOLD = 5
CRITICAL_ZONE = 4  # steps before death → absolute override


# ------------------------------------------------------------------
# Scoring helpers
# ------------------------------------------------------------------

def death_risk(wait_time: int) -> float:
    """Urgency that ramps up sharply as patient nears the lost threshold."""
    remaining = DEATH_THRESHOLD - wait_time
    if remaining <= 1:
        return 500.0   # IMMINENT — one step from death
    if remaining <= 2:
        return 300.0   # ALMOST DEAD
    if remaining <= CRITICAL_ZONE:
        return 150.0   # ENTERING CRITICAL ZONE
    if remaining <= 6:
        return 30.0    # HIGH RISK
    if remaining <= 9:
        return 8.0     # MODERATE RISK
    return 0.0


def emergency_delay_risk(patient: dict) -> float:
    """Extra urgency for emergencies approaching their delay-penalty threshold."""
    if not patient["is_emergency"]:
        return 0.0
    wait = patient["wait_time"]
    remaining = EMERGENCY_DELAY_THRESHOLD - wait
    if remaining <= 0:
        return 80.0    # ALREADY TAKING −5/STEP PENALTIES
    if remaining <= 1:
        return 40.0    # ABOUT TO START PENALTIES
    if remaining <= 2:
        return 20.0    # 2-STEP WARNING
    return 0.0


def score_patient(patient: dict) -> float:
    """
    Multi-factor scoring function.

    Components:
        severity × 5       — favour high-severity patients (boosts grader efficiency)
        emergency bonus     — +8 for emergency status
        wait urgency        — +0.3 per wait step (gentle ramp)
        death risk          — massive spike near death threshold
        emergency delay     — spike near emergency penalty threshold
    """
    severity = patient["severity"]
    wait = patient["wait_time"]
    is_emg = patient["is_emergency"]

    # Heavy severity weight → maximises severity_treated for hard grader
    base_score = severity * 5.0

    # Emergency bonus
    emg_bonus = 8.0 if is_emg else 0.0

    # Wait urgency (gentle so low-severity patients don't override high-severity ones too early)
    wait_urgency = wait * 0.3

    # Risk overrides
    d_risk = death_risk(wait)
    e_risk = emergency_delay_risk(patient)

    return base_score + emg_bonus + wait_urgency + d_risk + e_risk


def classify_patient(patient: dict) -> str:
    """Classify a patient's urgency level for logging."""
    wait = patient["wait_time"]
    is_emg = patient["is_emergency"]

    if DEATH_THRESHOLD - wait <= 2:
        return "💀 IMMINENT DEATH"
    if DEATH_THRESHOLD - wait <= CRITICAL_ZONE:
        return "💀 CRITICAL"
    if is_emg and wait >= EMERGENCY_DELAY_THRESHOLD:
        return "🚨 EMG DELAYED"
    if is_emg:
        return "🚨 EMERGENCY"
    if wait >= 8:
        return "⚠️  HIGH WAIT"
    return "   NORMAL"


# ------------------------------------------------------------------
# Smart agent
# ------------------------------------------------------------------

def smart_agent(state: dict, env) -> dict:
    """
    Multi-factor scoring agent with layered overrides.

    Decision flow:
      1. No patients or no doctors → wait
      2. CRITICAL override: any patient ≤ CRITICAL_ZONE steps from death
      3. EMERGENCY override: any emergency at or past delay threshold
      4. Otherwise: rank all patients by composite score (severity-heavy)
      5. Assign top-scored patient to first available doctor
    """
    patients = state.get("patients", [])
    doctors = state.get("doctors", [])

    if not patients:
        return {"type": "wait"}

    available_doctors = [d for d in doctors if d["available"]]
    if not available_doctors:
        return {"type": "wait"}

    # --- Override 1: DEATH PREVENTION ---
    critical = [
        p for p in patients
        if (DEATH_THRESHOLD - p["wait_time"]) <= CRITICAL_ZONE
    ]
    if critical:
        # Treat the one closest to death, break ties by severity (higher = more valuable)
        critical.sort(key=lambda p: (-p["wait_time"], -p["severity"]))
        return {
            "type": "assign",
            "patient_id": critical[0]["id"],
            "doctor_id": available_doctors[0]["id"],
        }

    # --- Override 2: EMERGENCY DELAY PREVENTION ---
    delayed_emg = [
        p for p in patients
        if p["is_emergency"] and p["wait_time"] >= EMERGENCY_DELAY_THRESHOLD - 1
    ]
    if delayed_emg:
        delayed_emg.sort(key=lambda p: (-p["wait_time"], -p["severity"]))
        return {
            "type": "assign",
            "patient_id": delayed_emg[0]["id"],
            "doctor_id": available_doctors[0]["id"],
        }

    # --- General: COMPOSITE SCORING (severity-heavy) ---
    scored = sorted(patients, key=lambda p: score_patient(p), reverse=True)
    return {
        "type": "assign",
        "patient_id": scored[0]["id"],
        "doctor_id": available_doctors[0]["id"],
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    env = HospitalEnv(max_steps=50, num_doctors=3, seed=42)
    state = env.reset()

    total_reward = 0.0
    step_num = 0
    done = False
    emergency_treated_count = 0

    print("=" * 74)
    print("  🏥 Smart Hospital v2 — Intelligent Agent Run")
    print("=" * 74)
    print()

    while not done:
        action = smart_agent(state, env)
        state, reward, done, info = env.step(action)
        total_reward += reward
        step_num += 1

        # --- Format step output ---
        action_desc = action["type"]
        if action["type"] == "assign":
            action_desc = f"assign P{action['patient_id']}→D{action['doctor_id']}"

        treated_info = ""
        if info.get("treated_patient"):
            p = info["treated_patient"]
            if p["is_emergency"]:
                emergency_treated_count += 1
            urgency = classify_patient(p)
            emg_tag = " 🚨" if p["is_emergency"] else ""
            treated_info = (
                f"\n         → Treated #{p['id']:>2d}"
                f"  sev={p['severity']:.2f}  wait={p['wait_time']}"
                f"  [{urgency}]{emg_tag}"
            )

        extra = ""
        if info.get("new_arrivals", 0) > 0:
            extra += "  (+1 new)"
        if info.get("invalid_action"):
            extra += "  ⚠ INVALID"
        lost = info.get("patients_lost", [])
        if lost:
            extra += f"  💀 {len(lost)} DIED!"

        print(
            f"  Step {step_num:>3d}  {action_desc:<22s}"
            f"  reward={reward:>+8.2f}"
            f"  queue={state['stats']['untreated_count']}"
            f"{extra}{treated_info}"
        )

    stats = state["stats"]
    print()
    print("=" * 74)
    print("  📊 FINAL RESULTS")
    print("=" * 74)
    print(f"  Total Reward              : {total_reward:>+.2f}")
    print(f"  Patients Treated          : {stats['total_treated']}")
    print(f"  Emergency Patients Treated: {stats['total_emergency_treated']}")
    print(f"  Deaths (patients lost)    : {stats['total_patients_lost']}")
    print(f"  Patients Still Waiting    : {stats['untreated_count']}")
    print(f"  Avg Wait (remaining)      : {stats['avg_wait_time']:.2f}")
    print(f"  Emergency Delay Penalties : {stats['total_emergency_delay_penalties']}")
    print(f"  Invalid Actions           : {stats['total_invalid_actions']}")
    print()
    if stats["total_patients_lost"] == 0:
        print("  ✅ ZERO DEATHS — All patients treated in time!")
    else:
        print(f"  ❌ {stats['total_patients_lost']} patient(s) lost.")
    print("=" * 74)

    # --- Run graders ---
    print()
    print("=" * 74)
    print("  🧪 Grader Scores (smart agent)")
    print("=" * 74)
    easy = grade_easy(smart_agent, seed=42)
    medium = grade_medium(smart_agent, seed=42)
    hard = grade_hard(smart_agent, seed=42)
    print(f"  Easy   (% treated)                   : {easy:.4f}")
    print(f"  Medium (treated − emergency penalty)  : {medium:.4f}")
    print(f"  Hard   (efficiency + wait + emg)       : {hard:.4f}")
    print()
    print("  ── vs Baseline ──")
    print(f"  Easy   : {easy:.4f}  (baseline: 0.9474)")
    print(f"  Medium : {medium:.4f}  (baseline: 0.8474)")
    print(f"  Hard   : {hard:.4f}  (baseline: 0.7492)")

    improved = all([
        easy >= 0.9474,
        medium >= 0.8474,
        hard >= 0.7492,
    ])
    print()
    if improved:
        print("  🎉 ALL SCORES IMPROVED OVER BASELINE!")
    if hard >= 0.80:
        print("  🏆 HARD GRADER TARGET (0.80+) ACHIEVED!")
    print("=" * 74)


if __name__ == "__main__":
    main()
