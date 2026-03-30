"""
Grading functions for the Smart Hospital Resource Management Environment (v2).

Each grader runs a full episode with the given agent function and returns
a normalised score in [0, 1].
"""

from env.hospital_env import HospitalEnv


def _run_episode(agent_fn, env=None, seed=42):
    """
    Run a full episode using *agent_fn* and return (env, total_reward).

    Parameters
    ----------
    agent_fn : callable
        A function that receives (state_dict, env) and returns an action dict.
    env : HospitalEnv or None
        If None a default environment is created.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple : (env, total_reward)
    """
    if env is None:
        env = HospitalEnv(seed=seed)

    state = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        action = agent_fn(state, env)
        state, reward, done, info = env.step(action)
        total_reward += reward

    return env, total_reward


# ------------------------------------------------------------------
# EASY — percentage of patients treated
# ------------------------------------------------------------------

def grade_easy(agent_fn, seed=42):
    """
    Grade based on the **percentage of patients treated** out of all
    patients that appeared during the episode.

    Scoring:
        treated_ratio = total_treated / total_patients_seen
        score = treated_ratio   (already in [0, 1])

    Parameters
    ----------
    agent_fn : callable   (state, env) -> action dict

    Returns
    -------
    float : score in [0, 1]
    """
    env, _ = _run_episode(agent_fn, seed=seed)
    total_seen = len(env.patients)
    if total_seen == 0:
        return 1.0
    score = env.total_treated / total_seen
    return round(min(max(score, 0.0), 1.0), 4)


# ------------------------------------------------------------------
# MEDIUM — penalise emergency delays
# ------------------------------------------------------------------

def grade_medium(agent_fn, seed=42):
    """
    Grade based on treatment rate **penalised by emergency delays**.

    Scoring:
        base      = treated_ratio
        penalty   = 0.05 * total_emergency_delay_penalties
        penalty  += 0.10 * total_patients_lost
        score     = max(0, base - penalty)

    Parameters
    ----------
    agent_fn : callable   (state, env) -> action dict

    Returns
    -------
    float : score in [0, 1]
    """
    env, _ = _run_episode(agent_fn, seed=seed)
    total_seen = len(env.patients)
    if total_seen == 0:
        return 1.0

    base = env.total_treated / total_seen
    penalty = (
        0.05 * env.total_emergency_delay_penalties
        + 0.10 * env.total_patients_lost
    )
    score = base - penalty
    return round(min(max(score, 0.0), 1.0), 4)


# ------------------------------------------------------------------
# HARD — composite: efficiency + avg wait + emergency handling
# ------------------------------------------------------------------

def grade_hard(agent_fn, seed=42):
    """
    Composite grade combining:

        1. **Efficiency**        (40 %) — severity_treated / max_possible_severity
        2. **Wait time**         (30 %) — 1 - normalised avg wait
        3. **Emergency handling** (30 %) — emergency_treated / total_emergencies
                                          minus delay penalty

    Parameters
    ----------
    agent_fn : callable   (state, env) -> action dict

    Returns
    -------
    float : score in [0, 1]
    """
    env, _ = _run_episode(agent_fn, seed=seed)

    all_patients = env.patients
    total_seen = len(all_patients)
    if total_seen == 0:
        return 1.0

    # --- 1. Efficiency (40 %) ---
    max_severity = total_seen * 1.0  # severity is normalised to [0, 1]
    efficiency = env.total_severity_treated / max_severity if max_severity else 0.0

    # --- 2. Wait time (30 %) ---
    avg_wait = env.total_wait_accumulated / total_seen if total_seen else 0.0
    # Normalise: 0 wait → 1.0, ≥10 wait → 0.0
    wait_score = max(0.0, 1.0 - avg_wait / 10.0)

    # --- 3. Emergency handling (30 %) ---
    total_emergencies = sum(1 for p in all_patients if p.is_emergency)
    if total_emergencies > 0:
        emg_treated_ratio = env.total_emergency_treated / total_emergencies
        emg_delay_penalty = 0.1 * env.total_emergency_delay_penalties
        emg_score = max(0.0, emg_treated_ratio - emg_delay_penalty)
    else:
        emg_score = 1.0  # no emergencies → perfect score on this axis

    # --- Weighted composite ---
    score = 0.40 * efficiency + 0.30 * wait_score + 0.30 * emg_score
    return round(min(max(score, 0.0), 1.0), 4)


# ------------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------------

if __name__ == "__main__":
    def always_treat(state, env):
        """Naive agent: assign first untreated patient to first available doctor."""
        patients = state.get("patients", [])
        doctors = state.get("doctors", [])
        if patients and doctors:
            # Pick highest severity
            best = max(patients, key=lambda p: (p["is_emergency"], p["severity"]))
            for d in doctors:
                if d["available"]:
                    return {"type": "assign", "patient_id": best["id"], "doctor_id": d["id"]}
        return {"type": "wait"}

    def always_wait(state, env):
        return {"type": "wait"}

    print("=== Grader Self-Test (v2) ===")
    print(f"  grade_easy  (treat): {grade_easy(always_treat)}")
    print(f"  grade_easy  (wait) : {grade_easy(always_wait)}")
    print(f"  grade_medium(treat): {grade_medium(always_treat)}")
    print(f"  grade_medium(wait) : {grade_medium(always_wait)}")
    print(f"  grade_hard  (treat): {grade_hard(always_treat)}")
    print(f"  grade_hard  (wait) : {grade_hard(always_wait)}")
