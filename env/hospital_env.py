"""
Smart Hospital Resource Management Environment — v2 (Production)

A reinforcement learning environment where an AI agent learns to:
  - Prioritize patients based on severity and emergency status
  - Reduce overall waiting time
  - Efficiently assign specific doctors to specific patients

Compatible with the OpenEnv specification.
Uses Pydantic typed models for all data structures.
"""

import random
import copy
from typing import Union

from env.models import (
    Patient,
    Doctor,
    HospitalState,
    AssignAction,
)


class HospitalEnv:
    """
    Smart Hospital Resource Management Environment (v2).

    Changes from v1
    ----------------
    - Pydantic-typed state, patients, doctors, actions.
    - Emergency patients (~20 % arrival chance per step).
    - Structured action space: {"type": "assign", "patient_id": ..., "doctor_id": ...}
      or {"type": "wait"}.
    - Improved reward shaping with emergency bonuses / penalties.
    - Invalid-action penalty.

    Observation (state) — HospitalState:
        time      : int
        patients  : list[Patient]   (untreated only)
        doctors   : list[Doctor]
        stats     : dict

    Actions:
        {"type": "assign", "patient_id": <id>, "doctor_id": <id>}
        {"type": "wait"}

    Reward design:
        Treat normal patient   : +5  × severity
        Treat emergency patient: +10 × severity
        Waiting penalty        : -0.2 × sum(wait_times)
        Emergency delay (>5)   : -5.0 per delayed emergency
        Idle doctor penalty    : -1.0 per idle doctor (when patients exist)
        Invalid action penalty : -2.0
        Patient lost (wait>15) : -20.0
        Queue cleared bonus    : +3.0
    """

    EMERGENCY_WAIT_THRESHOLD = 5  # steps before heavy penalty kicks in

    def __init__(
        self,
        max_steps: int = 50,
        num_doctors: int = 3,
        initial_patients: int = 5,
        arrival_rate: float = 0.6,
        emergency_rate: float = 0.2,
        treatment_cooldown: int = 2,
        max_wait_before_lost: int = 15,
        seed: int | None = None,
    ):
        self.max_steps = max_steps
        self.num_doctors = num_doctors
        self.initial_patients = initial_patients
        self.arrival_rate = arrival_rate
        self.emergency_rate = emergency_rate
        self.treatment_cooldown = treatment_cooldown
        self.max_wait_before_lost = max_wait_before_lost
        self.seed_value = seed

        # Runtime bookkeeping (set in reset)
        self._rng: random.Random = random.Random(seed)
        self._time: int = 0
        self._patients: list[Patient] = []
        self._doctors: list[Doctor] = []
        self._done: bool = False
        self._next_patient_id: int = 1
        self._total_treated: int = 0
        self._total_emergency_treated: int = 0
        self._total_wait_accumulated: int = 0
        self._total_severity_treated: float = 0.0
        self._total_emergency_delay_penalties: int = 0
        self._total_patients_lost: int = 0
        self._total_invalid_actions: int = 0
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_patient(self, is_emergency: bool = False) -> Patient:
        """Create a new patient with a unique ID."""
        severity = round(self._rng.uniform(0.1, 1.0), 2)
        if is_emergency:
            severity = round(max(severity, self._rng.uniform(0.6, 1.0)), 2)
        p = Patient(
            id=self._next_patient_id,
            severity=severity,
            is_emergency=is_emergency,
        )
        self._next_patient_id += 1
        return p

    def _untreated(self) -> list[Patient]:
        return [p for p in self._patients if not p.treated and not p.lost]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """Reset the environment and return the initial observation as a dict."""
        self._rng = random.Random(self.seed_value)
        self._time = 0
        self._done = False
        self._next_patient_id = 1
        self._total_treated = 0
        self._total_emergency_treated = 0
        self._total_wait_accumulated = 0
        self._total_severity_treated = 0.0
        self._total_emergency_delay_penalties = 0
        self._total_patients_lost = 0
        self._total_invalid_actions = 0
        self._history = []

        # Initial patients (some may be emergencies)
        self._patients = []
        for _ in range(self.initial_patients):
            is_emg = self._rng.random() < self.emergency_rate
            self._patients.append(self._make_patient(is_emergency=is_emg))

        # Doctors
        self._doctors = [Doctor(id=i + 1) for i in range(self.num_doctors)]

        return self.state()

    def step(self, action: Union[dict, AssignAction]) -> tuple[dict, float, bool, dict]:
        """
        Execute one time-step.

        Parameters
        ----------
        action : dict or AssignAction
            {"type": "assign", "patient_id": int, "doctor_id": int}
            or {"type": "wait"}

        Returns
        -------
        (state_dict, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode finished. Call reset().")

        # --- Normalise action into AssignAction ---
        if isinstance(action, dict):
            try:
                action_obj = AssignAction(**action)
            except Exception:
                action_obj = AssignAction(type="wait")
        elif isinstance(action, AssignAction):
            action_obj = action
        else:
            action_obj = AssignAction(type="wait")

        reward = 0.0
        info: dict = {
            "action": action_obj.model_dump(),
            "treated_patient": None,
            "new_arrivals": 0,
            "invalid_action": False,
            "patients_lost": [],
        }

        untreated = self._untreated()

        # --- 1. Update doctor cooldowns ---
        for doc in self._doctors:
            if not doc.available:
                doc.cooldown -= 1
                if doc.cooldown <= 0:
                    doc.available = True
                    doc.cooldown = 0

        # --- 2. Apply action ---
        if action_obj.type == "assign":
            # Validate patient_id
            target_patient = None
            for p in untreated:
                if p.id == action_obj.patient_id:
                    target_patient = p
                    break

            # Validate doctor_id
            target_doctor = None
            for d in self._doctors:
                if d.id == action_obj.doctor_id and d.available:
                    target_doctor = d
                    break

            if target_patient is not None and target_doctor is not None:
                # Successful treatment
                target_doctor.available = False
                target_doctor.cooldown = self.treatment_cooldown

                target_patient.treated = True
                self._total_treated += 1
                self._total_severity_treated += target_patient.severity

                if target_patient.is_emergency:
                    reward += 10.0 * target_patient.severity
                    self._total_emergency_treated += 1
                else:
                    reward += 5.0 * target_patient.severity

                info["treated_patient"] = target_patient.model_dump()
            else:
                # Invalid assignment
                reward -= 2.0
                info["invalid_action"] = True
                self._total_invalid_actions += 1

        elif action_obj.type == "wait":
            # Idle penalty per idle doctor when patients exist
            if untreated:
                idle_docs = sum(1 for d in self._doctors if d.available)
                reward -= 1.0 * idle_docs
        else:
            reward -= 2.0
            info["invalid_action"] = True
            self._total_invalid_actions += 1

        # --- 3. Increase waiting time for all untreated patients ---
        untreated_after = self._untreated()
        for p in untreated_after:
            p.wait_time += 1
            self._total_wait_accumulated += 1

        # --- 4. Waiting penalty proportional to cumulative wait ---
        current_wait_sum = sum(p.wait_time for p in untreated_after)
        reward -= 0.2 * current_wait_sum

        # --- 5. Emergency delay penalty ---
        for p in untreated_after:
            if p.is_emergency and p.wait_time > self.EMERGENCY_WAIT_THRESHOLD:
                reward -= 5.0
                self._total_emergency_delay_penalties += 1

        # --- 6. Patients lost (wait > threshold) ---
        for p in list(untreated_after):
            if p.wait_time > self.max_wait_before_lost:
                reward -= 20.0
                p.lost = True
                self._total_patients_lost += 1
                info["patients_lost"].append(p.model_dump())

        # --- 7. Queue cleared bonus ---
        remaining = self._untreated()
        if not remaining and self._total_treated > 0:
            reward += 3.0

        # --- 8. Stochastic patient arrivals ---
        arrivals = 0
        if self._rng.random() < self.arrival_rate:
            is_emg = self._rng.random() < self.emergency_rate
            self._patients.append(self._make_patient(is_emergency=is_emg))
            arrivals += 1
        info["new_arrivals"] = arrivals

        # --- 9. Advance time ---
        self._time += 1
        if self._time >= self.max_steps:
            self._done = True

        # --- 10. Record history ---
        self._history.append({
            "time": self._time,
            "action": action_obj.model_dump(),
            "reward": reward,
            "untreated_count": len(self._untreated()),
        })

        return self.state(), reward, self._done, info

    def state(self) -> dict:
        """Return the current observation as a plain dict (JSON-serialisable)."""
        untreated = self._untreated()
        return HospitalState(
            time=self._time,
            patients=untreated,
            doctors=list(self._doctors),
            stats={
                "total_patients_seen": len(self._patients),
                "total_treated": self._total_treated,
                "total_emergency_treated": self._total_emergency_treated,
                "untreated_count": len(untreated),
                "avg_wait_time": (
                    sum(p.wait_time for p in untreated) / len(untreated)
                    if untreated else 0.0
                ),
                "total_severity_treated": round(self._total_severity_treated, 4),
                "total_patients_lost": self._total_patients_lost,
                "total_emergency_delay_penalties": self._total_emergency_delay_penalties,
                "total_invalid_actions": self._total_invalid_actions,
            },
        ).model_dump()

    # ------------------------------------------------------------------
    # Convenience properties (used by graders)
    # ------------------------------------------------------------------

    @property
    def total_treated(self) -> int:
        return self._total_treated

    @property
    def total_emergency_treated(self) -> int:
        return self._total_emergency_treated

    @property
    def total_wait_accumulated(self) -> int:
        return self._total_wait_accumulated

    @property
    def total_severity_treated(self) -> float:
        return self._total_severity_treated

    @property
    def total_patients_lost(self) -> int:
        return self._total_patients_lost

    @property
    def total_emergency_delay_penalties(self) -> int:
        return self._total_emergency_delay_penalties

    @property
    def history(self) -> list[dict]:
        return copy.deepcopy(self._history)

    @property
    def patients(self) -> list[Patient]:
        return list(self._patients)

    @property
    def time(self) -> int:
        return self._time
