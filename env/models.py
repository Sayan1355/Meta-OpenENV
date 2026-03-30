"""
Typed data models for the Smart Hospital Resource Management Environment.

Uses Pydantic for runtime validation, serialisation, and documentation.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ------------------------------------------------------------------
# Domain Models
# ------------------------------------------------------------------

class Patient(BaseModel):
    """A patient waiting to be treated in the hospital."""

    id: int = Field(..., ge=1, description="Unique patient identifier")
    severity: float = Field(
        ..., ge=0.0, le=1.0,
        description="Normalised severity score (0 = minor, 1 = critical)"
    )
    wait_time: int = Field(default=0, ge=0, description="Steps spent waiting")
    is_emergency: bool = Field(
        default=False,
        description="Whether this patient arrived as an emergency case"
    )
    treated: bool = Field(default=False, description="Whether the patient has been treated")
    lost: bool = Field(default=False, description="Whether the patient was lost (waited too long)")


class Doctor(BaseModel):
    """A doctor available (or on cooldown) in the hospital."""

    id: int = Field(..., ge=1, description="Unique doctor identifier")
    available: bool = Field(default=True, description="Whether the doctor can treat a patient")
    cooldown: int = Field(default=0, ge=0, description="Steps remaining before available again")


# ------------------------------------------------------------------
# Observation / State
# ------------------------------------------------------------------

class HospitalState(BaseModel):
    """Full observable state returned by the environment each step."""

    time: int = Field(..., ge=0, description="Current discrete time step")
    patients: list[Patient] = Field(
        default_factory=list,
        description="List of untreated patients currently in the queue"
    )
    doctors: list[Doctor] = Field(
        default_factory=list,
        description="Current status of all doctors"
    )
    stats: dict = Field(
        default_factory=dict,
        description="Aggregate episode statistics"
    )


# ------------------------------------------------------------------
# Action
# ------------------------------------------------------------------

class AssignAction(BaseModel):
    """Structured action: assign a specific doctor to a specific patient."""

    type: str = Field(
        default="assign",
        pattern=r"^(assign|wait)$",
        description="Action type — 'assign' to treat or 'wait' to idle"
    )
    patient_id: Optional[int] = Field(
        default=None,
        description="ID of the patient to treat (required when type='assign')"
    )
    doctor_id: Optional[int] = Field(
        default=None,
        description="ID of the doctor to assign (required when type='assign')"
    )


# ------------------------------------------------------------------
# Step result
# ------------------------------------------------------------------

class StepResult(BaseModel):
    """Result returned by env.step()."""

    state: HospitalState
    reward: float
    done: bool
    info: dict = Field(default_factory=dict)
