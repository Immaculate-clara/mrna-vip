"""Pydantic output schema for the structured report (FR-4.1)."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field

from src.config import DISCLAIMER


class ClinicalSafetyReport(BaseModel):
    """Exactly five fields. Strict mode: a string is not silently coerced to a bool/float,
    and unknown extra fields are rejected."""

    model_config = ConfigDict(extra="forbid", strict=True)

    Eligible_For_Vaccine: bool = Field(
        ..., description="Whether the patient is eligible given risk and guidelines")
    Recommended_Dose_mg: float = Field(
        ..., ge=0,
        description="Recommended dose in MILLIGRAMS (1 mg = 1000 mcg); 0 when not eligible")
    Reasoning_Steps: List[str] = Field(
        ..., min_length=1, description="Ordered, human-readable reasoning behind the decision")
    FDA_Safety_Warnings_Cited: List[str] = Field(
        ..., description="Exact (verbatim) guideline snippets the recommendation relies on")
    Disclaimer: str = Field(
        default=DISCLAIMER, min_length=1, description="Must always be present in the final report")
