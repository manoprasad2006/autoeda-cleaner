"""Data Quality Agent: Computes quality score and detailed issue breakdown."""

from __future__ import annotations

from agents.base import BaseAgent
from agents.schemas import AgentResult
from tools.registry import registry
from workflow.events import ActorType, EventType
from workflow.state import WorkflowStage, WorkflowState


class QualityAgent(BaseAgent):
    name = "DataQualityAgent"
    description = "Profiles missing values, duplicates, mixed types, constant columns, high cardinality, and outliers."

    def run(self, state: WorkflowState) -> AgentResult:
        df = state.active_dataset

        # Ensure dataset profile is available
        if state.profile_result is None:
            state.profile_result = registry.execute("profile_dataset", df)
        profile = state.profile_result

        # Assess quality using registered tool
        quality = registry.execute("assess_quality", df, profile)
        state.quality_result = quality

        findings = []
        if quality.missing_cell_count > 0:
            findings.append(
                f"{quality.missing_cell_count} missing cell(s) across dataset."
            )
        if quality.duplicate_row_count > 0:
            findings.append(
                f"{quality.duplicate_row_count} duplicate row(s) identified."
            )
        if quality.inconsistent_columns:
            findings.append(
                f"Mixed python types in columns: {quality.inconsistent_columns}."
            )
        if quality.outlier_columns:
            findings.append(
                f"IQR outliers in numeric columns: {list(quality.outlier_columns.keys())}."
            )
        if profile.constant_columns:
            findings.append(
                f"Constant/zero-variance columns: {profile.constant_columns}."
            )
        if profile.high_cardinality_columns:
            findings.append(
                f"High cardinality text columns: {profile.high_cardinality_columns}."
            )

        event = state.record_audit(
            actor_type=ActorType.AGENT,
            actor_name=self.name,
            event_type=EventType.QUALITY_ANALYSIS_COMPLETED,
            message=f"Quality score: {quality.overall_score:.1f}/100. Issues detected: {len(findings)}.",
            metadata={
                "overall_score": quality.overall_score,
                "missing_cells": quality.missing_cell_count,
                "duplicates": quality.duplicate_row_count,
                "findings": findings,
            },
        )

        return AgentResult(
            agent_name=self.name,
            success=True,
            message=f"Quality assessment complete: {quality.overall_score:.1f}/100.",
            events=[event],
            next_stage=WorkflowStage.CLEANING_PLAN,
        )
