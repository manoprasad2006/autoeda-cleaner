"""Data Intake Agent: Validates dataset integrity and produces structural findings."""

from __future__ import annotations

import re

from agents.base import BaseAgent
from agents.schemas import AgentResult
from tools.registry import registry
from workflow.events import ActorType, EventType
from workflow.state import WorkflowStage, WorkflowState


class IntakeAgent(BaseAgent):
    name = "DataIntakeAgent"
    description = "Inspects schema, detects anomalies, suspicious IDs, empty columns, and date-like fields without modifying data."

    def run(self, state: WorkflowState) -> AgentResult:
        df = state.active_dataset
        findings: list[dict] = []
        warnings: list[str] = []

        # 1. Profile the dataset using the registered tool
        profile = registry.execute("profile_dataset", df)
        state.profile_result = profile

        # 2. Check duplicate column names
        cols = list(df.columns)
        if len(cols) != len(set(cols)):
            dups = [c for c in cols if cols.count(c) > 1]
            warnings.append(f"Duplicate column names detected: {set(dups)}")
            findings.append(
                {
                    "category": "Schema",
                    "severity": "high",
                    "detail": f"Duplicate columns: {dups}",
                }
            )

        # 3. Detect empty columns (100% missing)
        empty_cols = [c.name for c in profile.columns if c.missing_pct >= 100.0]
        if empty_cols:
            warnings.append(f"Columns with 100% missing values detected: {empty_cols}")
            findings.append(
                {
                    "category": "Missingness",
                    "severity": "medium",
                    "detail": f"Empty columns: {empty_cols}",
                }
            )

        # 4. Detect suspicious identifiers (e.g. id, uuid, key, ssn, customer_id, index)
        suspicious_ids = []
        for col in df.columns:
            col_clean = str(col).lower().replace("_", "").replace("-", "")
            if any(
                term in col_clean
                for term in ("id", "uuid", "guid", "ssn", "hash", "key", "token")
            ):
                suspicious_ids.append(col)
            elif df[col].nunique() == len(df) and len(df) > 20:
                suspicious_ids.append(col)
        if suspicious_ids:
            unique_ids = list(set(suspicious_ids))
            findings.append(
                {
                    "category": "Identifiers",
                    "severity": "info",
                    "detail": f"Potential identifier columns detected: {unique_ids}. Protect these from transformation.",
                }
            )

        # 5. Detect nested values (dict, list, set) in object columns
        nested_cols = []
        for col in df.select_dtypes(include=["object", "string"]).columns:
            non_null = df[col].dropna()
            if non_null.map(lambda v: isinstance(v, (list, dict, set))).any():
                nested_cols.append(col)
        if nested_cols:
            warnings.append(
                f"Columns containing nested data structures (lists/dicts) detected: {nested_cols}"
            )
            findings.append(
                {
                    "category": "Data Structure",
                    "severity": "high",
                    "detail": f"Nested values in: {nested_cols}",
                }
            )

        # 6. Detect date-like string columns
        date_like_cols = []
        for col in profile.categorical_columns:
            series = df[col].dropna().astype(str).str.strip()
            if len(series) > 0:
                sample = series.iloc[:50]
                date_patterns = [
                    r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}",
                    r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}",
                ]
                matches = sum(
                    any(re.match(p, val) for p in date_patterns) for val in sample
                )
                if matches / len(sample) >= 0.7:
                    date_like_cols.append(col)
        if date_like_cols:
            findings.append(
                {
                    "category": "Types",
                    "severity": "info",
                    "detail": f"Candidate date columns stored as text: {date_like_cols}. Propose datetime parsing.",
                }
            )

        state.intake_findings = findings
        state.intake_warnings = warnings

        # Record audit event
        event = state.record_audit(
            actor_type=ActorType.AGENT,
            actor_name=self.name,
            event_type=EventType.INTAKE_COMPLETED,
            message=f"Intake completed with {len(findings)} findings and {len(warnings)} warnings.",
            metadata={"findings_count": len(findings), "warnings": warnings},
        )

        return AgentResult(
            agent_name=self.name,
            success=True,
            message=f"Intake completed. {len(findings)} structural findings detected.",
            events=[event],
            next_stage=WorkflowStage.QUALITY,
        )
