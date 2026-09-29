"""Cleaning Planner Agent: Formulates governed cleaning proposals with risk levels."""

from __future__ import annotations


from agents.base import BaseAgent
from agents.schemas import AgentResult
from workflow.approvals import RiskLevel
from workflow.state import Proposal, WorkflowStage, WorkflowState


class CleaningPlannerAgent(BaseAgent):
    name = "CleaningPlannerAgent"
    description = "Analyzes quality findings and formulates explicit, governed proposals without applying mutations."

    def run(self, state: WorkflowState) -> AgentResult:
        df = state.active_dataset
        quality = state.quality_result
        profile = state.profile_result
        proposals: list[Proposal] = []

        # Identify protected columns (IDs, targets, user-specified)
        protected_cols = set()
        for f in state.intake_findings:
            if f.get("category") == "Identifiers":
                # Extract identified columns if any
                for col in df.columns:
                    if col.lower() in f.get("detail", "").lower():
                        protected_cols.add(col)

        # 1. Proposal: Protect identified ID columns if any
        if protected_cols:
            p_protect = Proposal.create(
                agent_name=self.name,
                action_type="protect_column",
                title="Protect Identifier Columns",
                description=f"Protect identifier columns {list(protected_cols)} from destructive value modification.",
                affected_columns=list(protected_cols),
                rows_affected=0,
                risk_level=RiskLevel.LOW,
                parameters={"protected_columns": list(protected_cols)},
                before_summary=f"{len(protected_cols)} identifier column(s) unprotected.",
                expected_after_summary=f"Mark {list(protected_cols)} as immutable during subsequent cleaning.",
            )
            proposals.append(p_protect)

        # 2. Proposal: Remove exact duplicate rows
        if quality and quality.duplicate_row_count > 0:
            dup_pct = (quality.duplicate_row_count / len(df) * 100) if len(df) else 0.0
            risk = RiskLevel.MEDIUM if dup_pct >= 5.0 else RiskLevel.LOW
            p_dedup = Proposal.create(
                agent_name=self.name,
                action_type="remove_duplicates",
                title="Remove Exact Duplicate Rows",
                description=f"Drop {quality.duplicate_row_count} duplicate row(s) keeping the first occurrence.",
                affected_columns=list(df.columns),
                rows_affected=quality.duplicate_row_count,
                risk_level=risk,
                parameters={"subset": None, "keep": "first"},
                before_summary=f"{len(df)} total rows ({quality.duplicate_row_count} duplicates).",
                expected_after_summary=f"{len(df) - quality.duplicate_row_count} unique rows.",
            )
            proposals.append(p_dedup)

        # 3. Proposal: Trim surrounding whitespace in text columns
        text_cols = [
            c
            for c in df.select_dtypes(include=["object", "string"]).columns
            if c not in protected_cols
        ]
        cols_with_whitespace = []
        rows_with_ws = 0
        for col in text_cols:
            series = df[col].dropna()
            str_series = series[series.map(lambda x: isinstance(x, str))]
            if not str_series.empty:
                ws_count = int(str_series.ne(str_series.str.strip()).sum())
                if ws_count > 0:
                    cols_with_whitespace.append(col)
                    rows_with_ws += ws_count

        if cols_with_whitespace:
            p_trim = Proposal.create(
                agent_name=self.name,
                action_type="trim_text",
                title="Trim Surrounding Whitespace",
                description=f"Strip leading and trailing whitespace from string columns {cols_with_whitespace}.",
                affected_columns=cols_with_whitespace,
                rows_affected=rows_with_ws,
                risk_level=RiskLevel.LOW,
                parameters={"columns": cols_with_whitespace},
                before_summary=f"{rows_with_ws} cell(s) contain extraneous surrounding whitespace.",
                expected_after_summary="All surrounding whitespace removed.",
            )
            proposals.append(p_trim)

        # 4. Proposal: Handle Missing Numeric Values
        if profile:
            for col in profile.numerical_columns:
                if col in protected_cols:
                    continue
                series = df[col]
                missing = int(series.isna().sum())
                if missing > 0:
                    pct = missing / len(df) * 100
                    # Never fill all-null columns
                    if pct >= 100.0:
                        continue
                    risk = (
                        RiskLevel.HIGH
                        if pct > 30
                        else (RiskLevel.MEDIUM if pct > 10 else RiskLevel.LOW)
                    )
                    p_num = Proposal.create(
                        agent_name=self.name,
                        action_type="fill_numeric_missing",
                        title=f"Impute Missing Numeric: {col}",
                        description=f"Impute {missing} missing value(s) in '{col}' using median strategy.",
                        affected_columns=[col],
                        rows_affected=missing,
                        risk_level=risk,
                        parameters={"strategy": "median", "column": col},
                        before_summary=f"{missing} missing cell(s) ({pct:.1f}%).",
                        expected_after_summary=f"Missing values filled with median ({series.median():.3g}).",
                    )
                    proposals.append(p_num)

        # 5. Proposal: Handle Missing Categorical Values
        if profile:
            for col in profile.categorical_columns:
                if col in protected_cols:
                    continue
                series = df[col]
                missing = int(series.isna().sum())
                if missing > 0:
                    pct = missing / len(df) * 100
                    if pct >= 100.0:
                        continue
                    p_cat = Proposal.create(
                        agent_name=self.name,
                        action_type="fill_categorical_missing",
                        title=f"Impute Missing Categorical: {col}",
                        description=f"Impute {missing} missing value(s) in '{col}' using mode (most frequent value).",
                        affected_columns=[col],
                        rows_affected=missing,
                        risk_level=RiskLevel.MEDIUM,
                        parameters={"strategy": "mode", "column": col},
                        before_summary=f"{missing} missing cell(s) ({pct:.1f}%).",
                        expected_after_summary="Missing values filled with mode.",
                    )
                    proposals.append(p_cat)

        # 6. Proposal: Cap extreme outliers (High / Medium risk)
        if quality and quality.outlier_columns:
            for col, count in quality.outlier_columns.items():
                if col in protected_cols:
                    continue
                p_outlier = Proposal.create(
                    agent_name=self.name,
                    action_type="cap_outliers",
                    title=f"Cap Extreme Outliers: {col}",
                    description=f"Cap {count} outlier value(s) in '{col}' to 1.5*IQR bounds.",
                    affected_columns=[col],
                    rows_affected=count,
                    risk_level=RiskLevel.HIGH,
                    parameters={"column": col, "iqr_multiplier": 1.5},
                    before_summary=f"{count} value(s) outside 1.5*IQR threshold.",
                    expected_after_summary="Extreme values clamped to lower/upper IQR limits.",
                )
                proposals.append(p_outlier)

        # 7. Proposal: Drop 100% empty columns if any (High risk)
        if profile:
            empty_cols = [c.name for c in profile.columns if c.missing_pct >= 100.0]
            if empty_cols:
                p_drop = Proposal.create(
                    agent_name=self.name,
                    action_type="drop_column",
                    title="Drop Empty Columns (100% Missing)",
                    description=f"Permanently remove all-null columns: {empty_cols}.",
                    affected_columns=empty_cols,
                    rows_affected=len(df),
                    risk_level=RiskLevel.HIGH,
                    parameters={"columns": empty_cols},
                    before_summary=f"Columns {empty_cols} contain zero data rows.",
                    expected_after_summary=f"Columns {empty_cols} removed from schema.",
                )
                proposals.append(p_drop)

        # Add all proposals to state (which also creates pending ApprovalRequests)
        for p in proposals:
            state.add_proposal(p)

        next_stage = (
            WorkflowStage.WAITING_FOR_CLEANING_APPROVAL
            if proposals
            else WorkflowStage.VISUALIZATION_AND_EDA
        )

        return AgentResult(
            agent_name=self.name,
            success=True,
            message=f"Formulated {len(proposals)} governed cleaning proposal(s).",
            proposals=proposals,
            next_stage=next_stage,
        )
