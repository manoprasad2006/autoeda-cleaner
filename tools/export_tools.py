"""Governed report generation and dataset export tools."""
from __future__ import annotations

from datetime import datetime, timezone
from html import escape
import io
import json
from typing import Any
import zipfile
import pandas as pd

from modules.exports import csv_bytes
from tools.registry import registry


@registry.register(
    name="generate_report",
    description="Compiles approved dataset findings, profiles, AI summaries, and audit trail into an HTML report.",
    requires_approval=False,
    destructive=False,
)
def generate_report_tool(state: Any) -> str:
    df = state.active_dataset
    filename = state.dataset_filename
    profile = state.profile_result
    quality = state.quality_result
    audit_events = state.audit_events

    rows_html = ""
    if profile:
        profile_df = pd.DataFrame(
            [
                {
                    "Column": c.name,
                    "Type": c.dtype,
                    "Missing %": c.missing_pct,
                    "Unique values": c.unique_count,
                    "Constant": c.is_constant,
                    "High cardinality": c.is_high_cardinality,
                }
                for c in profile.columns
            ]
        )
        rows_html = profile_df.to_html(index=False, escape=True)

    audit_rows = pd.DataFrame(
        [
            {
                "Timestamp": e.timestamp,
                "Actor": f"{e.actor_name} ({e.actor_type})",
                "Event": e.event_type,
                "Message": e.message,
            }
            for e in audit_events
        ]
    )
    audit_html = (
        audit_rows.to_html(index=False, escape=True)
        if not audit_rows.empty
        else "<p>No audit events recorded.</p>"
    )

    insights_block = ""
    if state.generated_insights:
        insights_block = f"""
        <h2>Governed AI Insights</h2>
        <div style="background:#eef2ff;border-left:4px solid #4f46e5;padding:16px;border-radius:8px;margin:16px 0;">
            <p><small style="color:#4f46e5;font-weight:bold;">✦ Generated with AI assistance · Verified by Human-in-the-Loop review</small></p>
            <div>{escape(state.generated_insights)}</div>
        </div>
        """

    q_score = f"{quality.overall_score:.1f}/100" if quality else "N/A"
    missing_cells = f"{quality.missing_cell_count:,}" if quality else "N/A"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>IntelliData Agentic Intelligence Report - {escape(filename)}</title>
<style>
body{{font:15px system-ui,-apple-system,sans-serif;color:#17233b;max-width:1100px;margin:40px auto;padding:24px;background:#f8fafc;line-height:1.5}}
h1{{font-size:32px;margin:8px 0}}h2{{font-size:20px;margin-top:32px;color:#0f172a;border-bottom:2px solid #e2e8f0;padding-bottom:8px}}
.hero{{background:linear-gradient(135deg,#0f172a,#1e293b);color:white;padding:32px;border-radius:16px}}
.kpis{{display:flex;gap:16px;flex-wrap:wrap;margin:24px 0}}
.kpis div{{background:white;padding:20px;border-radius:12px;border:1px solid #e2e8f0;min-width:160px;box-shadow:0 1px 3px rgba(0,0,0,0.05)}}
strong{{display:block;font-size:24px;color:#0f172a}}
table{{border-collapse:collapse;width:100%;background:white;border-radius:8px;overflow:hidden}}
td,th{{padding:10px 12px;border:1px solid #e2e8f0;text-align:left;font-size:14px}}
th{{background:#f1f5f9;font-weight:600}}
.table-wrapper{{overflow-x:auto;margin:16px 0;border:1px solid #e2e8f0;border-radius:8px}}
.badge{{display:inline-block;padding:4px 8px;border-radius:9999px;font-size:12px;font-weight:600;background:#dbeafe;color:#1e40af}}
</style>
</head>
<body>
<div class="hero">
    <span class="badge">INTELLIDATA AGENT PLATFORM</span>
    <h1>{escape(filename)}</h1>
    <p>Active Version: <strong>{escape(state.active_version_id)}</strong> | Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}</p>
</div>

<div class="kpis">
    <div><strong>{len(df):,}</strong>Records</div>
    <div><strong>{len(df.columns)}</strong>Columns</div>
    <div><strong>{q_score}</strong>Quality Score</div>
    <div><strong>{missing_cells}</strong>Missing Cells</div>
</div>

{insights_block}

<h2>Column Profiles</h2>
<div class="table-wrapper">{rows_html}</div>

<h2>Descriptive Statistics</h2>
<div class="table-wrapper">{df.describe(include="all").round(3).to_html(escape=True)}</div>

<h2>Governance & Audit Trail</h2>
<div class="table-wrapper">{audit_html}</div>

<footer style="margin-top:40px;color:#64748b;font-size:13px;">
    <p>✦ Generated with AI assistance · Governed by explicit Human-in-the-Loop approval · Original dataset preserved immutable.</p>
</footer>
</body>
</html>"""


@registry.register(
    name="export_dataset",
    description="Exports the approved active dataset package with full audit trail and report.",
    requires_approval=True,
    destructive=False,
)
def export_dataset_tool(
    state: Any,
    export_format: str = "zip",
) -> bytes:
    df = state.active_dataset
    filename = state.dataset_filename

    if export_format == "csv":
        return csv_bytes(df)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"approved_{filename}.csv", csv_bytes(df))
        report_content = generate_report_tool(state)
        archive.writestr("intelligence_report.html", report_content.encode("utf-8"))

        audit_df = pd.DataFrame(
            [
                {
                    "timestamp": e.timestamp,
                    "actor_type": e.actor_type,
                    "actor_name": e.actor_name,
                    "event_type": e.event_type,
                    "message": e.message,
                    "proposal_id": e.proposal_id,
                    "approval_id": e.approval_id,
                }
                for e in state.audit_events
            ]
        )
        archive.writestr("governance_audit.csv", csv_bytes(audit_df))

        proposals_data = [
            {
                "id": p.id,
                "agent_name": p.agent_name,
                "action_type": p.action_type,
                "title": p.title,
                "risk_level": p.risk_level,
                "status": p.status,
                "created_at": p.created_at,
                "executed_at": p.executed_at,
            }
            for p in state.proposals
        ]
        archive.writestr("proposals_log.json", json.dumps(proposals_data, indent=2))

    return buffer.getvalue()
