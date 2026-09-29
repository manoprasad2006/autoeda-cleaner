from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
from html import escape
import io
import json
import zipfile
import pandas as pd
from modules.profiler import profile_dataset
from modules.quality import assess_quality


def csv_bytes(df):
    """Neutralize spreadsheet formulas in text exports; numeric cells stay numeric."""
    safe = df.copy()
    for col in safe.columns:
        if not pd.api.types.is_numeric_dtype(safe[col]):
            safe[col] = safe[col].map(
                lambda x: "'" + x
                if isinstance(x, str)
                and x.lstrip().startswith(("=", "+", "-", "@", "\t", "\r"))
                else x
            )
    safe.columns = [
        "'" + str(c) if str(c).lstrip().startswith(("=", "+", "-", "@")) else str(c)
        for c in safe.columns
    ]
    return safe.to_csv(index=False).encode("utf-8-sig")


def report_html(df, filename, log=None):
    p = profile_dataset(df)
    q = assess_quality(df, p)
    rows = pd.DataFrame(
        [
            {
                "Column": c.name,
                "Type": c.dtype,
                "Missing %": c.missing_pct,
                "Unique values": c.unique_count,
            }
            for c in p.columns
        ]
    )
    audit = (
        log.to_dataframe().to_html(index=False, escape=True)
        if log and log.actions
        else "<p>No cleaning operations applied.</p>"
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>IntelliData report</title><style>
    body{{font:15px system-ui;color:#17233b;max-width:1100px;margin:50px auto;padding:24px;background:#f5f7fb}}h1{{font-size:38px}}h2{{margin-top:36px}}.hero{{background:#14233a;color:white;padding:32px;border-radius:20px}}.kpis{{display:flex;gap:20px;flex-wrap:wrap;margin:24px 0}}.kpis div{{background:white;padding:24px;border-radius:14px;min-width:160px}}strong{{display:block;font-size:26px}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:10px;border:1px solid #dbe1ec;text-align:left}}.table{{overflow:auto}}small{{color:#61708a}}@media print{{body{{margin:0;background:white}}.hero{{background:white;color:#17233b}}}}</style></head><body>
    <div class="hero"><p>INTELLIDATA STUDIO / ANALYSIS REPORT</p><h1>{escape(filename)}</h1><p>Generated {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}</p></div>
    <div class="kpis"><div><strong>{len(df):,}</strong>Records</div><div><strong>{len(df.columns)}</strong>Columns</div><div><strong>{q.overall_score:.1f}/100</strong>Quality score</div><div><strong>{q.missing_cell_count:,}</strong>Missing cells</div></div>
    <p>Quality is a heuristic weighted score: completeness 40%, duplicate checks 25%, type consistency 20%, and numeric IQR outlier checks 15%. It does not establish business correctness or model performance.</p>
    <h2>Column profile</h2><div class="table">{rows.to_html(index=False, escape=True)}</div>
    <h2>Descriptive statistics</h2><div class="table">{df.describe(include="all").round(3).to_html(escape=True)}</div>
    <h2>Cleaning audit</h2><div class="table">{audit}</div><p><small>Original data should be retained for review. This report includes column names and statistical values.</small></p></body></html>"""


def export_bundle(df, filename, log=None, recipe=None):
    buffer = io.BytesIO()
    p = profile_dataset(df)
    q = assess_quality(df, p)
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("dataset.csv", csv_bytes(df))
        archive.writestr("report.html", report_html(df, filename, log))
        archive.writestr("quality.json", json.dumps(asdict(q), indent=2))
        archive.writestr("recipe.json", json.dumps(recipe or {}, indent=2))
        if log:
            archive.writestr("cleaning_audit.csv", csv_bytes(log.to_dataframe()))
    return buffer.getvalue()
