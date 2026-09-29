"""Tom Lizard AI Pet Assistant component for IntelliData Studio.
Integrates the OpenPets 'tom-lizard' sprite sheet with contextual animations,
speech bubbles, and decision-making assistance powered by live Gemini API.
"""
from __future__ import annotations

import base64
from functools import lru_cache
import os
from typing import Any
import streamlit as st

from modules.ai.client import get_ai_client


SPRITESHEET_PATH = os.path.join(
    os.path.dirname(__file__), "assets", "pet", "tom-lizard", "spritesheet.webp"
)

# Dimensions from pet.json
FRAME_WIDTH = 192
FRAME_HEIGHT = 208
SHEET_COLS = 8
SHEET_ROWS = 9
SCALE = 0.5  # 96px x 104px display size

DISPLAY_W = int(FRAME_WIDTH * SCALE)    # 96px
DISPLAY_H = int(FRAME_HEIGHT * SCALE)   # 104px
SHEET_W = int(1536 * SCALE)             # 768px
SHEET_H = int(1872 * SCALE)             # 936px

ANIMATIONS: dict[str, dict[str, Any]] = {
    "idle": {"row": 0, "frames": 6, "duration": "1.0s", "loop": "infinite"},
    "running-right": {"row": 1, "frames": 8, "duration": "0.8s", "loop": "infinite"},
    "running-left": {"row": 2, "frames": 8, "duration": "0.8s", "loop": "infinite"},
    "waving": {"row": 3, "frames": 4, "duration": "0.6s", "loop": "infinite"},
    "jumping": {"row": 4, "frames": 5, "duration": "0.6s", "loop": "infinite"},
    "failed": {"row": 5, "frames": 8, "duration": "1.0s", "loop": "infinite"},
    "waiting": {"row": 6, "frames": 6, "duration": "1.0s", "loop": "infinite"},
    "running": {"row": 7, "frames": 6, "duration": "0.6s", "loop": "infinite"},
    "review": {"row": 8, "frames": 6, "duration": "1.0s", "loop": "infinite"},
}


@lru_cache(maxsize=1)
def get_spritesheet_base64() -> str:
    """Read and cache the base64 string of the Tom Lizard sprite sheet."""
    if os.path.exists(SPRITESHEET_PATH):
        with open(SPRITESHEET_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""


def get_pet_css(b64_img: str) -> str:
    """Generate CSS keyframe animations for all 9 sprite sheet rows."""
    css_rules = [
        "<style>",
        ".tom-pet-container {",
        "    display: flex;",
        "    align-items: center;",
        "    gap: 10px;",
        "    background: linear-gradient(135deg, rgba(20, 29, 48, 0.85), rgba(11, 16, 32, 0.95));",
        "    border: 1px solid rgba(102, 224, 194, 0.3);",
        "    border-radius: 12px;",
        "    padding: 8px 10px;",
        "    margin: 8px 0 14px 0;",
        "    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);",
        "    position: relative;",
        "    transition: border-color 0.25s ease;",
        "}",
        ".tom-pet-container:hover {",
        "    border-color: rgba(102, 224, 194, 0.65);",
        "}",
        ".tom-pet-avatar {",
        f"    width: {DISPLAY_W}px;",
        f"    height: {DISPLAY_H}px;",
        "    flex-shrink: 0;",
        f"    background-image: url('data:image/webp;base64,{b64_img}');",
        f"    background-size: {SHEET_W}px {SHEET_H}px;",
        "    background-repeat: no-repeat;",
        "    image-rendering: -webkit-optimize-contrast;",
        "    image-rendering: crisp-edges;",
        "    image-rendering: pixelated;",
        "    cursor: pointer;",
        "    transition: transform 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275), filter 0.2s ease;",
        "}",
        ".tom-pet-avatar:hover {",
        "    transform: scale(1.12) translateY(-2px);",
        "    filter: drop-shadow(0 0 8px rgba(102, 224, 194, 0.6));",
        "}",
        ".tom-pet-avatar:active {",
        "    transform: scale(0.96);",
        "}",
        ".tom-pet-speech {",
        "    font-size: 11px;",
        "    line-height: 1.35;",
        "    color: #E8EDF7;",
        "    background: rgba(14, 22, 38, 0.9);",
        "    border: 1px solid rgba(102, 224, 194, 0.35);",
        "    border-radius: 8px;",
        "    padding: 6px 9px;",
        "    position: relative;",
        "    flex-grow: 1;",
        "    min-height: 48px;",
        "    display: flex;",
        "    flex-direction: column;",
        "    justify-content: center;",
        "}",
        ".tom-pet-speech::before {",
        "    content: '';",
        "    position: absolute;",
        "    left: -6px;",
        "    top: 50%;",
        "    transform: translateY(-50%);",
        "    border-top: 5px solid transparent;",
        "    border-bottom: 5px solid transparent;",
        "    border-right: 6px solid rgba(102, 224, 194, 0.35);",
        "}",
        ".tom-pet-badge {",
        "    display: inline-block;",
        "    font-size: 9px;",
        "    font-weight: 700;",
        "    color: #66E0C2;",
        "    text-transform: uppercase;",
        "    letter-spacing: 0.5px;",
        "    margin-bottom: 2px;",
        "}",
    ]

    for name, config in ANIMATIONS.items():
        row = config["row"]
        frames = config["frames"]
        duration = config["duration"]
        loop = config["loop"]
        y_offset = -row * DISPLAY_H
        total_x = -frames * DISPLAY_W

        css_rules.append(
            f"@keyframes play-tom-{name} {{\n"
            f"    from {{ background-position: 0px {y_offset}px; }}\n"
            f"    to {{ background-position: {total_x}px {y_offset}px; }}\n"
            f"}}\n"
            f".tom-anim-{name} {{\n"
            f"    background-position: 0px {y_offset}px;\n"
            f"    animation: play-tom-{name} {duration} steps({frames}) {loop};\n"
            f"}}"
        )

    css_rules.append("</style>")
    return "\n".join(css_rules)


def determine_pet_state(wf_state: Any, cycle_index: int = 0) -> tuple[str, str, str]:
    """Inspect current workflow state and determine Tom's animation, badge, and message.
    Supports dynamic ambient animations across all 9 sprite rows.
    Returns: (animation_name, badge_label, message)
    """
    if wf_state is None:
        # Dynamic rotation when no dataset is loaded
        empty_states = [
            (
                "waving",
                "Tom Lizard · Ready",
                "Hey there! I'm Tom, your Data Governance Assistant! Upload a dataset or load the demo workspace to get started.",
            ),
            (
                "jumping",
                "Tom Lizard · Energetic",
                "Ready to explore some data? Drag & drop a CSV or Excel in Import Data.",
            ),
            (
                "running-right",
                "Tom Lizard · Active",
                "Warming up my data crunching muscles! Let's build governed pipelines.",
            ),
            (
                "waiting",
                "Tom Lizard · Curious",
                "Waiting for data and tables to inspect.",
            ),
        ]
        return empty_states[cycle_index % len(empty_states)]

    status = getattr(wf_state, "status", "idle")
    stage = getattr(wf_state, "current_stage", "intake")
    approvals = getattr(wf_state, "approvals", [])
    pending = [a for a in approvals if getattr(a, "status", "") == "pending"]

    # 1. Waiting for Approval (HITL)
    if status == "waiting_for_approval" or len(pending) > 0:
        high_risk_count = sum(1 for a in pending if getattr(a, "risk_level", "") == "high")
        if high_risk_count > 0:
            return (
                "review",
                "Tom Lizard · Reviewing",
                f"Attention! {len(pending)} proposal(s) waiting. Watch out for {high_risk_count} high-risk action(s) like outlier capping.",
            )
        return (
            "waiting",
            "Tom Lizard · Waiting",
            f"You have {len(pending)} pending proposal(s) ready for your review in the Governance Hub.",
        )

    # 2. Running pipeline
    if status == "running":
        return (
            "running",
            "Tom Lizard · Working",
            f"Agents are actively executing the '{stage}' stage. Please wait a moment.",
        )

    # 3. Failed
    if status == "failed":
        return (
            "failed",
            "Tom Lizard · Alert",
            "An agent encountered an issue. Check the audit logs or try resetting the step.",
        )

    # 4. Completed
    if status == "completed":
        return (
            "jumping",
            "Tom Lizard · Completed",
            "All pipeline steps completed and verified. Ready for report export.",
        )

    # 5. Active dataset exists and cleaned (v1+)
    active_ver = getattr(wf_state, "active_version_id", "v0")
    if active_ver != "v0":
        cleaned_states = [
            ("jumping", f"Tom Lizard · Version {active_ver}", f"Dataset is cleaned at {active_ver}. Ready for modeling."),
            ("review", f"Tom Lizard · Version {active_ver}", "Audit trail is verified. Check out Chart Studio or ML Readiness."),
            ("waving", f"Tom Lizard · Version {active_ver}", f"Cleaned data active ({active_ver}). Ask me anything below."),
        ]
        return cleaned_states[cycle_index % len(cleaned_states)]

    # 6. Default idle with data loaded - Dynamic rotation
    fname = getattr(wf_state, "dataset_filename", "Dataset")
    idle_states = [
        ("idle", "Tom Lizard · Standing By", f"Loaded '{fname}'. Click 'Run Autonomous Pipeline' to start."),
        ("review", "Tom Lizard · Inspecting", f"Examining {fname} distributions and schema. Looking consistent."),
        ("jumping", "Tom Lizard · Active", f"{fname} is ready. Ask me for outlier or ML advice below."),
        ("waiting", "Tom Lizard · Observant", f"{fname} is ready. Standing by for your instructions."),
        ("running-left", "Tom Lizard · Exploring", "Navigating features. You can also explore Chart Studio."),
    ]
    return idle_states[cycle_index % len(idle_states)]


def get_ai_decision_advice(query: str, wf_state: Any) -> str:
    """Provide intelligent decision recommendations using live AI with rich dataset context."""
    if wf_state is None:
        return "**Tom's Advice:** Please upload or import a dataset first so I can analyze it."

    df = getattr(wf_state, "active_dataset", None)
    if df is None:
        df = getattr(wf_state, "original_dataset", None)

    fname = getattr(wf_state, "dataset_filename", "Dataset")
    q = getattr(wf_state, "quality_result", None)
    ml = getattr(wf_state, "ml_readiness_result", None)
    approvals = getattr(wf_state, "approvals", [])
    pending = [a for a in approvals if getattr(a, "status", "") == "pending"]

    # Gather dataset metadata
    n_rows = len(df) if df is not None else 0
    n_cols = len(df.columns) if df is not None else 0
    q_score = getattr(q, "overall_score", 80.0) if q else 80.0
    missing_cells = getattr(q, "missing_cell_count", 0) if q else 0
    outlier_dict = getattr(q, "outlier_columns", {}) if q else {}
    outlier_cols = list(outlier_dict.keys()) if isinstance(outlier_dict, dict) else []
    col_names = list(df.columns[:20]) if df is not None else []
    inferred_task = getattr(ml, "inferred_task", "classification") if ml else "classification"
    pending_descs = [
        f"{getattr(a, 'action_type', 'action')} on {getattr(a, 'target_columns', [])} (Risk: {getattr(a, 'risk_level', 'unknown')})"
        for a in pending[:4]
    ]

    # Map preset query tags to natural questions
    query_map = {
        "outliers": "How should I handle the outliers in this dataset? Which columns need capping or trimming and what ML models would care?",
        "missing": "What is the best strategy for the missing values in this dataset? Should I impute or drop?",
        "ml": "What ML task and models are best suited for this dataset, and what data prep steps should I take?",
        "next": "What is the recommended next step for cleaning and analyzing this dataset?",
    }
    actual_question = query_map.get(query.lower(), query)

    # 1. Attempt AI Client (Groq Rotation or Gemini)
    ai_client = get_ai_client()
    if ai_client is not None:
        try:
            prompt = f"""You are Tom, an expert data scientist and AI assistant in IntelliData Studio.
You give concise, actionable, razor-sharp advice to help the user make safe, governed data decisions.
Do NOT use emojis in your response. Keep formatting clean with standard markdown bullet points.

Dataset Context:
- Filename: {fname}
- Dimensions: {n_rows:,} rows, {n_cols} columns
- Sample Columns: {', '.join(col_names[:15])}
- Data Quality Score: {q_score:.1f}/100
- Missing cells: {missing_cells:,}
- Columns with Outliers: {', '.join(outlier_cols[:5]) if outlier_cols else 'None detected'}
- Inferred ML Task: {inferred_task}
- Pending Human Approvals: {'; '.join(pending_descs) if pending_descs else 'None'}

User Question:
"{actual_question}"

Instructions:
1. Provide a direct, professional answer in 2-3 short, high-value bullet points.
2. Ground your advice directly in the dataset's actual features and statistics above.
3. If decisions carry data-loss risk, remind the user about human-in-the-loop review.
4. Strictly do NOT use emojis.
"""
            res = ai_client.generate(prompt, max_retries=3)
            if res.success and res.text:
                return f"**Tom's Advice:**\n\n{res.text}"
        except Exception:
            pass  # Fall back to heuristic rule

    # 2. Heuristic fallback when offline / no API key
    if query == "outliers":
        if outlier_cols:
            return (
                f"**Tom's Advice on Outliers:**\n\n"
                f"Outliers detected in **{', '.join(outlier_cols[:3])}**.\n\n"
                "- **Tree models (Random Forest, XGBoost):** Usually do not require capping; tree algorithms split thresholds naturally.\n"
                "- **Linear models / regression:** High-leverage outliers can distort slope coefficients. Consider IQR winsorization or log-transform.\n"
                "- **Validation:** Always verify whether extremes represent genuine data before removal."
            )
        return "**Tom's Advice:** No severe IQR outliers detected in continuous numeric features. You are safe to proceed."

    elif query == "missing":
        if missing_cells > 0:
            return (
                f"**Tom's Advice on Missing Data:**\n\n"
                f"Found **{missing_cells:,} missing cell(s)** in {fname}.\n\n"
                "- **Low missing rate (< 5%):** Median imputation for numeric, mode for categorical is standard practice.\n"
                "- **High missing rate (> 60%):** Imputing can induce artificial bias. Consider column drop or adding an indicator column.\n"
                "- **Keys:** Never impute unique record identifiers."
            )
        return "**Tom's Advice:** Completeness is 100%. No missing values need imputation."

    elif query == "ml":
        return (
            f"**Tom's Advice on ML Modeling:**\n\n"
            f"- **Inferred Task:** {inferred_task.title()}.\n"
            "- **Baseline Suggestion:** Start with an interpretable baseline model before moving to ensemble methods.\n"
            "- **Leakage Warning:** Exclude post-event attributes or record identifiers prior to training."
        )

    # Generic fallback
    return (
        f"**Tom's Advice:** For '{fname}', examine feature distributions in Chart Studio, "
        "verify any high-risk proposals in the Governance Hub, and note that all versions are preserved in lineage."
    )


def render_pet_assistant() -> None:
    """Render the animated Tom Lizard assistant in the Streamlit sidebar."""
    b64_img = get_spritesheet_base64()
    if not b64_img:
        st.caption("🦎 Tom Lizard companion active.")
        return

    wf_state = st.session_state.get("workflow_state")

    # Dynamic ambient cycling counter
    if "tom_cycle_count" not in st.session_state:
        st.session_state["tom_cycle_count"] = 0

    cycle_index = st.session_state["tom_cycle_count"]
    auto_anim, badge, message = determine_pet_state(wf_state, cycle_index=cycle_index)

    # Active animation resolution
    active_anim = st.session_state.get("tom_manual_anim", auto_anim)
    # If state changes to approval or running, reset override to reflect urgency
    if auto_anim in ("waiting", "review", "running", "failed"):
        active_anim = auto_anim

    # Dynamic speech override if Tom recently answered a question
    if "tom_speech_override" in st.session_state:
        message = st.session_state["tom_speech_override"]
        badge = "Tom Lizard · AI Answering"

    # Render CSS + Avatar + Speech bubble
    css = get_pet_css(b64_img)
    markup = (
        f"{css}\n"
        f'<div class="tom-pet-container" title="Click \'Ask Tom\' below for AI decision advice!">'
        f'<div class="tom-pet-avatar tom-anim-{active_anim}"></div>'
        f'<div class="tom-pet-speech">'
        f'<span class="tom-pet-badge">{badge}</span><br>'
        f"{message}"
        f'</div>'
        f'</div>'
    )
    if hasattr(st, "html"):
        st.html(markup)
    else:
        st.markdown(markup, unsafe_allow_html=True)

    # Interactive decision assistant drawer
    with st.expander("Ask Tom (Decision Helper)", expanded=False):
        # Quick preset buttons
        c1, c2 = st.columns(2)
        if c1.button("Outliers", key="tom_q_outliers", help="Ask Tom advice on capping outliers"):
            with st.spinner("Tom is thinking..."):
                st.session_state["tom_advice"] = get_ai_decision_advice("outliers", wf_state)
            st.session_state["tom_manual_anim"] = "review"
            st.session_state["tom_speech_override"] = "Analyzed your outliers. Details below."
            st.rerun()

        if c2.button("Missing Data", key="tom_q_missing", help="Ask Tom advice on missing values"):
            with st.spinner("Tom is thinking..."):
                st.session_state["tom_advice"] = get_ai_decision_advice("missing", wf_state)
            st.session_state["tom_manual_anim"] = "waiting"
            st.session_state["tom_speech_override"] = "Here is my advice on handling missing data."
            st.rerun()

        c3, c4 = st.columns(2)
        if c3.button("ML Strategy", key="tom_q_ml", help="Ask Tom advice on machine learning"):
            with st.spinner("Tom is thinking..."):
                st.session_state["tom_advice"] = get_ai_decision_advice("ml", wf_state)
            st.session_state["tom_manual_anim"] = "jumping"
            st.session_state["tom_speech_override"] = "Recommended ML strategy ready."
            st.rerun()

        if c4.button("Next Steps", key="tom_q_next", help="Ask Tom advice on next steps"):
            with st.spinner("Tom is thinking..."):
                st.session_state["tom_advice"] = get_ai_decision_advice("next", wf_state)
            st.session_state["tom_manual_anim"] = "running-right"
            st.session_state["tom_speech_override"] = "Got your next pipeline roadmap ready."
            st.rerun()

        # Free-form custom question input
        st.markdown("---")
        custom_q = st.text_input(
            "Ask Tom a custom question:",
            placeholder="e.g. Should I normalize Age and Fare?",
            key="tom_custom_q_input",
        )
        if st.button("Ask Tom", key="tom_ask_custom_btn"):
            if custom_q.strip():
                with st.spinner("Tom is thinking..."):
                    st.session_state["tom_advice"] = get_ai_decision_advice(custom_q.strip(), wf_state)
                st.session_state["tom_manual_anim"] = "jumping"
                st.session_state["tom_speech_override"] = f"Here is my advice on '{custom_q[:30]}...'"
                st.rerun()

        # Display advice if available
        if "tom_advice" in st.session_state:
            st.info(st.session_state["tom_advice"])
            if st.button("Clear advice", key="tom_clear_advice"):
                st.session_state.pop("tom_advice", None)
                st.session_state.pop("tom_speech_override", None)
                st.session_state.pop("tom_manual_anim", None)
                st.rerun()
