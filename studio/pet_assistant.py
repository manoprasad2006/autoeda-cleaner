"""Tom Lizard AI Pet Assistant component for IntelliData Studio.
Integrates the OpenPets 'tom-lizard' sprite sheet with contextual animations,
speech bubbles, and decision-making assistance based on active workflow state.
"""
from __future__ import annotations

import base64
from functools import lru_cache
import os
from typing import Any
import streamlit as st

from utils.config import load_settings


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
        "    background: linear-gradient(135deg, rgba(20, 29, 48, 0.75), rgba(11, 16, 32, 0.9));",
        "    border: 1px solid rgba(102, 224, 194, 0.25);",
        "    border-radius: 12px;",
        "    padding: 8px 10px;",
        "    margin: 8px 0 14px 0;",
        "    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);",
        "    position: relative;",
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
        "    transition: transform 0.2s ease;",
        "}",
        ".tom-pet-avatar:hover {",
        "    transform: scale(1.08);",
        "}",
        ".tom-pet-speech {",
        "    font-size: 11px;",
        "    line-height: 1.35;",
        "    color: #E8EDF7;",
        "    background: rgba(14, 22, 38, 0.85);",
        "    border: 1px solid rgba(102, 224, 194, 0.35);",
        "    border-radius: 8px;",
        "    padding: 6px 9px;",
        "    position: relative;",
        "    flex-grow: 1;",
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


def determine_pet_state(wf_state: Any) -> tuple[str, str, str]:
    """Inspect current workflow state and determine Tom's animation, badge, and message.
    Returns: (animation_name, badge_label, message)
    """
    if wf_state is None:
        return (
            "waving",
            "Tom Lizard · Ready",
            "Hey there! I'm Tom, your Data Governance Assistant! Upload a dataset or load the demo workspace to get started 🦎",
        )

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
                f"Attention! {len(pending)} proposal(s) waiting. Watch out for {high_risk_count} high-risk action(s) like outlier capping!",
            )
        return (
            "waiting",
            "Tom Lizard · Waiting",
            f"You have {len(pending)} pending proposal(s) ready for your review in the Governance Hub!",
        )

    # 2. Running pipeline
    if status == "running":
        return (
            "running",
            "Tom Lizard · Working",
            f"Agents are actively executing the '{stage}' stage! Hang tight ⚡",
        )

    # 3. Failed
    if status == "failed":
        return (
            "failed",
            "Tom Lizard · Alert",
            "Uh-oh! An agent encountered an issue. Check the audit logs or try resetting the step.",
        )

    # 4. Completed
    if status == "completed":
        return (
            "jumping",
            "Tom Lizard · Cheerful",
            "Mission accomplished! All pipeline steps completed and verified. Ready for report export! 🎉",
        )

    # 5. Active dataset exists and cleaned
    active_ver = getattr(wf_state, "active_version_id", "v0")
    if active_ver != "v0":
        return (
            "waving",
            f"Tom Lizard · Version {active_ver}",
            f"Dataset is clean and active at version {active_ver}. Check out Chart Studio or ML Readiness!",
        )

    # 6. Default idle with data loaded
    return (
        "idle",
        "Tom Lizard · Standing By",
        f"Loaded '{getattr(wf_state, 'dataset_filename', 'Dataset')}'. Click 'Run Autonomous Pipeline' to start!",
    )


def get_ai_decision_advice(query_type: str, wf_state: Any) -> str:
    """Provide intelligent decision recommendations for common user questions."""
    if wf_state is None:
        return "Please upload a dataset first so I can analyze it for you!"

    df = getattr(wf_state, "active_dataset", None)
    if df is None:
        df = getattr(wf_state, "original_dataset", None)

    q = getattr(wf_state, "quality_result", None)

    if query_type == "outliers":
        if q and getattr(q, "outlier_columns", None):
            outlier_names = list(q.outlier_columns.keys())
            return (
                f"🦎 **Tom's Advice on Outliers:**\n\n"
                f"We detected outliers in **{', '.join(outlier_names[:3])}**.\n\n"
                "- **If building tree models (Random Forest, XGBoost):** You usually do **NOT** need to cap them; tree models handle non-linear extremes well.\n"
                "- **If building linear models / regression:** High-leverage outliers can skew your coefficients. Cap with IQR or apply log-transformation.\n"
                "- **Remember:** Always inspect whether outliers are valid real-world records!"
            )
        return "🦎 **Tom's Advice:** No severe IQR outliers detected in continuous numeric features. You're safe to proceed!"

    elif query_type == "missing":
        if q and getattr(q, "missing_cell_count", 0) > 0:
            return (
                f"🦎 **Tom's Advice on Missing Data:**\n\n"
                f"Found **{q.missing_cell_count:,} missing cell(s)**.\n\n"
                "- **For low missingness (< 5%):** Median imputation for numeric, mode for categorical is safe and conservative.\n"
                "- **For high missingness (> 60%):** Imputing mode can severely bias your data. Consider dropping the column or creating an indicator column (`is_missing`).\n"
                "- **Never impute:** Unique identifiers or primary keys!"
            )
        return "🦎 **Tom's Advice:** Completeness is 100%! No missing values need imputation."

    elif query_type == "ml":
        ml = getattr(wf_state, "ml_readiness_result", None)
        task = getattr(ml, "inferred_task", "classification") if ml else "classification"
        return (
            f"🦎 **Tom's Advice on ML Modeling:**\n\n"
            f"- **Inferred Task:** {task.title()}.\n"
            "- **Baseline Suggestion:** Start with a simple interpretable model (Logistic Regression or Decision Tree), then benchmark against Random Forest or Gradient Boosting.\n"
            "- **Leakage Warning:** Always ensure target identifiers or post-event features are excluded before training!"
        )

    # General / Custom
    settings = load_settings()
    if settings.gemini_api_key:
        from tools.registry import registry
        summary = registry.execute("generate_ai_summary", state=wf_state, api_key=settings.gemini_api_key)
        return f"🦎 **Tom's AI Summary:**\n\n{summary}"
    return (
        "🦎 **Tom's Advice:** Explore your correlations in Chart Studio, review pending proposals carefully, "
        "and remember you can always roll back any change!"
    )


def render_pet_assistant() -> None:
    """Render the animated Tom Lizard assistant in the Streamlit sidebar."""
    b64_img = get_spritesheet_base64()
    if not b64_img:
        st.caption("🦎 Tom Lizard companion active.")
        return

    wf_state = st.session_state.get("workflow_state")

    # Allow user to pick a manual animation or keep automatic
    auto_anim, badge, message = determine_pet_state(wf_state)

    # Check for manual user override in session_state
    active_anim = st.session_state.get("tom_manual_anim", auto_anim)
    # If state changes to approval or running, reset override to reflect urgency
    if auto_anim in ("waiting", "review", "running", "failed"):
        active_anim = auto_anim

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
    with st.expander("💬 Ask Tom (AI Decision Helper)", expanded=False):
        c1, c2, c3 = st.columns(3)
        if c1.button("Outliers?", key="tom_q_outliers", help="Advice on capping outliers"):
            st.session_state["tom_advice"] = get_ai_decision_advice("outliers", wf_state)
            st.session_state["tom_manual_anim"] = "review"
            st.rerun()

        if c2.button("Missing?", key="tom_q_missing", help="Advice on missing values"):
            st.session_state["tom_advice"] = get_ai_decision_advice("missing", wf_state)
            st.session_state["tom_manual_anim"] = "waiting"
            st.rerun()

        if c3.button("ML Tips?", key="tom_q_ml", help="Advice on machine learning"):
            st.session_state["tom_advice"] = get_ai_decision_advice("ml", wf_state)
            st.session_state["tom_manual_anim"] = "jumping"
            st.rerun()

        # Display advice if available
        if "tom_advice" in st.session_state:
            st.info(st.session_state["tom_advice"])
            if st.button("Clear advice", key="tom_clear_advice"):
                st.session_state.pop("tom_advice", None)
                st.session_state.pop("tom_manual_anim", None)
                st.rerun()

        # Mood / Animation switcher
        selected_anim = st.selectbox(
            "Tom's Animation",
            list(ANIMATIONS.keys()),
            index=list(ANIMATIONS.keys()).index(active_anim) if active_anim in ANIMATIONS else 0,
            key="tom_select_anim_box",
            label_visibility="collapsed",
        )
        if selected_anim != active_anim:
            st.session_state["tom_manual_anim"] = selected_anim
            st.rerun()
