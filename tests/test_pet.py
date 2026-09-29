"""Tests for Tom Lizard AI pet assistant."""

from __future__ import annotations

import pandas as pd
from studio.pet_assistant import (
    ANIMATIONS,
    determine_pet_state,
    get_ai_decision_advice,
    get_pet_css,
    get_spritesheet_base64,
)
from workflow.approvals import ApprovalRequest, RiskLevel
from workflow.state import WorkflowState, WorkflowStatus


def test_spritesheet_loaded():
    b64 = get_spritesheet_base64()
    assert isinstance(b64, str)
    assert len(b64) > 10000  # valid base64 image data


def test_pet_css_generation():
    css = get_pet_css("fakebase64")
    assert "tom-pet-avatar" in css
    # Verify all 9 animations are generated
    for name in ANIMATIONS:
        assert f"play-tom-{name}" in css
        assert f"tom-anim-{name}" in css


def test_determine_pet_state_transitions():
    # 1. No workflow state -> waving
    anim, badge, _ = determine_pet_state(None)
    assert anim == "waving"
    assert "Ready" in badge

    df = pd.DataFrame({"a": [1, 2, None]})
    state = WorkflowState.create_initial(df, "test.csv")

    # 2. Idle state with dataset -> idle
    anim, _, _ = determine_pet_state(state)
    assert anim == "idle"

    # 3. Running state -> running
    state.status = WorkflowStatus.RUNNING
    anim, badge, _ = determine_pet_state(state)
    assert anim == "running"

    # 4. Waiting for approval (normal) -> waiting
    state.status = WorkflowStatus.WAITING_FOR_APPROVAL
    appr_low = ApprovalRequest.create(
        "wf", "prop", "agent", "trim_text", "desc", ["a"], 1, RiskLevel.LOW
    )
    state.approvals = [appr_low]
    anim, badge, _ = determine_pet_state(state)
    assert anim == "waiting"

    # 5. Waiting for approval with High Risk -> review
    appr_high = ApprovalRequest.create(
        "wf", "prop2", "agent", "cap_outliers", "desc", ["a"], 1, RiskLevel.HIGH
    )
    state.approvals = [appr_low, appr_high]
    anim, badge, _ = determine_pet_state(state)
    assert anim == "review"

    # 6. Completed -> jumping
    state.status = WorkflowStatus.COMPLETED
    state.approvals = []
    anim, badge, _ = determine_pet_state(state)
    assert anim == "jumping"

    # 7. Failed -> failed
    state.status = WorkflowStatus.FAILED
    anim, badge, _ = determine_pet_state(state)
    assert anim == "failed"


def test_ai_decision_advice():
    df = pd.DataFrame({"age": [20, 30, 40], "score": [10.0, None, 90.0]})
    state = WorkflowState.create_initial(df, "test.csv")

    advice_outliers = get_ai_decision_advice("outliers", state)
    assert "Tom's Advice" in advice_outliers

    advice_missing = get_ai_decision_advice("missing", state)
    assert "Tom's Advice" in advice_missing

    advice_ml = get_ai_decision_advice("ml", state)
    assert "Tom's Advice" in advice_ml
