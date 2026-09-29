import pandas as pd
from utils import session
from modules.loader import get_metadata


def test_new_dataset_invalidates_all_derived_results(monkeypatch):
    state = {
        "cleaned_dataset": pd.DataFrame({"old": [1]}),
        "chat_session": "old chat",
        "ai_output": {"x": "old"},
        "applied_recipe": {"x": 1},
        "ai_consent": True,
    }
    monkeypatch.setattr(session.st, "session_state", state)
    df = pd.DataFrame({"new": [2]})
    session.set_dataset(df, get_metadata(df, "new.csv", 0), "new.csv")
    assert session.get_active_dataset().columns.tolist() == ["new"]
    assert "chat_session" not in state
    assert "ai_consent" not in state
    assert "applied_recipe" not in state
    df.iloc[0, 0] = 99
    assert session.get_dataset().iloc[0, 0] == 2


def test_cleaning_and_restore_reset_ai(monkeypatch):
    raw = pd.DataFrame({"x": [1]})
    state = {"dataset": raw, "chat_session": "stale"}
    monkeypatch.setattr(session.st, "session_state", state)
    session.set_cleaned_dataset(pd.DataFrame({"x": [2]}), None)
    assert session.get_active_dataset().iloc[0, 0] == 2
    assert "chat_session" not in state
    session.restore_original()
    assert session.get_active_dataset().iloc[0, 0] == 1
