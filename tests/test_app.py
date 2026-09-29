from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_landing_and_demo():
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    assert not app.exception
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert app.session_state["dataset"].shape == (1818, 9)


def test_all_workspace_pages_with_demo():
    from studio.common import demo_data
    from modules.loader import get_metadata

    data = demo_data()
    for page in [
        "overview",
        "explorer",
        "quality",
        "charts",
        "intelligence",
        "report",
        "import",
    ]:
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
        app.session_state["dataset"] = data
        app.session_state["dataset_filename"] = "Demo"
        app.session_state["dataset_metadata"] = get_metadata(data, "Demo", 0)
        app.switch_page(f"studio/{page}_page.py").run(timeout=30)
        assert not app.exception, (page, app.exception)


def workspace(page):
    from studio.common import demo_data
    from modules.loader import get_metadata

    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    data = demo_data()
    app.session_state["dataset"] = data
    app.session_state["dataset_filename"] = "Demo"
    app.session_state["dataset_metadata"] = get_metadata(data, "Demo", 0)
    return app.switch_page(f"studio/{page}_page.py").run(timeout=30)


def test_preview_apply_restore_flow():
    app = workspace("quality")
    next(b for b in app.button if b.label == "Preview cleaning").click().run(timeout=30)
    assert not app.exception
    next(b for b in app.button if b.label == "Apply this preview").click().run(
        timeout=30
    )
    assert not app.exception
    assert len(app.session_state["cleaned_dataset"]) == 1800
    next(b for b in app.button if b.label == "Restore original dataset").click().run(
        timeout=30
    )
    assert not app.exception
    assert app.session_state["dataset"].shape == (1818, 9)


def test_every_chart_mode():
    app = workspace("charts")
    for kind in [
        "Relationship",
        "Category comparison",
        "Box plot",
        "Correlation",
        "Time trend",
        "Distribution",
    ]:
        next(s for s in app.selectbox if s.label == "Visualization").select(kind).run(
            timeout=30
        )
        assert not app.exception, kind


def test_search_returns_matching_records():
    app = workspace("explorer")
    next(t for t in app.text_input if t.label == "Search records").input("Europe").run(
        timeout=30
    )
    assert not app.exception
    assert app.dataframe[0].value["region"].eq("Europe").all()


def test_production_requires_login(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    assert not app.exception
    assert any(b.label == "Sign in" for b in app.button)
    assert "dataset" not in app.session_state
