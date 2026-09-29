import streamlit as st
import importlib
import studio.pet_assistant
importlib.reload(studio.pet_assistant)
from studio.pet_assistant import render_pet_assistant
from studio.common import style
from utils.config import load_settings
from utils.session import init_session_state, has_dataset, has_cleaned_dataset

st.set_page_config(
    page_title="IntelliData Studio",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)
style()
from utils.auth import enforce_auth

enforce_auth()
init_session_state()
settings = load_settings()
with st.sidebar:
    st.markdown("## ✦ IntelliData")
    st.caption("D A T A   S T U D I O")
    render_pet_assistant()
    st.divider()
    if has_dataset():
        st.caption("ACTIVE WORKSPACE")
        st.write(st.session_state.get("dataset_filename", "Dataset"))
        st.caption(
            "● Cleaned version" if has_cleaned_dataset() else "● Original version"
        )
        wf_state = st.session_state.get("workflow_state")
        if wf_state:
            pending_n = len([a for a in getattr(wf_state, "approvals", []) if getattr(a, "status", "") == "pending"])
            if pending_n > 0:
                st.warning(f"⚠️ {pending_n} decision(s) pending review")
            else:
                st.caption(f"Stage: {wf_state.current_stage}")
    else:
        st.caption("Your next insight starts here.")
pages = {
    "Workspace": [
        st.Page(
            "studio/overview_page.py",
            title="Overview",
            icon=":material/dashboard:",
            default=True,
        ),
        st.Page(
            "studio/agent_page.py",
            title="Agent Governance Hub",
            icon=":material/smart_toy:",
        ),
        st.Page(
            "studio/import_page.py", title="Import data", icon=":material/upload_file:"
        ),
    ],
    "Prepare": [
        st.Page(
            "studio/explorer_page.py",
            title="Data explorer",
            icon=":material/table_chart:",
        ),
        st.Page(
            "studio/quality_page.py",
            title="Quality & cleaning",
            icon=":material/auto_fix_high:",
        ),
    ],
    "Explore & share": [
        st.Page(
            "studio/charts_page.py", title="Chart studio", icon=":material/monitoring:"
        ),
        st.Page(
            "studio/intelligence_page.py",
            title="AI & ML readiness",
            icon=":material/psychology:",
        ),
        st.Page(
            "studio/report_page.py", title="Export center", icon=":material/ios_share:"
        ),
    ],
}
page = st.navigation(pages, position="hidden")
with st.sidebar:
    for section, items in pages.items():
        st.caption(section.upper())
        for item in items:
            st.page_link(item)
with st.sidebar:
    st.divider()
    st.caption("Import → Prepare → Explore → Share")
    st.caption(
        "AI available · opt-in per dataset"
        if settings.gemini_api_key
        else "Local analytics · AI key not configured"
    )
page.run()
