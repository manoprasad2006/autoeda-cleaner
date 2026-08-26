import streamlit as st

from utils.config import load_settings
from utils.logger import get_logger

# st.set_page_config must be the first Streamlit command executed,
# before any other st.* call — Streamlit enforces this and raises an
# error otherwise.
st.set_page_config(
    page_title="IntelliData AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_settings():
    """
    Load settings once per app process, not once per rerun.

    st.cache_resource is Streamlit's mechanism for caching objects that
    should be shared across all users and all reruns of the script
    (as opposed to st.cache_data, which is for cacheable data values).
    Settings are a perfect fit: they don't change while the app is
    running, and reloading .env on every click would be wasteful.
    """
    return load_settings()


def main() -> None:
    settings = get_settings()
    logger = get_logger("app", settings.logs_dir, settings.log_level)

    logger.info("App started | env=%s", settings.app_env)

    if not settings.gemini_api_key:
        # Fail loudly in the UI (not just the log) — a missing API key
        # is the single most common setup mistake, and every AI feature
        # in later phases depends on it.
        st.warning(
            "GEMINI_API_KEY is not set. Copy `.env.example` to `.env` "
            "and add your key before AI features will work.",
            icon="⚠️",
        )

    st.title("IntelliData AI")
    st.caption("An AI-Powered Data Intelligence Platform for Data Analytics & ML")

    st.markdown(
        """
        Welcome — this is the project scaffold from **Phase 1**.

        Nothing functional lives here yet. Upload, profiling, cleaning,
        EDA, visualization, and the AI features arrive in the phases
        that follow, each as its own page under `pages/`.
        """
    )


if __name__ == "__main__":
    main()
