import pandas as pd
import streamlit as st

from modules.ai.chatbot import ChatSession, ask_chatbot, build_system_context
from modules.ai.client import GeminiClient
from modules.cleaner import auto_clean
from modules.eda import run_eda
from modules.ml_readiness import assess_ml_readiness
from modules.profiler import profile_dataset
from modules.quality import assess_quality
from utils.config import load_settings
from utils.logger import get_logger
from utils.session import (
    get_active_dataset,
    get_chat_session,
    get_cleaning_log,
    has_dataset,
    init_session_state,
    set_chat_session,
)

settings = load_settings()
logger = get_logger("pages.chat", settings.logs_dir, settings.log_level)
init_session_state()

st.title("?? Chat with Your Dataset")

if not has_dataset():
    st.info("Upload a dataset first on the **Upload** page.")
    st.stop()

if not settings.gemini_api_key:
    st.info("Add a GEMINI_API_KEY in your .env file to use the chatbot.")
    st.stop()

df = get_active_dataset()


@st.cache_data(show_spinner="Preparing dataset context...")
def _cached_context(data: pd.DataFrame):
    profile = profile_dataset(data)
    quality = assess_quality(data, profile)
    eda = run_eda(data, profile.numerical_columns, profile.categorical_columns)
    readiness = assess_ml_readiness(data, profile, quality, eda)
    log = get_cleaning_log()
    if log is None:
        _, log = auto_clean(data)  # fallback: build a log if cleaning wasn't run
    return build_system_context(profile, quality, log, eda, readiness)


if get_chat_session() is None:
    context = _cached_context(df)
    set_chat_session(ChatSession(system_context=context))

session = get_chat_session()

for message in session.messages:
    with st.chat_message(message.role):
        st.write(message.content)

question = st.chat_input("Ask about your dataset...")

if question:
    with st.chat_message("user"):
        st.write(question)

    client = GeminiClient(api_key=settings.gemini_api_key, model=settings.gemini_model)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask_chatbot(client, session, question)
        st.write(answer)

    logger.info("Chat exchange | question_len=%d answer_len=%d", len(question), len(answer))