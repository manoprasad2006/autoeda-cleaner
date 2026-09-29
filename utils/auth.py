"""Optional OIDC gate. Production mode always requires authentication."""

import os
import streamlit as st


def enforce_auth():
    required = (
        os.getenv("APP_ENV", "development").lower() == "production"
        or os.getenv("REQUIRE_AUTH", "false").lower() == "true"
    )
    if not required:
        return
    if not st.user.get("is_logged_in", False):
        st.title("Welcome to IntelliData Studio")
        st.write("Sign in to open your data workspace.")
        if st.button("Sign in", type="primary"):
            try:
                st.login()
            except Exception:
                st.error(
                    "Sign-in is not configured. Ask the operator to configure OIDC in Streamlit secrets."
                )
        st.stop()
    allowed = {
        s.strip() for s in os.getenv("ALLOWED_SUBJECTS", "").split(",") if s.strip()
    }
    if not allowed or st.user.get("sub") not in allowed:
        st.error("This account has not been granted workspace access.")
        if st.button("Sign out"):
            st.logout()
        st.stop()
    with st.sidebar:
        if st.button("Sign out", key="sign_out"):
            st.logout()
