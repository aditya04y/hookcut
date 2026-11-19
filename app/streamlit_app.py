"""HookCut: the app entry point (Streamlit multipage). Run: streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="HookCut", page_icon="✂️", layout="wide")

from app.hr import data, style  # noqa: E402

style.inject()
data.copy_cloud_secrets()
st.navigation([
    st.Page("hr/page_try.py", title="Try one episode", default=True),
    st.Page("hr/page_show.py", title="A whole show"),
    st.Page("hr/page_how.py", title="How it works"),
    st.Page("workbench.py", title="Workbench"),
], position="top").run()
