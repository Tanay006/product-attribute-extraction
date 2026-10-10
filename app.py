"""CA3 - E-commerce Product Attribute Extraction (spaCy NER) - Streamlit app.

Run:  streamlit run app.py
"""
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)                       # the project scripts use paths relative to the project folder
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Attribute Extraction | CA3", page_icon=":material/checkroom:",
                   layout="wide", initial_sidebar_state="collapsed")

import ui  # noqa: E402

ui.inject_css()

pages = [
    st.Page("views/extraction_page.py", title="NLP Model + Attribute Extraction",
            icon=":material/manage_search:", default=True),          # default page is served at "/"
    st.Page("views/evaluation_page.py", title="Model Evaluation + Audit",
            icon=":material/fact_check:", url_path="evaluation"),
]
st.navigation(pages, position="top").run()
