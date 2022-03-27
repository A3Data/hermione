import os
from pathlib import Path
import streamlit as st
from ..graphic_elements.st_functions import *

DATA_DIR = os.path.abspath(os.path.join(Path(__file__).resolve().parents[3], "data"))


class AnalysisPage:
    def __init__(self):
        st.title("Exploratory Analysis")
        self.df = load_data(os.path.join(DATA_DIR, "raw", "train.csv"))
        self.analysis_type = st.sidebar.radio("", ["Dataset Exploration", "Profiling"])

    def dataset_explo(self):
        st.write("## Dataset Exploration")
        st.write("")
        dataset_analysis(self.df)

    def profiling(self):
        st.write("## Pandas Profiling")
        st.write("")
        with st.form(key="profile_form"):
            st.write("### Profiling options")
            profiling_opts = {}
            col1, col2 = st.columns(2)
            with col1:
                st.write("Minimal computation?")
                minimal = st.radio(
                    "",
                    (True, False),
                    index=1,
                    format_func=lambda x: "Yes" if x else "No",
                )
                profiling_opts["minimal"] = minimal
            with col2:
                st.write("Should a sample of the data be displayed?")
                sample = st.radio(
                    "", (True, False), format_func=lambda x: "Yes" if x else "No"
                )
                if not sample:
                    profiling_opts["samples"] = None
            submit_button = st.form_submit_button(label="Run")
        if submit_button:
            profilling_analysis(self.df, **profiling_opts)

    def write(self):
        if self.analysis_type == "Dataset Exploration":
            self.dataset_explo()
        elif self.analysis_type == "Profiling":
            self.profiling()
