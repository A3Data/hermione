import streamlit as st
import pandas as pd
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import altair as alt
import plotly.graph_objects as go


@st.cache
def load_data(path):
    dataframe = pd.read_csv(path)
    dataframe["Survived"] = dataframe["Survived"].replace([0, 1], ["Died", "Survived"])
    return dataframe


def profilling_analysis(df, **kwargs):
    try:
        pr = ProfileReport(df, explorative=True, **kwargs)
        st_profile_report(pr)
    except:
        st.error("Error - Pandas profiling was not generated")


def dataset_analysis(df):
    survived = ["All"]
    survived.extend(df["Survived"].unique())

    selected = st.selectbox("Survived:", survived)
    if selected == "All":
        st.write(df)
    else:
        st.write(df[df["Survived"] == selected])

    if st.checkbox("Graphical Display", False):
        st.subheader("Dataset Graphical Display")

        st.altair_chart(
            alt.Chart(df)
            .mark_circle()
            .encode(
                alt.X("Age", scale=alt.Scale(zero=False)),
                alt.Y("Fare", scale=alt.Scale(zero=False, padding=1)),
                color="Survived",
                size="Pclass",
                tooltip=["Age", "Survived", "Sex", "Pclass"],
            )
            .interactive(),
            use_container_width=True,
        )
    if st.checkbox("Show Summary", False):
        st.write(df.describe())


def velocimeter_chart(value):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": "Probability of Survival"},
            gauge={"axis": {"range": [0, 1]}},
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )
    st.plotly_chart(fig)
