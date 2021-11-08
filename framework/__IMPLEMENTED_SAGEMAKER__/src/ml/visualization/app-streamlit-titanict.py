import streamlit as st
from vega_datasets import data
import pandas as pd
import altair as alt
import sys
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

@st.cache
def load_data():
    dataframe = pd.read_csv("../../../data/raw/train.csv")
    dataframe["Survived"] = dataframe["Survived"].replace([0,1],["Died", "Survived"])
    return dataframe

def instructions():
  
    st.markdown(
    """
    Sample streamlit page using the Titanic dataset.
    This library is interesting for presenting the results and generating a web page.
     
    
    ### Questions?
    
    Streamlit community  -> https://discuss.streamlit.io
    """)

def dataset_analysis(df):
    survived = ["All"]
    survived.extend(df["Survived"].unique())
    
    selected = st.selectbox("Survived:", survived)
    if selected == "All":
        st.write('## Dataset Titanic', df)
    else:
        st.write('## Dataset Titanic', df[df["Survived"] == selected])
    
    if st.checkbox("Graphical Display", False):
        st.subheader("Dataset Graphical Display")
        
        st.altair_chart(alt.Chart(df).mark_circle().encode(
                        alt.X('Age', scale=alt.Scale(zero=False)),
                        alt.Y('Fare', scale=alt.Scale(zero=False, padding=1)),
                        color='Survived',
                        size='Pclass',
                        tooltip=['Age','Survived', 'Sex', 'Pclass'],
                    ).interactive(), use_container_width=True)
    if st.checkbox("Show Summary", False):
        st.write(df.describe())

        
def profilling_analysis(df):
    try:  
        pr = ProfileReport(df, explorative=True)
        st.title("Pandas Profiling in Streamlit")
        st.write(df)
        st_profile_report(pr)
    except:
        st.title("Error - Pandas profiling was not generated")


def main(): 
    st.title("Titanic Dataset")
    
    df = load_data()
        
    st.sidebar.title("What to do")
    menu = ["Instructions", "DataSet Exploration - Profilling", "DataSet Exploration - General"]
    app_mode = st.sidebar.selectbox("Select an option:",
        menu)
    if app_mode == menu[0]:
        st.sidebar.success('Next "'+menu[1]+'".')
        instructions()
    elif app_mode == menu[1]:
        st.sidebar.success('Next "'+menu[2]+'".')
        profilling_analysis(df)
    elif app_mode == menu[2]:
        #st.sidebar.success('Para continuar selecione "'+menu[3]+'".')
        dataset_analysis(df)
        
            

if __name__ == "__main__":
    main()