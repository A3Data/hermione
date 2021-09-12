import streamlit as st
from PIL import Image

class IntroPage:

    def __init__(self):
        pass

    def write(self):
        image = Image.open("images/hermione_logo.png")
        st.title("Hermione Titanic Example")
        st.write("")
        st.image(image, width=500)
        st.write("")
        st.write(
            """
            ### Purpose
            This is an example app used to show how it is possible to make interactive apps to serve as a 
            high-level, user-friendly and easy-to-make deploy for machine learning models or analytic applications.

            ### Context
            The dataset explored is the commonly used Titanic dataset, that carries out information about some of the passengers present
            in the ship, and whether they die or not.

            ### Questions?
            You can learn more in the Streamlit [official website](https://streamlit.io/) and 
            [community forum](https://discuss.streamlit.io).

            """
        )