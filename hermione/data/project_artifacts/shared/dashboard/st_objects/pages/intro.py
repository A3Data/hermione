import streamlit as st
from PIL import Image
import requests
from io import BytesIO

url = "https://github.com/A3Data/hermione/blob/master/images/vertical_logo.png?raw=true"
response = requests.get(url)
img = Image.open(BytesIO(response.content))
img.load()


class IntroPage:
    def __init__(self):
        pass

    def write(self):
        st.title("Hermione Titanic Example")
        st.write("")
        st.image(img, width=500)
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
