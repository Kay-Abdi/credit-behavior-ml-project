import streamlit as st

st.set_page_config(
    page_title="Customer Financial Behavior & Credit Risk",
    layout="wide",
)

pages = [
    st.Page("pages/Introduction.py", title="Introduction"),
    st.Page("pages/Data_Prep_EDA.py", title="Data Prep EDA"),
    st.Page("pages/ARM.py", title="ARM"),
    st.Page("pages/Clustering.py", title="Clustering"),
    st.Page("pages/DT.py", title="DT"),
    st.Page("pages/NB.py", title="NB"),
    st.Page("pages/PCA.py", title="PCA"),
    st.Page("pages/Regression.py", title="Regression"),
    st.Page("pages/SVM.py", title="SVM"),
    st.Page("pages/Conclusions.py", title="Conclusions"),
]

nav = st.navigation(pages, position="sidebar")
nav.run()
