
import streamlit as st

st.set_page_config(
    page_title="NovaTech Assistant",
    page_icon="💬",
    layout="centered"
)

st.title("NovaTech Internal Assistant")
st.caption("Choose your RAG implementation.")

st.divider()

option = st.selectbox(
    "Select RAG Approach",
    [
        "ChromaDB - Semantic Search",
        "TF-IDF - Keyword Search"
    ]
)

if st.button("Launch Assistant"):

    if option == "ChromaDB - Semantic Search":
        st.switch_page("streamlit_app.py")

    elif option == "TF-IDF - Keyword Search":
        st.switch_page("streamlit_app_tfidf.py")
