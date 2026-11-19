import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "data")

st.set_page_config(
    page_title="NovaTech Assistant - TF-IDF",
    page_icon="💬",
    layout="centered"
)

st.title("NovaTech Internal Assistant")
st.caption("Ask anything about policies, products, or procedures using TF-IDF.")
st.divider()


# --------------------------------------------------
# LOAD DOCUMENTS AND CREATE TF-IDF VECTORS
# --------------------------------------------------

@st.cache_resource
def load_tfidf():

    if not os.path.exists(DATA_FOLDER):
        st.error(f"_data/ folder not found at: {DATA_FOLDER}")
        st.stop()

    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY is missing. Check your .env file.")
        st.stop()

    groq_client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )

    all_chunks = []
    all_sources = []

    for filename in sorted(os.listdir(DATA_FOLDER)):

        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(DATA_FOLDER, filename)

        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        for para in text.strip().split("\n\n"):

            para = para.strip()

            if len(para) < 50:
                continue

            all_chunks.append(para)
            all_sources.append(filename)

    if not all_chunks:
        st.error("No valid text documents found in _data folder.")
        st.stop()

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(all_chunks)

    return groq_client, vectorizer, tfidf_matrix, all_chunks, all_sources


with st.spinner("Loading and processing company documents..."):

    groq_client, vectorizer, tfidf_matrix, all_chunks, all_sources = load_tfidf()

st.success(
    f"Ready - {len(all_chunks)} document chunks indexed.",
    icon="✅"
)

st.divider()


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# TF-IDF RAG PIPELINE
# --------------------------------------------------

def ask_rag(question: str) -> dict:

    # RETRIEVE
    query_vector = vectorizer.transform([question])

    similarity_scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    top_indices = similarity_scores.argsort()[-3:][::-1]

    relevant_indices = [
        i for i in top_indices
        if similarity_scores[i] > 0
    ]

    if not relevant_indices:
        return {
            "answer": "I don't have enough information to answer this.",
            "sources": [],
            "chunks": []
        }

    chunks = [all_chunks[i] for i in relevant_indices]

    sources = [all_sources[i] for i in relevant_indices]

    # AUGMENT
    context = "\n\n---\n\n".join(chunks)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful NovaTech company assistant. "
                "Answer questions using ONLY the provided context. "
                "If the context does not contain enough information, "
                "say 'I don't have enough information to answer this.' "
                "Be concise and direct."
            )
        },
        {
            "role": "user",
            "content": f"Context: {context}\n\nQuestion: {question}"
        }
    ]

    # GENERATE
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=0.2
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": list(set(sources)),
        "chunks": list(zip(chunks, sources))
    }


# --------------------------------------------------
# DISPLAY PREVIOUS CHAT MESSAGES
# --------------------------------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("sources"):

            st.caption(
                f"Sources: {', '.join(msg['sources'])}"
            )

            with st.expander("View retrieved document chunks"):

                for i, (chunk_text, source_file) in enumerate(
                    msg["chunks"], 1
                ):

                    st.markdown(
                        f"**Chunk {i} - '{source_file}'**"
                    )

                    st.info(chunk_text)


# --------------------------------------------------
# USER INPUT AND CHAT RESPONSE
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about NovaTech policies or products..."
)

if question:

    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("assistant"):

        with st.spinner("Searching documents and generating answer..."):

            try:
                result = ask_rag(question)

                st.markdown(result["answer"])

                if result["sources"]:

                    st.caption(
                        f"Sources: {', '.join(result['sources'])}"
                    )

                    with st.expander("View retrieved document chunks"):

                        for i, (chunk_text, source_file) in enumerate(
                            result["chunks"], 1
                        ):

                            st.markdown(
                                f"**Chunk {i} - '{source_file}'**"
                            )

                            st.info(chunk_text)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "chunks": result["chunks"]
                })

            except Exception as e:
                st.error(f"An error occurred: {e}")