
import os
import json
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
import chromadb

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "data")

st.set_page_config(
    page_title="NovaTech Assistant",
    page_icon="💬",
    layout="centered"
)

st.title("NovaTech Internal Assistant")
st.caption("Ask anything about policies, products, or procedures.")
st.divider()


@st.cache_resource
def load_rag():
    """
    Read all .txt files from _data/, chunk by paragraph,
    and index in ChromaDB.
    """

    if not os.path.exists(DATA_FOLDER):
        st.error(f"_data/ folder not found at: {DATA_FOLDER}")
        st.stop()

    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )

    chroma = chromadb.EphemeralClient()

    collection = chroma.create_collection("docs")

    all_chunks = []
    all_ids = []
    all_metadatas = []

    chunk_id = 0

    for filename in sorted(os.listdir(DATA_FOLDER)):

        if not filename.endswith(".txt"):
            continue

        with open(os.path.join(DATA_FOLDER, filename), "r") as f:
            text = f.read()

        for para in text.strip().split("\n\n"):

            para = para.strip()

            if len(para) < 50:
                continue

            all_chunks.append(para)
            all_ids.append(f"chunk_{chunk_id}")

            all_metadatas.append({
                "source": filename,
                "text": para
            })

            chunk_id += 1

    collection.add(
        documents=all_chunks,
        ids=all_ids,
        metadatas=all_metadatas
    )

    return collection, client, chunk_id


with st.spinner("Loading and indexing company documents..."):
    collection, groq_client, total_chunks = load_rag()

st.success(
    f"Ready - {total_chunks} document chunks indexed.",
    icon="✅"
)

st.divider()


# CHAT HISTORY

if "messages" not in st.session_state:
    st.session_state.messages = []


def ask_rag(question: str) -> dict:
    """
    Execute the full RAG pipeline: Retrieve -> Augment -> Generate.
    """

    # RETRIEVE
    results = collection.query(
        query_texts=[question],
        n_results=3
    )

    chunks = results["documents"][0]
    sources = [
        m["source"] for m in results["metadatas"][0]
    ]

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


# DISPLAY CHAT HISTORY

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("sources"):

            st.caption(f"Sources: {', '.join(msg['sources'])}")

            with st.expander("View retrieved document chunks"):

                for i, (chunk_text, source_file) in enumerate(
                    msg["chunks"], 1
                ):
                    st.markdown(
                        f"**Chunk {i} - '{source_file}'**"
                    )
                    st.info(chunk_text)


# USER INPUT

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

        with st.spinner(
            "Searching documents and generating answer..."
        ):
            try:
                result = ask_rag(question)

                st.markdown(result["answer"])

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
