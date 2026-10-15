import os
import time

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from google import genai
from google.genai import errors
from sentence_transformers import SentenceTransformer


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="IPC GenAI Assistant",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ IPC GenAI Assistant")
st.write(
    "Ask questions about your IPC dataset. The application uses "
    "semantic retrieval (RAG) and Google Gemini to generate answers."
)


# =========================================================
# API KEY
# Local development: .streamlit/secrets.toml
# Streamlit Cloud: App settings -> Secrets
# Example TOML: GEMINI_API_KEY = "your-key"
# =========================================================
try:
    api_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    api_key = os.getenv("GEMINI_API_KEY", "")

if not api_key:
    st.error(
        "Gemini API key is missing. For local use, create "
        ".streamlit/secrets.toml with GEMINI_API_KEY = "
        '"your-key". For Streamlit Cloud, add the same key in App Settings → Secrets.'
    )
    st.stop()

client = genai.Client(api_key=api_key)


# =========================================================
# LOAD DATA, EMBEDDING MODEL, AND FAISS INDEX
# =========================================================
@st.cache_resource(show_spinner="Loading the IPC dataset and search model...")
def load_ipc_system():
    dataset_path = "ipc_dataset.csv"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Could not find '{dataset_path}'. Put your IPC CSV file "
            "in the same folder as app.py, or update dataset_path in app.py."
        )

    df = pd.read_csv(dataset_path)
    df = df.dropna(how="all").fillna("")

    if df.empty:
        raise ValueError("The IPC dataset has no records.")

    for column in df.columns:
        df[column] = df[column].astype(str)

    documents = []
    for _, row in df.iterrows():
        record = "\n".join(
            f"{column}: {row[column]}" for column in df.columns
        )
        documents.append(record)

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = embedding_model.encode(
        documents,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    vectors = np.asarray(vectors, dtype="float32")

    # Normalized vectors + inner product approximate cosine similarity.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    return df, documents, embedding_model, index


try:
    df, documents, embedding_model, index = load_ipc_system()
except Exception as exc:
    st.error(f"Could not initialize the IPC search system: {exc}")
    st.info(
        "Check that ipc_dataset.csv exists beside app.py and that "
        "requirements.txt dependencies installed successfully."
    )
    st.stop()

st.success(f"IPC dataset loaded successfully — {len(df):,} records.")


# =========================================================
# RETRIEVE RELEVANT DATASET RECORDS
# =========================================================
def search_ipc(question: str, top_k: int = 5):
    query_vector = embedding_model.encode(
        [question],
        normalize_embeddings=True,
    )
    query_vector = np.asarray(query_vector, dtype="float32")

    k = min(top_k, len(documents))
    scores, positions = index.search(query_vector, k)

    matches = []
    for score, position in zip(scores[0], positions[0]):
        if position >= 0:
            matches.append(
                {
                    "text": documents[int(position)],
                    "score": float(score),
                }
            )
    return matches


# =========================================================
# GEMINI GENERATION WITH RETRIES FOR TEMPORARY 503/429 ERRORS
# =========================================================
def ask_ipc_ai(question: str):
    sources = search_ipc(question, top_k=5)
    context = "\n\n---\n\n".join(item["text"] for item in sources)

    prompt = f"""
You are an educational assistant that answers questions using an IPC dataset.

Use the retrieved records below as the primary source. Do not invent IPC
section numbers, descriptions, or punishments. If the records do not contain
enough information, say that the information is not available in the dataset.
Explain clearly and simply. Do not present the response as legal advice.
If relevant, note that the IPC was replaced by the Bharatiya Nyaya Sanhita
(BNS) for most new offences from 1 July 2024; current law should be verified.

RETRIEVED DATASET RECORDS:
{context}

USER QUESTION:
{question}

Organize the answer with these headings where applicable:
- Relevant IPC Section
- Description
- Punishment (only if supported by the records)
- Simple Explanation
- Example (only if supported by the records)

End with a short educational-purpose disclaimer.
"""

    # Model names can change or be unavailable for some API keys.
    # Override with GEMINI_MODEL in environment/secrets if needed.
    model_name = st.secrets.get(
        "GEMINI_MODEL",
        os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
    )

    last_error = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            if not getattr(response, "text", None):
                raise RuntimeError(
                    "Gemini returned an empty response. Please try again."
                )
            return response.text, sources

        except errors.APIError as exc:
            last_error = exc
            code = getattr(exc, "code", None)

            # Retry temporary overload or rate-limit errors only.
            if code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(2 ** (attempt + 1))  # 2 sec, then 4 sec
                continue
            raise

    raise last_error


# =========================================================
# STREAMLIT UI
# =========================================================
with st.form("ipc_question_form"):
    question = st.text_area(
        "Enter your IPC question",
        placeholder="Example: What does IPC Section 420 cover?",
        height=90,
    )
    submitted = st.form_submit_button("🔎 Find Answer", type="primary")

if submitted:
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        try:
            with st.spinner(
                "Searching your dataset and asking Gemini. "
                "The first request may take a little longer..."
            ):
                answer, sources = ask_ipc_ai(question.strip())

            st.subheader("🤖 AI Answer")
            st.markdown(answer)

            with st.expander("📚 Retrieved dataset records"):
                for number, source in enumerate(sources, start=1):
                    st.markdown(f"**Source {number}**")
                    st.code(source["text"])
                    st.caption(
                        f"Similarity score: {source['score']:.3f}"
                    )

        except errors.APIError as exc:
            code = getattr(exc, "code", None)
            if code == 503:
                st.error(
                    "Gemini is temporarily overloaded (503). Please wait a "
                    "short while and try again. The app retried automatically."
                )
            elif code == 429:
                st.error(
                    "Gemini API rate limit or quota reached (429). Check the "
                    "quota and billing/usage limits for your API key, then retry."
                )
            else:
                st.error(f"Gemini API error: {exc}")
                st.caption(
                    "Check the model name and API key permissions. You can set "
                    "GEMINI_MODEL in Streamlit Secrets to a model available to your account."
                )
        except Exception as exc:
            st.error(f"An error occurred: {exc}")


# =========================================================
# FOOTER
# =========================================================
st.divider()
st.caption(
    "Educational reference only — not legal advice. The IPC was replaced "
    "by the Bharatiya Nyaya Sanhita (BNS) for most new offences from "
    "1 July 2024. Verify current law with authoritative sources."
)