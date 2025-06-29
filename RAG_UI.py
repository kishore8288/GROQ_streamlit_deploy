import streamlit as st
#import fitz  # PyMuPDF
import pdfplumber
import faiss
import numpy as np
import requests
import os
from dotenv import load_dotenv
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize

# Load environment variable
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)
API_KEY = os.getenv("GROQ_API_KEY")

# Set up Streamlit
st.set_page_config(page_title="RAG Assistant", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff;
        color: #1a1a1a;
        font-family: "Segoe UI", sans-serif;
    }

    /* Transparent input and textarea */
    input, textarea, .stTextInput>div>div>input {
        background-color: rgba(255, 255, 255, 0.7);
        color: #1a1a1a;
        border: 1px solid #ccc;
        border-radius: 6px;
    }

    .stTextInput>div>div {
        background-color: transparent;
    }

    /* Transparent file uploader */
    .stFileUploader {
        background-color: transparent;
    }

    /* Optional: hide the label of file uploader */
    label[for^="fileUploader"] {
        display: none;
    }

    /* Upload icon styling */
    .upload-wrapper {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1rem;
    }

    /* Custom button style */
    .stButton>button {
        background-color: rgba(0,0,0,0.8);
        color: white;
        border-radius: 6px;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧠 Coding & Document Q&A Assistant")

# Load models
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return embedder

embedder = load_models()

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Split text into chunks
def better_split(text, chunk_size=300, overlap=50):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-(overlap // len(sentence.split())):]
            current_len = sum(len(s) for s in current_chunk)
        current_chunk.append(sentence)
        current_len += len(sentence)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# Build prompt from query and context
def build_prompt(query, chunks, index, top_k=3):
    if not query.strip():
        return None

    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    selected_chunks = [chunks[i] for i in indices[0][:top_k]]
    context = "\n".join(selected_chunks)

    prompt = f"""
You are a helpful assistant. Use the document context below if it's relevant to the question.
Otherwise, use your general knowledge. Don't mention it in your answer.

[Context Start]
{context}
[Context End]

User's Question:
{query}

Answer:
"""
    return prompt

# Initialize session state
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index" not in st.session_state:
    st.session_state.index = None
if "history" not in st.session_state:
    st.session_state.history = []

# File upload and processing
with st.container():
    st.markdown('<div class="upload-wrapper">📎 Upload Document:', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

if uploaded_file:
    with st.spinner("📖 Reading and processing your document..."):
        text = extract_text_from_pdf(uploaded_file)
        chunks = better_split(text)
        chroma_client = chromadb.Client()
        embed_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        collection = chroma_client.create_collection(name="mydocs", embedding_function=embed_fn)
        collection.add(documents=chunks, ids=[str(i) for i in range(len(chunks))])
        st.session_state.chunks = chunks
        st.session_state.collections = collection
        st.success("✅ Document processed successfully!")

# Input query
query = st.text_input("💬 Ask a question:", placeholder="E.g., What are symptoms of low blood sugar?")

# On generate
if st.button("Generate"):
    if not query.strip():
        st.warning("Please enter a valid question.")
        st.stop()

    # Build prompt using RAG
    prompt = build_prompt(query, st.session_state.chunks, st.session_state.index)

    # Build full chat message history
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.extend(st.session_state.history)  # Keep past turns
    messages.append({"role": "user", "content": prompt})  # New turn with context

    # Groq API call
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-70b-8192",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 5000
    }

    with st.spinner("Generating response from LLaMA 3..."):
        res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)

    if res.status_code == 200:
        result = res.json()["choices"][0]["message"]["content"]
        # Save to history
        st.session_state.history.append({"role": "user", "content": prompt})
        st.session_state.history.append({"role": "assistant", "content": result})

        # Display result
        st.success("🤖 Assistant says:")
        st.markdown(result)
    else:
        st.error(f"Error {res.status_code}: {res.text}")

# Display history
if st.session_state.history:
    st.markdown("### 🗃️ Chat History")
    for i in range(0, len(st.session_state.history), 2):
        user = st.session_state.history[i]
        assistant = st.session_state.history[i+1] if i+1 < len(st.session_state.history) else {"content": ""}
        st.markdown(f"**🧑‍💻 You:** {user['content']}")
        st.markdown(f"**🤖 Assistant:** {assistant['content']}")
        st.markdown("---")
