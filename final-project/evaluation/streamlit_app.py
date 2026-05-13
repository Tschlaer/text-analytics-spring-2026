"""
Assignment 5 -- Option A: "Ask My Resume" RAG Chatbot
BSAN 6200 | Spring 2026 | Thomas Schlaerth

Run with: python -m streamlit run streamlit_app.py
"""

import streamlit as st
import chromadb
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Ask My Resume", page_icon="📄")

# ── Config ──
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


# ══════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════

def load_text_file(filepath):
    """Load a .txt file and return its content."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf_file(filepath):
    """Load a .pdf file and return its text content."""
    from pypdf import PdfReader
    reader = PdfReader(filepath)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_all_documents(data_dir="data"):
    """Load all .txt and .pdf files from the data directory."""
    docs = []
    if not os.path.exists(data_dir):
        return docs
    for filename in sorted(os.listdir(data_dir)):
        filepath = os.path.join(data_dir, filename)
        if filename.endswith(".txt"):
            text = load_text_file(filepath)
        elif filename.endswith(".pdf"):
            text = load_pdf_file(filepath)
        else:
            continue
        if text.strip():
            docs.append({"text": text, "source": filename})
    return docs


def ask_llm(hf_client, prompt):
    """Send a prompt to the LLM and return the response."""
    try:
        response = hf_client.chat_completion(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.1,
            stop=["Context:", "Question:", "\n\n\n"],
            provider="hf-inference",
        )
    except TypeError:
        response = hf_client.chat_completion(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.1,
            stop=["Context:", "Question:", "\n\n\n"],
        )
    answer = response.choices[0].message.content.strip()
    for cutoff in ["Context:", "Question:", "\n\n\n"]:
        if cutoff in answer:
            answer = answer[:answer.index(cutoff)].strip()
    return answer


# ══════════════════════════════════════════
# Chunking strategy — Recursive character splitting
#
# Compared against fixed-size in notebook Section 3:
#   Fixed-size:  14 chunks, avg 1,063 chars
#   Recursive:   37 chunks, avg   430 chars  ← chosen
#
# Recursive splitting tries paragraph breaks (\n\n) first,
# then newlines (\n), then spaces — preserving bullet points
# and sentence boundaries that fixed-size splitting breaks.
# ══════════════════════════════════════════

def _recursive_split(text, chunk_size, chunk_overlap, separators):
    """Recursively split text, trying each separator in order."""
    if len(text) <= chunk_size:
        return [text]

    for i, sep in enumerate(separators):
        if sep == "":
            # Last resort: hard cut
            return [
                text[j:j + chunk_size]
                for j in range(0, len(text), chunk_size - chunk_overlap)
            ]

        splits = text.split(sep)
        if len(splits) == 1:
            continue  # Separator not found — try next

        chunks = []
        current = ""
        for split in splits:
            candidate = current + (sep if current else "") + split
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(split) > chunk_size:
                    chunks.extend(_recursive_split(split, chunk_size, chunk_overlap, separators[i + 1:]))
                    current = ""
                else:
                    current = split
        if current:
            chunks.append(current)

        # Add overlap between adjacent chunks
        if chunk_overlap > 0 and len(chunks) > 1:
            overlapped = [chunks[0]]
            for j in range(1, len(chunks)):
                tail = chunks[j - 1][-chunk_overlap:]
                overlapped.append(tail + " " + chunks[j])
            return overlapped

        return chunks

    return [text]


def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    """
    Split documents into chunks using recursive character splitting.

    Tries to split on paragraph breaks first (\n\n), then newlines (\n),
    then spaces — preserving bullet points and paragraph structure.

    Input:  list of dicts [{"text": "...", "source": "filename.pdf"}, ...]
    Output: list of dicts [{"text": "chunk text", "source": "filename.pdf"}, ...]
    """
    separators = ["\n\n", "\n", " ", ""]
    chunks = []

    for doc in documents:
        doc_chunks = _recursive_split(doc["text"], chunk_size, chunk_overlap, separators)
        for chunk_text in doc_chunks:
            if chunk_text.strip():
                chunks.append({"text": chunk_text.strip(), "source": doc["source"]})

    return chunks


# ══════════════════════════════════════════
# System prompt — v3 (Final)
#
# Iteration history (documented in notebook Section 6):
#   v1: Basic instruction — no grounding, model drew from training data
#   v2: Added grounding constraint + fallback phrase — context-bound
#   v3: Added professional tone, third-person, sensitive topic guard ← this
# ══════════════════════════════════════════

SYSTEM_PROMPT = """You are a professional assistant representing a job candidate to recruiters and hiring managers.

INSTRUCTIONS:
- Answer using ONLY the information provided in the context below.
- Do not invent, assume, or extrapolate facts not in the context.
- If the context does not contain enough information, respond with:
  "That information is not available in the candidate's documents."
- Write in third-person professional tone (e.g., "The candidate has...").
- Cite the specific document or project when relevant.
- Do not discuss salary, compensation, or personal contact information."""


# ══════════════════════════════════════════
# Load resources (cached)
# ══════════════════════════════════════════

@st.cache_resource
def load_vectorstore():
    documents = load_all_documents("data")
    if not documents:
        return None, []

    chunks = chunk_documents(documents)
    client = chromadb.Client()
    collection = client.create_collection("resume_rag")
    collection.add(
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"]} for c in chunks],
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )
    return collection, documents


@st.cache_resource
def load_llm():
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        return None
    return InferenceClient(token=token)


# ══════════════════════════════════════════
# RAG logic
# ══════════════════════════════════════════

def search(collection, query, k=3):
    """Retrieve top-k chunks from the vector store."""
    results = collection.query(query_texts=[query], n_results=k)
    return results["documents"][0], [m["source"] for m in results["metadatas"][0]]


def ask_rag(collection, hf_client, question, k=3):
    """Full RAG pipeline: retrieve -> build prompt -> generate."""
    docs, sources = search(collection, question, k=k)
    context = "\n\n".join(docs)

    full_prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

Question: {question}

Answer:"""

    answer = ask_llm(hf_client, full_prompt)
    return answer, sources, docs


# ══════════════════════════════════════════
# UI
# ══════════════════════════════════════════

collection, raw_docs = load_vectorstore()
hf_client = load_llm()

st.title("📄 Ask My Resume")
st.caption("Ask me anything about my skills, experience, and projects.")

# ── Error checks ──
if not hf_client:
    st.error("HF_TOKEN not found. Add it to your .env file.")
    st.stop()

if collection is None:
    st.error("No documents found in data/ folder. Add your resume and other files there.")
    st.stop()

# ── Sidebar ──
with st.sidebar:
    st.header("About")
    st.write("This chatbot answers questions about my professional background using RAG.")
    st.write(f"**Documents loaded:** {len(raw_docs)}")
    for d in raw_docs:
        st.write(f"- {d['source']}")
    st.divider()
    st.write(f"**Chunks in vector store:** {collection.count()}")
    st.write(f"**Model:** {MODEL_ID}")
    st.divider()
    st.caption("BSAN 6200 | Assignment 5 | Option A")

# ── Sample questions ──
st.write("**Try a sample question:**")
samples = [
    "What technical skills does this person have?",
    "Does this person have leadership experience?",
    "What industry is this person trying to enter?",
]
cols = st.columns(len(samples))
for i, q in enumerate(samples):
    if cols[i].button(q, key=f"sample_{i}"):
        st.session_state["pending_question"] = q

# ── Chat interface ──
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📎 Retrieved chunks"):
                st.markdown(msg["sources"])

user_input = st.chat_input("Ask me about my background...")

if "pending_question" in st.session_state:
    user_input = st.session_state.pop("pending_question")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            answer, sources, docs = ask_rag(collection, hf_client, user_input)

            st.markdown(answer)

            source_text = ""
            for i, (doc, src) in enumerate(zip(docs, sources)):
                source_text += f"**Chunk {i+1}** ({src}):\n> {doc[:200]}...\n\n"

            with st.expander("📎 Retrieved chunks"):
                st.markdown(source_text)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": source_text,
            })

        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")
