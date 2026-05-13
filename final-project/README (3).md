# Ask My Resume — RAG Chatbot

**Assignment 5 | Option A | BSAN 6200 Text Mining & Social Media Analytics | Spring 2026**

---

## 1. Project Title and Option

**"Ask My Resume" RAG Chatbot — Option A**

## 2. Your Name

Thomas Schlaerth

## 3. Project Description

A Retrieval-Augmented Generation (RAG) chatbot that lets recruiters and hiring managers ask natural language questions about my professional background. The chatbot retrieves relevant passages from four personal career documents — my resume, two cover letters, and a professional reference letter — and generates grounded, professional answers using a free HuggingFace LLM.

The system is designed so that answers are always traceable to actual documents. If a question cannot be answered from the documents (e.g., salary expectations or personal contact info), the chatbot declines rather than guessing.

## 4. Setup Instructions

**Prerequisites:** Python 3.10+, a free HuggingFace account with a READ token

```bash
# 1. Clone the repo
git clone https://github.com/Tschlaer/text-analytics-spring-2026.git
cd text-analytics-spring-2026/final-project

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your HuggingFace token — create a .env file in the project root
echo "HF_TOKEN=your_token_here" > .env

# 5. Confirm career documents are in the data/ folder
ls data/

# 6. Run the Streamlit app
streamlit run streamlit_app.py
```

**Get a HuggingFace READ token:** https://huggingface.co/settings/tokens

**Google Colab users:** Use Colab Secrets instead of a .env file:
1. Click the 🔑 key icon in the left sidebar
2. Add secret name `HF_TOKEN` with your token as the value
3. Toggle "Notebook access" to ON

## 5. Models and Tools Used

| Component | Tool / Model |
|-----------|-------------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| Vector Store | ChromaDB (local, persisted to disk) |
| LLM | `Qwen/Qwen2.5-7B-Instruct` via HuggingFace Inference API |
| RAG Framework | LangChain + LangChain Community |
| UI | Streamlit |
| Development Environment | Google Colab |

## 6. Paid vs. Free Path

**Free path used throughout — total API cost: $0.00**

| Component | Free Solution Used |
|-----------|-------------------|
| Embeddings | `all-MiniLM-L6-v2` runs fully locally via `sentence-transformers` |
| LLM | HuggingFace Inference API with a free READ token |
| Vector Store | ChromaDB persisted locally — no cloud cost |

Note: Two other HuggingFace models were attempted before Qwen (`zephyr-7b-beta` and `Mistral-7B-Instruct-v0.3`) but were rejected by HuggingFace's free router. `Qwen/Qwen2.5-7B-Instruct` was confirmed compatible.

## 7. Key Findings

- **Recursive chunking (37 chunks, avg 430 chars) outperformed fixed-size chunking (14 chunks, avg 1,063 chars)** — smaller chunks aligned better with resume bullet points and paragraph structure, improving retrieval precision
- **Prompt v3 (professional tone + grounding constraint) outperformed v1 and v2** — out-of-scope questions (salary, address) were declined correctly in every test, and factual answers were consistently written in third-person professional tone
- **Out-of-scope and inference questions scored highest** — the chatbot scored 5/5/5 on both out-of-scope questions and performed well on inference questions requiring reasoning across documents
- **Factual retrieval was the primary weakness** — all three factual questions scored 1 on answer quality because the vector store returned chunks from the wrong document; the resume skills section was frequently bypassed in favor of cover letter chunks
- **Faithfulness remained high (avg 4.1/5) even when retrieval failed** — the model correctly declined to answer rather than hallucinating, which validates the grounding constraint

## 8. File Descriptions

```
final-project/
├── README.md                   — This file (all 8 required sections)
├── requirements.txt            — All Python dependencies with pinned versions
├── .gitignore                  — Excludes .env, __pycache__, chroma_db/
├── .env                        — HF_TOKEN (not committed to GitHub)
├── streamlit_app.py            — Streamlit chat UI with sidebar, sample
│                                 questions, chunk display, error handling
├── memo.md                     — Business memo with technical findings
├── ai_log.md                   — AI usage log (8 entries with progression)
├── data/
│   ├── Thomas_Schlaerth_Resume_2.16.26.pdf
│   ├── Thomas Schlaerth Cover Letter - Carson .pdf
│   ├── DBS Cover Letter.pdf
│   └── Reference for Thomas Schlaerth.pdf
├── notebooks/
│   └── Thomas_Schlaerth_A5_OptionA_Resume_RAG.ipynb
│       — Full RAG pipeline: document loading, chunking comparison,
│         embedding, vector store, retrieval chain, 3 prompt iterations,
│         and 10-question evaluation with analysis
└── evaluation/
    ├── test_results.csv        — 10-question evaluation scores (machine-readable)
    └── test_results.md         — Full evaluation table, LLM outputs,
                                  category averages, and written analysis
```
