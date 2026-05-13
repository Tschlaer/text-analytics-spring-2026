# Business Memo: "Ask My Resume" RAG Chatbot

**To:** To Who It May Concern
**From:** Thomas Schlaerth
**Date:** May 13, 2026
**Re:** Assignment 5 Option A — RAG Chatbot Implementation & Findings

---

## Executive Summary

This project implements a Retrieval-Augmented Generation (RAG) chatbot designed to let recruiters and hiring managers ask natural language questions about my professional background. The chatbot retrieves relevant passages from four personal career documents and generates grounded, professional answers using a free HuggingFace LLM. Evaluation across 10 test questions revealed strong performance on out-of-scope handling and inference reasoning, while factual retrieval from specific documents was the system's primary weakness due to chunk-level retrieval mismatches.

---

## Project Overview

A recruiter or hiring manager should be able to interact with this chatbot the way they would with a knowledgeable HR contact — asking plain questions and receiving accurate, document-grounded answers without hallucinated details. The system was built entirely on the free HuggingFace path at zero API cost.

**Documents loaded (4 PDFs):**
- `Thomas_Schlaerth_Resume_2.16.26.pdf` — primary skills and experience document
- `Thomas Schlaerth Cover Letter - Carson .pdf` — logistics role application
- `DBS Cover Letter.pdf` — Dublin Business School internship cover letter
- `Reference for Thomas Schlaerth.pdf` — professional reference letter

---

## Technical Approach

### Document Loading
All four PDFs were loaded using LangChain's `PyPDFLoader`, with each page tagged with its source filename in metadata for traceability during retrieval.

### Chunking Strategy
Two strategies were compared at identical settings (`chunk_size=500, chunk_overlap=50`):

| Strategy | Chunk Count | Avg Length | Min / Max |
|----------|-------------|------------|-----------|
| Fixed-size (CharacterTextSplitter) | 14 | 1,063 chars | 123 / 3,581 |
| Recursive (RecursiveCharacterTextSplitter) | 37 | 430 chars | 93 / 500 |

The **Recursive strategy was selected** because career documents are organized into short, meaningful paragraphs and bullet points. The fixed-size splitter produced 14 very large chunks that blended unrelated content, while the recursive splitter produced 37 focused chunks that preserved paragraph and bullet-point boundaries — resulting in more targeted retrieval.

### Embedding and Vector Store
Chunks were embedded using `sentence-transformers/all-MiniLM-L6-v2` running locally on CPU and stored in ChromaDB (185 total vectors). This model was selected for its strong semantic similarity performance on short passages and zero API cost.

### LLM
`Qwen/Qwen2.5-7B-Instruct` via HuggingFace free Inference API. Two prior models (`zephyr-7b-beta`, `Mistral-7B-Instruct-v0.3`) were found unsupported by HuggingFace's current free router; Qwen was confirmed compatible and produced high-quality instruction-following responses.

### Prompt Engineering
Three system prompt iterations were developed and tested using the question *"What industry is this person trying to enter?"*:

**v1 — Basic instruction only**
- Prompt: "You are a helpful assistant. Answer questions about this person's resume."
- Result: Verbose answers that drew from training data in addition to retrieved context. No grounding constraint.

**v2 — Grounding constraint added**
- Added: Answer ONLY from context; fallback phrase if not found.
- Result: Concise, context-bound responses. Out-of-scope questions declined correctly. Tone inconsistent (mixed first/second-person).

**v3 — Professional tone + format rules (Final)**
- Added: Third-person professional tone, document citation instruction, sensitive topic block.
- Result: Consistently professional, recruiter-appropriate responses. v2 returned bullet lists in first-person; v3 returned prose in third-person. Example: v2 → "This person has the following skills: [list]" vs. v3 → "The candidate has proficiency in the Microsoft Office Suite, Python 3, Tableau, Figma, and Jira."

---

## Evaluation Results

10 questions were tested across 4 categories using the final prompt (v3) and k=3 retrieval:

| Category | Question | Retrieval | Faithfulness | Quality | Notes |
|----------|----------|:---------:|:------------:|:-------:|-------|
| Factual | What programming languages does this person know? | 1 | 4 | 1 | Wrong document retrieved |
| Factual | What is this person's educational background? | 1 | 5 | 1 | Answer not in retrieved chunks |
| Factual | What tools or software does this person have experience with? | 1 | 3 | 1 | Wrong document retrieved |
| Inference | Would this person be a good fit for a data engineering role? | 4 | 5 | 5 | Strong reasoning from context |
| Inference | Does this person have leadership or team experience? | 5 | 5 | 5 | Correct source, clear answer |
| Inference | What industries or domains has this person worked in? | 1 | 1 | 1 | Not retrieved from correct doc |
| Out-of-scope | What is this person's salary expectation? | 5 | 5 | 5 | Declined correctly |
| Out-of-scope | What is this person's home address? | 5 | 5 | 5 | Declined correctly |
| Specificity | Describe the most recent project this person worked on. | 3 | 4 | 3 | Retrieved an older project |
| Specificity | What were this person's responsibilities in their last role? | 2 | 4 | 3 | Retrieved wrong role |

*Scale: 1 (poor) – 5 (excellent)*

---

## Key Findings

**1. Out-of-scope handling was excellent.** Both sensitive questions (salary, home address) were declined with the correct fallback phrase every time. The v3 grounding constraint proved reliable for protecting against inappropriate responses.

**2. Inference questions outperformed factual questions.** Leadership and data engineering fit questions scored 5/5/5 because they required synthesizing evidence rather than locating a specific fact. When retrieval was correct, the LLM reasoned well across chunks.

**3. Factual retrieval was the primary failure mode.** Three of four factual questions scored 1 on answer quality because the retriever returned chunks from the wrong document. The programming languages question retrieved cover letter text instead of the resume skills section. Critically, this is a retrieval problem, not a generation problem — the LLM correctly declined when the context it received did not contain the answer.

**4. Specificity questions showed partial retrieval.** Both specificity questions retrieved plausible but imprecise documents — returning an older project instead of the most recent one, for example — suggesting the semantic embeddings struggle to distinguish temporal ordering ("most recent") from general topic similarity.

---

## Limitations and Future Improvements

- **Increase k from 3 to 5** to cast a wider retrieval net across all four documents for factual questions
- **Add metadata filtering** to direct skill queries to the resume and experience queries to cover letters
- **Improve PDF chunking** for the DBS Cover Letter, which had spacing artifacts causing character-by-character tokenization that degraded chunk quality
- **Add reranking** of retrieved chunks before generation to surface the most relevant passage beyond embedding similarity alone

---

*Total API cost: $0.00 — free HuggingFace Inference API + local sentence-transformers embeddings*
*LLM: Qwen/Qwen2.5-7B-Instruct | Embeddings: all-MiniLM-L6-v2 | Vector store: ChromaDB*

