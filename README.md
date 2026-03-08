<div align='center'>
    <h1 align='center'> GenAI RAG Document Assistant by Aman </h1>
    <p align='center'> A modern Retrieval-Augmented Generation (RAG) web app where users upload one or more PDF documents, index them with FAISS + HuggingFace embeddings, and ask questions answered by a local Ollama LLM with ChatGPT-style streaming responses. </p>
    <div>
        <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white">
        <img src="https://img.shields.io/badge/Flask-Web%20App-000000?logo=flask&logoColor=white">
        <img src="https://img.shields.io/badge/LangChain-RAG-121212">
        <img src="https://img.shields.io/badge/FAISS-Vector%20Store-0467DF">
        <img src="https://img.shields.io/badge/Ollama-Local%20LLM-111111">
        <img src="https://img.shields.io/badge/Socket.IO-Streaming-010101?logo=socketdotio&logoColor=white">
    </div>
</div>

### Project Overview

*This project focuses on building an end-to-end local RAG assistant with a polished SaaS-like frontend, multi-file document ingestion, vector search, and real-time streamed answers.*

- **Multi-PDF Upload:** Upload and index one or more PDF files.
- **RAG Pipeline:** Splits text into chunks, creates embeddings, stores them in FAISS, and retrieves relevant context for answers.
- **Local AI Inference:** Uses Ollama (`phi3`) so answers are generated locally.
- **Streaming UX:** Chat responses stream token-by-token using Flask-SocketIO.
- **Chat Timeline:** User and AI messages appear as chat bubbles with source citations.
- **Persistent Index:** Vectorstore is saved and auto-loaded across app restarts.

<div align='center'>
    <img src="./ui-preview.png" alt="GenAI RAG UI Preview" width="900" align="center">
    <p align="center"><em>GenAI RAG Document Assistant UI</em></p>
</div>

### Tools and Technologies

| Tool / Library        | Purpose |
|-----------------------|---------|
| Flask                 | Backend web server and routing |
| Flask-SocketIO        | Real-time streaming responses |
| LangChain             | RAG orchestration and document processing |
| FAISS                 | Vector similarity search |
| HuggingFace Embeddings | Embedding generation (`all-MiniLM-L6-v2`) |
| Ollama                | Local LLM inference (`phi3`) |
| PyPDF                 | PDF text loading |
| HTML/CSS/JS           | Interactive frontend (drag/drop, progress, chat UI) |

### Pipeline Workflow

**(1) Upload & Ingestion**

- User uploads one or more PDFs from the web UI.
- Files are saved under `uploads/`.

**(2) Processing & Indexing**

- PDFs are parsed into pages.
- Text is chunked via LangChain text splitters.
- Embeddings are generated and indexed in FAISS.
- Index is persisted in `vectorstore/`.

**(3) Retrieval**

- On each question, relevant chunks are retrieved using similarity search.

**(4) Generation**

- Retrieved context + question are sent to Ollama (`phi3`).
- Response is streamed word-by-word over Socket.IO.

**(5) UI Rendering**

- Chat bubbles update in real-time with typing animation.
- Sources are shown under each AI response.

### Setup on Your Machine

#### Prerequisites

- Python 3.10+ (recommended)
- Ollama installed and running locally
- `phi3` model available in Ollama

#### Clone Repository

```bash
git clone https://github.com/Amanzuu/GenAI-RAG-Document-Assistant.git
cd GenAI-RAG-Document-Assistant
```

#### Create Virtual Environment (Recommended)

```bash
python -m venv .venv
```

Activate:

- Windows (PowerShell):

```bash
.\.venv\Scripts\Activate.ps1
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Pull Ollama Model

```bash
ollama run phi3
```

#### Start the App

```bash
python app.py
```

Open:

`http://127.0.0.1:5000`

### Project Structure

| Path | Description |
|------|-------------|
| `app.py` | Main Flask + SocketIO backend |
| `templates/index.html` | Frontend UI template |
| `static/style.css` | Application styling |
| `uploads/` | Uploaded PDF files |
| `vectorstore/` | Persisted FAISS index files |
| `ui-preview.png` | Screenshot shown in README |

### Stopping the App

| Action | Command |
|--------|---------|
| Stop Flask-SocketIO server | Press `CTRL+C` in terminal |

