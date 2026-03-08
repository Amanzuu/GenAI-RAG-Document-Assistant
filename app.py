from flask import Flask, jsonify, render_template, request
import json
import os
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO, emit

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_DIR = "uploads"
VECTORSTORE_DIR = "vectorstore"
VECTORSTORE_META = os.path.join(VECTORSTORE_DIR, "meta.json")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = None
indexed_files = []
chat_history = []


def load_persisted_state():
    global vectorstore, indexed_files
    index_file = os.path.join(VECTORSTORE_DIR, "index.faiss")
    store_file = os.path.join(VECTORSTORE_DIR, "index.pkl")

    if os.path.exists(index_file) and os.path.exists(store_file):
        vectorstore = FAISS.load_local(
            VECTORSTORE_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )

    if os.path.exists(VECTORSTORE_META):
        with open(VECTORSTORE_META, "r", encoding="utf-8") as f:
            meta = json.load(f)
            indexed_files = meta.get("indexed_files", [])


def build_prompt(context, question):
    return f"""
You are a helpful assistant.
Answer using only the provided context.
Use short paragraphs and simple wording.
If the answer is missing in context, say so clearly.

Context:
{context}

Question:
{question}
"""


def get_retrieved_docs(question, k=3):
    return vectorstore.similarity_search(question, k=k)


def build_sources(docs):
    sources = []
    for doc in docs:
        file = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "?")
        sources.append(f"{file} (Page {page})")
    return ", ".join(sorted(set(sources)))


def clear_vectorstore_files():
    index_file = os.path.join(VECTORSTORE_DIR, "index.faiss")
    store_file = os.path.join(VECTORSTORE_DIR, "index.pkl")
    if os.path.exists(index_file):
        os.remove(index_file)
    if os.path.exists(store_file):
        os.remove(store_file)
    if os.path.exists(VECTORSTORE_META):
        os.remove(VECTORSTORE_META)


@app.route("/", methods=["GET","POST"])
def index():
    global vectorstore, indexed_files, chat_history
    answer = None
    source = None
    message = None
    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"

    if request.method == "POST":

        # MULTIPLE FILE UPLOAD
        if "document" in request.files:
            files = request.files.getlist("document")

            all_docs = []
            uploaded_names = []

            for file in files:
                filename = secure_filename(file.filename or "")
                if not filename:
                    continue

                os.makedirs(UPLOAD_DIR, exist_ok=True)
                path = os.path.join(UPLOAD_DIR, filename)
                file.save(path)

                loader = PyPDFLoader(path)
                documents = loader.load()

                for doc in documents:
                    doc.metadata["source_file"] = filename

                all_docs.extend(documents)
                uploaded_names.append(filename)

            if not uploaded_names:
                message = "Please choose at least one PDF before clicking Upload."
                if is_ajax:
                    return jsonify({"ok": False, "message": message}), 400
                return render_template(
                    "index.html",
                    answer=answer,
                    source=source,
                    message=message,
                    indexed_files=indexed_files,
                    chat_history=chat_history
                )

            splitter = CharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            docs = splitter.split_documents(all_docs)

            vectorstore = FAISS.from_documents(docs, embeddings)
            os.makedirs(VECTORSTORE_DIR, exist_ok=True)
            vectorstore.save_local(VECTORSTORE_DIR)
            indexed_files = uploaded_names
            with open(VECTORSTORE_META, "w", encoding="utf-8") as f:
                json.dump({"indexed_files": indexed_files}, f, indent=2)
            chat_history = []
            message = f"Indexed {len(uploaded_names)} file(s)."
            if is_ajax:
                return jsonify(
                    {
                        "ok": True,
                        "message": message,
                        "indexed_files": indexed_files
                    }
                )

        # QUESTION ASK
        elif "question" in request.form:

            question = request.form["question"].strip()
            if not question:
                message = "Please type a question."
                return render_template(
                    "index.html",
                    answer=answer,
                    source=source,
                    message=message,
                    indexed_files=indexed_files,
                    chat_history=chat_history
                )

            if vectorstore is None:
                message = "Upload PDF file(s) first, then ask a question."
                return render_template(
                    "index.html",
                    answer=answer,
                    source=source,
                    message=message,
                    indexed_files=indexed_files,
                    chat_history=chat_history
                )

            llm = Ollama(model="phi3")
            docs = get_retrieved_docs(question, k=3)
            context = "\n".join([doc.page_content for doc in docs])
            prompt = build_prompt(context, question)
            answer = llm.invoke(prompt)
            source = build_sources(docs)
            chat_history.append(
                {
                    "question": question,
                    "answer": answer,
                    "source": source
                }
            )

    return render_template(
        "index.html",
        answer=answer,
        source=source,
        message=message,
        indexed_files=indexed_files,
        chat_history=chat_history
    )


@socketio.on("ask_question")
def handle_question(data):
    global chat_history

    if vectorstore is None:
        emit("stream_error", {"message": "Upload PDF file(s) first, then ask a question."})
        return

    question = str(data.get("question", "")).strip()
    if not question:
        emit("stream_error", {"message": "Please type a question."})
        return

    try:
        docs = get_retrieved_docs(question, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        prompt = build_prompt(context, question)
        source = build_sources(docs)
        llm = Ollama(model="phi3")

        answer_parts = []
        for chunk in llm.stream(prompt):
            token = str(chunk)
            if not token:
                continue
            answer_parts.append(token)
            emit("stream_response", {"word": token})

        final_answer = "".join(answer_parts).strip()
        chat_history.append(
            {
                "question": question,
                "answer": final_answer,
                "source": source
            }
        )
        emit("stream_done", {"source": source})
    except Exception as exc:
        emit("stream_error", {"message": str(exc)})


@app.route("/clear-index", methods=["POST"])
def clear_index():
    global vectorstore, indexed_files, chat_history
    vectorstore = None
    indexed_files = []
    chat_history = []
    clear_vectorstore_files()
    return jsonify({"ok": True, "message": "Cleared indexed documents."})


if __name__ == "__main__":
    load_persisted_state()
    socketio.run(app, debug=True)
