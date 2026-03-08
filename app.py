from flask import Flask, jsonify, render_template, request
import json
import os
from werkzeug.utils import secure_filename

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings

app = Flask(__name__)

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

            docs = vectorstore.similarity_search(question, k=3)

            context = "\n".join([doc.page_content for doc in docs])

            prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{question}
"""

            answer = llm.invoke(prompt)

            # SOURCE CITATION
            sources = []
            for doc in docs:
                file = doc.metadata.get("source_file","unknown")
                page = doc.metadata.get("page","?")
                sources.append(f"{file} (Page {page})")

            source = ", ".join(set(sources))
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


if __name__ == "__main__":
    load_persisted_state()
    app.run(debug=True)
