from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

app = Flask(__name__)

vectorstore = None
qa_chain = None
current_filename = None
QA_PROMPT = PromptTemplate.from_template(
    """You are a helpful assistant.
Use only the context below to answer.
Give a simple, clear answer in 5-8 lines.
Include key points, but avoid long paragraphs.
If the answer is not in the context, say: "I couldn't find that in the uploaded PDF."

Context:
{context}

Question:
{question}

Answer:"""
)

@app.route("/", methods=["GET","POST"])
def index():
    global vectorstore, qa_chain, current_filename
    answer = None
    message = None

    if request.method == "POST":

        if "document" in request.files:
            file = request.files["document"]
            filename = secure_filename(file.filename or "")
            if not filename:
                message = "Please choose a PDF file before clicking Upload."
                return render_template("index.html", answer=answer, message=message)

            os.makedirs("uploads", exist_ok=True)
            path = os.path.join("uploads", filename)
            file.save(path)

            loader = PyPDFLoader(path)
            documents = loader.load()

            splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=100)
            docs = splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            vectorstore = FAISS.from_documents(docs, embeddings)
            llm = Ollama(model="phi3", num_predict=220, temperature=0.15)
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={"prompt": QA_PROMPT}
            )
            current_filename = filename
            message = f"Uploaded and indexed: {filename}"

        elif "question" in request.form:
            question = request.form["question"].strip()
            if not question:
                message = "Please type a question."
                return render_template(
                    "index.html",
                    answer=answer,
                    message=message,
                    current_filename=current_filename
                )

            if qa_chain is None:
                message = "Upload a PDF first, then ask a question."
                return render_template(
                    "index.html",
                    answer=answer,
                    message=message,
                    current_filename=current_filename
                )

            answer = qa_chain.run(question)

    return render_template(
        "index.html",
        answer=answer,
        message=message,
        current_filename=current_filename
    )

if __name__ == "__main__":
    app.run(debug=True)
