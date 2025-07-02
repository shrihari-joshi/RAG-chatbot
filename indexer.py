import os
import re
import json
import hashlib
import streamlit as st
from pinecone import Pinecone
from dotenv import load_dotenv

# Updated LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from groq import Groq

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
INDEX_NAME = "basicrag"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Paths
INDEXED_FILES_PATH = "indexed_files.json"
TEMP_DIR = "./temp/"
os.makedirs(TEMP_DIR, exist_ok=True)
if not os.path.exists(INDEXED_FILES_PATH):
    with open(INDEXED_FILES_PATH, "w") as f:
        json.dump({}, f)

# Utility functions
def file_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def is_already_indexed(file_hash):
    with open(INDEXED_FILES_PATH) as f:
        indexed = json.load(f)
    return indexed.get(file_hash, False)

def mark_as_indexed(file_hash):
    with open(INDEXED_FILES_PATH) as f:
        indexed = json.load(f)
    indexed[file_hash] = True
    with open(INDEXED_FILES_PATH, "w") as f:
        json.dump(indexed, f)

# Streamlit UI
st.set_page_config(page_title="RAG with Groq", layout="centered")
st.title("📄 RAG-Powered Q&A using Groq LLM")
st.markdown("Upload a PDF and ask questions based on its content.")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
query = st.text_input("Ask a question based on the document")

if uploaded_file and query:
    with st.spinner("Processing document..."):
        file_path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        doc_hash = file_hash(file_path)
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        if not is_already_indexed(doc_hash):
            with st.spinner("Indexing document..."):
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                docs = splitter.split_documents(documents)

                PineconeVectorStore.from_documents(
                    docs, embedding=embedding_model, index_name=INDEX_NAME
                )
                mark_as_indexed(doc_hash)
                st.success("✅ Document indexed!")
        else:
            st.info("ℹ️ Document already indexed. Skipping re-indexing.")

    with st.spinner("Searching and generating answer..."):
        vector_store = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME, embedding=embedding_model
        )
        results = vector_store.similarity_search(query, k=3)

        context = ""
        for doc in results:
            cleaned = re.sub(r'\s+', ' ', doc.page_content).strip()
            context += cleaned + "\n"

        prompt = f"""You are a helpful assistant. Answer the user's question based only on the following context.

Context:
{context}

Question: {query}
Answer:"""

        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content

    st.success("✅ Answer Generated:")
    st.markdown(f"**{answer}**")
