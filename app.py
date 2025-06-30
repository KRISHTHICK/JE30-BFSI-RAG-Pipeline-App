# RAG Pipeline for BFSI Document Analysis using Python + ChromaDB + Ollama

import os
import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import ollama

# Step 1: Initialize ChromaDB Vector Store
def init_vector_store():
    embedding_func = OllamaEmbeddingFunction(model_name="nomic-embed-text")
    client = chromadb.Client()
    collection = client.get_or_create_collection("bfsi_rag_knowledgebase", embedding_function=embedding_func)
    return collection

# Step 2: Extract text from PDF
def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    full_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return full_text

# Step 3: Split text into chunks
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    return splitter.split_text(text)

# Step 4: Add chunks to vector DB
def add_to_vector_db(collection, chunks, source="Uploaded File"):
    for i, chunk in enumerate(chunks):
        collection.add(documents=[chunk], ids=[f"doc_{source}_{i}"], metadatas=[{"source": source}])

# Step 5: RAG Retrieval + LLM Answer Generation
def query_with_rag(collection, query):
    results = collection.query(query_texts=[query], n_results=4)
    context = "\n\n".join(results["documents"][0])
    prompt = f"""
You are a BFSI domain expert assistant. Based on the documents below, answer the user's query:

Documents:
{context}

Question:
{query}

Give a clear and concise answer with financial domain context.
"""
    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

# Streamlit UI for RAG Pipeline
st.set_page_config(page_title="🔍 BFSI RAG Pipeline", layout="wide")
st.title("📂 RAG Pipeline for BFSI Document Analysis")

collection = init_vector_store()

uploaded_file = st.file_uploader("Upload BFSI-related Document (PDF)", type=["pdf"])
if uploaded_file:
    with st.spinner("Processing PDF and indexing..."):
        raw_text = extract_pdf_text(uploaded_file)
        chunks = split_text(raw_text)
        add_to_vector_db(collection, chunks, source=uploaded_file.name)
        st.success("Document indexed to knowledgebase!")

st.markdown("---")
user_query = st.text_input("Ask a question based on uploaded BFSI content:", "What are the key risk factors in this policy?")
if st.button("🔎 Generate Answer") and user_query:
    with st.spinner("Querying RAG system..."):
        output = query_with_rag(collection, user_query)
        st.markdown("### 🧠 Answer from BFSI RAG Agent")
        st.markdown(output)
