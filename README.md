# JE30-BFSI-RAG-Pipeline-App
GEN AI

# 🧠 BFSI RAG Pipeline

This project is an AI-powered Retrieval-Augmented Generation (RAG) system for analyzing BFSI (Banking, Financial Services, Insurance) documents. Built with Python, Streamlit, and Ollama, it allows you to upload PDFs, index their content, and ask contextual questions.

## 🚀 Features

- Upload BFSI documents (e.g. policies, reports)
- Local LLM inference with Ollama (LLaMA3)
- Vector search with ChromaDB
- Streamlit frontend for user interaction
- Clear answers from retrieved content using RAG

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/bfsi-rag-pipeline.git
cd bfsi-rag-pipeline
pip install -r requirements.txt
ollama pull llama3
