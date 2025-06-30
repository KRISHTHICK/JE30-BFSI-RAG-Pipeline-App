bfsi-rag-pipeline/
│
├── app.py                        # Main Streamlit app
├── requirements.txt             # Python dependencies
├── README.md                    # Project overview and instructions
│
├── utils/
│   ├── extract.py               # PDF text extraction
│   ├── split.py                 # Text chunking logic
│   └── vector_store.py          # ChromaDB initialization and management
│
├── agent/
│   └── rag_agent.py             # RAG logic using Ollama + ChromaDB
│
└── sample_docs/
    └── sample_policy.pdf        # Sample BFSI document for testing
