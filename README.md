# Customer Service Chatbot - Dynamic Knowledge Base

Internship Task: Dynamically expanding chatbot knowledge base.
Base Project: customer_service_chatbot_LLM (Google PaLM + LangChain + FAISS)

## Problem Statement
The original Nullclass chatbot uses a static FAISS index. New information requires a full manual rebuild. This project adds automatic incremental updates.

## Dataset
- File: dataset/dataset.csv
- Format: prompt and response columns
- Size: 200+ Nullclass FAQ entries
- Source: Real support data from Nullclass

## Features
- Multi-source ingestion: CSV, text, URL, JSON, inline
- Incremental FAISS merge via FAISS.merge_from
- SHA-256 deduplication per chunk
- SQLite audit log for every update run
- APScheduler background auto-updates
- Streamlit UI for source management

## Installation
git clone https://github.com/aravindsunilkumar2003-ops/customer-service-chatbot-dynamic-kb.git
pip install -r requirements.txt

## Usage
streamlit run app.py
python app.py --update
python app.py --schedule 3600

## Methodology
sources_config.json -> SourceLoader -> SHA256 Dedup -> HuggingFace Embeddings -> FAISS.merge_from -> RetrievalQA (Google PaLM)

## Baseline vs Extended

| Metric | Baseline | This Project |
|---|---|---|
| Update method | Full rebuild | Incremental merge |
| Speed | 5 minutes | Seconds |
| Deduplication | None | SHA-256 |
| Sources | CSV only | CSV JSON URL text inline |
| Auto-update | No | Yes APScheduler |
| Audit trail | No | Yes SQLite |

## Results
Streamlit Update History tab shows total runs, success rate, chunks added over time, and per-run duration.

## Credits
Base project by Nullclass - extended for Elevance Skills GenAI Internship.
