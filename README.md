# Customer Service Chatbot - Dynamic Knowledge Base Extension

Internship Task: Implement a system for dynamically expanding the chatbot's knowledge base with periodic auto-updates.
Base Project: customer_service_chatbot_LLM (Google PaLM + LangChain + FAISS)

## Problem Statement

The original Nullclass chatbot answers questions from a static FAISS vector index built once from a CSV file. When new FAQs or product info are added, the entire index must be rebuilt manually causing stale responses.

Goal: Automatically pull new information from multiple sources, deduplicate it, and incrementally merge it into the live FAISS index without rebuilding from scratch.

## Dataset

- Name: Nullclass Customer Service FAQ
- File: dataset/dataset.csv
- Format: prompt (customer question) and response (staff answer) columns
- Size: 200+ FAQ entries covering courses, internships, payments, and tools
- Source: Real FAQ data used by Nullclass support staff

## Features Added

- Multi-source ingestion: CSV, plain-text files, live URLs, JSON Q&A files, inline text
- Incremental FAISS update: New chunks merged via FAISS.merge_from - no full rebuild
- Content-hash deduplication: SHA-256 per chunk, already-indexed content skipped
- SQLite update log: Every run records start time, finish time, chunks added, status
- APScheduler: Background scheduler with interval from 15 min to 1 day
- Streamlit UI: Add sources, toggle on/off, trigger updates, view history chart

## Project Structure

- app.py: single all-in-one file (LLM + updater + Streamlit UI)
- dataset/dataset.csv: Nullclass FAQ dataset
- requirements.txt: all dependencies
- .env.example: rename to .env and add your API key

## Installation

git clone https://github.com/aravindsunilkumar2003-ops/customer-service-chatbot-dynamic-kb.git
cd customer-service-chatbot-dynamic-kb
pip install -r requirements.txt

Add your key to .env:
GOOGLE_API_KEY="your_google_palm_api_key"

## Usage

Run the app:
streamlit run app.py

CLI one-shot update:
python app.py --update

CLI with scheduler:
python app.py --schedule 3600

## Methodology

Pipeline:
sources_config.json
-> SourceLoader (csv/text/url/json/inline)
-> RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
-> SHA-256 Deduplicator + SQLite History DB
-> HuggingFace Embeddings (hkunlp/instructor-large)
-> FAISS.merge_from (incremental, no full rebuild)
-> RetrievalQA Chain (Google PaLM LLM)

## Baseline vs Extended Model

| Metric | Baseline | This Project |
|---|---|---|
| Update method | Full manual rebuild | Incremental merge |
| Speed | 5 minutes | Seconds |
| Deduplication | None | SHA-256 |
| Sources | CSV only | CSV, JSON, URL, text, inline |
| Auto-update | No | Yes - APScheduler |
| Audit trail | No | Yes - SQLite log |

## Results

The Update History tab in the Streamlit app shows:
- Total runs and success rate
- Total new chunks added over time (line chart)
- Per-run duration, chunk count, and status

## Credits

Base project by Nullclass - extended as part of the Elevance Skills GenAI Internship.
