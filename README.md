# 🤖 Customer Service Chatbot — Dynamic Knowledge Base Extension
> **Internship Task:** Implement a system for dynamically expanding the chatbot's knowledge base with periodic auto-updates.
> **Base Project:** customer_service_chatbot_LLM (Google PaLM + LangChain + FAISS)
---
## 📌 Problem Statement
The original Nullclass chatbot answers questions from a static FAISS vector index built once from a CSV file. When the company adds new FAQs, policy updates, or product info, the entire index must be rebuilt manually — a slow, error-prone process that causes stale responses.
**Goal:** Add a mechanism that automatically pulls new information from multiple source types, deduplicates it, and incrementally merges it into the live FAISS index without rebuilding from scratch.
---
## 📂 Dataset
- **Name:** Nullclass Customer Service FAQ
- **File:** dataset/dataset.csv
- **Format:** Two columns — prompt (customer question) and response (staff answer)
- **Size:** 200+ FAQ entries covering courses, internships, payments, tools, and platform support
- **Source:** Real FAQ data used by Nullclass support staff
---
## ✨ Features Added
| Feature | Description |
|---|---|
| Multi-source ingestion | CSV files, plain-text files, live URLs, JSON Q&A files, inline text snippets |
| Incremental FAISS update | New chunks merged into existing index via FAISS.merge_from — no full rebuild |
| Content-hash deduplication | SHA-256 per chunk; already-indexed content skipped silently |
| SQLite update log | Every run records start time, finish time, new chunks added, and status |
| APScheduler integration | Background scheduler with configurable interval (15 min to 1 day) |
| Streamlit management UI | Add sources, toggle on/off, trigger manual updates, view history chart |
| sources_config.json | Declarative config — add/enable/disable sources without touching code |
---
## 🗂 Project Structure
customer-service-chatbot-dynamic-kb/
├── dataset/
│   └── dataset.csv
├── app.py
├── requirements.txt
├── .env.example
└── .gitignore
---
## 🔧 Installation
git clone https://github.com/aravindsunilkumar2003-ops/customer-service-chatbot-dynamic-kb.git
cd customer-service-chatbot-dynamic-kb
pip install -r requirements.txt
Rename .env.example to .env and add: GOOGLE_API_KEY="your_key"
---
## 🚀 Usage
streamlit run app.py
CLI: python app.py --update
Scheduler: python app.py --schedule 3600
---
## ⚙️ Methodology
Pipeline: sources_config.json → SourceLoader → SHA-256 Dedup → HuggingFace Embeddings → FAISS.merge_from → RetrievalQA (Google PaLM)
## 📊 Baseline vs Extended Model
| Metric | Baseline | This Project |
|---|---|---|
| Update method | Full manual rebuild | Incremental merge |
| Speed | ~5 minutes | Seconds |
| Deduplication | None | SHA-256 |
| Sources | CSV only | CSV, JSON, URL, text, inline |
| Auto-update | No | Yes (APScheduler) |
| Audit trail | No | Yes (SQLite) |
---
## 🤝 Credits
Base project by Nullclass — extended as part of the Elevance Skills GenAI Internship.
