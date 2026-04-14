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
- **File:** `dataset/dataset.csv`
- **Format:** Two columns — `prompt` (customer question) and `response` (staff answer)
- **Size:** 200+ FAQ entries covering courses, internships, payments, tools, and platform support
- **Source:** Real FAQ data used by Nullclass support staff

---

## ✨ Features Added

| Feature | Description |
|---|---|
| Multi-source ingestion | CSV files, plain-text files, live URLs (web scraping), JSON Q&A files, inline text snippets |
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
│   └── dataset.csv          ← Nullclass FAQ dataset (prompt/response pairs)
├── faiss_index/             ← auto-created FAISS vector index
├── kb_update_history.db     ← SQLite run log (auto-created)
├── sources_config.json      ← source registry (auto-created on first run)
├── app.py                   ← single all-in-one file (LLM + updater + UI)
├── requirements.txt
├── .env.example             ← rename to .env and add your API key
└── .gitignore

---

## 🔧 Installation
git clone https://github.com/aravindsunilkumar2003-ops/customer-service-chatbot-dynamic-kb.git
cd customer-service-chatbot-dynamic-kb
pip install -r requirements.txt

Rename .env.example to .env and add your key:
GOOGLE_API_KEY="your_google_palm_api_key"

---

## 🚀 Usage

### 1. Run the Streamlit app
streamlit run app.py

### 2. First-time setup
Click "Build from scratch" in the sidebar to index the base CSV.

### 3. Add a new knowledge source
- Choose a source type in the sidebar (CSV / text / URL / JSON / inline)
- Fill in the path or URL and click "Add Source"
- Click "Update now" — only new, unseen chunks are added

### 4. Enable the auto-scheduler
Select an interval and click "Start Scheduler". The background thread runs automatically.

### 5. CLI usage (no UI)
python app.py --update
python app.py --schedule 3600

---

## ⚙️ Methodology & How It Works

### Pipeline
sources_config.json
│
▼
SourceLoader (csv / text / url / json / inline)
│
▼  RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
│
▼
SHA-256 Deduplicator ──► SQLite History DB (skip seen chunks)
│
▼  new chunks only
│
▼
HuggingFace Embeddings (hkunlp/instructor-large)
│
▼
FAISS.merge_from(existing_index)   ← incremental, no full rebuild
│
▼
faiss_index/ saved to disk
│
▼
RetrievalQA Chain (Google PaLM LLM)

### Key Design Choices

| Choice | Rationale |
|---|---|
| Incremental FAISS merge | FAISS.merge_from is O(new_chunks) not O(all_chunks) — stays fast as index grows |
| SHA-256 content dedup | Prevents duplicate embeddings when the same source is re-scraped |
| SQLite audit log | Lightweight, zero extra dependencies, survives app restarts |
| APScheduler BackgroundScheduler | Runs inside the Streamlit process — no separate daemon or cron job needed |
| Single app.py file | All logic in one file makes it easy to review, run, and submit |

---

## 📊 Results & Metrics

### Update History Dashboard (built into the app)
The Update History tab tracks every update run and shows:
- Total runs completed and success rate
- Total new chunks added across all runs
- Line chart: chunks added over time
- Table: per-run duration, chunk count, status

### Baseline vs Extended Model

| Metric | Baseline (static index) | This Project (dynamic index) |
|---|---|---|
| Knowledge update method | Full manual rebuild | Incremental merge only |
| Rebuild time on new data | ~5 min (all docs) | ~seconds (new chunks only) |
| Duplicate prevention | None | SHA-256 hash dedup |
| Sources supported | CSV only | CSV, JSON, URL, text, inline |
| Auto-update | No | Yes (APScheduler) |
| Audit trail | No | Yes (SQLite log) |

---

## 📦 Extending with New Source Types

Add a method to SourceLoader in app.py:

```python
@classmethod
def from_pdf(cls, path: str) -> List[Document]:
    from pypdf import PdfReader
    text = "\n".join(p.extract_text() for p in PdfReader(path).pages)
    chunks = cls.splitter.split_text(text)
    return [Document(page_content=c, metadata={"source": path, "type": "pdf"}) for c in chunks]
```

Then add a "type": "pdf" branch in _load_all_documents() and a new entry in sources_config.json.

---

## 🤝 Credits

Base project by Nullclass — extended as part of the Elevance Skills GenAI Internship.