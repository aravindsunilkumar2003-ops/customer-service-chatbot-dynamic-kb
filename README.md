# 🤖 Customer Service Chatbot — Dynamic Knowledge Base Extension

> **Internship Task:** Implement a system for dynamically expanding the chatbot's knowledge base with periodic auto-updates.  
> **Base Project:** `customer_service_chatbot_LLM` (Google PaLM + LangChain + FAISS)

---

## 📌 Problem Statement

The original chatbot answers questions from a static FAISS vector index built once from a CSV file. When the company adds new FAQs, policy updates, or product info, the entire index must be rebuilt manually — a slow, error-prone process that causes stale responses.

**Goal:** Add a mechanism that automatically pulls new information from multiple source types, deduplicates it, and incrementally merges it into the live FAISS index without rebuilding from scratch.

---

## ✨ Features Added

| Feature | Description |
|---|---|
| **Multi-source ingestion** | CSV files, plain-text files, live URLs (web scraping), JSON Q&A files, inline text snippets |
| **Incremental FAISS update** | New chunks are merged into the existing index (`FAISS.merge_from`) — no full rebuild |
| **Content-hash deduplication** | SHA-256 per chunk; already-indexed content is skipped silently |
| **SQLite update log** | Every run records start time, finish time, new chunks added, and status |
| **APScheduler integration** | Background scheduler with configurable interval (15 min → 1 day) |
| **Streamlit management UI** | Add sources, toggle sources on/off, trigger manual updates, view history chart |
| **`sources_config.json`** | Declarative config file — add/enable/disable sources without touching code |

---

## 🗂 Project Structure

```
customer-service-chatbot-dynamic-kb/
├── dataset/
│   └── dataset.csv               ← original FAQ dataset (Nullclass)
├── faiss_index/                  ← auto-created FAISS index
├── kb_update_history.db          ← SQLite run log (auto-created)
├── sources_config.json           ← source registry (auto-created)
├── app.py                        ← ★ single all-in-one file (LLM + updater + UI)
├── requirements.txt
├── .env.example                  ← rename to .env and add your key
└── .gitignore
```

---

## 🔧 Installation

```bash
git clone https://github.com/aravindsunilkumar2003-ops/customer-service-chatbot-dynamic-kb.git
cd customer-service-chatbot-dynamic-kb
pip install -r requirements.txt
```

Rename `.env.example` to `.env` and add your key:
```
GOOGLE_API_KEY="your_google_palm_api_key"
```

---

## 🚀 Usage

### 1 · Run the Streamlit app
```bash
streamlit run app.py
```

### 2 · First-time setup
Click **"🔨 Build from scratch"** in the sidebar to index the base CSV.

### 3 · Add a new knowledge source
- In the sidebar choose a source type (CSV / text / URL / JSON / inline).
- Fill in the path or URL and click **"Add Source"**.
- Click **"🔄 Update now"** — only the new, unseen chunks are added.

### 4 · Enable the scheduler
Select an interval and click **"▶ Start Scheduler"**. The background thread calls `run_update_cycle()` automatically.

### 5 · CLI usage (without UI)
```bash
# one-shot update
python app.py --update

# update + start hourly background scheduler
python app.py --schedule 3600
```

---

## ⚙ How It Works

```
sources_config.json
       │
       ▼
SourceLoader (csv / text / url / json / inline)
       │
       ▼  (RecursiveCharacterTextSplitter → chunks)
Deduplicator (SHA-256 hash → SQLite lookup)
       │ skips already-seen chunks
       ▼
FAISS.from_documents(new_chunks)
       │
       ▼
FAISS.merge_from(existing_index)   ← incremental, no rebuild
       │
       ▼
faiss_index/   (saved to disk)
```

### Key design choices

| Choice | Rationale |
|---|---|
| **Incremental merge** | `FAISS.merge_from` is O(new_chunks) not O(all_chunks) — fast on large indexes |
| **Content-hash dedup** | Prevents duplicate embeddings when a source is re-scraped |
| **SQLite log** | Lightweight, zero dependencies, survives restarts |
| **APScheduler BackgroundScheduler** | Runs in the same Python process as Streamlit — no extra daemon needed |
| **`sources_config.json`** | Non-developers can add sources by editing a JSON file |

---

## 📊 Results / Metrics

The **Update History** tab in the Streamlit app tracks every update run and displays:

- ✅ Total update runs and success rate
- 📈 Chunks added over time (line chart)
- ⏱ Per-run duration
- 🗂 Source breakdown per run

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                   Knowledge Sources                  │
│   CSV │ JSON │ Plain Text │ URL (scrape) │ Inline    │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
          ┌─────────────────────────┐
          │   RecursiveTextSplitter  │  chunk_size=500
          └─────────────┬───────────┘
                        │
                        ▼
          ┌─────────────────────────┐
          │  SHA-256 Deduplicator   │◄──── SQLite History DB
          │  (skip seen chunks)     │
          └─────────────┬───────────┘
                        │ new chunks only
                        ▼
          ┌─────────────────────────┐
          │  HuggingFace Embeddings │  hkunlp/instructor-large
          └─────────────┬───────────┘
                        │
                        ▼
          ┌─────────────────────────┐
          │   FAISS.merge_from()    │  incremental, no rebuild
          │   faiss_index/          │
          └─────────────┬───────────┘
                        │
                        ▼
          ┌─────────────────────────┐
          │   RetrievalQA Chain     │  Google PaLM LLM
          │   (RAG)                 │
          └─────────────────────────┘
```

### Dataset

- **Source:** Nullclass FAQ dataset (`dataset/dataset.csv`)
- **Format:** prompt/response pairs
- **Size:** ~200+ FAQ entries covering courses, internships, payments, tools

---

## 📦 Adding New Source Types (extensibility)

Extend `SourceLoader` in `app.py`:

```python
@classmethod
def from_pdf(cls, path: str) -> List[Document]:
    from pypdf import PdfReader
    text = "\n".join(p.extract_text() for p in PdfReader(path).pages)
    chunks = cls.splitter.split_text(text)
    return [Document(page_content=c, metadata={"source": path, "type": "pdf"}) for c in chunks]
```

Then add a `"type": "pdf"` branch in `_load_all_documents()` and a new entry in `sources_config.json`.

---

## 🤝 Credits

Base project by [Nullclass](https://nullclass.com) — extended as part of the Elevance Skills GenAI Internship.