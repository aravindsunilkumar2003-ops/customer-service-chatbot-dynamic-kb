"""
app.py  –  Customer Service Chatbot with Dynamic Knowledge Base
===============================================================
All-in-one file combining:
  • LangChain / FAISS helpers  (originally langchain_helper.py)
  • Dynamic KB updater          (originally kb_updater.py)
  • Streamlit UI                (originally main.py)

Run:
    streamlit run app.py

CLI update (without UI):
    python app.py --update
    python app.py --schedule 3600
"""

# ═══════════════════════════════════════════════════════════════
# Standard-library imports
# ═══════════════════════════════════════════════════════════════
import os
import csv
import json
import sys
import time
import hashlib
import sqlite3
import logging
import argparse
import datetime
from pathlib import Path
from typing import List, Dict, Optional

# ═══════════════════════════════════════════════════════════════
# Third-party imports
# ═══════════════════════════════════════════════════════════════
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from apscheduler.schedulers.background import BackgroundScheduler

load_dotenv()

# ═══════════════════════════════════════════════════════════════
# ① CONFIGURATION
# ═══════════════════════════════════════════════════════════════

BASE_DIR            = Path(__file__).parent
FAISS_INDEX_PATH    = str(BASE_DIR / "faiss_index")
DB_PATH             = str(BASE_DIR / "kb_update_history.db")
SOURCES_CONFIG_PATH = str(BASE_DIR / "sources_config.json")
LOG_PATH            = str(BASE_DIR / "kb_updater.log")
DATASET_CSV         = str(BASE_DIR / "dataset" / "dataset.csv")

CHUNK_SIZE          = 500
CHUNK_OVERLAP       = 50
SCORE_THRESHOLD     = 0.7

# ═══════════════════════════════════════════════════════════════
# ② LOGGING
# ═══════════════════════════════════════════════════════════════

_log_handler_file = logging.FileHandler(LOG_PATH)
_log_handler_file.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
_log_handler_console = logging.StreamHandler()
_log_handler_console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

log = logging.getLogger("dynamic_kb")
if not log.handlers:   # avoid duplicate handlers on Streamlit hot-reload
    log.setLevel(logging.INFO)
    log.addHandler(_log_handler_file)
    log.addHandler(_log_handler_console)

# ═══════════════════════════════════════════════════════════════
# ③ SHARED EMBEDDING MODEL  (loaded once, reused everywhere)
# ═══════════════════════════════════════════════════════════════

_embeddings: Optional[HuggingFaceInstructEmbeddings] = None

def get_embeddings() -> HuggingFaceInstructEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    return _embeddings

# ═══════════════════════════════════════════════════════════════
# ④ LLM  (originally langchain_helper.py)
# ═══════════════════════════════════════════════════════════════

def get_llm() -> GooglePalm:
    return GooglePalm(
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.1
    )


def create_vector_db():
    """Build the FAISS index from scratch using the base CSV dataset."""
    loader = CSVLoader(file_path=DATASET_CSV, source_column="prompt")
    data   = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=get_embeddings())
    vectordb.save_local(FAISS_INDEX_PATH)
    log.info("FAISS index built from scratch (%d docs).", len(data))


def get_qa_chain() -> RetrievalQA:
    """Load the FAISS index and return a ready-to-use RetrievalQA chain."""
    vectordb  = FAISS.load_local(
        FAISS_INDEX_PATH,
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
    retriever = vectordb.as_retriever(search_kwargs={"score_threshold": SCORE_THRESHOLD, "k": 4})

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from the "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    return RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )

# ═══════════════════════════════════════════════════════════════
# ⑤ HISTORY DATABASE  (originally part of kb_updater.py)
# ═══════════════════════════════════════════════════════════════

class UpdateHistoryDB:
    """SQLite store tracking indexed chunks and update run logs."""

    def __init__(self, db_path: str = DB_PATH):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS indexed_chunks (
                chunk_hash  TEXT PRIMARY KEY,
                source      TEXT,
                added_at    TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS update_runs (
                run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at  TEXT,
                finished_at TEXT,
                new_chunks  INTEGER,
                status      TEXT
            )
        """)
        self.conn.commit()

    def is_indexed(self, chunk_hash: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM indexed_chunks WHERE chunk_hash=?", (chunk_hash,)
        ).fetchone() is not None

    def mark_indexed(self, chunk_hash: str, source: str):
        self.conn.execute(
            "INSERT OR IGNORE INTO indexed_chunks VALUES (?,?,?)",
            (chunk_hash, source, datetime.datetime.utcnow().isoformat()),
        )
        self.conn.commit()

    def log_run(self, started: str, finished: str, new_chunks: int, status: str):
        self.conn.execute(
            "INSERT INTO update_runs(started_at,finished_at,new_chunks,status) VALUES (?,?,?,?)",
            (started, finished, new_chunks, status),
        )
        self.conn.commit()

    def recent_runs(self, n: int = 20) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT run_id,started_at,finished_at,new_chunks,status "
            "FROM update_runs ORDER BY run_id DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(zip(["run_id","started_at","finished_at","new_chunks","status"], r)) for r in rows]

# ═══════════════════════════════════════════════════════════════
# ⑥ SOURCE LOADERS  (originally part of kb_updater.py)
# ═══════════════════════════════════════════════════════════════

class SourceLoader:
    """Loads raw content from different source types into LangChain Documents."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    @staticmethod
    def from_csv(path: str, prompt_col: str = "prompt", response_col: str = "response") -> List[Document]:
        docs = []
        try:
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    text = f"Q: {row.get(prompt_col,'').strip()}\nA: {row.get(response_col,'').strip()}"
                    docs.append(Document(page_content=text, metadata={"source": path, "type": "csv"}))
        except Exception as e:
            log.error("CSV error (%s): %s", path, e)
        return docs

    @classmethod
    def from_text_file(cls, path: str) -> List[Document]:
        try:
            text   = Path(path).read_text(encoding="utf-8")
            chunks = cls.splitter.split_text(text)
            return [Document(page_content=c, metadata={"source": path, "type": "text"}) for c in chunks]
        except Exception as e:
            log.error("Text file error (%s): %s", path, e)
            return []

    @classmethod
    def from_url(cls, url: str) -> List[Document]:
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            text   = " ".join(soup.get_text(separator=" ").split())
            chunks = cls.splitter.split_text(text)
            return [Document(page_content=c, metadata={"source": url, "type": "url"}) for c in chunks]
        except Exception as e:
            log.error("URL error (%s): %s", url, e)
            return []

    @staticmethod
    def from_json(path: str) -> List[Document]:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(data, list):
                data = [data]
            docs = []
            for item in data:
                q    = item.get("question") or item.get("prompt") or ""
                a    = item.get("answer")   or item.get("response") or ""
                text = f"Q: {q.strip()}\nA: {a.strip()}" if (q or a) else json.dumps(item)
                docs.append(Document(page_content=text, metadata={"source": path, "type": "json"}))
            return docs
        except Exception as e:
            log.error("JSON error (%s): %s", path, e)
            return []

    @classmethod
    def from_inline(cls, text: str, label: str = "inline") -> List[Document]:
        chunks = cls.splitter.split_text(text)
        return [Document(page_content=c, metadata={"source": label, "type": "inline"}) for c in chunks]

# ═══════════════════════════════════════════════════════════════
# ⑦ KNOWLEDGE BASE UPDATER  (originally kb_updater.py)
# ═══════════════════════════════════════════════════════════════

class KnowledgeBaseUpdater:
    """
    Orchestrates: load sources → deduplicate → incrementally update FAISS.
    Sources are declared in sources_config.json (auto-created on first run).
    """

    def __init__(self,
                 faiss_path:     str = FAISS_INDEX_PATH,
                 db_path:        str = DB_PATH,
                 sources_config: str = SOURCES_CONFIG_PATH):
        self.faiss_path     = faiss_path
        self.sources_config = sources_config
        self.db             = UpdateHistoryDB(db_path)
        self.embeddings     = get_embeddings()
        self._ensure_sources_config()

    # ── Config bootstrap ──────────────────────────────────────
    def _ensure_sources_config(self):
        if not Path(self.sources_config).exists():
            default = {"sources": [
                {"type": "csv",    "path": DATASET_CSV,
                 "enabled": True,  "description": "Base FAQ CSV – re-checked each run"},
                {"type": "text",   "path": "new_knowledge/announcements.txt",
                 "enabled": False, "description": "Plain-text file; set enabled=true"},
                {"type": "url",    "url": "https://nullclass.com/faq",
                 "enabled": False, "description": "Scraped on every update cycle"},
                {"type": "json",   "path": "new_knowledge/extra_qa.json",
                 "enabled": False, "description": "JSON list of {question, answer}"},
                {"type": "inline", "text": "Nullclass offers a 30-day money-back guarantee.",
                 "label": "policy_2024",
                 "enabled": False, "description": "Inline text snippet"},
            ]}
            Path(self.sources_config).write_text(json.dumps(default, indent=2))
            log.info("Created default sources_config.json")

    def _load_sources_config(self) -> List[Dict]:
        try:
            cfg = json.loads(Path(self.sources_config).read_text())
            return [s for s in cfg.get("sources", []) if s.get("enabled", False)]
        except Exception as e:
            log.error("sources_config read error: %s", e)
            return []

    # ── Document loading ──────────────────────────────────────
    def _load_all_documents(self) -> List[Document]:
        all_docs: List[Document] = []
        for src in self._load_sources_config():
            stype = src.get("type", "")
            label = src.get("path") or src.get("url") or src.get("label", "?")
            log.info("Loading [%s] %s", stype, label)
            if   stype == "csv":    all_docs.extend(SourceLoader.from_csv(src["path"]))
            elif stype == "text":   all_docs.extend(SourceLoader.from_text_file(src["path"]))
            elif stype == "url":    all_docs.extend(SourceLoader.from_url(src["url"]))
            elif stype == "json":   all_docs.extend(SourceLoader.from_json(src["path"]))
            elif stype == "inline": all_docs.extend(SourceLoader.from_inline(src["text"], src.get("label","inline")))
            else: log.warning("Unknown source type: %s", stype)
        log.info("Candidate docs: %d", len(all_docs))
        return all_docs

    # ── Deduplication ─────────────────────────────────────────
    def _deduplicate(self, docs: List[Document]) -> List[Document]:
        fresh = []
        for doc in docs:
            h = hashlib.sha256(doc.page_content.encode()).hexdigest()
            if not self.db.is_indexed(h):
                doc.metadata["chunk_hash"] = h
                fresh.append(doc)
        log.info("New chunks: %d / %d", len(fresh), len(docs))
        return fresh

    # ── Incremental FAISS merge ───────────────────────────────
    def _update_faiss(self, new_docs: List[Document]) -> int:
        if not new_docs:
            log.info("Nothing new to index.")
            return 0
        new_index = FAISS.from_documents(new_docs, self.embeddings)
        if Path(self.faiss_path).exists():
            existing = FAISS.load_local(self.faiss_path, self.embeddings,
                                        allow_dangerous_deserialization=True)
            existing.merge_from(new_index)
            existing.save_local(self.faiss_path)
            log.info("Merged %d chunks into existing index.", len(new_docs))
        else:
            new_index.save_local(self.faiss_path)
            log.info("Created new index with %d chunks.", len(new_docs))
        for doc in new_docs:
            h = doc.metadata.get("chunk_hash",
                hashlib.sha256(doc.page_content.encode()).hexdigest())
            self.db.mark_indexed(h, doc.metadata.get("source", "unknown"))
        return len(new_docs)

    # ── Public API ────────────────────────────────────────────
    def run_update_cycle(self) -> Dict:
        started = datetime.datetime.utcnow().isoformat()
        status, new_chunks, finished = "success", 0, None
        try:
            docs       = self._load_all_documents()
            fresh      = self._deduplicate(docs)
            new_chunks = self._update_faiss(fresh)
        except Exception as e:
            log.exception("Update cycle failed: %s", e)
            status = f"error: {e}"
        finally:
            finished = datetime.datetime.utcnow().isoformat()
            self.db.log_run(started, finished, new_chunks, status)
        summary = {"started_at": started, "finished_at": finished,
                   "new_chunks_added": new_chunks, "status": status}
        log.info("Cycle complete: %s", summary)
        return summary

    def get_update_history(self, n: int = 20) -> List[Dict]:
        return self.db.recent_runs(n)

# ═══════════════════════════════════════════════════════════════
# ⑧ SCHEDULER HELPER
# ═══════════════════════════════════════════════════════════════

def start_scheduler(interval_seconds: int = 3600, run_now: bool = False) -> BackgroundScheduler:
    updater   = KnowledgeBaseUpdater()
    if run_now:
        updater.run_update_cycle()
    scheduler = BackgroundScheduler()
    scheduler.add_job(updater.run_update_cycle, "interval",
                      seconds=interval_seconds, id="kb_update", replace_existing=True)
    scheduler.start()
    log.info("Scheduler started – interval %ds", interval_seconds)
    return scheduler

# ═══════════════════════════════════════════════════════════════
# ⑨ SOURCES CONFIG HELPERS  (used by the Streamlit UI)
# ═══════════════════════════════════════════════════════════════

def load_sources_config() -> Dict:
    if Path(SOURCES_CONFIG_PATH).exists():
        return json.loads(Path(SOURCES_CONFIG_PATH).read_text())
    return {"sources": []}

def save_sources_config(cfg: Dict):
    Path(SOURCES_CONFIG_PATH).write_text(json.dumps(cfg, indent=2))

# ═══════════════════════════════════════════════════════════════
# ⑩ STREAMLIT UI  (originally main.py)
# ═══════════════════════════════════════════════════════════════

def run_streamlit_app():
    import streamlit as st

    st.set_page_config(
        page_title="Customer Service Chatbot 🤖",
        page_icon="🤖",
        layout="wide",
    )

    # ── Session state ────────────────────────────────────────
    if "scheduler"    not in st.session_state: st.session_state.scheduler    = None
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    if "last_update"  not in st.session_state: st.session_state.last_update  = None

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.title("🛠 KB Management")

        # Index controls
        st.subheader("Index Controls")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔨 Build from scratch"):
                with st.spinner("Building…"):
                    create_vector_db()
                st.success("Index built!")
        with col2:
            if st.button("🔄 Update now"):
                with st.spinner("Updating…"):
                    result = KnowledgeBaseUpdater().run_update_cycle()
                st.session_state.last_update = result
                added = result.get("new_chunks_added", 0)
                st.success(f"+{added} new chunks" if added else "Already up to date")

        # Scheduler
        st.subheader("⏰ Auto-Update Scheduler")
        interval_opts  = {"Every 15 min": 900, "Every hour": 3600,
                          "Every 6 hours": 21600, "Every day": 86400}
        interval_label = st.selectbox("Interval", list(interval_opts.keys()), index=1)
        interval_secs  = interval_opts[interval_label]

        sched_on = st.session_state.scheduler is not None and st.session_state.scheduler.running
        if not sched_on:
            if st.button("▶ Start Scheduler"):
                st.session_state.scheduler = start_scheduler(interval_secs)
                st.success(f"Started ({interval_label})")
        else:
            if st.button("⏹ Stop Scheduler"):
                st.session_state.scheduler.shutdown(wait=False)
                st.session_state.scheduler = None
                st.info("Stopped.")
            st.success(f"🟢 Running ({interval_label})")

        # Last update badge
        if st.session_state.last_update:
            lu = st.session_state.last_update
            st.subheader("Last Update")
            st.json({"status": lu["status"],
                     "new_chunks": lu["new_chunks_added"],
                     "finished":   lu["finished_at"]})

        # ── Add source form ──────────────────────────────────
        st.subheader("➕ Add Knowledge Source")
        src_type   = st.selectbox("Type", ["csv", "text file", "url", "json", "inline text"])
        new_source = {"type": src_type.split()[0], "enabled": True}

        if   src_type == "csv":        new_source["path"]  = st.text_input("CSV path", "new_knowledge/extra.csv")
        elif src_type == "text file":  new_source["type"]  = "text"; new_source["path"] = st.text_input("File path", "new_knowledge/notes.txt")
        elif src_type == "url":        new_source["url"]   = st.text_input("URL", "https://example.com/faq")
        elif src_type == "json":       new_source["path"]  = st.text_input("JSON path", "new_knowledge/qa.json")
        elif src_type == "inline text":
            new_source["type"]  = "inline"
            new_source["text"]  = st.text_area("Paste text", height=80)
            new_source["label"] = st.text_input("Label", "manual_entry")

        new_source["description"] = st.text_input("Description (optional)", "")

        if st.button("Add Source"):
            cfg = load_sources_config()
            cfg["sources"].append(new_source)
            save_sources_config(cfg)
            st.success("Added! Click 'Update now' to index it.")

        # ── Source list ──────────────────────────────────────
        st.subheader("📋 Active Sources")
        cfg = load_sources_config()
        for i, src in enumerate(cfg["sources"]):
            enabled = src.get("enabled", False)
            label   = src.get("path") or src.get("url") or src.get("label", f"src_{i}")
            ca, cb  = st.columns([5, 1])
            with ca: st.text(f"{'✅' if enabled else '❌'} [{src['type']}] {label}")
            with cb:
                toggled = st.checkbox("on", value=enabled, key=f"src_{i}", label_visibility="collapsed")
                if toggled != enabled:
                    cfg["sources"][i]["enabled"] = toggled
                    save_sources_config(cfg)
                    st.rerun()

    # ── Main area ────────────────────────────────────────────
    st.title("🤖 Customer Service Chatbot")
    st.caption("Google PaLM + LangChain + Dynamic FAISS Knowledge Base")

    tab_chat, tab_history = st.tabs(["💬 Chat", "📊 Update History"])

    # Chat tab
    with tab_chat:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        question = st.chat_input("Ask anything about Nullclass courses…")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        chain    = get_qa_chain()
                        response = chain(question)
                        answer   = response["result"]
                        sources  = response.get("source_documents", [])
                    except Exception as e:
                        answer  = f"⚠️ Error: {e}. Make sure the knowledge base is built first."
                        sources = []
                st.write(answer)
                if sources:
                    with st.expander("📚 Source snippets"):
                        for doc in sources[:3]:
                            st.caption(f"Source: {doc.metadata.get('source','unknown')}")
                            st.text(doc.page_content[:300] + "…")

            st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # History tab
    with tab_history:
        st.subheader("📊 Update Run History")
        history = KnowledgeBaseUpdater().get_update_history()
        if not history:
            st.info("No runs yet. Click 'Update now' in the sidebar.")
        else:
            import pandas as pd
            df = pd.DataFrame(history)
            df["started_at"]  = pd.to_datetime(df["started_at"])
            df["finished_at"] = pd.to_datetime(df["finished_at"])
            df["duration_s"]  = (df["finished_at"] - df["started_at"]).dt.total_seconds().round(1)

            c1, c2, c3 = st.columns(3)
            c1.metric("Total runs",        len(df))
            c2.metric("Successful",        f"{(df['status']=='success').sum()}/{len(df)}")
            c3.metric("Total chunks added", int(df["new_chunks"].sum()))

            st.dataframe(
                df[["run_id","started_at","new_chunks","duration_s","status"]].rename(columns={
                    "run_id":"Run #","started_at":"Started","new_chunks":"New Chunks",
                    "duration_s":"Duration (s)","status":"Status"}),
                use_container_width=True,
            )
            st.line_chart(df.set_index("started_at")["new_chunks"].sort_index(),
                          use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# ⑪ ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def _cli_main():
    """CLI entry point: python app.py --update / --schedule N"""
    parser = argparse.ArgumentParser(description="Dynamic KB Chatbot – CLI")
    parser.add_argument("--update",   action="store_true", help="Run one update cycle and exit.")
    parser.add_argument("--schedule", type=int, default=0,  help="Start scheduler with this interval (seconds).")
    args = parser.parse_args()

    if not args.update and not args.schedule:
        parser.print_help()
        return

    updater = KnowledgeBaseUpdater()
    result  = updater.run_update_cycle()
    print("\n=== Update Summary ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

    if args.schedule > 0:
        sched = start_scheduler(args.schedule, run_now=False)
        print(f"\nScheduler running every {args.schedule}s – Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            sched.shutdown()
            print("Stopped.")


# Streamlit runs this file as a module (not __main__), so we detect
# it by checking sys.argv[0] for the streamlit runner.
if sys.argv[0].endswith(("streamlit", "streamlit/__main__.py", "_stcore/bootstrap.py")) \
        or "streamlit" in sys.modules:
    # Launched via `streamlit run app.py`
    run_streamlit_app()
elif __name__ == "__main__":
    # Launched via `python app.py --update` etc.
    _cli_main()
