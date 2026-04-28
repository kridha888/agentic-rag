"""
app.py  —  Streamlit UI for Agentic RAG
Run:  streamlit run app.py
"""

import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

import streamlit as st
import time, uuid, traceback
from pathlib import Path

st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Sora:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family:'Sora',sans-serif; background:#0d0f14; color:#e2e8f0; }
.block-container { padding-top:1.5rem; max-width:1100px; }
section[data-testid="stSidebar"] { background:#111420; border-right:1px solid #1e2130; }
section[data-testid="stSidebar"] * { color:#cbd5e1 !important; }

.user-bubble {
    background:#1a2035; border:1px solid #2a3050;
    border-radius:16px 16px 4px 16px; padding:14px 18px;
    margin:8px 0 8px 60px; font-size:0.95rem; line-height:1.7; color:#e2e8f0;
}
.assistant-bubble {
    background:#131825; border:1px solid #1e2a40;
    border-left:3px solid #38bdf8; border-radius:4px 16px 16px 16px;
    padding:16px 20px; margin:8px 60px 8px 0;
    font-size:0.95rem; line-height:1.8; color:#e2e8f0;
}
.assistant-bubble code {
    font-family:'JetBrains Mono',monospace; background:#1e2a3a;
    padding:2px 6px; border-radius:4px; font-size:0.85em; color:#7dd3fc;
}
.role-label {
    font-size:0.72rem; font-weight:600; letter-spacing:0.08em;
    text-transform:uppercase; margin-bottom:4px; font-family:'JetBrains Mono',monospace;
}
.role-user { color:#94a3b8; text-align:right; }
.role-bot  { color:#38bdf8; }

/* ── Step list ── */
.step-row { display:flex; align-items:center; gap:10px; padding:6px 0; font-family:'JetBrains Mono',monospace; font-size:0.82rem; }
.s-done { color:#4ade80; } .s-run { color:#facc15; } .s-wait { color:#2d3748; }
.l-done { color:#94a3b8; } .l-run { color:#facc15; font-weight:600; } .l-wait { color:#2d3748; }

/* ── Error box ── */
.err-box {
    background:#1a0a0a; border:1px solid #7f1d1d; border-left:4px solid #ef4444;
    border-radius:8px; padding:14px 16px; font-family:'JetBrains Mono',monospace;
    font-size:0.8rem; color:#fca5a5; line-height:1.7; white-space:pre-wrap;
}
.err-title { color:#ef4444; font-weight:600; font-size:0.9rem; margin-bottom:8px; display:block; }

/* ── File badge ── */
.file-badge {
    display:inline-flex; align-items:center; gap:5px;
    background:#0f2a1a; border:1px solid #166534; border-radius:8px;
    padding:3px 10px; font-family:'JetBrains Mono',monospace;
    font-size:0.74rem; color:#4ade80; margin:3px 3px 3px 0;
}

/* ── Trace / pills / chips ── */
.trace-box {
    background:#0a0c12; border:1px solid #1a2030; border-radius:10px;
    padding:12px 16px; font-family:'JetBrains Mono',monospace;
    font-size:0.78rem; color:#64748b; line-height:1.8; margin-top:8px;
}
.stat-row { display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }
.stat-pill {
    background:#131825; border:1px solid #1e2a40; border-radius:20px;
    padding:3px 12px; font-size:0.72rem; font-family:'JetBrains Mono',monospace; color:#7dd3fc;
}
.source-chip {
    display:inline-block; background:#162032; border:1px solid #1e3a55;
    border-radius:6px; padding:2px 8px; font-size:0.72rem;
    font-family:'JetBrains Mono',monospace; color:#7dd3fc; margin:2px 3px;
}

.stTextInput > div > div > input {
    background:#111420 !important; border:1px solid #1e2540 !important;
    border-radius:12px !important; color:#e2e8f0 !important;
    font-family:'Sora',sans-serif !important; padding:12px 16px !important;
}
.stTextInput > div > div > input:focus {
    border-color:#38bdf8 !important; box-shadow:0 0 0 2px rgba(56,189,248,0.15) !important;
}
.stButton > button {
    background:#0f172a !important; border:1px solid #1e3a55 !important;
    color:#38bdf8 !important; border-radius:10px !important;
    font-family:'Sora',sans-serif !important; font-size:0.85rem !important;
    padding:8px 18px !important; transition:all 0.2s;
}
.stButton > button:hover { background:#162032 !important; border-color:#38bdf8 !important; }
.main-header { text-align:center; padding:1.5rem 0 0.5rem; }
.main-header h1 {
    font-family:'Sora',sans-serif; font-weight:600; font-size:2rem;
    background:linear-gradient(135deg,#38bdf8,#818cf8);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;
}
.main-header p { color:#475569; font-size:0.85rem; margin-top:4px; font-family:'JetBrains Mono',monospace; }
hr { border-color:#1e2130 !important; }
details { background:#0a0c12 !important; border-radius:8px !important; }
.stSelectbox div[data-baseweb="select"] > div { background:#111420 !important; border-color:#1e2540 !important; color:#e2e8f0 !important; }
.stProgress > div > div > div > div { background:linear-gradient(90deg,#38bdf8,#818cf8) !important; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  STEP DEFINITIONS
# ════════════════════════════════════════════════════════════
STEPS = [
    ("🔑", "Validating API key & importing modules"),
    ("📂", "Resolving document sources"),
    ("🤗", "Loading sentence-transformer model (~80 MB first run)"),
    ("🔢", "Chunking & embedding documents → ChromaDB"),
    ("🕸️", "Compiling LangGraph state machine"),
    ("☁️",  "Verifying Groq API connection"),
]

def render_steps_html(done: int, current: int, failed: bool = False) -> str:
    rows = ""
    for i, (icon, label) in enumerate(STEPS):
        if i < done:
            rows += f'<div class="step-row"><span class="s-done">✔</span><span class="l-done">{icon} {label}</span></div>'
        elif i == current:
            if failed:
                rows += f'<div class="step-row"><span class="s-run" style="color:#ef4444">✖</span><span class="l-run" style="color:#ef4444">{icon} {label} ← failed here</span></div>'
            else:
                rows += f'<div class="step-row"><span class="s-run">⟳</span><span class="l-run">{icon} {label}</span></div>'
        else:
            rows += f'<div class="step-row"><span class="s-wait">○</span><span class="l-wait">{icon} {label}</span></div>'
    return rows


# ════════════════════════════════════════════════════════════
#  SESSION STATE
# ════════════════════════════════════════════════════════════
for k, v in {
    "messages":     [],
    "chat_history": [],
    "graph":        None,
    "vectorstore":  None,
    "ready":        False,
    "thread_id":    str(uuid.uuid4()),
    "init_error":   None,
    "loaded_files": [],
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    groq_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Free at console.groq.com → API Keys",
    )

    model_choice = st.selectbox(
        "Groq Model",
        ["llama-3.3-70b-versatile", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"],
    )

    st.markdown("---")
    st.markdown("## 📂 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload .txt files (multiple allowed)",
        type=["txt"],
        accept_multiple_files=True,
        help="Upload one or more .txt documents to index",
    )
    if uploaded_files:
        badges = "".join(f'<span class="file-badge">📄 {f.name}</span>' for f in uploaded_files)
        st.markdown(badges, unsafe_allow_html=True)

    use_sample = st.checkbox(
        "Also include sample knowledge_base.txt",
        value=(not bool(uploaded_files)),
    )

    st.markdown("---")
    st.markdown("## 🔧 RAG Settings")
    retrieval_k         = st.slider("Top-K chunks", 3, 10, 5)
    relevance_threshold = st.slider("Relevance threshold", 0.1, 0.9, 0.4, 0.05)
    max_rewrites        = st.slider("Max query rewrites", 0, 3, 2)

    st.markdown("---")
    init_btn = st.button("🚀 Initialize System", use_container_width=True)

    st.markdown("---")
    st.markdown("**Stack**")
    st.code("LLM      → Groq\nEmbed    → sentence-transformers\nVectorDB → ChromaDB\nSearch   → DuckDuckGo\nGraph    → LangGraph\nUI       → Streamlit", language=None)

    if st.session_state.ready:
        st.success("✅ System ready")
        for fn in st.session_state.loaded_files:
            st.caption(f"📄 {fn}")
    elif st.session_state.init_error:
        st.error("❌ Init failed")
    else:
        st.warning("⚠️ Not initialised")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages     = []
            st.session_state.chat_history = []
            st.session_state.thread_id    = str(uuid.uuid4())
            st.rerun()
    with c2:
        if st.button("🔄 Re-index", use_container_width=True):
            st.session_state.ready        = False
            st.session_state.graph        = None
            st.session_state.vectorstore  = None
            st.session_state.init_error   = None
            st.session_state.loaded_files = []
            import shutil; shutil.rmtree("./.chroma_db_v2", ignore_errors=True)
            st.rerun()


# ════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
  <h1>🧠 Agentic RAG</h1>
  <p>LangGraph · Groq · ChromaDB · DuckDuckGo · sentence-transformers</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")


# ════════════════════════════════════════════════════════════
#  INITIALIZATION WITH PROGRESS BAR
# ════════════════════════════════════════════════════════════
if init_btn:
    st.session_state.init_error = None

    prog_bar   = st.progress(0)
    step_box   = st.empty()
    status_msg = st.empty()

    def tick(done, current, msg, failed=False):
        prog_bar.progress(int(done / len(STEPS) * 100))
        step_box.markdown(render_steps_html(done, current, failed), unsafe_allow_html=True)
        status_msg.caption(f"⏳ {msg}")

    current_step = 0
    try:
        # ── Step 0 ───────────────────────────────────────────
        tick(0, 0, "Checking Groq key format and importing modules…")
        time.sleep(0.2)

        if not groq_key or not groq_key.strip().startswith("gsk_"):
            raise ValueError(
                "Groq API key is missing or invalid.\n"
                "It must start with: gsk_\n"
                "Get a FREE key at:  https://console.groq.com"
            )
        from rag_engine import build_vectorstore, build_graph

        # ── Step 1 ───────────────────────────────────────────
        current_step = 1
        tick(1, 1, "Resolving document sources…")

        txt_paths = []
        if uploaded_files:
            for uf in uploaded_files:
                import tempfile
                _tmp = Path(tempfile.gettempdir())
                p = _tmp / uf.name
                p.write_bytes(uf.read())
                txt_paths.append(str(p))

        if use_sample and Path("knowledge_base.txt").exists():
            txt_paths.append("knowledge_base.txt")

        if not txt_paths:
            raise FileNotFoundError(
                "No documents found!\n"
                "• Upload a .txt file via the sidebar, OR\n"
                "• Place knowledge_base.txt next to app.py and check 'include sample'"
            )

        file_names = [Path(p).name for p in txt_paths]

        # ── Step 2 ───────────────────────────────────────────
        current_step = 2
        tick(2, 2, "Loading sentence-transformer model (downloads ~80 MB on first run — be patient)…")

        # ── Step 3 ───────────────────────────────────────────
        current_step = 3
        tick(3, 3, f"Embedding {len(txt_paths)} document(s) into ChromaDB…")

        vs = build_vectorstore(txt_paths[0])

        if len(txt_paths) > 1:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_community.embeddings import SentenceTransformerEmbeddings
            emb = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            spl = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60)
            for extra in txt_paths[1:]:
                text   = Path(extra).read_text(encoding="utf-8")
                chunks = spl.create_documents([text])
                for i, c in enumerate(chunks):
                    c.metadata["chunk_id"] = i
                    c.metadata["source"]   = Path(extra).name
                vs.add_documents(chunks)

        # ── Step 4 ───────────────────────────────────────────
        current_step = 4
        tick(4, 4, "Compiling LangGraph state machine…")

        g = build_graph(
            groq_api_key=groq_key.strip(),
            vectorstore=vs,
            model=model_choice,
            retrieval_k=retrieval_k,
            relevance_threshold=relevance_threshold,
            max_rewrites=max_rewrites,
        )

        # ── Step 5 ───────────────────────────────────────────
        current_step = 5
        tick(5, 5, "Pinging Groq API with a test message…")

        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage as HM
        ChatGroq(model=model_choice, temperature=0, api_key=groq_key.strip()).invoke(
            [HM(content="Reply with one word: OK")]
        )

        # ── All done ─────────────────────────────────────────
        prog_bar.progress(100)
        step_box.markdown(render_steps_html(len(STEPS), -1), unsafe_allow_html=True)
        status_msg.caption("✅ All done!")
        time.sleep(0.5)

        prog_bar.empty(); step_box.empty(); status_msg.empty()

        st.session_state.graph        = g
        st.session_state.vectorstore  = vs
        st.session_state.ready        = True
        st.session_state.loaded_files = file_names

        st.success(f"✅ Ready! Indexed: **{', '.join(file_names)}**")
        time.sleep(1)
        st.rerun()

    except Exception as exc:
        short = str(exc)
        full  = traceback.format_exc()

        prog_bar.progress(int(current_step / len(STEPS) * 100))
        step_box.markdown(render_steps_html(current_step, current_step, failed=True), unsafe_allow_html=True)
        status_msg.caption(f"❌ Failed at step {current_step + 1}")
        time.sleep(0.4)
        prog_bar.empty(); step_box.empty(); status_msg.empty()

        st.session_state.init_error = full
        st.session_state.ready      = False

        st.markdown(
            f'<div class="err-box"><span class="err-title">❌ Failed at Step {current_step + 1} — {STEPS[current_step][1]}</span>{short}</div>',
            unsafe_allow_html=True,
        )

        with st.expander("🔍 Full traceback", expanded=True):
            st.code(full, language="python")

        st.markdown("#### 💡 Common fixes:")
        if "gsk_" in short or "401" in short or "api key" in short.lower():
            st.info("🔑 Copy your full Groq key (`gsk_...`) from [console.groq.com](https://console.groq.com)")
        elif "file" in short.lower() or "document" in short.lower() or "No documents" in short:
            st.info("📂 Upload a `.txt` file in the sidebar, or place `knowledge_base.txt` in the same folder as `app.py`")
        elif "import" in short.lower() or "module" in short.lower() or "sentence" in short.lower():
            st.info("📦 Run: `pip install -r requirements.txt`  \nFirst run downloads ~80 MB — needs internet.")
        elif "connection" in short.lower() or "timeout" in short.lower() or "network" in short.lower():
            st.info("🌐 Check your internet connection. Groq API requires outbound HTTPS.")
        else:
            st.info("🛠️ Run in terminal to isolate the issue:\n```\npython -c \"from rag_engine import build_vectorstore, build_graph; print('OK')\"\n```")


# ════════════════════════════════════════════════════════════
#  CONVERSATION HISTORY
# ════════════════════════════════════════════════════════════
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown('<div class="role-label role-user">You</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="role-label role-bot">Assistant</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="assistant-bubble">{msg["content"]}</div>', unsafe_allow_html=True)

        if msg.get("trace"):
            with st.expander("🔎 Agent Trace", expanded=False):
                st.markdown(
                    f'<div class="trace-box">{"<br>".join(msg["trace"])}</div>',
                    unsafe_allow_html=True,
                )

        meta = msg.get("meta", {})
        if meta:
            pills = []
            if meta.get("query_type"):              pills.append(f"type: {meta['query_type']}")
            if meta.get("relevance_score") is not None: pills.append(f"relevance: {meta['relevance_score']}")
            if meta.get("iteration") is not None:   pills.append(f"rewrites: {meta['iteration']}")
            if meta.get("needs_web"):               pills.append("🌐 web used")
            pill_html = "".join(f'<span class="stat-pill">{p}</span>' for p in pills)
            st.markdown(f'<div class="stat-row">{pill_html}</div>', unsafe_allow_html=True)
            if meta.get("sources"):
                chip_html = "".join(f'<span class="source-chip">📄 {s}</span>' for s in meta["sources"][:6])
                st.markdown(
                    f"<div>{chip_html}</div>",
                    unsafe_allow_html=True,
                )
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  EXAMPLE QUESTIONS
# ════════════════════════════════════════════════════════════
if not st.session_state.messages:
    if st.session_state.ready:
        st.markdown("#### 💡 Try asking:")
        for i, ex in enumerate([
            "What products does TechCorp offer?",  "Compare MediAI and FinanceGuard.",
            "What is the interview process?",       "What are TechCorp's HR benefits?",
            "Who is the CEO of TechCorp?",          "What is the latest news in AI?",
            "How much do senior engineers earn?",   "Hello! What can you help me with?",
        ]):
            with st.columns(2)[i % 2]:
                if st.button(ex, key=f"ex_{i}", use_container_width=True):
                    st.session_state["_prefill"] = ex
                    st.rerun()
    else:
        st.info("👈 Enter your Groq API key in the sidebar and click **🚀 Initialize System** to get started.")


# ════════════════════════════════════════════════════════════
#  CHAT INPUT
# ════════════════════════════════════════════════════════════
prefill = st.session_state.pop("_prefill", "")

col_q, col_s = st.columns([6, 1])
with col_q:
    question = st.text_input(
        "q", value=prefill,
        placeholder="Type your question and press Enter…",
        label_visibility="collapsed",
        key="chat_input",
        disabled=not st.session_state.ready,
    )
with col_s:
    send = st.button("Send →", use_container_width=True, disabled=not st.session_state.ready)


# ════════════════════════════════════════════════════════════
#  QUERY EXECUTION
# ════════════════════════════════════════════════════════════
# ── Guard: only run when Send button is explicitly clicked ──
if send and question.strip() and st.session_state.ready:
    user_q = question.strip()

    # Prevent duplicate submission on rerun
    if user_q != st.session_state.get("_last_processed", ""):
        st.session_state["_last_processed"] = user_q
        st.session_state.messages.append({"role": "user", "content": user_q})

        with st.spinner("🤖 Agents thinking…"):
            try:
                from rag_engine import run_query
                result = run_query(
                    graph=st.session_state.graph,
                    question=user_q,
                    chat_history=st.session_state.chat_history,
                    thread_id=st.session_state.thread_id,
                    relevance_threshold=relevance_threshold,
                    max_rewrites=max_rewrites,
                )
                answer = result.get("answer", "No answer generated.")
                docs   = result.get("documents", [])
                sources = list(dict.fromkeys(
                    f"{d.metadata.get('source','?')} #chunk{d.metadata.get('chunk_id','?')}"
                    for d in docs
                ))
                st.session_state.messages.append({
                    "role": "assistant", "content": answer,
                    "trace": result.get("agent_log", []),
                    "meta": {
                        "query_type":      result.get("query_type", ""),
                        "relevance_score": result.get("relevance_score", 0),
                        "iteration":       result.get("iteration", 0),
                        "needs_web":       result.get("needs_web", False),
                        "sources":         sources,
                    },
                })
                st.session_state.chat_history.append({"user": user_q, "assistant": answer})

            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ **Query error:** {e}\n```\n{traceback.format_exc()}\n```",
                    "trace": [], "meta": {},
                })

        st.rerun()