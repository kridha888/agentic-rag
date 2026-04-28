"""
rag_engine.py
─────────────────────────────────────────────────────────────
Pure backend — LangGraph Agentic RAG engine.
No UI code lives here; Streamlit imports this module.

Free stack:
  LLM        → Groq (llama-3.3-70b-versatile, free tier)
  Embeddings → sentence-transformers/all-MiniLM-L6-v2 (local)
  Vector DB  → ChromaDB (local)
  Web Search → DuckDuckGo (no API key)
─────────────────────────────────────────────────────────────
"""

import os, re, json, operator
from pathlib import Path
from typing import TypedDict, Annotated, List, Optional

# ── LangChain / LangGraph ────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# ════════════════════════════════════════════════════════════
#  STATE
# ════════════════════════════════════════════════════════════
class AgentState(TypedDict):
    question        : str
    documents       : Annotated[List, operator.add]
    answer          : str
    query_type      : str
    needs_web       : bool
    web_results     : str
    relevance_score : float
    iteration       : int
    chat_history    : Annotated[List, operator.add]
    agent_log       : Annotated[List, operator.add]
    _threshold      : float
    _max_rewrites   : int


# ════════════════════════════════════════════════════════════
#  KNOWLEDGE BASE
# ════════════════════════════════════════════════════════════
def build_vectorstore(txt_path: str, collection: str = "agentic_rag_v2") -> Chroma:
    """Chunk the .txt file, embed with sentence-transformers, persist in ChromaDB."""
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"   # 22 MB, runs on CPU, no API key
    )

    persist_dir = "./.chroma_db_v2"
    # If collection already exists and the file hasn't changed, reuse it
    if Path(persist_dir).exists():
        try:
            vs = Chroma(
                collection_name=collection,
                embedding_function=embeddings,
                persist_directory=persist_dir,
            )
            if vs._collection.count() > 0:
                return vs
        except Exception:
            pass

    text = Path(txt_path).read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=60,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.create_documents([text])
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
        c.metadata["source"]   = Path(txt_path).name

    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection,
        persist_directory=persist_dir,
    )
    return vs


# ════════════════════════════════════════════════════════════
#  AGENT NODES
# ════════════════════════════════════════════════════════════
class Nodes:
    def __init__(self, llm: ChatGroq, vectorstore: Chroma,
                 retrieval_k: int = 5, relevance_threshold: float = 0.40,
                 max_rewrites: int = 2):
        self.llm      = llm
        self.relevance_threshold = relevance_threshold
        self.max_rewrites        = max_rewrites
        self.retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": retrieval_k, "fetch_k": retrieval_k * 4}
        )
        self.web_tool = DuckDuckGoSearchRun()

    # ── 1. CLASSIFY ──────────────────────────────────────────
    def classify_query(self, state: AgentState) -> dict:
        q = state["question"]
        prompt = f"""Classify the user question and return ONLY a JSON object.
Question: "{q}"

{{
  "query_type": "factual" | "analytical" | "conversational",
  "needs_web": true | false,
  "reasoning": "<one sentence>"
}}

Rules:
- conversational → greetings, thanks, meta
- factual → specific facts, numbers, names from a document
- analytical → comparisons, explanations, summaries
- needs_web → true ONLY for current events, stock prices, news, or topics clearly outside a company KB"""

        raw = self.llm.invoke([HumanMessage(content=prompt)]).content.strip()
        raw = re.sub(r"```[a-z]*|```", "", raw).strip()
        try:
            data = json.loads(raw)
        except Exception:
            data = {"query_type": "factual", "needs_web": False, "reasoning": "fallback"}

        qt  = data.get("query_type", "factual")
        nw  = bool(data.get("needs_web", False))
        log = f"🔍 **Classifier** → type=`{qt}` | web={nw} | _{data.get('reasoning','')}_"
        return {
            "query_type": qt,
            "needs_web":  nw,
            "iteration":  state.get("iteration", 0),
            "agent_log":  [log],
        }

    # ── 2. WEB SEARCH ────────────────────────────────────────
    def web_search(self, state: AgentState) -> dict:
        q = state["question"]
        try:
            result = self.web_tool.run(q)
        except Exception as e:
            result = f"Web search failed: {e}"
        log = f"🌐 **Web Search** → fetched {len(result)} chars from DuckDuckGo"
        return {"web_results": result, "agent_log": [log]}

    # ── 3. RETRIEVE ──────────────────────────────────────────
    def retrieve(self, state: AgentState) -> dict:
        docs = self.retriever.invoke(state["question"])
        log  = f"📄 **Retriever** → {len(docs)} chunks via MMR from ChromaDB"
        return {"documents": docs, "agent_log": [log]}

    # ── 4. GRADE ─────────────────────────────────────────────
    def grade_documents(self, state: AgentState) -> dict:
        q    = state["question"]
        docs = state["documents"]

        if not docs:
            return {"relevance_score": 0.0, "agent_log": ["⚖️ **Grader** → no docs, score=0.0"]}

        snippets = "\n".join(
            f"[{i}] {d.page_content[:180]}" for i, d in enumerate(docs[:4])
        )
        prompt = f"""Score how relevant these document chunks are to the question (0.0 → 1.0).
Question: {q}
Chunks:
{snippets}

Reply with ONLY a decimal number like 0.75. Nothing else."""

        raw   = self.llm.invoke([HumanMessage(content=prompt)]).content
        nums  = re.findall(r"\d+\.?\d*", raw)
        score = float(nums[0]) if nums else 0.5
        score = round(min(max(score, 0.0), 1.0), 2)
        log   = f"⚖️ **Grader** → relevance score = `{score}` (threshold={self.relevance_threshold})"
        return {"relevance_score": score, "agent_log": [log]}

    # ── 5. REWRITE ───────────────────────────────────────────
    def rewrite_query(self, state: AgentState) -> dict:
        orig = state["question"]
        prompt = f"""The question failed to retrieve useful documents.
Original: "{orig}"
Rewrite it to be more specific, use synonyms, and improve retrieval accuracy.
Return ONLY the rewritten question."""

        new_q     = self.llm.invoke([HumanMessage(content=prompt)]).content.strip().strip('"')
        iteration = state.get("iteration", 0) + 1
        log       = f"✏️ **Rewriter** → iteration {iteration} | new query: _{new_q}_"
        return {
            "question":  new_q,
            "documents": [],          # clear stale docs
            "iteration": iteration,
            "agent_log": [log],
        }

    # ── 6. GENERATE ──────────────────────────────────────────
    def generate_answer(self, state: AgentState) -> dict:
        q          = state["question"]
        docs       = state["documents"]
        web        = state.get("web_results", "")
        history    = state.get("chat_history", [])

        kb_context = "\n\n".join(
            f"[Chunk {d.metadata.get('chunk_id','?')} | {d.metadata.get('source','?')}]\n{d.page_content}"
            for d in docs
        ) if docs else "No KB documents retrieved."

        web_context = f"\n\n[Web Search Results]\n{web}" if web else ""

        history_text = "\n".join(
            f"User: {h['user']}\nAssistant: {h['assistant']}"
            for h in history[-4:]
        )

        system = """You are an expert AI assistant with access to a company knowledge base and optionally web search results.
Answer using the provided context. Be structured, precise, and cite chunk IDs when relevant.
If context is insufficient, say so honestly — do not hallucinate."""

        user_msg = f"""
Previous conversation:
{history_text or "None"}

Knowledge Base Context:
{kb_context}{web_context}

Question: {q}

Provide a well-structured, thorough answer:"""

        answer = self.llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=user_msg)
        ]).content.strip()

        log = f"✅ **Generator** → answer produced ({len(answer)} chars)"
        return {"answer": answer, "agent_log": [log]}

    # ── 7. CONVERSATIONAL ────────────────────────────────────
    def handle_conversation(self, state: AgentState) -> dict:
        resp = self.llm.invoke([
            SystemMessage(content="You are a friendly, helpful AI assistant."),
            HumanMessage(content=state["question"])
        ]).content.strip()
        log = "💬 **Conversational** → direct reply, no retrieval"
        return {"answer": resp, "documents": [], "agent_log": [log]}


# ════════════════════════════════════════════════════════════
#  ROUTING
# ════════════════════════════════════════════════════════════
def route_after_classify(state: AgentState) -> str:
    if state["query_type"] == "conversational":
        return "conversational"
    if state.get("needs_web"):
        return "web_search"
    return "retrieve"

def route_after_grade(state: AgentState) -> str:
    if state.get("relevance_score", 0) >= state.get("_threshold", 0.40):
        return "generate"
    if state.get("iteration", 0) >= state.get("_max_rewrites", 2):
        return "generate"
    return "rewrite"


# ════════════════════════════════════════════════════════════
#  GRAPH FACTORY
# ════════════════════════════════════════════════════════════
def build_graph(groq_api_key: str, vectorstore: Chroma,
                model: str = "llama-3.3-70b-versatile",
                retrieval_k: int = 5,
                relevance_threshold: float = 0.40,
                max_rewrites: int = 2):
    llm = ChatGroq(
        model=model,
        temperature=0.1,
        api_key=groq_api_key,
    )
    n = Nodes(llm, vectorstore, retrieval_k, relevance_threshold, max_rewrites)

    g = StateGraph(AgentState)
    g.add_node("classify",       n.classify_query)
    g.add_node("web_search",     n.web_search)
    g.add_node("retrieve",       n.retrieve)
    g.add_node("grade",          n.grade_documents)
    g.add_node("rewrite",        n.rewrite_query)
    g.add_node("generate",       n.generate_answer)
    g.add_node("conversational", n.handle_conversation)

    g.set_entry_point("classify")
    g.add_conditional_edges("classify", route_after_classify, {
        "retrieve":       "retrieve",
        "web_search":     "web_search",
        "conversational": "conversational",
    })
    g.add_edge("web_search", "retrieve")
    g.add_edge("retrieve",   "grade")
    g.add_conditional_edges("grade", route_after_grade, {
        "generate": "generate",
        "rewrite":  "rewrite",
    })
    g.add_edge("rewrite",        "retrieve")
    g.add_edge("generate",       END)
    g.add_edge("conversational", END)

    return g.compile(checkpointer=MemorySaver())


# ════════════════════════════════════════════════════════════
#  CONVENIENCE RUNNER  (used by Streamlit)
# ════════════════════════════════════════════════════════════
def run_query(
    graph,
    question: str,
    chat_history: list,
    thread_id: str = "st-session",
    relevance_threshold: float = 0.40,
    max_rewrites: int = 2,
) -> dict:
    state_input = {
        "question":       question,
        "documents":      [],
        "answer":         "",
        "query_type":     "",
        "needs_web":      False,
        "web_results":    "",
        "relevance_score": 0.0,
        "iteration":      0,
        "chat_history":   chat_history.copy(),
        "agent_log":      [],
        "_threshold":     relevance_threshold,
        "_max_rewrites":  max_rewrites,
    }
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(state_input, config=config)