import os
import json
import time
import numpy as np
import streamlit as st
from dotenv import load_dotenv

# LangChain
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Optional local open‑source LLM via Ollama (HTTP)
import requests

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in .env or Streamlit Secrets.")
    st.stop()

st.set_page_config(page_title="Advanced: LangChain + RAG", page_icon="🧩")

st.title("🧩 Advanced: Chains / Agents + RAG ‘Seen‑Before’ Check")

# ---------- Settings ----------
with st.sidebar:
    st.subheader("Model Settings")
    lc_model = st.selectbox("OpenAI Chat Model", ["gpt-3.5-turbo", "gpt-4o-mini"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.5, 0.1)
    max_tokens = st.number_input("Max tokens", 64, 4000, 512, 64)
    st.markdown("---")
    use_ollama = st.checkbox("Use local open‑source LLM via Ollama (instead of OpenAI)", value=False)
    ollama_model = st.text_input("Ollama model name", value="llama3", help="Requires `ollama run llama3` locally.")
    st.caption("If Ollama is enabled, requests go to http://localhost:11434")

# ---------- LangChain LLM ----------
def get_lc_llm():
    if use_ollama:
        return None  # we’ll hit HTTP below instead
    return ChatOpenAI(api_key=OPENAI_API_KEY, model=lc_model, temperature=temperature, max_tokens=max_tokens)

# ---------- Simple Chain (LangChain) ----------
st.header("1) LangChain Chat (prompt template + memory-ish)")

SYSTEM = """You are an expert interview coach. Be concise, structured, and constructive.
If the user asks for questions, provide one at a time and wait for their answer before moving on."""

if "lc_history" not in st.session_state:
    st.session_state.lc_history = []  # list[dict]: role/content

user_msg = st.text_input("Your message (LangChain)", key="lc_user_msg")
send_lc = st.button("Send (LangChain)")

if send_lc and user_msg.strip():
    if use_ollama:
        # Local open‑source path via Ollama HTTP
        payload = {
            "model": ollama_model,
            "prompt": f"{SYSTEM}\n\nUser: {user_msg}\nAssistant:",
            "stream": False
        }
        try:
            r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            reply = data.get("response", "").strip()
        except Exception as e:
            st.error(f"Ollama error: {e}")
            reply = ""
    else:
        llm = get_lc_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM),
            ("human", "{question}")
        ])
        chain = prompt | llm
        reply = chain.invoke({"question": user_msg}).content

    st.session_state.lc_history.append({"role": "user", "content": user_msg})
    st.session_state.lc_history.append({"role": "assistant", "content": reply})

if st.session_state.lc_history:
    st.markdown("#### Conversation")
    for m in st.session_state.lc_history:
        st.markdown(f"**{'You' if m['role']=='user' else 'Assistant'}:** {m['content']}")

st.markdown("---")

# ---------- RAG “Seen‑Before” Check with lightweight vector index ----------
st.header("2) Vector ‘Seen‑Before’ Check (novelty gate)")

st.caption(
    "Paste interview prep material (notes, Q&A, JD highlights). "
    "We’ll embed and check against previously stored chunks to detect near‑duplicates."
)

if "rag_index" not in st.session_state:
    st.session_state.rag_index = []  # list of dict: {"text": str, "embedding": list[float]}

emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")

def embed(text: str) -> np.ndarray:
    vec = emb.embed_query(text)
    return np.array(vec, dtype=np.float32)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)

new_chunk = st.text_area("New prep chunk", height=140, placeholder="Paste a question, answer, rubric, or note…")
thr = st.slider("Duplicate threshold (cosine ≥)", 0.70, 0.99, 0.85, 0.01)
col_a, col_b = st.columns(2)
btn_check = col_a.button("Check & Add")
btn_list = col_b.button("Show Index")

if btn_check and new_chunk.strip():
    with st.spinner("Embedding & comparing…"):
        vec = embed(new_chunk)
        sim = 0.0
        nearest = None
        for item in st.session_state.rag_index:
            s = cosine(vec, np.array(item["embedding"], dtype=np.float32))
            if s > sim:
                sim = s
                nearest = item
        if sim >= thr:
            st.warning(f"⚠️ Looks like a duplicate (similarity {sim:.2f}).\n\n**Closest match:**\n\n{nearest['text'][:300]}…")
        else:
            st.session_state.rag_index.append({"text": new_chunk, "embedding": vec.tolist()})
            st.success(f"✅ New chunk added (max similarity {sim:.2f} < {thr:.2f}).")

if btn_list:
    st.json([{"text": i["text"][:160] + ("…" if len(i["text"]) > 160 else "")} for i in st.session_state.rag_index])

st.markdown("##### Ask with Context (top‑k retrieval)")
q = st.text_input("Your question (RAG)", key="rag_q")
k = st.slider("Top‑k", 1, 5, 3)
ask_btn = st.button("Answer with retrieval")

if ask_btn and q.strip():
    if not st.session_state.rag_index:
        st.info("Index is empty. Add some prep chunks first.")
    else:
        with st.spinner("Retrieving & answering…"):
            qv = embed(q)
            scored = []
            for item in st.session_state.rag_index:
                s = cosine(qv, np.array(item["embedding"], dtype=np.float32))
                scored.append((s, item["text"]))
            scored.sort(reverse=True, key=lambda t: t[0])
            ctx = "\n\n".join([f"- {t[1]}" for t in scored[:k]])

            if use_ollama:
                payload = {
                    "model": ollama_model,
                    "prompt": f"{SYSTEM}\nUse only this context if relevant:\n{ctx}\n\nUser: {q}\nAssistant:",
                    "stream": False
                }
                try:
                    r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
                    r.raise_for_status()
                    ans = r.json().get("response", "").strip()
                except Exception as e:
                    st.error(f"Ollama error: {e}")
                    ans = ""
            else:
                llm = get_lc_llm()
                messages = [
                    SystemMessage(content=SYSTEM),
                    HumanMessage(content=f"Context (use if relevant):\n{ctx}\n\nQuestion: {q}")
                ]
                ans = llm.invoke(messages).content

            st.markdown("**Answer:**")
            st.write(ans)

st.info("This page covers: **Full chat via components, LangChain chains, vector ‘seen‑before’ check, open‑source LLM via Ollama.**")