import streamlit as st
import os
import time
import voyageai
from chromadb import PersistentClient
from groq import Groq
from dotenv import load_dotenv

# --- 1. SETUP INIZIALE ---
load_dotenv()

# Configurazione Pagina (Barclays Inspired)
st.set_page_config(
    page_title="Financial AI Analyst",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (Advanced Theme with Menu Fix) ---
st.markdown("""
<style>
    /* Background generale */
    .stApp { background-color: #f8f9fa; }

    /* --- SIDEBAR STYLE (Blu Scuro) --- */
    section[data-testid="stSidebar"] { background-color: #00395D; }
    
    /* Testo Bianco nella Sidebar */
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div { color: #ffffff !important; }
    
    /* --- FIX MENU A TENDINA (SELECTBOX) - Testo Scuro su Bianco --- */
    div[data-baseweb="select"] span { color: #00395D !important; }
    div[data-baseweb="select"] div { color: #00395D !important; }
    ul[data-baseweb="menu"] li span { color: #00395D !important; }
    
    /* Radio buttons text fix */
    div[role="radiogroup"] label span { color: white !important; }

    /* --- WELCOME MESSAGE BOX (Blu Chiaro) --- */
    .welcome-box {
        background-color: #e3f2fd; border-left: 6px solid #00AEEF;
        padding: 20px; border-radius: 10px; margin-bottom: 20px;
        color: #0d47a1; font-size: 16px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .welcome-box strong { color: #00395D; }
    .welcome-box h3, .welcome-box p, .welcome-box ul { color: #00395D !important; margin-top: 0; }

    /* --- CHAT BUBBLES --- */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #e1f5fe; border: 1px solid #b3e5fc; border-radius: 15px 15px 0 15px;
    }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 15px 15px 15px 0;
    }

    h1 { color: #00395D !important; font-family: 'Segoe UI', sans-serif; }
    .stButton button { background-color: #00AEEF !important; color: white !important; border: none; font-weight: bold; }
    .stButton button:hover { background-color: #0077c2 !important; transform: translateY(-2px); }
    
    /* --- ICONA MONETA --- */
    .coin-icon { font-size: 80px; text-align: center; margin-bottom: 20px; display: block; animation: float 3s ease-in-out infinite; }
    @keyframes float { 0% { transform: translateY(0px); } 50% { transform: translateY(-10px); } 100% { transform: translateY(0px); } }
</style>
""", unsafe_allow_html=True)

# Percorsi
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = CURRENT_DIR 
CHROMA_VOYAGE_DIR = os.path.join(BASE_PATH, "RAG BANK FINAL", "chroma_db")
CHROMA_E5_DIR = os.path.join(BASE_PATH, "RAG BANK FINAL", "chroma_db_e5")

# --- 2. IL MOTORE RAG ---
class UnifiedFinancialRAG:
    def __init__(self, embedding_type="voyage", llm_model="llama-3.3-70b-versatile", use_multiquery=True):
        self.embedding_type = embedding_type
        self.llm_model = llm_model
        self.use_multiquery = use_multiquery
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        if embedding_type == "voyage":
            if not os.path.exists(CHROMA_VOYAGE_DIR):
                st.error(f"❌ Database Voyage non trovato in: {CHROMA_VOYAGE_DIR}")
                st.stop()
            self.client_db = PersistentClient(path=CHROMA_VOYAGE_DIR)
            self.collection = self.client_db.get_collection("finance_docs")
            self.vo_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
        else:
            if not os.path.exists(CHROMA_E5_DIR):
                st.error(f"❌ Database E5 non trovato in: {CHROMA_E5_DIR}")
                st.stop()
            self.client_db = PersistentClient(path=CHROMA_E5_DIR)
            self.collection = self.client_db.get_collection("finance_docs_e5")
            
            # Lazy import CON FIX PER WINDOWS/CPU
            from sentence_transformers import SentenceTransformer, CrossEncoder
            if "e5_model" not in st.session_state:
                # FORZIAMO DEVICE='CPU' PER EVITARE ERRORE META TENSOR
                st.session_state.e5_model = SentenceTransformer("intfloat/e5-large-v2", device="cpu")
            self.local_embedder = st.session_state.e5_model
            
            if "reranker" not in st.session_state:
                 # FORZIAMO DEVICE='CPU' ANCHE QUI
                 st.session_state.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
            self.local_reranker = st.session_state.reranker

    def _decompose_query(self, query: str):
        if not self.use_multiquery: return [query]
        prompt = f"Act as a financial analyst. Break down this query into 2-3 simple search queries.\nUser Query: '{query}'\nOutput ONLY the sub-queries, one per line."
        try:
            resp = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role":"user", "content":prompt}],
                temperature=0.1
            )
            return [q.strip() for q in resp.choices[0].message.content.split('\n') if q.strip()]
        except: return [query]

    def retrieve(self, query: str):
        sub_queries = self._decompose_query(query)
        all_candidates = []
        for sq in sub_queries:
            try:
                if self.embedding_type == "voyage":
                    emb = self.vo_client.embed([sq], model="voyage-3", input_type="query").embeddings[0]
                    res = self.collection.query(query_embeddings=[emb], n_results=10)
                else:
                    emb = self.local_embedder.encode(f"query: {sq}", normalize_embeddings=True).tolist()
                    res = self.collection.query(query_embeddings=[emb], n_results=10)
                
                for doc, meta in zip(res['documents'][0], res['metadatas'][0]):
                    all_candidates.append({'text': doc, 'filename': meta['filename']})
            except: continue

        unique_docs = {c['text']: c for c in all_candidates}.values()
        candidates = list(unique_docs)
        if not candidates: return []

        if self.embedding_type == "voyage":
            try:
                docs_text = [c['text'] for c in candidates]
                reranked = self.vo_client.rerank(query, docs_text, model="rerank-2", top_k=5)
                final = []
                for r in reranked.results:
                    c = candidates[r.index]
                    c['score'] = r.relevance_score
                    final.append(c)
                return final
            except: 
                 for c in candidates[:5]: c['score'] = 0.0
                 return candidates[:5]
        else:
            try:
                pairs = [[query, c['text']] for c in candidates]
                scores = self.local_reranker.predict(pairs)
                scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
                final = []
                for c, s in scored[:5]:
                    c['score'] = s
                    final.append(c)
                return final
            except: return candidates[:5]

    def answer(self, query: str):
        chunks = self.retrieve(query)
        if not chunks: return "I couldn't find relevant information in the documents.", []
        
        context = "\n\n".join([f"[Source: {c['filename']}]\n{c['text']}" for c in chunks])
        
        sys_prompt = """You are a Senior Financial Analyst. 
        Answer the question strictly based on the provided context. 
        Format tables in Markdown. Cite the source document."""
        
        try:
            resp = self.groq_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUERY: {query}"}
                ],
                temperature=0.1
            )
            return resp.choices[0].message.content, chunks
        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg:
                return "⚠️ **API Rate Limit Reached.** Please wait a moment or switch LLM model in the sidebar.", []
            elif "model_decommissioned" in error_msg:
                return f"⚠️ **Model Error:** The model '{self.llm_model}' is unavailable.", []
            return f"Error: {e}", []

# --- 3. INTERFACCIA STREAMLIT ---

# Sidebar
with st.sidebar:
    st.markdown('<div class="coin-icon">🪙</div>', unsafe_allow_html=True)
    st.markdown("## ⚙️ Control Panel")
    
    embedding_opt = st.radio("Search Engine:", ["voyage", "e5"], index=0, 
                             format_func=lambda x: "Voyage AI (Pro)" if x=="voyage" else "Local (Free)")
    
    # MODELLO QWEN 3 CORRETTO
    llm_opt = st.selectbox("LLM Model:", 
                           ["llama-3.3-70b-versatile", "qwen/qwen3-32b"], 
                           index=0)
    
    multiquery = st.checkbox("Multi-Step Reasoning", value=True)
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main Content
st.title("🏦 AI Financial Analyst Pro")
st.caption(f"Active Engine: {embedding_opt.upper()} + {llm_opt}")

welcome_html = """
<div class="welcome-box">
    <h3>👋 Hello! I am your AI Financial Assistant.</h3>
    <p>I have direct access to <strong>2022-2024 Annual Reports & Pillar 3 Disclosures</strong> for:</p>
    <ul>
        <li>🇬🇧 <strong>Barclays</strong> & <strong>HSBC</strong></li>
        <li>🇺🇸 <strong>Bank of America</strong>, <strong>Wells Fargo</strong> & <strong>Goldman Sachs</strong></li>
        <li>🇳🇱 <strong>ING Group</strong></li>
    </ul>
    <p><em>How can I help you analyze their performance, risks, or capital ratios today?</em></p>
</div>
"""

# Initialize Bot
config_key = f"{embedding_opt}_{llm_opt}_{multiquery}"
if "current_config" not in st.session_state or st.session_state.current_config != config_key:
    with st.spinner("Initializing Neural Engines..."):
        try:
            st.session_state.bot = UnifiedFinancialRAG(embedding_opt, llm_opt, multiquery)
            st.session_state.current_config = config_key
        except Exception as e:
            st.error(f"Startup Error: {e}")

if "messages" not in st.session_state: st.session_state.messages = [] 

st.markdown(welcome_html, unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a financial question (e.g. LCR of Barclays 2023)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing financial documents..."):
            if "bot" in st.session_state:
                response_text, sources = st.session_state.bot.answer(prompt)
                
                if "Error:" in response_text or "⚠️" in response_text:
                     st.error(response_text)
                else:
                    st.markdown(response_text)
                    if sources:
                        with st.expander("📚 Source Documents & Relevance"):
                            for i, s in enumerate(sources, 1):
                                score_val = s.get('score', 0.0)
                                st.markdown(f"**{i}. {s['filename']}** (Score: {score_val:.2f})")
                                st.caption(s['text'][:200] + "...")
                
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            else:
                st.error("System not initialized.")