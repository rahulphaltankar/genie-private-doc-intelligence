import streamlit as st
import os
import json
import io
import re
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import nltk

# Genie Modules
from ingestion_pipeline import process_uploaded_files
from bm25_index import BM25Index
from hybrid_retriever import hybrid_search
try:
    from reranker import Reranker
except Exception as e:
    Reranker = None
    import traceback
    st.error(f"Genie System Error (Reranker Import): {str(e)}")
    st.code(traceback.format_exc())
# from quiz_generator import generate_quiz
from mode_router import detect_mode
from citation_formatter import format_harvard_citation
from grounding import compute_grounding_score
from gatekeeper import run_gatekeeper
from trace_logger import log_trace

# UI Helpers
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import requests

# Load environment variables
load_dotenv()

# Pre-load NLTK
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass

# Configure page
st.set_page_config(
    page_title="Genie AI",
    page_icon="🧞",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- IDE-Themed Aesthetic Overhaul ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global IDE Dark Theme */
    .stApp {
        background-color: #1e1e1e;
        color: #d4d4d4;
        font-family: 'Inter', sans-serif;
    }
    
    .block-container {
        padding-top: 2rem;
        max-width: 900px;
    }

    /* Branding - Clean & High Contrast */
    .genie-header {
        font-size: 2.2rem;
        font-weight: 500;
        text-align: left;
        color: #e0e0e0;
        margin-bottom: 0.2rem;
        letter-spacing: -0.04rem;
    }
    .genie-sub {
        color: #858585;
        font-size: 0.95rem;
        margin-bottom: 2.5rem;
    }

    /* Step Containers - IDE Panels */
    .step-box {
        background-color: #252526;
        border: 1px solid #333333;
        border-radius: 6px;
        padding: 30px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }

    /* High-Legibility Form Elements */
    .stFileUploader section {
        background-color: #2d2d2d !important;
        border: 1px dashed #444444 !important;
        color: #d4d4d4 !important;
    }
    /* Native IDE Action Buttons */
    [data-testid='stFileUploaderDropzone'] button {
        color: #cccccc !important;
        background-color: #3e3e42 !important;
        border: 1px solid #454545 !important;
        border-radius: 4px !important;
    }
    [data-testid='stFileUploaderDropzone'] span {
        color: #cccccc !important;
    }

    /* Message Bubbles - IDE Terminal Consistency */
    .stChatMessage {
        background-color: transparent !important;
        padding: 1rem 0 !important;
        border-bottom: 1px solid #2d2d2d !important;
    }
    .stChatMessage div[data-testid="stMarkdownContainer"] p,
    .stChatMessage div[data-testid="stMarkdownContainer"] li,
    .stChatMessage div[data-testid="stMarkdownContainer"] span {
        color: #d4d4d4 !important;
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
    }
    .stChatMessage div[data-testid="stMarkdownContainer"] h1,
    .stChatMessage div[data-testid="stMarkdownContainer"] h2,
    .stChatMessage div[data-testid="stMarkdownContainer"] h3 {
        color: #e0e0e0 !important;
        font-weight: 500 !important;
    }
    .stChatMessage div[data-testid="stMarkdownContainer"] strong,
    .stChatMessage div[data-testid="stMarkdownContainer"] b {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* Inline Code Blocks - IDE Syntax Highlighting */
    .stChatMessage div[data-testid="stMarkdownContainer"] code {
        color: #ce9178 !important; /* VSCode string color */
        background-color: #2d2d2d !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 4px !important;
        border: 1px solid #404040 !important;
    }
    .stChatMessage div[data-testid="stMarkdownContainer"] pre {
        background-color: #1e1e1e !important;
        border: 1px solid #333333 !important;
        border-radius: 6px !important;
    }

    /* Tables - IDE Structural Data */
    .stChatMessage div[data-testid="stMarkdownContainer"] table {
        color: #d4d4d4 !important;
        border-collapse: collapse !important;
        width: 100% !important;
        margin-top: 1rem !important;
        margin-bottom: 1rem !important;
    }
    .stChatMessage div[data-testid="stMarkdownContainer"] th {
        background-color: #252526 !important;
        color: #9cdcfe !important; /* VSCode variable color */
        font-weight: 600 !important;
        border: 1px solid #404040 !important;
        padding: 0.75rem !important;
        text-align: left !important;
    }
    .stChatMessage div[data-testid="stMarkdownContainer"] td {
        border: 1px solid #333333 !important;
        padding: 0.75rem !important;
    }
    .stChatMessage div[data-testid="stMarkdownContainer"] tr:nth-child(even) {
        background-color: #222222 !important;
    }

    /* Primary Assistant Color Bubble */
    .stChatMessage [data-testid="stChatMessageAvatarAssistant"] {
        background-color: #007acc !important; /* VSCode Blue */
    }

    /* General Default Buttons */
    .stButton > button {
        background-color: #0e639c !important; /* VSCode Button Blue */
        color: #ffffff !important;
        border: 1px solid #1177bb !important;
        border-radius: 4px !important;
        font-weight: 500 !important;
        padding: 4px 12px !important;
    }
    .stButton > button p {
        color: #ffffff !important;
    }
    .stButton > button:hover {
        background-color: #1177bb !important;
        border-color: #1177bb !important;
    }

    /* Download Buttons Customization - Muted Action Links */
    div.stDownloadButton > button {
        background-color: transparent !important;
        border: 1px solid #404040 !important;
        color: #cccccc !important;
        border-radius: 4px !important;
        padding: 4px 10px !important;
        font-size: 0.85rem !important;
        display: inline-flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 0.3rem !important;
        height: 32px !important;
        white-space: nowrap !important;
    }
    div.stDownloadButton > button p {
        color: #cccccc !important;
        font-size: 0.85rem !important;
        margin: 0 !important;
    }
    div.stDownloadButton > button:hover {
        background-color: #333333 !important;
        border-color: #858585 !important;
    }

    /* Global Input - IDE Command Palette Style */
    div[data-testid="stChatInput"] {
        background-color: #1e1e1e !important;
        padding-bottom: 2rem !important;
    }
    div[data-testid="stChatInput"] textarea {
        background-color: #252526 !important;
        color: #d4d4d4 !important;
        border: 1px solid #3c3c3c !important;
        border-radius: 4px !important;
        caret-color: #d4d4d4 !important;
        font-family: 'Inter', sans-serif !important;
    }
    div[data-testid="stChatInput"] textarea:focus {
        border-color: #007acc !important;
        box-shadow: 0 0 0 1px #007acc !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #858585 !important;
        opacity: 1 !important;
    }
    
    /* Hide the light-colored streamlit footer bar background */
    [data-testid="stBottomBlockContainer"] {
        background-color: #1e1e1e !important;
        border-top: 1px solid #2d2d2d !important;
    }
    
    /* Knowledge Bar */
    .knowledge-bar {
        background-color: #000000;
        border-bottom: 1px solid #111111;
        padding: 8px 0;
        display: flex;
        gap: 8px;
    }
    .knowledge-pill {
        background-color: #111111;
        color: #888888;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.7rem;
        border: 1px solid #1a1a1a;
    }

    /* Reset default st content spacing */
    .stMarkdown p { color: #ffffff !important; }
    
    header, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    return SentenceTransformer('all-MiniLM-L6-v2')

def call_mistral_api(query, context):
    api_key = os.getenv("MISTRAL_API_KEY")
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    system_prompt = (
        "You are Genie, a precision AI assistant. "
        "Use ONLY the provided context. If the fact is not in the context, say you cannot find it.\n"
        "FORMATTING:\n"
        "- Use clear headings and lists.\n"
        "- Use Markdown tables if requested.\n"
        "- Use LaTeX ($$ ... $$) for all math.\n"
        "- ALWAYS include inline citations like (Filename, Page X).\n"
    )
    
    data = {
        "model": "mistral-tiny",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nUSER INSTRUCTION: {query}"}
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Genie Error: {str(e)}"

def extract_number(text):
    """Dynamic detection of question counts (e.g., '10 MCQs' -> 10)"""
    nums = re.findall(r'\d+', text)
    if nums:
        return int(nums[0])
    return 5 # fallback

def main():
    embedding_model = load_models()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "indexed" not in st.session_state:
        st.session_state.indexed = False
    if "stage" not in st.session_state:
        st.session_state.stage = "upload"

    # --- Header ---
    if st.session_state.stage != "chat":
        st.markdown('<div class="genie-header">genie</div>', unsafe_allow_html=True)
        st.markdown('<div class="genie-sub">Simplistic, premium document intelligence.</div>', unsafe_allow_html=True)

    # --- Step 1: Upload ---
    if st.session_state.stage == "upload":
        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.markdown("### Add Knowledge")
        uploaded_files = st.file_uploader("", accept_multiple_files=True, label_visibility="collapsed")
        if uploaded_files:
            st.session_state.uploader = uploaded_files
            if st.button("Continue to Indexing"):
                st.session_state.stage = "indexing"
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Step 2: Index ---
    elif st.session_state.stage == "indexing":
        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.markdown("### Analyze Patterns")
        st.write(f"Knowledge Source: {len(st.session_state.uploader)} documents successfully staged.")
        
        if st.button("Build Assistant Brain"):
            with st.spinner("Processing..."):
                try:
                    chunks = process_uploaded_files(st.session_state.uploader)
                    st.session_state.chunks = chunks
                    chunk_texts = [c.text for c in chunks]
                    embeddings = embedding_model.encode(chunk_texts)
                    st.session_state.embeddings = embeddings
                    index = faiss.IndexFlatL2(embeddings.shape[1])
                    index.add(np.array(embeddings).astype('float32'))
                    st.session_state.vector_store = index
                    
                    # Initialize BM25 index (Hackathon-3 Step 4)
                    from bm25_index import BM25Index
                    # Convert ChunkMeta to (text, filename) tuples for BM25
                    bm25_chunks = [(c.text, c.filename) for c in chunks]
                    st.session_state.bm25_index = BM25Index(bm25_chunks)
                    
                    st.session_state.indexed = True
                    st.session_state.stage = "chat"
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        if st.button("← Modify Selection"):
            st.session_state.stage = "upload"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Step 3: Chat ---
    elif st.session_state.stage == "chat":
        # Compact top header with Knowledge Bar
        cols = st.columns([0.8, 0.2])
        with cols[0]:
            st.markdown('**genie** · active')
            # Knowledge Bar: Always visible pills
            files = [f.name for f in st.session_state.uploader]
            kb_html = '<div class="knowledge-bar">' + "".join([f'<div class="knowledge-pill">📄 {name}</div>' for name in files]) + '</div>'
            st.markdown(kb_html, unsafe_allow_html=True)
        with cols[1]:
            if st.button("Reset", use_container_width=True):
                st.session_state.clear()
                st.rerun()

        st.divider()

        # Chat view
        for idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=True)
                
                # Add download buttons for assistant messages (only if it's a real content response)
                if msg["role"] == "assistant" and msg["content"] and "I'm sorry, I cannot verify" not in msg["content"] and "Error" not in msg["content"]:
                    dl_cols = st.columns([0.15, 0.15, 0.7])
                    with dl_cols[0]:
                        st.download_button(
                            label="⬇️ .md",
                            data=msg["content"],
                            file_name=f"genie_response_{idx}.md",
                            mime="text/markdown",
                            key=f"dl_md_{idx}"
                        )
                    with dl_cols[1]:
                        # Generate simple PDF bytes
                        pdf_buffer = io.BytesIO()
                        c = canvas.Canvas(pdf_buffer, pagesize=letter)
                        c.setFont("Helvetica", 10)
                        
                        # Very basic word wrap for PDF
                        text = msg["content"].replace('\n\n', '\n')
                        lines = text.split('\n')
                        y = 750
                        for line in lines:
                            # basic chunking for long lines
                            words = line.split(' ')
                            current_line = ""
                            for word in words:
                                if len(current_line) + len(word) > 90:
                                    c.drawString(50, y, current_line)
                                    y -= 15
                                    current_line = word + " "
                                else:
                                    current_line += word + " "
                            if current_line:
                                c.drawString(50, y, current_line)
                                y -= 15
                                
                            if y < 50:
                                c.showPage()
                                c.setFont("Helvetica", 10)
                                y = 750
                        c.save()
                        pdf_bytes = pdf_buffer.getvalue()
                        
                        st.download_button(
                            label="⬇️ .pdf",
                            data=pdf_bytes,
                            file_name=f"genie_response_{idx}.pdf",
                            mime="application/pdf",
                            key=f"dl_pdf_{idx}"
                        )

        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            user_msg = st.session_state.messages[-1]["content"]
            with st.chat_message("assistant"):
                with st.spinner(""):
                    intent = detect_mode(user_msg)
                    
                    # Step 5: Hybrid Search replacement
                    from hybrid_retriever import hybrid_search
                    indices = hybrid_search(
                        query=user_msg,
                        vector_store=st.session_state.vector_store,
                        encoder_model=embedding_model,
                        bm25_index=st.session_state.bm25_index,
                        chunks=st.session_state.chunks,
                        top_k=15 if intent == "quiz" else 10
                    )
                    
                    # Map indices back to ChunkMeta objects
                    candidates = [st.session_state.chunks[i] for i in indices]
                    
                    from reranker import Reranker
                    rer = Reranker()
                    scored = rer.rerank(user_msg, candidates)
                    top = [c for c, s in scored]
                    
                    if intent == "quiz":
                        try:
                            num_q = extract_number(user_msg)
                            
                            from structured_extractor import extract_atomic_facts
                            facts = []
                            for chunk in top:
                                facts.extend(extract_atomic_facts(chunk))
                            
                            unique = {}
                            for f in facts:
                                key = f["text"]
                                if key not in unique:
                                    unique[key] = f
                            facts = list(unique.values())
                            
                            from quiz_generator import generate_mcqs_from_facts
                            mcqs = generate_mcqs_from_facts(facts, max_questions=num_q * 3)
                            
                            from per_output_validator import validate_mcq
                            chunks_store = {c.chunk_id: c for c in st.session_state.chunks}
                            
                            final_mcqs = []
                            from trace_logger import log_trace
                            for m in mcqs:
                                ok, reason, score = validate_mcq(m, chunks_store)
                                if ok:
                                    final_mcqs.append(m)
                                else:
                                    log_trace(
                                        query="quiz_generation_distractor",
                                        answer=json.dumps(m),
                                        grounding_score=score,
                                        decision="BLOCK_MCQ",
                                        sources=[m.get("source", {}).get("filename", "unknown")]
                                    )
                                    
                            final_mcqs = final_mcqs[:num_q]
                            
                            if len(final_mcqs) < num_q:
                                ans = f"I could only generate {len(final_mcqs)} verified questions from your documents. For broader question types, allow synthesis mode (may draw on multiple sections). Or upload additional documents.\n\n"
                            else:
                                ans = f"### 🧞 Quiz Generated ({len(final_mcqs)} questions)\n\n"
                                
                            for i, q in enumerate(final_mcqs):
                                opts = q['options']
                                ans += f"**{i+1}. {q['question']}**\n- " + "\n- ".join(opts) + f"\n*Correct Answer: {q['answer']}*\n\n"
                        except Exception as e:
                            import traceback
                            st.error(f"Genie System Error (Quiz Gen): {str(e)}")
                            st.code(traceback.format_exc())
                            ans = f"Error during quiz generation: {str(e)}"
                    else:
                        top = top[:10]
                        ctx = "\n".join([f"Source ({c.filename}, Page {c.page}): {c.text}" for c in top])
                        raw_ans = call_mistral_api(user_msg, ctx)
                        gs = compute_grounding_score(raw_ans, [c.text for c in top])
                        decision, _ = run_gatekeeper(raw_ans, [c.text for c in top], gs, mode=intent)
                        
                        if decision == "BLOCK":
                            ans = "I'm sorry, I cannot verify that information in your documents."
                        else:
                            ans = raw_ans
                    
                    st.session_state.messages.append({"role": "assistant", "content": ans})
                    st.rerun()

        if prompt := st.chat_input("Ask about your documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()

if __name__ == "__main__":
    main()
