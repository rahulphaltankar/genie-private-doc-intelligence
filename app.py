import streamlit as st
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import os
from dotenv import load_dotenv
import json
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io
from citation_formatter import format_harvard_citation
from grounding import compute_grounding_score
from citation_validator import has_valid_citations
from gatekeeper import run_gatekeeper
from trace_logger import log_trace


# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(page_title="Genie — Private Document Intelligence Assistant", layout="wide")

def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_docx_text(docx_file):
    doc = Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])

def get_txt_text(txt_file):
    return str(txt_file.read(), "utf-8")

def get_text_chunks(text):
    # Simple chunking by character count with overlap
    chunk_size = 1000
    chunk_overlap = 200
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks


def call_mistral_api(prompt, context):
    """Strict factual grounding — for specific questions. Refuses to answer from prior knowledge."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return "Error: MISTRAL_API_KEY not found in environment variables."

    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    full_prompt = f"""You are a strict document question-answering assistant.
You are given context extracted from uploaded documents. Your job is to answer the user's question ONLY using information explicitly present in the context below.

STRICT RULES:
- Do NOT use any prior knowledge or information outside the provided context.
- If the context does not contain enough information to answer the question, you MUST respond with exactly: "ANSWER_NOT_IN_DOCUMENTS"
- Do NOT speculate, infer, or provide a general answer.
- Do NOT acknowledge that you could answer from general knowledge.
- You MUST cite the source filename inline in your answer using this format: (filename, n.d.)
  Example: The interest rate is 5.25% (policy.pdf, n.d.)
- If you cannot cite a source, respond with: "ANSWER_NOT_IN_DOCUMENTS"

Context:
---------------------
{context}
---------------------

Question: {prompt}
Answer (cite sources inline as (filename, n.d.), or ANSWER_NOT_IN_DOCUMENTS):"""

    data = {
        "model": "mistral-tiny",
        "messages": [{"role": "user", "content": full_prompt}]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error calling Mistral API: {str(e)}"

def call_mistral_comprehension(prompt, context):
    """Permissive comprehension mode — for summarisation, explanation, and synthesis tasks."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return "Error: MISTRAL_API_KEY not found in environment variables."

    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    full_prompt = f"""You are an intelligent document assistant. You have been given content extracted from one or more uploaded documents.
Your task is to read the content carefully and respond to the user's request using the document content as your primary source.

You may synthesise, infer, and summarise from the content. Be concise, clear, and structured.

CITATION RULE:
- You MUST reference the source filenames inline in your response using this format: (filename, n.d.)
  Example: The document discusses transformer architecture (attention_is_all_you_need.pdf, n.d.)
- Only reference sources that appear in the document content below.
- Do not fabricate source names.

Document content:
---------------------
{context}
---------------------

User request: {prompt}
Response (cite sources inline as (filename, n.d.)):"""

    data = {
        "model": "mistral-tiny",
        "messages": [{"role": "user", "content": full_prompt}]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error calling Mistral API: {str(e)}"


def create_pdf(text):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 40
    c.drawString(30, y, "Genie Answer Export")
    y -= 40
    
    # Simple text wrapping
    text_object = c.beginText(30, y)
    text_object.setFont("Helvetica", 12)
    
    # Split by newlines first
    lines = text.split('\n')
    for line in lines:
        # Simple wrapping by length (could be improved)
        while len(line) > 90:
            text_object.textLine(line[:90])
            line = line[90:]
        text_object.textLine(line)
        
    c.drawText(text_object)
    c.save()
    buffer.seek(0)
    return buffer

def main():
    st.title("Genie — Private Document Intelligence Assistant")

    # Initialize session state for conversation
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "chunks" not in st.session_state:
        st.session_state.chunks = []

    with st.sidebar:
        st.subheader("Your Documents")
        uploaded_files = st.file_uploader(
            "Upload up to 3 files (PDF, DOCX, TXT)", 
            type=["pdf", "docx", "txt"], 
            accept_multiple_files=True
        )

        if st.button("Process Documents"):
            if not uploaded_files:
                st.warning("Please upload files first.")
            elif len(uploaded_files) > 3:
                st.error("Please upload maximum 3 files.")
            else:
                with st.spinner("Processing documents..."):
                    all_text = ""
                    chunks = []
                    
                    for file in uploaded_files:
                        if file.type == "application/pdf":
                            text = get_pdf_text(file)
                        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                            text = get_docx_text(file)
                        else: # txt
                            text = get_txt_text(file)
                        
                        file_chunks = get_text_chunks(text)
                        # Store source with chunk -> (chunk_text, source_filename)
                        chunks.extend([(c, file.name) for c in file_chunks])
                    
                    st.session_state.chunks = chunks
                    
                    # Create Embeddings
                    model = SentenceTransformer('all-MiniLM-L6-v2')
                    # We just need the text for embedding
                    chunk_texts = [c[0] for c in chunks]
                    embeddings = model.encode(chunk_texts)
                    
                    # Create FAISS index
                    dimension = embeddings.shape[1]
                    index = faiss.IndexFlatL2(dimension)
                    index.add(np.array(embeddings).astype('float32'))
                    
                    st.session_state.vector_store = index
                    # Storing model in session state might use memory, but needed for query encoding
                    # Alternatively we can reload model. For now reload model to save state size or keep it?
                    # Streamlit reloads script on interaction. Model load is expensive.
                    # Better to cache the model resource.
                    
                    st.success("Documents processed successfully!")

    # Main Chat Interface
    question = st.text_input("Ask a question about your documents:")

    if question:
        if st.session_state.vector_store is None:
            st.error("Please process documents first.")
        else:
            with st.spinner("Thinking..."):
                from citation_formatter import format_harvard_citation

                model = SentenceTransformer('all-MiniLM-L6-v2')
                question_embedding = model.encode([question])

                # Always do FAISS search first
                D, I = st.session_state.vector_store.search(
                    np.array(question_embedding).astype('float32'), k=5
                )

                FACTUAL_THRESHOLD = 1.2    # Below this -> precise factual match found
                SCOPE_THRESHOLD = 2.0      # Above this -> truly out of scope (even for synthesis)
                min_distance = float(D[0][0])

                sources: set = set()
                context_text: str = ""
                answer_with_citations: str = ""

                def show_answer(ans: str, srcs: set, decision: str,
                               grounding: float = -1.0, confidence: float = -1.0) -> None:
                    """
                    Render the answer with full provenance transparency.
                    decision: 'PASS' | 'SYNTHESIS'
                    """
                    is_grounded = (decision == "PASS")

                    if is_grounded:
                        harvard = " ".join([format_harvard_citation(s) for s in srcs])
                        combined = f"{ans}\n\nSources: {harvard}"
                        badge = "✅ GROUNDED"
                        st.success(f"{badge} — Answer verified against uploaded documents.")
                    else:
                        combined = ans
                        badge = "⚠️ SYNTHESIS"
                        st.warning(f"{badge} — Answer synthesised from document content. "
                                   "No direct source citation available.")

                    st.write("### Answer")
                    st.write(combined)
                    st.write("---")

                    # Always show provenance state
                    meta_cols = st.columns(3)
                    with meta_cols[0]:
                        st.write(f"**Decision:** {badge}")
                    with meta_cols[1]:
                        if is_grounded and confidence >= 0:
                            st.write(f"**Confidence:** {confidence:.1f}%")
                        else:
                            st.write("**Confidence:** LOW")
                    with meta_cols[2]:
                        if is_grounded:
                            st.write(f"**Grounding:** {grounding:.3f}")
                            st.write(f"**Sources:** {', '.join(format_harvard_citation(s) for s in srcs)}")
                        else:
                            st.write(f"**Grounding:** {grounding:.3f}")
                            st.write("**Sources:** No direct document citation")

                    st.write("### Export Answer")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.download_button("Download TXT", data=combined,
                                           file_name="answer.txt", mime="text/plain")
                    with c2:
                        jd = json.dumps({
                            "question": question,
                            "answer": ans,
                            "decision": decision,
                            "grounding_score": round(grounding, 3),
                            "sources": list(srcs) if is_grounded else []
                        }, indent=2)
                        st.download_button("Download JSON", data=jd,
                                           file_name="answer.json", mime="application/json")
                    with c3:
                        st.download_button("Download PDF", data=create_pdf(combined),
                                           file_name="answer.pdf", mime="application/pdf")

                def show_sme_escalation(reason: str, grounding: float = -1.0) -> None:
                    """Render the SME escalation message with provenance state."""
                    st.error("🚫 BLOCK — Answer cannot be verified against uploaded documents.")
                    st.warning(f"⚠️ {reason}")
                    if grounding >= 0:
                        st.caption(f"Grounding score: {grounding:.3f}")
                    st.info(
                        "**Next step:** Please consult the appropriate **Subject Matter Expert (SME)** "
                        "or upload additional authoritative documents related to this topic."
                    )

                # ── PATH A: Good factual match — strict grounding ──────────────
                if min_distance <= FACTUAL_THRESHOLD:
                    relevant_chunks_text = []
                    for i_chunk, idx in enumerate(I[0]):
                        if idx < len(st.session_state.chunks) and float(D[0][i_chunk]) <= FACTUAL_THRESHOLD * 1.5:
                            chunk_text, source = st.session_state.chunks[idx]
                            sources.add(str(source))
                            context_text = context_text + f"\nSource ({source}): {chunk_text}\n"
                            relevant_chunks_text.append(str(chunk_text))

                    answer = call_mistral_api(question, context_text)
                    grounding_score = compute_grounding_score(answer, relevant_chunks_text)
                    confidence = round(grounding_score * 100, 1)  # cosine similarity as %
                    decision, reason = run_gatekeeper(answer, relevant_chunks_text, grounding_score, mode="factual")

                    if decision == "PASS":
                        log_trace(question, answer, grounding_score, decision, sources)
                        show_answer(answer, sources, decision="PASS",
                                    grounding=grounding_score, confidence=confidence)
                    else:
                        # Factual path blocked — try comprehension fallback before giving up
                        all_chunks = st.session_state.chunks
                        step = max(1, len(all_chunks) // 15)
                        sampled = all_chunks[::step][:15]
                        ctx2: str = ""
                        srcs2: set = set()
                        chunks2_text = []
                        for ct, src in sampled:
                            srcs2.add(str(src))
                            ctx2 = ctx2 + f"\nSource ({src}): {ct}\n"
                            chunks2_text.append(str(ct))
                        answer2 = call_mistral_comprehension(question, ctx2)
                        gs2 = compute_grounding_score(answer2, chunks2_text)
                        dec2, reason2 = run_gatekeeper(answer2, chunks2_text, gs2, mode="synthesis")
                        log_trace(question, answer2, gs2, dec2, srcs2)
                        if dec2 in ["PASS", "SYNTHESIS"]:
                            show_answer(answer2, srcs2, decision=dec2, grounding=gs2)
                        else:
                            show_sme_escalation(reason2, grounding=gs2)

                # ── PATH B: No close match — auto comprehension fallback ────────
                elif min_distance <= SCOPE_THRESHOLD:
                    all_chunks = st.session_state.chunks
                    step = max(1, len(all_chunks) // 15)
                    sampled = all_chunks[::step][:15]
                    srcs3: set = set()
                    ctx3: str = ""
                    chunks3_text = []
                    for ct, src in sampled:
                        srcs3.add(str(src))
                        ctx3 = ctx3 + f"\nSource ({src}): {ct}\n"
                        chunks3_text.append(str(ct))

                    answer = call_mistral_comprehension(question, ctx3)
                    grounding_score = compute_grounding_score(answer, chunks3_text)
                    decision, reason = run_gatekeeper(answer, chunks3_text, grounding_score, mode="synthesis")

                    if decision in ["PASS", "SYNTHESIS"]:
                        log_trace(question, answer, grounding_score, decision, srcs3)
                        show_answer(answer, srcs3, decision=decision, grounding=grounding_score)
                    else:
                        log_trace(question, answer, grounding_score, decision, srcs3)
                        show_sme_escalation(reason, grounding=grounding_score)

                # ── PATH C: Truly out of scope (very high distance) ───────────
                else:
                    log_trace(question, "", -1.0, "BLOCK", set())
                    show_sme_escalation(
                        "This question doesn't appear to be related to your uploaded documents."
                    )



if __name__ == "__main__":
    main()

