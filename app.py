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
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return "Error: MISTRAL_API_KEY not found in environment variables."

    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    full_prompt = f"""Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {prompt}
Answer:"""

    data = {
        "model": "mistral-tiny",
        "messages": [
            {"role": "user", "content": full_prompt}
        ]
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
                # Retrieve relevant chunks
                model = SentenceTransformer('all-MiniLM-L6-v2')
                question_embedding = model.encode([question])
                
                # Search
                D, I = st.session_state.vector_store.search(np.array(question_embedding).astype('float32'), k=3)
                
                relevant_chunks = []
                sources = set()
                # D represents L2 distance. Lower is better.
                # Approximate confidence: 1 / (1 + distance) or similar. 
                # Let's take the closest match distance for "Confidence".
                min_distance = D[0][0]
                confidence_score = max(0, 100 - (min_distance * 10)) # Heuristic
                
                context_text = ""
                for idx in I[0]:
                    if idx < len(st.session_state.chunks):
                        chunk_text, source = st.session_state.chunks[idx]
                        relevant_chunks.append(chunk_text)
                        sources.add(source)
                        context_text += f"\nSource ({source}): {chunk_text}\n"

                # Call Mistral
                answer = call_mistral_api(question, context_text)
                
                # Display Results
                st.write("### Answer")
                st.write(answer)
                
                st.write("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Confidence:** {confidence_score:.1f}%")
                with col2:
                    st.write(f"**Sources:** {', '.join(sources)}")
                
                # Export options
                st.write("### Export Answer")
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    st.download_button(
                        "Download TXT", 
                        data=answer, 
                        file_name="answer.txt", 
                        mime="text/plain"
                    )
                
                with c2:
                    json_data = json.dumps({
                        "question": question,
                        "answer": answer,
                        "sources": list(sources),
                        "confidence": confidence_score
                    }, indent=2)
                    st.download_button(
                        "Download JSON", 
                        data=json_data, 
                        file_name="answer.json", 
                        mime="application/json"
                    )
                    
                with c3:
                    pdf_data = create_pdf(answer)
                    st.download_button(
                        "Download PDF",
                        data=pdf_data,
                        file_name="answer.pdf",
                        mime="application/pdf"
                    )

if __name__ == "__main__":
    main()
