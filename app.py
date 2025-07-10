import streamlit as st
from loaders.pdf_loader import extract_text_from_pdf
from rag.chunker import chunk_text
from rag.embed_store import embed_and_store
from rag.query import query_documents, get_all_sources
from rag.utils import clear_database
import requests

st.set_page_config(layout="wide")

st.title("📚 Research Paper Query System")

# Initialize session state
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

# --- SIDEBAR --- #
st.sidebar.header("Controls")

if st.sidebar.button("❌ Clear all vectors"):
    with st.spinner("Clearing database..."):
        clear_database()
        st.session_state.processed_files = []
    st.sidebar.success("All data has been cleared.")

# --- File Upload and Processing --- #
st.sidebar.subheader("1. Upload Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files", type="pdf", accept_multiple_files=True
)

if uploaded_files:
    st.sidebar.subheader("2. Select Files to Process")
    files_to_process = {
        file.name: file for file in uploaded_files 
        if file.name not in st.session_state.processed_files
    }
    
    selected_for_processing = []
    for name, file in files_to_process.items():
        if st.sidebar.checkbox(f"Process `{name}`", key=f"process_{name}"):
            selected_for_processing.append(file)

    if st.sidebar.button("⚙️ Process Selected Files"):
        if not selected_for_processing:
            st.sidebar.warning("No new files selected to process.")
        else:
            with st.spinner("Processing selected files..."):
                for file in selected_for_processing:
                    st.sidebar.write(f"> Processing {file.name}...")
                    pdf_text = extract_text_from_pdf(file)
                    chunks = chunk_text(pdf_text)
                    embed_and_store(chunks, source_name=file.name)
                    st.session_state.processed_files.append(file.name)
            st.sidebar.success("Selected files have been processed.")
            st.rerun() # Force a rerun to update the source list

# --- Source Selection for Querying --- #
st.sidebar.subheader("3. Select Sources for Query")
try:
    all_sources = get_all_sources()
except Exception as e:
    all_sources = []
    st.sidebar.warning(f"Could not get sources: {e}")

if all_sources:
    selected_sources = st.sidebar.multiselect(
        "Select files to include in context:", 
        options=all_sources, 
        default=all_sources
    )
else:
    st.sidebar.warning("No sources found. Upload and process files first.")
    selected_sources = []

# --- MAIN CONTENT --- #
st.header("Query Interface")

query = st.text_input("Enter your query:", key="query_input")

if query:
    if not selected_sources:
        st.error("Please select at least one source file from the sidebar.")
    else:
        with st.spinner("Searching for relevant documents..."):
            results = query_documents(query, sources=selected_sources)

        if results and results.get("documents") and results.get("distances"):
            docs = results["documents"][0]
            scores = results["distances"][0]

            if docs and scores:
                st.subheader("Top Results")
                for i, (doc, score) in enumerate(zip(docs, scores)):
                    with st.expander(f"Result {i+1} (Score: {score:.4f})"):
                        st.write(doc)

                # Generate with local LLM
                context = "\n\n".join(docs)
                prompt = f"Use the following research context to answer:\n\n{context}\n\nQuestion: {query}"

                with st.spinner("Generating answer with LLM..."):
                    try:
                        response = requests.post("http://localhost:11434/api/generate", json={
                            "model": "mistral:7b",
                            "prompt": prompt,
                            "stream": False
                        })
                        response.raise_for_status()
                        answer = response.json().get("response", "No response from model.")
                        st.subheader("🧠 Answer")
                        st.write(answer)
                    except requests.exceptions.RequestException as e:
                        st.error(f"Failed to connect to LLM: {e}")
            else:
                st.warning("No relevant documents found for the selected sources.")
        else:
            st.warning("No results returned. Try re-uploading or re-querying.")