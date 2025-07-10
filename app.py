import streamlit as st
from loaders.pdf_loader import extract_text_from_pdf
from rag.chunker import chunk_text
from rag.embed_store import embed_and_store
from rag.query import query_documents

st.title("📄 Research PDF Analyzer")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# ========== STEP 1: Upload and process ==========
if uploaded_file and "chunks_stored" not in st.session_state:
    with st.spinner("Extracting text..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
        st.session_state["pdf_text"] = pdf_text

    st.subheader("Extracted Text Preview")
    st.text_area("Raw Text from PDF", pdf_text[:1000], height=300)

    with st.spinner("Chunking text..."):
        chunks = chunk_text(pdf_text)
        st.session_state["chunks"] = chunks
    st.success(f"✅ Extracted {len(chunks)} chunks.")

    with st.spinner("Embedding and storing chunks..."):
        embed_and_store(chunks)
        st.session_state["chunks_stored"] = True
    st.success("✅ Chunks embedded and stored successfully.")

# ========== STEP 2: Query ==========
query = st.text_input("Enter your query:")

if query:
    if "chunks_stored" not in st.session_state:
        st.error("Please upload and process a PDF first.")
    else:
        with st.spinner("Searching for relevant documents..."):
            results = query_documents(query)

        if results and results.get("documents") and results.get("distances"):
            docs = results.get("documents", [[]])[0]
            scores = results.get("distances", [[]])[0]

            if docs and scores:
                st.subheader("Top Results")
                for i, (doc, score) in enumerate(zip(docs, scores)):
                    with st.expander(f"Result {i+1} (Score: {score:.4f})"):
                        st.write(doc)
            else:
                st.warning("No relevant documents found.")
        else:
            st.warning("No results returned. Try re-uploading or re-querying.")
