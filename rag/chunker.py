from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document 

def chunk_text(raw_text, chunk_size = 500 , chunk_overlap =100 ):
    """Splits raw text into overlapping chunks using LangChain's RecursiveCharacterTextSplitter.
    Returns a list of LangChain Document objects."""
    
    doc = Document(page_content=raw_text)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_documents([doc])
    return chunks