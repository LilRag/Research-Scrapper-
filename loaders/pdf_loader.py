
import fitz 


def extract_text_from_pdf(uploaded_file):
    """
    Extract text from a PDF uploaded via Streamlit's file_uploader.
    Returns the full text as a single string.
    """
    with fitz.open(stream = uploaded_file.read() ,filetype = "pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
        return text

