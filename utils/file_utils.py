from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
import markdown

def process_uploaded_file(uploaded_file):
    """Process the uploaded file and extract text."""
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        text = "\n".join(page.extract_text() for page in reader.pages)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(uploaded_file)
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        df = pd.read_excel(uploaded_file)
        text = df.to_string()
    elif uploaded_file.type == "text/markdown":
        text = markdown.markdown(uploaded_file.read().decode())
    elif uploaded_file.type.startswith("image/"):
        # Use OCR to extract text from images (requires libraries like pytesseract)
        text = "Extracted text from image (OCR not implemented in this example)"
    else:
        text = uploaded_file.read().decode()
    return text