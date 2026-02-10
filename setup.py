import os
import numpy as np
import faiss
import pytesseract
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer

# Paths
PDF_PATH = "invoice.pdf"
POPPLER_PATH = r"D:\poppler\poppler-25.12.0\Library\bin"
TESSERACT_PATH = r"D:\tesseract\tesseract.exe"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# 1️⃣ OCR (slow – once)
print("Running OCR...")
pages = convert_from_path(PDF_PATH, dpi=300, poppler_path=POPPLER_PATH)
invoice_text = ""
for p in pages:
    invoice_text += pytesseract.image_to_string(p)

with open("invoice_text.txt", "w", encoding="utf-8") as f:
    f.write(invoice_text)

# 2️⃣ Chunking
def chunk_text(text, size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+size])
        start += size - overlap
    return chunks

chunks = chunk_text(invoice_text)
np.save("chunks.npy", chunks)

# 3️⃣ Embeddings (slow – once)
print("Creating embeddings...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(chunks)

# 4️⃣ FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))
faiss.write_index(index, "faiss.index")

print("SETUP COMPLETE ✅")
