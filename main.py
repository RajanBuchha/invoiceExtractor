import os
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

# Load saved data (FAST)
chunks = np.load("chunks.npy", allow_pickle=True)
index = faiss.read_index("faiss.index")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def retrieve_chunks(query, k=1):
    query_embedding = embedder.encode([query])
    _, indices = index.search(query_embedding, k)

    results = [chunks[i] for i in indices[0]]

    # FORCE include first chunk for seller/company questions
    if any(word in query.lower() for word in ["seller", "company", "from"]):
        if chunks[0] not in results:
            results.insert(0, chunks[0])

    return results


print("Invoice Q&A ready ⚡ (type 'exit' to quit)")

while True:
    query = input("\nAsk a question: ")
    if query.lower() == "exit":
        break

    context = retrieve_chunks(query)

    prompt = f"""
Answer ONLY using the invoice text below.
Always answer in a full sentence.

If the question is about seller or company:
- extract the company name exactly as written
- ignore address lines unless asked

If the answer is not present, say "Not mentioned in invoice".

Invoice Text:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    print(response.choices[0].message.content)
