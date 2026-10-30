from turtle import mode

from fastapi import HTTPException
from pypdf import PdfReader
from collections import Counter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np

file_path = r"C:\Projects\LLM_Rag\ecommerce_policy.pdf"

MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120
TOP_K = 3
THRESHOLD = 0.3

# ---------------- Step 1: load the PDF ----------------
def load_pages(file_path):
    """Read a PDF and return a list of (page_number, text)."""
    
    try:
        reader = PdfReader(file_path)
        if not reader.pages:
            raise ValueError(f"No pages found in PDF: {file_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Not a valid PDF: {file_path}") from e

    
    pages = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""      # empty/None pages become ""
        pages.append((page_no, text))
    print("Pages loaded:", len(pages))
    return pages

 
# ---------------- Step 2: chunking ----------------
def chunk_pages(pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split each page into chunks. Returns a list of {"text", "page"}."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = []
    for page_no, text in pages:
        for piece in splitter.split_text(text):
            chunks.append({"text": piece, "page": page_no})
 
    print("Total chunks:", len(chunks))
    print("Chunks per page:", dict(sorted(Counter(c["page"] for c in chunks).items())))
    for c in chunks[:5]:
        print(f"--- page {c['page']} ({len(c['text'])} chars) ---")
        print(c["text"])
    return chunks

# ---------------- Step 3: embeddings ----------------
def embed_texts(model, texts, verbose=True):
    """Turn a list of strings into a (len(texts), 384) array of unit-length vectors."""
    vectors = model.encode(texts, normalize_embeddings=True)
    if verbose:
        print("Vectors shape:", vectors.shape)
    return vectors

# ---------------- Step 4: retrieval ----------------
def retrieve(model, vectors, chunks, question, k=TOP_K):
    """Return the k most similar chunks, best first."""
    q_vec = embed_texts(model, [question], verbose=False)[0]
    scores = vectors @ q_vec                      # one similarity score per chunk
    top = np.argsort(scores)[::-1][:k]            # positions of the best k
 
    hits = [
        {"score": float(scores[i]), "page": chunks[i]["page"], "text": chunks[i]["text"]}
        for i in top
    ]
 
    print("\nQuestion:", question)
    for h in hits:
        print(f"score={h['score']:.3f}  page={h['page']}")
        print(h["text"][:150].replace("\n", " "), "\n")
    if hits[0]["score"] < THRESHOLD:
        print("No relevant chunk found. I don't know.")
    return hits
if __name__ == "__main__":
    model = SentenceTransformer(MODEL_NAME)
    pages = load_pages(file_path)
    chunks = chunk_pages(pages)
    vector = embed_texts(model, [c["text"] for c in chunks])

    # full_text = " "
    # for page in reader.pages:
    #     text = page.extract_text()
    #     if text:
    #         full_text += text+ "\n"
    # print(full_text)    
      
    # full_text = reader.pages[1].extract_text()  
    

#     splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
#     all_chunks = []
#     for page_no, page in enumerate(reader.pages, start=1):
#         text = page.extract_text()
#         for chunk in splitter.split_text(text):
#             all_chunks.append({"text": chunk, "page": page_no})
#     print("Total chunks:", len(all_chunks))
#     for c in all_chunks[:5]:
#         print(f"--- page {c['page']} ({len(c['text'])} chars) ---")
#         print(c["text"])
#         #counter to check the number of chunks per page
#     print(Counter(c["page"] for c in all_chunks))   
    
#     texts = [c["text"] for c in all_chunks]
#     vectors = model.encode(texts, normalize_embeddings=True)
#     print(vectors,"check vector output")
#     print("Vectors shape:", vectors.shape)    

#     # question = "How long do I have to return a laptop?"
#     question = input("Enter your question: ")
#     q_vec = model.encode([question], normalize_embeddings=True)[0]

#     scores = vectors @ q_vec
#     # print("Scores:", scores)
#     k=5
#     top = np.argsort(scores)[::-1][:k]
#     print("Top indices:", top)
#     print("\nQuestion:", question)
#     for idx in top:
#         print(f"score={scores[idx]:.4f}  page={all_chunks[idx]['page']}")
#         print(all_chunks[idx]["text"][:150].replace("\n", " "), "\n")

#     THRESHOLD = 0.3
#     if scores[top[0]] < THRESHOLD:
#         print("No relevant chunk found. I don't know.")    
     
# except Exception as e:
#     print(f"The file {file_path} caused an error: {e}")


