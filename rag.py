"""
Mini RAG project - Stage 1: "Chat with your PDFs"

Pipeline:
  INGEST: PDF -> pages -> chunks -> embeddings -> ChromaDB
  QUERY : question -> embedding -> top-k chunks -> LLM -> answer + sources

Usage:
  python rag.py ingest ./docs
  python rag.py ask "How many days of paternity leave do I get?"
  python rag.py ask "..." --show-chunks      # debug retrieval
  python rag.py chat
"""
import argparse
import os
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------------- Config (experiment with these in Stage 2) ----------------
EMBED_MODEL = "all-MiniLM-L6-v2"                    # 384-dim, fast, free, local
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))     # characters (~250 tokens)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
TOP_K = int(os.getenv("TOP_K", 4))
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")
DB_DIR = "chroma_db"
COLLECTION = "docs"

_embedder = None


def get_embedder():
    """Load the embedding model once. SAME model must be used for index + query."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_client():
    return chromadb.PersistentClient(path=DB_DIR)


# ---------------------------- INGESTION ----------------------------
def load_pdf(path: Path):
    """Yield (page_number, text) for each page."""
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            yield i, text


def ingest(folder: str):
    pdfs = sorted(Path(folder).glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {folder}")
        return

    # Recursive splitter: tries paragraph -> line -> sentence -> word
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    ids, docs, metas = [], [], []
    for pdf in pdfs:
        for page_no, text in load_pdf(pdf):
            for j, chunk in enumerate(splitter.split_text(text)):
                ids.append(f"{pdf.stem}-p{page_no}-c{j}")
                docs.append(chunk)
                metas.append({"source": pdf.name, "page": page_no})
        print(f"Read {pdf.name}")

    print(f"Created {len(docs)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # Embed all chunks (normalized, so cosine == dot product)
    embeddings = get_embedder().encode(
        docs, normalize_embeddings=True, batch_size=64, show_progress_bar=True
    )

    # Fresh collection each run so re-ingesting with new settings is clean
    client = get_client()
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    B = 500
    for s in range(0, len(docs), B):
        col.add(
            ids=ids[s:s + B],
            documents=docs[s:s + B],
            embeddings=embeddings[s:s + B].tolist(),
            metadatas=metas[s:s + B],
        )
    print(f"Stored {col.count()} chunks in ChromaDB ('{DB_DIR}')")


# ---------------------------- RETRIEVAL ----------------------------
def retrieve(question: str, k: int = TOP_K):
    col = get_client().get_collection(COLLECTION)
    q_emb = get_embedder().encode([question], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=q_emb, n_results=k)
    hits = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        # Chroma returns cosine DISTANCE; similarity = 1 - distance
        hits.append({"text": doc, "meta": meta, "score": 1 - dist})
    return hits


# ---------------------------- GENERATION ----------------------------
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided "
    "context. If the answer is not in the context, say \"I don't know based on "
    "the provided documents.\" Do not use outside knowledge. Cite sources like "
    "[file.pdf, p.3] after the facts they support."
)


def build_prompt(question: str, hits):
    context = "\n\n".join(
        f"[{h['meta']['source']}, p.{h['meta']['page']}]\n{h['text']}" for h in hits
    )
    return f"Context:\n{context}\n\nQuestion: {question}"


def generate(question: str, hits) -> str:
    import anthropic  # needs ANTHROPIC_API_KEY in your environment

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=LLM_MODEL,
        max_tokens=800,
        temperature=0,  # low temperature = factual, consistent answers
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(question, hits)}],
    )
    return msg.content[0].text


def ask(question: str, show_chunks: bool = False):
    hits = retrieve(question)
    if show_chunks:
        print("\n--- Retrieved chunks ---")
        for i, h in enumerate(hits, 1):
            print(f"{i}. score={h['score']:.3f}  {h['meta']}")
            print(f"   {h['text'][:200].replace(chr(10), ' ')}...\n")
    print("\n--- Answer ---")
    print(generate(question, hits))
    print("\n--- Sources ---")
    for h in hits:
        print(f"  {h['meta']['source']} p.{h['meta']['page']} (score {h['score']:.2f})")


# ---------------------------- CLI ----------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest")
    s.add_argument("folder")

    s = sub.add_parser("ask")
    s.add_argument("question")
    s.add_argument("--show-chunks", action="store_true")

    sub.add_parser("chat")

    a = p.parse_args()
    if a.cmd == "ingest":
        ingest(a.folder)
    elif a.cmd == "ask":
        ask(a.question, a.show_chunks)
    elif a.cmd == "chat":
        print("Type a question (or 'quit').")
        while (q := input("\n> ").strip()).lower() not in {"quit", "exit"}:
            if q:
                ask(q, show_chunks=True)