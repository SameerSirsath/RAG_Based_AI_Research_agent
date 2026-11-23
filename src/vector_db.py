"""
vector_db.py
Persistent ChromaDB Vector Store, Hybrid Retrieval (Dense + Sparse TF-IDF),
and Dynamic Live arXiv Paper Ingestion.

Course: Natural Language Processing (NLP)
Team: Sameer Sirsath (PRN: 202401110023) & Rameshwar Sanap (PRN: 202401110021)
"""

import os
import re
import urllib.request
import logging
from typing import List, Dict, Any, Optional, Tuple

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_chroma import Chroma
import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

try:
    import arxiv
except ImportError:
    arxiv = None

from src.preprocessing import extract_text_from_pdf, ensure_sample_pdf

logger = logging.getLogger(__name__)

# Constants
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "academic_research_assistant"
ARXIV_CACHE_DIR = "data/arxiv_cache"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Global Singletons
_EMBEDDINGS = None
_VECTOR_STORE = None
_CHUNKS_CACHE: List[Document] = []
_TFIDF_VECTORIZER: Optional[TfidfVectorizer] = None
_TFIDF_MATRIX = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Returns singleton HuggingFaceEmbeddings instance."""
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _EMBEDDINGS


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Returns standard chunk splitter (800 chars, 150 overlap)."""
    return RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )


def build_corpus_documents(pdf_path: str = "data/sample_book.pdf") -> List[Document]:
    """
    Extracts text from PDF page-by-page and splits all pages into Document chunks
    with explicit [Page X] boundary tags and comprehensive metadata.
    """
    pages = extract_text_from_pdf(pdf_path)
    splitter = get_text_splitter()
    documents = []

    for page_info in pages:
        page_num = page_info["page_num"]
        source_name = page_info["source"]
        tagged_text = page_info["tagged_text"]

        splits = splitter.split_text(tagged_text)
        for idx, split_text in enumerate(splits):
            doc = Document(
                page_content=split_text.strip(),
                metadata={
                    "source": source_name,
                    "page": page_num,
                    "chunk_id": f"{source_name}_p{page_num}_c{idx+1}",
                    "char_count": len(split_text.strip())
                }
            )
            documents.append(doc)

    return documents


def get_or_create_vector_store(force_reindex: bool = False,
                               pdf_path: str = "data/sample_book.pdf") -> Chroma:
    """
    Initializes persistent ChromaDB. If collection already contains indexed chunks
    and force_reindex is False, re-uses existing vector store for instant startup.
    Otherwise, chunks ALL pages from the PDF and indexes them.
    """
    global _VECTOR_STORE, _CHUNKS_CACHE, _TFIDF_VECTORIZER, _TFIDF_MATRIX
    embeddings = get_embedding_model()
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Check existing collection
    if not force_reindex:
        try:
            existing_col = client.get_collection(COLLECTION_NAME)
            doc_count = existing_col.count()
            if doc_count > 0:
                logger.info(f"Loaded existing ChromaDB collection with {doc_count} chunks.")
                _VECTOR_STORE = Chroma(
                    client=client,
                    collection_name=COLLECTION_NAME,
                    embedding_function=embeddings
                )
                refresh_lexical_cache(_VECTOR_STORE)
                return _VECTOR_STORE
        except Exception:
            pass

    # Build fresh chunks and index
    logger.info("Initializing fresh ChromaDB index from PDF corpus...")
    docs = build_corpus_documents(pdf_path)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    _VECTOR_STORE = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        client=client
    )

    refresh_lexical_cache(_VECTOR_STORE)
    return _VECTOR_STORE


def refresh_lexical_cache(vector_store: Chroma):
    """
    Synchronizes in-memory documents and builds a TF-IDF matrix over all indexed chunks
    to support fast hybrid retrieval.
    """
    global _CHUNKS_CACHE, _TFIDF_VECTORIZER, _TFIDF_MATRIX
    try:
        col = vector_store._collection
        all_data = col.get(include=["documents", "metadatas"])
        docs = []
        raw_texts = []
        for text, meta in zip(all_data.get("documents", []), all_data.get("metadatas", [])):
            docs.append(Document(page_content=text, metadata=meta or {}))
            raw_texts.append(text)

        _CHUNKS_CACHE = docs
        if raw_texts:
            _TFIDF_VECTORIZER = TfidfVectorizer(stop_words="english", max_features=5000)
            _TFIDF_MATRIX = _TFIDF_VECTORIZER.fit_transform(raw_texts)
    except Exception as e:
        logger.warning(f"Could not build TF-IDF lexical cache: {e}")


def ingest_single_pdf(pdf_path: str, max_pages: Optional[int] = 12) -> Dict[str, Any]:
    """
    Ingests an individual reference academic PDF into ChromaDB without overwriting
    existing documents. Extracts pages, splits into chunks, indexes embeddings,
    and refreshes the hybrid lexical cache.
    Limits to core pages (default: 12) to ensure fast indexing while capturing core methodology.
    """
    global _VECTOR_STORE
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    if _VECTOR_STORE is None:
        get_or_create_vector_store()

    doc = fitz.open(pdf_path)
    source_name = os.path.basename(pdf_path)
    total_doc_pages = len(doc)
    pages_to_index = min(total_doc_pages, max_pages) if max_pages else total_doc_pages
    splitter = get_text_splitter()
    new_docs = []

    for page_idx in range(pages_to_index):
        page_num = page_idx + 1
        raw_text = doc[page_idx].get_text().strip()
        if not raw_text:
            continue
        tagged_text = f"[{source_name} | Page {page_num}]\n{raw_text}"
        splits = splitter.split_text(tagged_text)
        for c_idx, split_text in enumerate(splits):
            new_docs.append(Document(
                page_content=split_text.strip(),
                metadata={
                    "source": source_name,
                    "page": page_num,
                    "chunk_id": f"{source_name}_p{page_num}_c{c_idx+1}",
                    "char_count": len(split_text.strip())
                }
            ))

    doc.close()

    if new_docs:
        _VECTOR_STORE.add_documents(new_docs)
        refresh_lexical_cache(_VECTOR_STORE)
        logger.info(f"Ingested {len(new_docs)} chunks from {source_name} ({pages_to_index} pages) into ChromaDB.")

    return {
        "source": source_name,
        "pdf_path": pdf_path,
        "pages_processed": pages_to_index,
        "total_doc_pages": total_doc_pages,
        "chunks_indexed": len(new_docs),
        "total_chunks_in_db": len(_CHUNKS_CACHE)
    }


def ingest_references_directory(dir_path: str = "data/references") -> Dict[str, Any]:
    """
    Scans a directory for all .pdf files, extracts, chunks, and appends them
    to the persistent ChromaDB collection.
    """
    os.makedirs(dir_path, exist_ok=True)
    pdf_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith(".pdf")]

    results = []
    total_new_chunks = 0

    for pdf_path in pdf_files:
        try:
            res = ingest_single_pdf(pdf_path)
            results.append(res)
            total_new_chunks += res["chunks_indexed"]
        except Exception as e:
            logger.error(f"Failed to ingest {pdf_path}: {e}")
            results.append({"source": os.path.basename(pdf_path), "error": str(e), "chunks_indexed": 0})

    return {
        "directory": dir_path,
        "pdf_count": len(pdf_files),
        "total_new_chunks": total_new_chunks,
        "files_detail": results
    }


def get_indexed_sources_summary() -> Dict[str, Any]:
    """
    Returns a breakdown of all sources and chunk counts currently indexed in ChromaDB.
    """
    global _VECTOR_STORE, _CHUNKS_CACHE
    if _VECTOR_STORE is None:
        get_or_create_vector_store()

    sources: Dict[str, int] = {}
    for doc in _CHUNKS_CACHE:
        src = doc.metadata.get("source", "Unknown")
        sources[src] = sources.get(src, 0) + 1

    return {
        "total_chunks": len(_CHUNKS_CACHE),
        "unique_sources_count": len(sources),
        "sources": sources
    }


def hybrid_search(query: str,
                  k: int = 4,
                  dense_weight: float = 0.65,
                  sparse_weight: float = 0.35) -> List[Dict[str, Any]]:
    """
    Executes Hybrid Retrieval combining:
    1. Dense semantic search via ChromaDB cosine distance (top k*2 candidates).
    2. Sparse lexical search via TF-IDF cosine similarity.
    Blends both ranking scores to return the top k most relevant context chunks.
    """
    global _VECTOR_STORE, _CHUNKS_CACHE, _TFIDF_VECTORIZER, _TFIDF_MATRIX

    if _VECTOR_STORE is None:
        get_or_create_vector_store()

    # 1. Dense retrieval
    try:
        dense_results_raw = _VECTOR_STORE.similarity_search_with_score(query, k=k * 2)
        dense_results = []
        for doc, dist in dense_results_raw:
            # Cosine distance to similarity (0 to 1)
            sim = max(0.0, min(1.0, 1.0 - (float(dist) / 2.0)))
            dense_results.append((doc, sim))
    except Exception:
        dense_docs = _VECTOR_STORE.similarity_search(query, k=k * 2)
        dense_results = [(doc, 0.75) for doc in dense_docs]

    # Map candidate contents
    candidates: Dict[str, Dict[str, Any]] = {}
    for doc, rel_score in dense_results:
        content = doc.page_content
        candidates[content] = {
            "content": content,
            "metadata": doc.metadata,
            "dense_score": max(0.0, float(rel_score)),
            "sparse_score": 0.0
        }

    # 2. Sparse lexical TF-IDF scoring
    if _TFIDF_VECTORIZER is not None and _TFIDF_MATRIX is not None and _CHUNKS_CACHE:
        try:
            q_vec = _TFIDF_VECTORIZER.transform([query])
            sim_scores = cosine_similarity(q_vec, _TFIDF_MATRIX).flatten()
            top_sparse_indices = np.argsort(sim_scores)[::-1][:k * 2]

            for s_idx in top_sparse_indices:
                sparse_score = float(sim_scores[s_idx])
                if sparse_score > 0.05:
                    doc = _CHUNKS_CACHE[s_idx]
                    content = doc.page_content
                    if content in candidates:
                        candidates[content]["sparse_score"] = sparse_score
                    else:
                        candidates[content] = {
                            "content": content,
                            "metadata": doc.metadata,
                            "dense_score": 0.0,
                            "sparse_score": sparse_score
                        }
        except Exception as e:
            logger.warning(f"Sparse retrieval encountered notice: {e}")

    # 3. Calculate Hybrid Combined Score
    ranked_results = []
    for item in candidates.values():
        hybrid_score = (dense_weight * item["dense_score"]) + (sparse_weight * item["sparse_score"])
        ranked_results.append({
            "content": item["content"],
            "metadata": item["metadata"],
            "page": item["metadata"].get("page", "N/A"),
            "source": item["metadata"].get("source", "PDF Corpus"),
            "score": round(hybrid_score, 4),
            "dense_score": round(item["dense_score"], 4),
            "sparse_score": round(item["sparse_score"], 4)
        })

    # Sort descending by hybrid score
    ranked_results.sort(key=lambda x: x["score"], reverse=True)
    return ranked_results[:k]


def fetch_and_ingest_arxiv(query: str, max_results: int = 2) -> List[Dict[str, Any]]:
    """
    Searches arXiv for scholarly papers relevant to the query, downloads the PDFs into data/arxiv_cache/,
    extracts text using PyMuPDF, chunks the papers with [arXiv: ID] tags, and indexes them into ChromaDB.
    """
    global _VECTOR_STORE
    os.makedirs(ARXIV_CACHE_DIR, exist_ok=True)
    ingested_papers = []

    if arxiv is None:
        logger.warning("arxiv package not available; skipping dynamic arXiv fetch.")
        return []

    try:
        client = arxiv.Client(page_size=max_results, delay_seconds=2.0, num_retries=2)
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        results = list(client.results(search))
    except Exception as e:
        logger.warning(f"arXiv search request failed or timed out: {e}")
        return []

    splitter = get_text_splitter()
    new_docs: List[Document] = []

    for paper in results:
        arxiv_id = paper.get_short_id()
        title_clean = re.sub(r"[^\w\s-]", "", paper.title)[:40].strip().replace(" ", "_")
        pdf_filename = f"arxiv_{arxiv_id}_{title_clean}.pdf"
        pdf_path = os.path.join(ARXIV_CACHE_DIR, pdf_filename)

        # Download if not cached
        if not os.path.exists(pdf_path):
            try:
                paper.download_pdf(dirpath=ARXIV_CACHE_DIR, filename=pdf_filename)
            except Exception as dl_err:
                logger.warning(f"Could not download PDF for {arxiv_id}: {dl_err}. Using abstract.")
                # Fallback to abstract text
                abstract_doc = Document(
                    page_content=f"[arXiv: {arxiv_id}] Title: {paper.title}\nAuthors: {', '.join(a.name for a in paper.authors)}\nAbstract:\n{paper.summary}",
                    metadata={
                        "source": f"arXiv:{arxiv_id}",
                        "page": 1,
                        "title": paper.title,
                        "arxiv_id": arxiv_id
                    }
                )
                new_docs.append(abstract_doc)
                ingested_papers.append({
                    "arxiv_id": arxiv_id,
                    "title": paper.title,
                    "pdf_path": None,
                    "chunks": 1
                })
                continue

        # Extract text from downloaded PDF
        try:
            doc = fitz.open(pdf_path)
            paper_chunk_count = 0
            for page_idx in range(min(len(doc), 6)):  # Ingest first 6 pages for speed
                p_text = doc[page_idx].get_text().strip()
                if not p_text:
                    continue
                splits = splitter.split_text(p_text)
                for c_idx, chunk in enumerate(splits):
                    tagged_chunk = f"[arXiv: {arxiv_id} | Page {page_idx+1}]\n{chunk}"
                    new_docs.append(Document(
                        page_content=tagged_chunk,
                        metadata={
                            "source": f"arXiv:{arxiv_id}",
                            "page": page_idx + 1,
                            "title": paper.title,
                            "arxiv_id": arxiv_id
                        }
                    ))
                    paper_chunk_count += 1
            doc.close()

            ingested_papers.append({
                "arxiv_id": arxiv_id,
                "title": paper.title,
                "pdf_path": pdf_path,
                "chunks": paper_chunk_count
            })
        except Exception as read_err:
            logger.warning(f"Error parsing PDF for {arxiv_id}: {read_err}")

    # Add new documents to ChromaDB and refresh cache
    if new_docs:
        if _VECTOR_STORE is None:
            get_or_create_vector_store()
        _VECTOR_STORE.add_documents(new_docs)
        refresh_lexical_cache(_VECTOR_STORE)
        logger.info(f"Successfully ingested {len(new_docs)} arXiv chunks into ChromaDB.")

    return ingested_papers


if __name__ == "__main__":
    print("Testing src/vector_db.py...")
    store = get_or_create_vector_store()
    print("Vector store ready. Running hybrid search test...")
    results = hybrid_search("What is the Transformer architecture and self-attention?", k=3)
    for r in results:
        print(f"\n[Source: {r['source']} | Page: {r['page']} | Score: {r['score']}]")
        print(r['content'][:150] + "...")
    print("\nsrc/vector_db.py executed successfully!")
