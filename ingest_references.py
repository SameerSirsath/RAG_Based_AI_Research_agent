"""
ingest_references.py
Command-Line Utility to Ingest Reference Academic PDFs into ChromaDB.
Expands the Agentic RAG Vector Store with additional literature, survey papers, and lecture notes.

Usage:
    # 1. Ingest all PDFs placed inside data/references/
    python ingest_references.py

    # 2. Ingest a specific PDF file
    python ingest_references.py --file path/to/paper.pdf

    # 3. Ingest all PDFs from a custom directory
    python ingest_references.py --dir path/to/my_papers/

    # 4. List all currently indexed document sources and chunk stats
    python ingest_references.py --list

    # 5. Generate a sample reference PDF in data/references/ and ingest it
    python ingest_references.py --create-sample

Course: Natural Language Processing (NLP)
Team: Sameer Sirsath (PRN: 202401110023) & Rameshwar Sanap (PRN: 202401110021)
"""

import os
import sys
import argparse
import fitz  # PyMuPDF
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

from src.vector_db import (
    get_or_create_vector_store,
    ingest_single_pdf,
    ingest_references_directory,
    get_indexed_sources_summary,
    reindex_all_books_completely
)


def create_sample_reference_pdf(output_path: str = "data/references/sample_attention_mechanisms_survey.pdf"):
    """
    Creates a sample 3-page academic reference PDF in data/references/
    covering Deep Attention & FlashAttention to demonstrate reference ingestion.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = fitz.open()

    pages = [
        ("FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
         "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness\n"
         "Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re (Stanford University)\n\n"
         "Abstract & Core Thesis:\n"
         "Standard scaled dot-product attention scales quadratically O(N^2) in sequence length N, "
         "imposing severe memory bottlenecks on modern Graphics Processing Units (GPUs). "
         "Standard attention materializes the intermediate N x N attention matrix S = Q * K^T in high-bandwidth "
         "GPU High-Bandwidth Memory (HBM), leading to memory IO memory bandwidth saturation rather than compute saturation.\n\n"
         "IO-Aware Algorithm:\n"
         "FlashAttention redesigns the attention computation to account for GPU memory hierarchy levels: "
         "fast on-chip SRAM (Static RAM, ~19 TB/s) versus large but slower HBM (~1.5-2.0 TB/s). "
         "By applying tiling and incremental softmax normalization via online softmax algorithms, FlashAttention computes "
         "exact attention without materializing the large N x N intermediate matrix into HBM. "
         "This yields a 2x-4x wall-clock speedup and enables context length scaling from 2K to 64K tokens."),

        ("Multi-Query and Grouped-Query Attention Mechanisms",
         "Chapter 2: Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)\n\n"
         "During autoregressive inference in Large Language Models (LLMs), generating each token requires fetching "
         "the Key-Value (KV) cache from GPU memory for all preceding sequence tokens. As batch size and context window grow, "
         "the memory footprint of the KV cache dominates inference VRAM.\n\n"
         "1. Multi-Head Attention (MHA):\n"
         "Standard MHA uses H query heads, H key heads, and H value heads. Memory complexity is O(b * s * h * d_k).\n\n"
         "2. Multi-Query Attention (MQA):\n"
         "Noam Shazeer (2019) introduced MQA, which shares a single Key head and single Value head across all H Query heads. "
         "This slashes KV cache memory by a factor of H, enabling dramatically higher inference batch sizes with slight quality trade-offs.\n\n"
         "3. Grouped-Query Attention (GQA):\n"
         "Ainslie et al. (2023) introduced GQA as an optimal Pareto compromise: query heads are partitioned into G groups, "
         "each sharing one KV head pair. GQA matches MHA quality while approaching MQA inference throughput."),

        ("Linear Attention and State Space Models (Mamba / S4)",
         "Chapter 3: Beyond Quadratic Attention: Linear Attention and State Space Models\n\n"
         "To escape the quadratic complexity of self-attention entirely, alternative sequence architectures have emerged.\n\n"
         "1. Kernelized Linear Attention:\n"
         "Katharopoulos et al. (2020) demonstrated that applying feature maps phi(x) allows reordering matrix multiplications via associativity: "
         "(phi(Q) * phi(K)^T) * V = phi(Q) * (phi(K)^T * V), reducing complexity from O(N^2) to O(N * d).\n\n"
         "2. Selective State Space Models (Mamba):\n"
         "Albert Gu and Tri Dao (2023) developed Mamba, which replaces attention layers with continuous-time state space models (SSMs) "
         "featuring time-varying selection parameters. Mamba computes representations via hardware-aware associative parallel scans, "
         "achieving linear inference throughput O(1) per token generation and quadratic training parallelization without an explicit attention matrix.")
    ]

    for title, text in pages:
        page = doc.new_page()
        rect = fitz.Rect(50, 50, 545, 792)
        page.insert_textbox(rect, text, fontsize=10, fontname="helv", color=(0.1, 0.1, 0.1), align=fitz.TEXT_ALIGN_LEFT)

    doc.save(output_path)
    doc.close()
    print(f"✅ Generated sample reference PDF at: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Reference PDFs into ChromaDB Vector Store for Agentic RAG Assistant"
    )
    parser.add_argument("--file", type=str, help="Path to a single PDF file to ingest")
    parser.add_argument("--dir", type=str, default="data/references",
                        help="Directory containing reference PDFs (default: data/references)")
    parser.add_argument("--list", action="store_true",
                        help="List all currently indexed document sources in ChromaDB")
    parser.add_argument("--complete", action="store_true",
                        help="Completely index 100% of all pages from all books in data/, data/references/, and data/references_nlp/")
    parser.add_argument("--create-sample", action="store_true",
                        help="Generate a sample reference academic PDF in data/references/ and ingest it")

    args = parser.parse_args()

    print("=" * 70)
    print("      AGENTIC RAG: ACADEMIC REFERENCE INGESTION UTILITY")
    print("Course: Natural Language Processing (NLP)")
    print("Team: Sameer Sirsath (202401110023) & Rameshwar Sanap (202401110021)")
    print("=" * 70)

    # 1. Complete Re-indexing of All Books
    if args.complete:
        print("\n🚀 Commencing 100% COMPLETE indexing of all reference books & papers...")
        print("Corpus directories: data/sample_book.pdf, data/references/, data/references_nlp/")
        res = reindex_all_books_completely()
        print(f"\n✅ Complete Ingestion Finished!")
        print(f"   - Total Documents Processed : {res['total_documents']}")
        print(f"   - Total Pages Indexed       : {res['total_pages_indexed']}")
        print(f"   - Total Chunks in ChromaDB  : {res['total_chunks_indexed']}")
        print("\nDocument Breakdown:")
        print("-" * 60)
        for d in res["documents_detail"]:
            print(f"  • {d['source']:<42} : {d['pages']:>4} pages -> {d['chunks']:>5} chunks")
        print("-" * 60)
        return

    # 1. List Sources
    if args.list:
        print("\n🔍 Querying active ChromaDB vector store...")
        summary = get_indexed_sources_summary()
        print(f"\n📊 Total Indexed Chunks in Database: {summary['total_chunks']}")
        print(f"📚 Unique Document Sources: {summary['unique_sources_count']}\n")
        print("Indexed Sources Breakdown:")
        print("-" * 55)
        for src, count in summary["sources"].items():
            print(f"  • {src:<38} : {count:>4} chunks")
        print("-" * 55)
        return

    # 2. Create Sample Reference PDF
    if args.create_sample:
        sample_path = create_sample_reference_pdf()
        print(f"\n📥 Ingesting newly created sample PDF into ChromaDB...")
        result = ingest_single_pdf(sample_path)
        print(f"✨ Successfully indexed '{result['source']}':")
        print(f"   - Pages Processed : {result['pages_processed']}")
        print(f"   - Chunks Created  : {result['chunks_indexed']}")
        print(f"   - Total DB Chunks : {result['total_chunks_in_db']}")
        return

    # 3. Ingest Single File
    if args.file:
        if not os.path.exists(args.file):
            print(f"❌ Error: File not found: {args.file}")
            sys.exit(1)
        print(f"\n📥 Ingesting single reference PDF: {args.file}...")
        result = ingest_single_pdf(args.file)
        print(f"✅ Ingestion Complete for '{result['source']}':")
        print(f"   - Pages Processed : {result['pages_processed']}")
        print(f"   - Chunks Created  : {result['chunks_indexed']}")
        print(f"   - Total DB Chunks : {result['total_chunks_in_db']}")
        return

    # 4. Ingest Directory (Default: data/references)
    target_dir = args.dir
    os.makedirs(target_dir, exist_ok=True)
    pdf_files = [f for f in os.listdir(target_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"\n⚠️  No PDF files found in '{target_dir}'.")
        print("👉 You can place your research papers into 'data/references/' and rerun this script,")
        print("   OR run:  python ingest_references.py --create-sample")
        print("   OR run:  python ingest_references.py --file path/to/paper.pdf")
        return

    print(f"\n📂 Scanning '{target_dir}' for reference PDFs ({len(pdf_files)} found)...")
    res = ingest_references_directory(target_dir)

    print(f"\n✅ Directory Ingestion Complete!")
    print(f"   - PDFs Ingested     : {res['pdf_count']}")
    print(f"   - New Chunks Added  : {res['total_new_chunks']}")
    print("\nFile Ingestion Details:")
    for f in res["files_detail"]:
        if "error" in f:
            print(f"   ❌ {f['source']}: {f['error']}")
        else:
            print(f"   • {f['source']}: {f['chunks_indexed']} chunks (from {f['pages_processed']} pages)")

    # Print summary of updated database
    summary = get_indexed_sources_summary()
    print(f"\n🎉 Total Vector Store Capacity: {summary['total_chunks']} chunks across {summary['unique_sources_count']} documents.")


if __name__ == "__main__":
    main()
