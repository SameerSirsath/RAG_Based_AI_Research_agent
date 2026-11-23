"""
download_references.py
Downloads 15 seminal open-access textbooks & foundation papers in AI/ML, Deep Learning, and NLP,
generates a comprehensive Markdown documentation index, and indexes them into ChromaDB.

Course: Natural Language Processing (NLP)
Team: Sameer Sirsath (PRN: 202401110023) & Rameshwar Sanap (PRN: 202401110021)
"""

import os
import sys
import time
import urllib.request
import fitz  # PyMuPDF
from dotenv import load_dotenv

load_dotenv()

from src.vector_db import get_or_create_vector_store, ingest_single_pdf, get_indexed_sources_summary

REFERENCES_DIR = "data/references"
INDEX_MD_PATH = "data/references/ACADEMIC_REFERENCES_INDEX.md"
ROOT_MD_PATH = "REFERENCES.md"

# List of 15 Seminal Foundation Papers & Books in AI/ML, Deep Learning, and NLP
REFERENCE_PAPERS = [
    {
        "id": "ref_01",
        "title": "Attention Is All You Need",
        "filename": "vaswani2017_attention_is_all_you_need.pdf",
        "authors": "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin",
        "year": "2017",
        "domain": "NLP / Deep Learning",
        "url": "https://arxiv.org/pdf/1706.03762.pdf",
        "description": "The seminal paper that introduced the Transformer architecture, replacing recurrent connections with Scaled Dot-Product and Multi-Head Self-Attention mechanisms."
    },
    {
        "id": "ref_02",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "filename": "devlin2018_bert.pdf",
        "authors": "Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova (Google AI)",
        "year": "2018",
        "domain": "NLP / Foundation Models",
        "url": "https://arxiv.org/pdf/1810.04805.pdf",
        "description": "Introduced bidirectional contextual representations via Masked Language Modeling (MLM) and Next Sentence Prediction (NSP), transforming modern NLP."
    },
    {
        "id": "ref_03",
        "title": "Language Models are Few-Shot Learners (GPT-3)",
        "filename": "brown2020_gpt3_few_shot_learners.pdf",
        "authors": "Tom B. Brown et al. (OpenAI)",
        "year": "2020",
        "domain": "LLMs / Generative AI",
        "url": "https://arxiv.org/pdf/2005.14165.pdf",
        "description": "Demonstrated that scaling autoregressive transformers to 175 billion parameters yields emergent in-context few-shot learning without gradient updates."
    },
    {
        "id": "ref_04",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG)",
        "filename": "lewis2020_rag.pdf",
        "authors": "Patrick Lewis, Ethan Perez, Aleksandara Piktus, Fabio Petroni, Vladimir Karpukhin et al. (Meta AI / UCL / NYU)",
        "year": "2020",
        "domain": "Agentic NLP / RAG",
        "url": "https://arxiv.org/pdf/2005.11401.pdf",
        "description": "Foundational paper introducing the Retrieval-Augmented Generation paradigm, marrying dense neural retrieval with sequence-to-sequence generators."
    },
    {
        "id": "ref_05",
        "title": "Deep Residual Learning for Image Recognition (ResNet)",
        "filename": "he2015_resnet_deep_residual_learning.pdf",
        "authors": "Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft Research)",
        "year": "2015",
        "domain": "Deep Learning / Computer Vision",
        "url": "https://arxiv.org/pdf/1512.03385.pdf",
        "description": "Introduced residual skip connections to resolve the degradation problem in extremely deep neural networks (152 layers)."
    },
    {
        "id": "ref_06",
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "filename": "hu2021_lora.pdf",
        "authors": "Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (Microsoft)",
        "year": "2021",
        "domain": "PEFT / Efficient Deep Learning",
        "url": "https://arxiv.org/pdf/2106.09685.pdf",
        "description": "Parameter-efficient fine-tuning technique freezing pre-trained model weights and injecting trainable rank decomposition matrices into attention layers."
    },
    {
        "id": "ref_07",
        "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "filename": "wei2022_chain_of_thought.pdf",
        "authors": "Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou (Google Research)",
        "year": "2022",
        "domain": "Prompting / LLM Reasoning",
        "url": "https://arxiv.org/pdf/2201.11903.pdf",
        "description": "Demonstrated that generating a series of intermediate reasoning steps dramatically enhances multi-step problem solving in arithmetic and symbolic tasks."
    },
    {
        "id": "ref_08",
        "title": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
        "filename": "dao2022_flashattention.pdf",
        "authors": "Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré (Stanford University)",
        "year": "2022",
        "domain": "Hardware Acceleration / Deep Learning",
        "url": "https://arxiv.org/pdf/2205.14135.pdf",
        "description": "IO-aware exact attention algorithm leveraging GPU SRAM memory tiling to achieve 2x-4x speedups without materializing large attention matrices in HBM."
    },
    {
        "id": "ref_09",
        "title": "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
        "filename": "reimers2019_sentence_bert.pdf",
        "authors": "Nils Reimers, Iryna Gurevych (UKP Lab, TU Darmstadt)",
        "year": "2019",
        "domain": "Dense Semantic Embeddings",
        "url": "https://arxiv.org/pdf/1908.10084.pdf",
        "description": "Modification of pre-trained BERT using Siamese and triplet network structures to derive semantically meaningful sentence embeddings compared via cosine distance."
    },
    {
        "id": "ref_10",
        "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        "filename": "rafailov2023_direct_preference_optimization.pdf",
        "authors": "Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn (Stanford University)",
        "year": "2023",
        "domain": "LLM Alignment / RLHF",
        "url": "https://arxiv.org/pdf/2305.18290.pdf",
        "description": "Replaces complex reinforcement learning from human feedback (RLHF) with a closed-form implicit reward loss function for preference alignment."
    },
    {
        "id": "ref_11",
        "title": "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter",
        "filename": "sanh2019_distilbert.pdf",
        "authors": "Victor Sanh, Lysandre Debut, Julien Chaumond, Thomas Wolf (Hugging Face)",
        "year": "2019",
        "domain": "Model Compression / NLP",
        "url": "https://arxiv.org/pdf/1910.01108.pdf",
        "description": "Applies knowledge distillation during pre-training to compress BERT by 40% while preserving 97% of language understanding and accelerating inference by 60%."
    },
    {
        "id": "ref_12",
        "title": "LLaMA: Open and Efficient Foundation Language Models",
        "filename": "touvron2023_llama.pdf",
        "authors": "Hugo Touvron et al. (Meta AI)",
        "year": "2023",
        "domain": "Foundation Models / Open LLMs",
        "url": "https://arxiv.org/pdf/2302.13971.pdf",
        "description": "Seminal open foundation model suite trained exclusively on publicly available datasets using RMSNorm, SwiGLU activations, and Rotary Positional Embeddings."
    },
    {
        "id": "ref_13",
        "title": "Adam: A Method for Stochastic Optimization",
        "filename": "kingma2014_adam_optimizer.pdf",
        "authors": "Diederik P. Kingma, Jimmy Ba (University of Amsterdam / University of Toronto)",
        "year": "2014",
        "domain": "Optimization / Machine Learning",
        "url": "https://arxiv.org/pdf/1412.6980.pdf",
        "description": "The dominant first-order gradient-based optimization algorithm computing individual adaptive learning rates from estimates of first and second moments of gradients."
    },
    {
        "id": "ref_14",
        "title": "A Survey of Large Language Models",
        "filename": "zhao2023_survey_large_language_models.pdf",
        "authors": "Wayne Xin Zhao et al. (Renmin University / University of Montreal / AI Box)",
        "year": "2023",
        "domain": "AI/ML & NLP Comprehensive Survey",
        "url": "https://arxiv.org/pdf/2303.18223.pdf",
        "description": "Exhaustive survey exploring the background, evolutionary tree, pre-training, adaptation, utilization, and emergent capabilities of modern Large Language Models."
    },
    {
        "id": "ref_15",
        "title": "Deep Learning: An Introduction for Applied Mathematicians",
        "filename": "higham2018_deep_learning_primer.pdf",
        "authors": "Catherine F. Higham, Desmond J. Higham (University of Edinburgh)",
        "year": "2018",
        "domain": "Foundations of AI/ML & Deep Learning",
        "url": "https://arxiv.org/pdf/1801.05894.pdf",
        "description": "Rigorous academic introduction to deep neural networks, backpropagation calculus, stochastic gradient descent, cross-entropy, and regularization mathematics."
    }
]


def download_paper(paper: dict) -> str:
    """
    Downloads paper PDF with proper headers and error handling.
    """
    os.makedirs(REFERENCES_DIR, exist_ok=True)
    file_path = os.path.join(REFERENCES_DIR, paper["filename"])

    if os.path.exists(file_path) and os.path.getsize(file_path) > 10000:
        print(f"  ⏩ Already cached: {paper['filename']} ({os.path.getsize(file_path):,} bytes)")
        return file_path

    print(f"  📥 Downloading '{paper['title']}'...")
    req = urllib.request.Request(
        paper["url"],
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response, open(file_path, "wb") as out_file:
            out_file.write(response.read())
        
        # Verify valid PDF
        doc = fitz.open(file_path)
        p_count = len(doc)
        doc.close()
        print(f"  ✅ Downloaded {paper['filename']} ({p_count} pages, {os.path.getsize(file_path):,} bytes)")
        return file_path
    except Exception as e:
        print(f"  ❌ Error downloading {paper['title']}: {e}")
        return None


def generate_markdown_index(papers: list) -> str:
    """
    Generates an exhaustive, beautifully formatted Markdown file of all 15 reference books
    and papers with titles, authors, domains, descriptions, and direct download links.
    """
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    
    content = (
        "# 📚 Academic Literature & Reference Corpus Index\n"
        "## Foundational Books & Papers in AI/ML, Deep Learning, and Natural Language Processing\n\n"
        "> **Project:** Agentic RAG-Based AI Research Assistant  \n"
        "> **Course:** Natural Language Processing (NLP)  \n"
        "> **Authors:** Sameer Sirsath (PRN: `202401110023`) & Rameshwar Sanap (PRN: `202401110021`)  \n"
        f"> **Generated:** {now}  \n\n"
        "---\n\n"
        "### 📌 Overview\n"
        "This repository incorporates a curated collection of **15 seminal academic foundation papers and textbooks** "
        "downloaded directly from official open-access academic repositories (arXiv / Open Science). "
        "All references are indexed into the persistent **ChromaDB vector store** using `sentence-transformers/all-MiniLM-L6-v2` "
        "embeddings and balanced with sparse **TF-IDF lexical search**, enabling the multi-agent swarm to answer questions "
        "grounded in verified scientific literature.\n\n"
        "---\n\n"
        "## 📑 Curated Literature Matrix\n\n"
        "| # | Paper / Book Title | Domain | Year | Authors | Direct PDF Link |\n"
        "| :-: | :--- | :--- | :-: | :--- | :--- |\n"
    )

    for idx, p in enumerate(papers, 1):
        content += f"| {idx} | **{p['title']}** | `{p['domain']}` | {p['year']} | {p['authors'][:45]}... | [Download PDF]({p['url']}) |\n"

    content += "\n---\n\n## 📖 Detailed Annotations & Scientific Summaries\n\n"

    for idx, p in enumerate(papers, 1):
        content += (
            f"### {idx}. {p['title']} ({p['year']})\n"
            f"- **Domain:** `{p['domain']}`\n"
            f"- **Authors:** {p['authors']}\n"
            f"- **Direct Download Link:** [{p['url']}]({p['url']})\n"
            f"- **Local Filename:** `{p['filename']}`\n"
            f"- **Significance & Abstract Summary:**\n"
            f"  > {p['description']}\n\n"
        )

    content += (
        "---\n\n"
        "## 🛠️ How to Ingest Additional References\n"
        "To index additional PDF books or papers into the ChromaDB vector database:\n"
        "```powershell\n"
        "# 1. Place any academic PDF into data/references/\n"
        "# 2. Run the ingestion utility:\n"
        "python ingest_references.py\n\n"
        "# 3. Or inspect the updated database:\n"
        "python ingest_references.py --list\n"
        "```\n\n"
        "---\n*Curated for NLP Academic Research by Sameer Sirsath & Rameshwar Sanap.*"
    )

    with open(INDEX_MD_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    with open(ROOT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n📄 Generated Markdown Literature Index at:\n  - {INDEX_MD_PATH}\n  - {ROOT_MD_PATH}")
    return content


def ingest_papers_into_chromadb(downloaded_paths: list):
    """
    Ingests downloaded reference papers into ChromaDB.
    """
    print("\n" + "=" * 70)
    print("      INGESTING DOWNLOADED REFERENCES INTO CHROMADB VECTOR STORE")
    print("=" * 70)

    # Ensure vector store is initialized
    store = get_or_create_vector_store()
    total_added = 0

    for idx, p_path in enumerate(downloaded_paths, 1):
        if not p_path or not os.path.exists(p_path):
            continue
        try:
            filename = os.path.basename(p_path)
            print(f"[{idx}/{len(downloaded_paths)}] Indexing {filename}...")
            res = ingest_single_pdf(p_path)
            print(f"    ✨ Added {res['chunks_indexed']} chunks from {res['pages_processed']} pages.")
            total_added += res['chunks_indexed']
        except Exception as e:
            print(f"    ⚠️ Warning indexing {p_path}: {e}")

    summary = get_indexed_sources_summary()
    print("\n" + "=" * 70)
    print(f"🎉 Ingestion Complete! Added {total_added} new chunks.")
    print(f"📊 Vector Database Total: {summary['total_chunks']} chunks across {summary['unique_sources_count']} sources.")
    print("=" * 70)


def main():
    print("=" * 70)
    print("   DOWNLOADING 15 SEMINAL AI/ML, DEEP LEARNING & NLP REFERENCES")
    print("Course: Natural Language Processing (NLP)")
    print("Team: Sameer Sirsath (202401110023) & Rameshwar Sanap (202401110021)")
    print("=" * 70 + "\n")

    downloaded = []
    for idx, paper in enumerate(REFERENCE_PAPERS, 1):
        print(f"[{idx}/15] {paper['title']}")
        path = download_paper(paper)
        if path:
            downloaded.append(path)
        time.sleep(1.2)  # Polite delay to respect arXiv rate limits

    # Generate Markdown documentation
    generate_markdown_index(REFERENCE_PAPERS)

    # Ingest downloaded papers into ChromaDB
    ingest_papers_into_chromadb(downloaded)


if __name__ == "__main__":
    main()
