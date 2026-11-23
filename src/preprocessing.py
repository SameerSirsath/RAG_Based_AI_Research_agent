"""
preprocessing.py
Academic NLP Module for Text Extraction, Morphological Preprocessing, and Vectorization Benchmarks.
Course: Natural Language Processing (NLP)
Team: Sameer Sirsath (PRN: 202401110023) & Rameshwar Sanap (PRN: 202401110021)
"""

import os
import re
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Ensure required NLTK data resources are present
for pkg in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass


def ensure_sample_pdf(pdf_path: str = "data/sample_book.pdf") -> str:
    """
    Ensures that a valid academic NLP PDF exists at pdf_path.
    If missing, creates a comprehensive 15-page academic reference PDF automatically.
    """
    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000:
        return pdf_path

    os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)
    doc = fitz.open()

    pages_data = [
        ("Natural Language Processing: Foundations and Advanced Architectures",
         "Natural Language Processing: Foundations and Advanced Architectures\n"
         "Academic Reference Handbook - Volume 1\n"
         "Authors: NLP Research Group (Sameer Sirsath & Rameshwar Sanap)\n\n"
         "Chapter 1: Foundations of Natural Language Processing\n"
         "Natural Language Processing (NLP) is an interdisciplinary subfield of artificial intelligence, "
         "computer science, and computational linguistics concerned with the interactions between computers "
         "and human natural language. The overarching objective of NLP is to construct computational models "
         "capable of understanding, interpreting, manipulating, and generating natural language in a manner that "
         "is both syntactically coherent and semantically meaningful.\n\n"
         "Historical NLP approaches relied heavily on hand-crafted symbolic rules and formal grammars devised by linguists, "
         "exemplified by context-free grammars (CFGs) and rule-based expert systems. However, symbolic paradigms suffered "
         "from extreme brittleness, inability to generalize to unseen vocabularies, and poor scalability across domain shifts. "
         "The statistical revolution in the 1990s superseded symbolic systems by introducing probabilistic language models, "
         "hidden Markov models (HMMs), and maximum entropy models.\n\n"
         "In modern computational linguistics, natural language data is treated as high-dimensional sequential signals. "
         "Modern deep learning architectures treat text as structured token sequences transformed through dense mathematical spaces. "
         "Understanding the fundamental transition from symbolic rule systems to statistical modeling and modern deep transformer "
         "architectures constitutes the foundation of contemporary research in artificial intelligence."),

        ("Morphological Processing and Lexical Normalization",
         "Chapter 2: Morphological Processing, Tokenization, and Normalization\n\n"
         "Morphology is the study of words, how they are formed, and their relationship to other words in the same language. "
         "In NLP pipelines, raw text strings cannot be directly processed by statistical or neural models without rigorous "
         "morphological normalization and lexical segmentation.\n\n"
         "1. Tokenization:\n"
         "Tokenization represents the foundational stage of any linguistic processing pipeline. Word tokenization segments continuous "
         "text streams into distinct lexical units (tokens). Sentence tokenization identifies grammatical sentence boundaries using "
         "punctuation heuristics and honorific abbreviations. Modern neural systems employ subword tokenization algorithms, including "
         "Byte-Pair Encoding (BPE), WordPiece, and SentencePiece, which mitigate the out-of-vocabulary (OOV) challenge by decomposing "
         "rare words into frequent subword substrings.\n\n"
         "2. Stemming and Lemmatization:\n"
         "Stemming is a crude heuristic process that strips word affixes (prefixes and suffixes) to arrive at a common base form or stem. "
         "The Porter Stemmer, developed by Martin Porter in 1980, operates via a cascade of five sequential algorithmic rules based on "
         "consonant-vowel transitions (measure m). In contrast, the Lancaster Stemmer employs a significantly more aggressive, iterative "
         "rule set that frequently results in over-stemming. Lemmatization, unlike stemming, leverages full morphological analysis and "
         "lexical databases such as WordNet to reduce an inflected word to its valid grammatical lemma based on its part of speech (POS)."),

        ("Lexical Semantics and Vocabulary Statistics",
         "Chapter 3: Lexical Semantics, Stopwords, and Vocabulary Statistics\n\n"
         "The quantitative distribution of words in natural language corpora exhibits universal mathematical regularities governed "
         "by statistical mechanics and information theory.\n\n"
         "1. Zipf's Law:\n"
         "George Kingsley Zipf formulated the empirical observation that in any natural language corpus, the frequency of any word "
         "is inversely proportional to its rank in the frequency table. Formally: f(r) = C / r^s, where f is the word frequency, "
         "r is the frequency rank, s is an exponent close to 1, and C is a normalizing constant. This implies that the top 100 most "
         "frequent words account for approximately 50% of all word occurrences in a typical corpus.\n\n"
         "2. Stopword Filtering:\n"
         "Because functional words (e.g., 'the', 'is', 'in', 'at', 'which') carry minimal domain-specific semantic entropy while dominating "
         "token counts, classical NLP pipelines filter them using curated stopword lexicons. Removing stopwords drastically reduces the "
         "dimensionality of the vector space and diminishes computational overhead for statistical analysis.\n\n"
         "3. Heaps' Law:\n"
         "Heaps' Law models the growth of vocabulary size V as a function of total text length N: V = K * N^beta (with 0.4 <= beta <= 0.6). "
         "This sub-linear growth explains why vocabulary pruning and subword vocabularies are essential to prevent combinatorial explosion."),

        ("Sparse Vector Space Models and TF-IDF",
         "Chapter 4: Vector Space Models and Term Frequency-Inverse Document Frequency (TF-IDF)\n\n"
         "To enable mathematical computation over qualitative linguistic tokens, texts must be projected into numerical vector spaces. "
         "Gerard Salton introduced the Vector Space Model (VSM), where documents are represented as high-dimensional vectors in a space "
         "spanned by orthogonal vocabulary axes.\n\n"
         "1. Term Frequency (TF):\n"
         "Term frequency measures the local importance of term t within document d. Standard representations include raw count and sublinear "
         "log-scaled term frequency: TF(t, d) = 1 + log(count(t, d)) if count(t, d) > 0 else 0.\n\n"
         "2. Inverse Document Frequency (IDF):\n"
         "Inverse document frequency penalizes terms that occur ubiquitously across the entire corpus D, elevating terms that possess high "
         "discriminatory power: IDF(t, D) = log((1 + |D|) / (1 + DF(t, D))) + 1.\n\n"
         "3. TF-IDF Weighting and Cosine Similarity:\n"
         "The composite TF-IDF score is the product TF(t, d) * IDF(t, D), typically normalized using L2 Euclidean norm. The semantic "
         "similarity between two document vectors A and B is quantified using the Cosine Similarity metric: cos(theta) = (A . B) / (||A|| * ||B||)."),

        ("Dense Vector Representations and Semantic Embeddings",
         "Chapter 5: Dense Vector Representations and Distributed Semantics\n\n"
         "The distributional hypothesis popularized by J.R. Firth in 1957 asserts: 'You shall know a word by the company it keeps.' "
         "Distributed representations project words from high-dimensional sparse spaces into low-dimensional continuous dense vector spaces.\n\n"
         "1. Word2Vec:\n"
         "Tomas Mikolov et al. (2013) introduced Word2Vec with two neural architectures: Continuous Bag of Words (CBOW) and Skip-Gram. "
         "Optimization via negative sampling yields vector arithmetic properties such as vec('King') - vec('Man') + vec('Woman') approx vec('Queen').\n\n"
         "2. GloVe (Global Vectors):\n"
         "Pennington et al. (2014) combined global matrix factorization with local window-based architectures by training on log-bilinear ratios "
         "of co-occurrence probabilities across the global corpus.\n\n"
         "3. Sentence Transformers:\n"
         "Nils Reimers and Iryna Gurevych (2019) introduced Sentence-BERT (SBERT), fine-tuning transformer models using Siamese network structures "
         "to generate fixed-size sentence embeddings (such as all-MiniLM-L6-v2) that enable efficient semantic similarity computation."),

        ("Sequential Architectures: RNNs, LSTMs, and GRUs",
         "Chapter 6: Sequential Deep Learning Architectures in NLP\n\n"
         "Natural language is inherently sequential, with long-range syntactic and semantic dependencies. Recurrent Neural Networks (RNNs) "
         "process sequences by iteratively updating a hidden state vector h_t based on current input x_t and prior state h_{t-1}.\n\n"
         "1. Vanishing and Exploding Gradients:\n"
         "Standard Jordan/Elman RNNs suffer from the vanishing gradient problem during Backpropagation Through Time (BPTT), preventing the "
         "model from capturing long-range dependencies.\n\n"
         "2. Long Short-Term Memory (LSTM):\n"
         "Hochreiter and Schmidhuber (1997) resolved this limitation by designing the Long Short-Term Memory cell. The LSTM architecture "
         "incorporates an internal cell state c_t regulated by three specialized gating mechanisms: Forget Gate f_t, Input Gate i_t, and "
         "Output Gate o_t.\n\n"
         "3. Gated Recurrent Unit (GRU):\n"
         "Cho et al. (2014) proposed the GRU, which merges the cell state and hidden state, employing only two gates (Reset and Update)."),

        ("The Transformer Architecture and Self-Attention",
         "Chapter 7: The Transformer Architecture and Self-Attention Mechanism\n\n"
         "In 2017, Vaswani et al. published the seminal paper 'Attention Is All You Need', completely replacing recurrent connections with the "
         "Multi-Head Self-Attention mechanism. This breakthrough eliminated sequential processing constraints, enabling massively parallelized "
         "computation across GPU clusters.\n\n"
         "1. Scaled Dot-Product Attention:\n"
         "Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V. The scaling factor sqrt(d_k) prevents dot products from growing excessively "
         "large in high dimensions, preventing vanishing gradients in softmax.\n\n"
         "2. Multi-Head Attention:\n"
         "Multi-head attention projects Queries, Keys, and Values h times with distinct learned linear projections: "
         "MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W^O.\n\n"
         "3. Positional Encodings:\n"
         "Sinusoidal functions or learned embeddings inject positional sequence awareness into permutation-invariant self-attention layers."),

        ("Bidirectional Pre-trained Models: BERT and DistilBERT",
         "Chapter 8: Bidirectional Encoders: BERT, RoBERTa, and DistilBERT\n\n"
         "Pre-training large neural models on vast unannotated text corpora followed by fine-tuning on downstream tasks transformed NLP.\n\n"
         "1. BERT:\n"
         "Devlin et al. (2018) introduced BERT, a bidirectional transformer encoder trained via Masked Language Model (MLM) and Next Sentence "
         "Prediction (NSP). Unlike unidirectional decoders, BERT attends to context from both left and right directions simultaneously.\n\n"
         "2. DistilBERT:\n"
         "Sanh et al. (2019) developed DistilBERT using knowledge distillation. DistilBERT retains 97% of BERT's language understanding "
         "capabilities while reducing model parameters by 40% and accelerating inference speed by 60%."),

        ("Autoregressive Transformers: The GPT and LLaMA Family",
         "Chapter 9: Autoregressive Generative Models: GPT and LLaMA\n\n"
         "Generative pre-trained models rely on causal (unidirectional) transformer decoders optimized to predict the next token given "
         "preceding context tokens.\n\n"
         "1. The GPT Series:\n"
         "Radford et al. demonstrated that unsupervised generative pre-training yields strong transfer learning. Large models exhibit emergent "
         "capabilities, including in-context learning, few-shot prompting, and chain-of-thought reasoning.\n\n"
         "2. Modern Open Architectures: LLaMA:\n"
         "Touvron et al. (Meta AI, 2023) developed LLaMA foundation models featuring RMSNorm pre-normalization, SwiGLU activations, "
         "Rotary Positional Embeddings (RoPE), and Grouped-Query Attention (GQA). Compact variants like LLaMA 3.2 enable fast local inference."),

        ("Retrieval-Augmented Generation (RAG) Architecture",
         "Chapter 10: Retrieval-Augmented Generation (RAG) Architectures\n\n"
         "While large language models possess extensive parametric memory, they suffer from knowledge cutoffs, domain gaps, and stochastic hallucinations. "
         "Patrick Lewis et al. (2020) proposed Retrieval-Augmented Generation (RAG) to ground model generation in non-parametric external knowledge.\n\n"
         "1. Architecture and Operational Flow:\n"
         "A production RAG framework consists of ingestion and chunking, semantic retrieval from vector stores, and grounded context synthesis.\n\n"
         "2. Hybrid Retrieval:\n"
         "Dense retrieval using neural embeddings excels at semantic abstraction, while sparse lexical matching (TF-IDF/BM25) ensures exact keyword "
         "recall. Hybrid retrieval combines both scores to achieve superior precision and recall."),

        ("Vector Databases and Indexing: ChromaDB and HNSW",
         "Chapter 11: Vector Stores, Indexing Mechanisms, and ChromaDB\n\n"
         "Managing millions of high-dimensional dense vectors requires specialized vector database systems optimized for approximate nearest "
         "neighbor (ANN) search.\n\n"
         "1. Hierarchical Navigable Small World (HNSW):\n"
         "Malkov and Yashunin (2018) developed HNSW, a multi-layer graph-based index structure achieving logarithmic O(log N) search complexity "
         "with high recall (>95%).\n\n"
         "2. ChromaDB Architecture:\n"
         "ChromaDB is an open-source, AI-native embedding database that decouples metadata storage (SQLite) from vector indices (HNSW), "
         "providing fast persistent storage and querying for LLM applications."),

        ("Named Entity Recognition and Information Extraction",
         "Chapter 12: Named Entity Recognition and spaCy NLP Framework\n\n"
         "Information extraction transforms unstructured natural language text into structured, queryable knowledge graphs and entity tables.\n\n"
         "1. Named Entity Recognition (NER):\n"
         "NER is a sequence tagging task that classifies named entities into categories such as PERSON, ORGANIZATION (ORG), GEOPOLITICAL "
         "ENTITY (GPE), DATE, and LAW.\n\n"
         "2. spaCy Architecture:\n"
         "Explosion AI developed spaCy with optimized Cython pipelines, combining rule-based matchers with convolutional neural models "
         "for fast entity identification and syntactic dependency parsing."),

        ("Topic Modeling: Latent Dirichlet Allocation (LDA)",
         "Chapter 13: Probabilistic Topic Modeling with Latent Dirichlet Allocation\n\n"
         "Topic modeling is an unsupervised statistical technique that discovers abstract latent topics occurring within a collection of documents.\n\n"
         "1. Generative Probabilistic Model:\n"
         "David Blei, Andrew Ng, and Michael Jordan (2003) introduced Latent Dirichlet Allocation (LDA). Documents are modeled as mixtures over "
         "latent topics, and topics are modeled as multinomial distributions over vocabulary tokens.\n\n"
         "2. Inference via Collapsed Gibbs Sampling:\n"
         "Posterior inference is carried out using Collapsed Gibbs Sampling or Variational Bayes, estimating topic assignments and term distributions.\n\n"
         "3. Topic Coherence:\n"
         "Metrics such as C_v or U_mass evaluate semantic consistency among top topic keywords."),

        ("Sentiment Analysis and Rhetorical Stance Classification",
         "Chapter 14: Sentiment Analysis, Rhetorical Polarity, and Stance\n\n"
         "Sentiment analysis determines the subjective emotional attitude, sentiment polarity, or rhetorical stance conveyed by a written passage.\n\n"
         "1. Polarity in Academic Discourse:\n"
         "Academic sentiment analysis identifies argumentative stance, methodological confidence, critical skepticism, and affirmative empirical claims.\n\n"
         "2. Transformer-Based Fine-Tuning:\n"
         "DistilBERT fine-tuned on SST-2 classifies rhetorical polarity into positive (affirmative/supportive) and negative (critical/skeptical) stance."),

        ("Agentic NLP Systems and Multi-Agent LangGraph Workflows",
         "Chapter 15: Agentic NLP Systems, Cyclic LangGraph Workflows, and Future Frontiers\n\n"
         "Traditional RAG pipelines operate as rigid linear chains. In complex research settings, linear chains fail when queries require iterative "
         "decomposition, self-correction, multi-source verification, and language translation.\n\n"
         "1. The Multi-Agent Paradigm:\n"
         "Agentic AI architectures model reasoning as a cyclic state machine where specialized autonomous agents collaborate by sharing a centralized state:\n"
         "- Research Agent: intent detection, multilingual gateway, hybrid retrieval, and arXiv self-correction fallback.\n"
         "- Analysis Agent: spaCy NER and DistilBERT stance analysis on retrieved evidence.\n"
         "- Writing Agent: evidence-grounded academic synthesis citing exact page and arXiv numbers.\n"
         "- Proofreader Agent: source verification, hallucination checks, provenance formatting, and target language translation.\n\n"
         "2. Conclusion:\n"
         "Collaborative agent teams provide verifiable, self-correcting research workflows for scientific literature.")
    ]

    for title, text in pages_data:
        page = doc.new_page()
        rect = fitz.Rect(50, 50, 545, 792)
        page.insert_textbox(rect, text, fontsize=10, fontname="helv", color=(0.1, 0.1, 0.1), align=fitz.TEXT_ALIGN_LEFT)

    doc.save(pdf_path)
    doc.close()
    return pdf_path


def extract_text_from_pdf(pdf_path: str = "data/sample_book.pdf") -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from an academic PDF using PyMuPDF (fitz),
    annotating each page with explicit [Page X] boundary tags and metadata.
    """
    pdf_path = ensure_sample_pdf(pdf_path)
    doc = fitz.open(pdf_path)
    pages = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1
        raw_text = page.get_text()
        tagged_text = f"[Page {page_num}]\n{raw_text.strip()}"
        pages.append({
            "page_num": page_num,
            "raw_text": raw_text.strip(),
            "tagged_text": tagged_text,
            "char_count": len(raw_text.strip()),
            "source": os.path.basename(pdf_path)
        })

    doc.close()
    return pages


def preprocess_text(text: str) -> Dict[str, Any]:
    """
    Performs full NLTK morphological preprocessing:
    1. Word tokenization
    2. Lowercasing
    3. Alphanumeric & punctuation normalization
    4. Stopword filtering (NLTK English stopwords)
    5. PorterStemmer stemming
    """
    # 1. Lowercase
    lowered = text.lower()

    # 2. Tokenize
    raw_tokens = word_tokenize(lowered)

    # 3. Filter alphanumeric
    alpha_tokens = [tok for tok in raw_tokens if tok.isalnum()]

    # 4. Stopwords
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [tok for tok in alpha_tokens if tok not in stop_words]

    # 5. Porter Stemmer
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(tok) for tok in filtered_tokens]

    return {
        "raw_text": text,
        "raw_tokens": raw_tokens,
        "alpha_tokens": alpha_tokens,
        "filtered_tokens": filtered_tokens,
        "stemmed_tokens": stemmed_tokens,
        "cleaned_text": " ".join(stemmed_tokens)
    }


def run_morphological_benchmarks(pdf_path: str = "data/sample_book.pdf",
                                 output_file: str = "outputs/preprocessing_analysis.txt") -> Dict[str, Any]:
    """
    Executes morphological benchmarks across the entire PDF corpus,
    calculating character count, token count, unique vocabulary, and reduction metrics.
    Saves empirical report to outputs/preprocessing_analysis.txt.
    """
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    pages = extract_text_from_pdf(pdf_path)
    full_raw_text = "\n\n".join(p["raw_text"] for p in pages)

    proc_result = preprocess_text(full_raw_text)

    raw_char_count = len(full_raw_text)
    raw_token_count = len(proc_result["raw_tokens"])
    raw_vocab_size = len(set(proc_result["raw_tokens"]))

    clean_char_count = len(proc_result["cleaned_text"])
    clean_token_count = len(proc_result["stemmed_tokens"])
    clean_vocab_size = len(set(proc_result["stemmed_tokens"]))

    token_reduction_pct = ((raw_token_count - clean_token_count) / max(raw_token_count, 1)) * 100
    vocab_reduction_pct = ((raw_vocab_size - clean_vocab_size) / max(raw_vocab_size, 1)) * 100
    char_reduction_pct = ((raw_char_count - clean_char_count) / max(raw_char_count, 1)) * 100

    report = (
        "========================================================================\n"
        "           EMPIRICAL NLP PREPROCESSING BENCHMARK REPORT                 \n"
        "Course: Natural Language Processing (NLP)                               \n"
        "Team: Sameer Sirsath (202401110023) & Rameshwar Sanap (202401110021)    \n"
        "========================================================================\n\n"
        f"Target PDF Corpus       : {pdf_path}\n"
        f"Total Pages Analyzed    : {len(pages)}\n\n"
        "------------------------------------------------------------------------\n"
        "METRIC                   | RAW CORPUS       | CLEANED (STEMMED) | REDUCTION (%)\n"
        "------------------------------------------------------------------------\n"
        f"Character Count          | {raw_char_count:<16} | {clean_char_count:<17} | {char_reduction_pct:.2f}%\n"
        f"Token Count              | {raw_token_count:<16} | {clean_token_count:<17} | {token_reduction_pct:.2f}%\n"
        f"Unique Vocabulary Size   | {raw_vocab_size:<16} | {clean_vocab_size:<17} | {vocab_reduction_pct:.2f}%\n"
        "------------------------------------------------------------------------\n\n"
        "Sample Stemming Transformations (Top 10):\n"
    )

    sample_transformations = []
    seen = set()
    for raw, stem in zip(proc_result["filtered_tokens"], proc_result["stemmed_tokens"]):
        if raw != stem and raw not in seen:
            seen.add(raw)
            sample_transformations.append(f"  - '{raw}' -> '{stem}'")
            if len(sample_transformations) >= 10:
                break

    report += "\n".join(sample_transformations) + "\n\n"
    report += "Morphological analysis confirms significant dimensional pruning suitable for vector space indexing.\n"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)

    return {
        "raw_char_count": raw_char_count,
        "clean_char_count": clean_char_count,
        "raw_token_count": raw_token_count,
        "clean_token_count": clean_token_count,
        "raw_vocab_size": raw_vocab_size,
        "clean_vocab_size": clean_vocab_size,
        "token_reduction_pct": token_reduction_pct,
        "vocab_reduction_pct": vocab_reduction_pct,
        "sample_transformations": sample_transformations,
        "report_path": output_file
    }


def build_tfidf_and_dense_models(pages: List[Dict[str, Any]] = None,
                                 pdf_path: str = "data/sample_book.pdf",
                                 matrix_path: str = "outputs/tfidf_matrix.pkl") -> Dict[str, Any]:
    """
    Builds:
    1. Scikit-learn TfidfVectorizer(max_features=2500, stop_words="english", sublinear_tf=True, ngram_range=(1, 2))
       and saves matrix and vectorizer to outputs/tfidf_matrix.pkl.
    2. SentenceTransformer (all-MiniLM-L6-v2) dense embeddings.
    3. Pairwise cosine similarity matrix comparison between sparse TF-IDF and dense embeddings.
    """
    if pages is None:
        pages = extract_text_from_pdf(pdf_path)

    corpus_texts = [p["raw_text"] for p in pages]

    # 1. Sparse TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=2500,
        stop_words="english",
        sublinear_tf=True,
        ngram_range=(1, 2)
    )
    tfidf_matrix = vectorizer.fit_transform(corpus_texts)

    os.makedirs(os.path.dirname(matrix_path) or ".", exist_ok=True)
    with open(matrix_path, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "matrix": tfidf_matrix, "pages": pages}, f)

    # 2. Dense SentenceTransformer
    dense_model = SentenceTransformer("all-MiniLM-L6-v2")
    dense_embeddings = dense_model.encode(corpus_texts, convert_to_numpy=True)

    # 3. Pairwise Cosine Similarities
    tfidf_cosine_sim = cosine_similarity(tfidf_matrix)
    dense_cosine_sim = cosine_similarity(dense_embeddings)

    return {
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "tfidf_matrix_shape": tfidf_matrix.shape,
        "dense_model": dense_model,
        "dense_embeddings_shape": dense_embeddings.shape,
        "tfidf_cosine_sim": tfidf_cosine_sim,
        "dense_cosine_sim": dense_cosine_sim,
        "matrix_path": matrix_path
    }


if __name__ == "__main__":
    print("Testing src/preprocessing.py...")
    benchmarks = run_morphological_benchmarks()
    print("Morphological benchmarks completed:")
    print(f"Token reduction: {benchmarks['token_reduction_pct']:.2f}%")
    print(f"Report saved to: {benchmarks['report_path']}")

    vec_results = build_tfidf_and_dense_models()
    print(f"TF-IDF Matrix Shape: {vec_results['tfidf_matrix_shape']}")
    print(f"Dense Embeddings Shape: {vec_results['dense_embeddings_shape']}")
    print("src/preprocessing.py executed successfully!")
