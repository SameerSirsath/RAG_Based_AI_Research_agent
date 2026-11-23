"""
app.py
Interactive Multi-Tab Streamlit Application for Academic NLP Research.
Title: Agentic RAG-Based AI Research Assistant
Course: Natural Language Processing (NLP)
Team: Sameer Sirsath (PRN: 202401110023) & Rameshwar Sanap (PRN: 202401110021)
"""

import os
import time
import json
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv

# Load backend environment configuration
load_dotenv()

# Also support Streamlit Secrets seamlessly if deployed on Streamlit Cloud
try:
    if hasattr(st, "secrets"):
        for key in ["GROQ_API_KEY", "GOOGLE_API_KEY", "OLLAMA_MODEL"]:
            if key in st.secrets and not os.environ.get(key):
                os.environ[key] = str(st.secrets[key])
except Exception:
    pass

# Page configuration
st.set_page_config(
    page_title="Agentic RAG AI Research Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Academic CSS
st.markdown("""
<style>
    .main-title {
        font-family: 'Georgia', serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .academic-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .badge-agent {
        background-color: #0284c7;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-source {
        background-color: #059669;
        color: white;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.75rem;
    }
    .badge-sentiment-pos {
        background-color: #10b981;
        color: white;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.75rem;
    }
    .badge-sentiment-neg {
        background-color: #f59e0b;
        color: white;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# Imports from src/
from src.preprocessing import (
    extract_text_from_pdf,
    run_morphological_benchmarks,
    build_tfidf_and_dense_models,
    ensure_sample_pdf
)
from src.nlp_analyzers import (
    run_ner_corpus_benchmark,
    train_lda_topic_model,
    analyze_sentiment_batch,
    LANGUAGE_NAME_MAP
)
from src.vector_db import (
    get_or_create_vector_store,
    hybrid_search,
    fetch_and_ingest_arxiv,
    ingest_single_pdf,
    get_indexed_sources_summary
)
from src.graph import run_research_workflow, generate_markdown_report


# Ensure default corpus exists on startup
ensure_sample_pdf("data/sample_book.pdf")


# -------------------------------------------------------------------------
# Sidebar Controls & Configuration
# -------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🎓 NLP Mini Project")
    st.markdown("**Agentic RAG Research Assistant**")
    st.caption("Course: Natural Language Processing (NLP)")
    
    st.markdown("---")
    st.markdown("#### 👥 Team Metadata")
    st.markdown("- **Sameer Sirsath** `202401110023`")
    st.markdown("- **Rameshwar Sanap** `202401110021`")

    st.markdown("---")
    st.markdown("#### ⚙️ LLM Backend Status")

    # Secure Backend Status Indicator (Keys configured securely in backend .env)
    if os.environ.get("GROQ_API_KEY"):
        backend_status = "⚡ Cloud (Groq: llama-3.1-8b)"
        backend_badge = "Cloud Acceleration Enabled"
    elif os.environ.get("GOOGLE_API_KEY"):
        backend_status = "🌐 Cloud (Gemini: gemini-2.5-flash)"
        backend_badge = "Cloud Acceleration Enabled"
    else:
        backend_status = "💻 Local / Edge (Ollama: llama3.2)"
        backend_badge = "Offline Local Execution"

    st.info(f"**Active LLM:** {backend_status}\n\n*Status:* `{backend_badge}`")
    st.caption("🔒 *LLM API credentials are configured securely in the backend (.env / server environment).*")

    st.markdown("---")
    st.markdown("#### 📚 Reference Corpus & Vector Store")

    # Display live vector store metrics
    try:
        db_summary = get_indexed_sources_summary()
        st.markdown(f"**Database Status:** `{db_summary['total_chunks']} chunks` across `{db_summary['unique_sources_count']} sources`")
    except Exception:
        pass

    uploaded_file = st.file_uploader("Add Academic Reference PDF", type=["pdf"])
    if uploaded_file is not None:
        custom_pdf_path = os.path.join("data/references", uploaded_file.name)
        os.makedirs("data/references", exist_ok=True)
        with open(custom_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded `{uploaded_file.name}`")

        c_up1, c_up2 = st.columns(2)
        with c_up1:
            if st.button("➕ Ingest into ChromaDB"):
                with st.spinner("Extracting, chunking and embedding into vector store..."):
                    res = ingest_single_pdf(custom_pdf_path)
                st.success(f"Added {res['chunks_indexed']} chunks!")
                st.rerun()
        with c_up2:
            if st.button("🔄 Reset as Sole Corpus"):
                with st.spinner("Rebuilding ChromaDB from uploaded PDF..."):
                    get_or_create_vector_store(force_reindex=True, pdf_path=custom_pdf_path)
                st.success("Re-indexed as primary corpus!")
                st.rerun()

    force_arxiv_toggle = st.checkbox("Force Live arXiv Search", value=False,
                                     help="If checked, the Research Agent will query arXiv even if local chunks have high score.")

    if st.button("Reset Default PDF Corpus"):
        with st.spinner("Rebuilding ChromaDB index..."):
            get_or_create_vector_store(force_reindex=True, pdf_path="data/sample_book.pdf")
        st.success("Default corpus re-indexed!")
        st.rerun()


# -------------------------------------------------------------------------
# Main Header
# -------------------------------------------------------------------------

st.markdown('<div class="main-title">🏛️ Agentic RAG-Based AI Research Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Autonomous 4-Agent Research Workflow Grounded in Academic Literature with Active NLP Pipelines</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Multi-Agent Research Assistant",
    "🔬 NLP Preprocessing & Vectorization Lab",
    "🏷️ NER & Topic Modeling Explorer",
    "📐 Architecture & Project Information"
])


# =========================================================================
# TAB 1: Multi-Agent Research Assistant
# =========================================================================

with tab1:
    st.markdown("### 🔍 Scholarly Research Assistant")
    st.caption("Ask technical questions in any language (e.g., English, French, Hindi, Spanish). The 4 agents will research, analyze, synthesize, and cite verified evidence.")

    # Suggested queries pills
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        if st.button("What is Self-Attention in Transformers?"):
            st.session_state["query_input"] = "What is Self-Attention in Transformers?"
    with col_s2:
        if st.button("Explain Hybrid RAG Retrieval"):
            st.session_state["query_input"] = "How does Hybrid Retrieval combine sparse and dense vector representations in RAG?"
    with col_s3:
        if st.button("DistilBERT vs BERT"):
            st.session_state["query_input"] = "Compare DistilBERT and BERT architecture and compression advantages."
    with col_s4:
        if st.button("Multilingual Query (French)"):
            st.session_state["query_input"] = "Quels sont les avantages de l'architecture Transformer par rapport aux RNNs?"

    # Chat history state
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_query = st.chat_input("Enter your academic inquiry or research topic...")

    # Handle preset clicks
    if "query_input" in st.session_state and st.session_state["query_input"]:
        user_query = st.session_state.pop("query_input")

    if user_query:
        # Display user question
        st.session_state["chat_history"].append({"role": "user", "content": user_query})

        # Run multi-agent graph
        with st.status("🚀 4-Agent Research Swarm Active...", expanded=True) as status_box:
            st.write("🕵️ **Research Agent**: Processing language gateway & hybrid ChromaDB search...")
            time.sleep(0.3)
            
            # Execute workflow
            state = run_research_workflow(user_query, force_arxiv=force_arxiv_toggle)
            
            st.write("🔬 **Analysis Agent**: Extracting spaCy entities & DistilBERT rhetorical stance...")
            time.sleep(0.2)
            st.write("✍️ **Writing Agent**: Drafting evidence-grounded academic synthesis...")
            time.sleep(0.2)
            st.write("🧐 **Proofreader Agent**: Verifying citations & preparing provenance block...")
            
            status_box.update(label="✅ Scholarly Research Completed!", state="complete", expanded=False)

        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": state["final_answer"],
            "state": state
        })

    # Render Chat History
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant" and "state" in msg:
                m_state = msg["state"]

                # Multi-Agent Trace Expander
                with st.expander("⏱️ Multi-Agent Execution Trace", expanded=False):
                    trace = m_state.get("execution_trace", [])
                    trace_df = pd.DataFrame(trace)
                    if not trace_df.empty:
                        st.table(trace_df)

                # Provenance & Evidence Expander
                chunks = m_state.get("retrieved_chunks", [])
                if chunks:
                    with st.expander(f"📚 Retrieved Academic Evidence Passages ({len(chunks)})", expanded=False):
                        for idx, chunk in enumerate(chunks):
                            c1, c2, c3 = st.columns([2, 1, 1])
                            with c1:
                                st.markdown(f"**Passage #{idx+1}:** `{chunk.get('source')}` (Page {chunk.get('page')})")
                            with c2:
                                st.markdown(f"**Relevance Score:** `{chunk.get('score', 0):.4f}`")
                            with c3:
                                # Find matching sentiment if any
                                s_list = m_state.get("chunk_sentiments", [])
                                s_match = next((s for s in s_list if s.get("chunk_idx") == idx + 1), None)
                                if s_match:
                                    badge_cls = "badge-sentiment-pos" if s_match["label"] == "POSITIVE" else "badge-sentiment-neg"
                                    st.markdown(f'<span class="{badge_cls}">{s_match["label"]} ({s_match["score"]:.2f})</span>', unsafe_allow_html=True)
                            st.caption(chunk.get("content", ""))
                            st.divider()

                # Extracted Entities
                ents = m_state.get("entities", [])
                if ents:
                    with st.expander(f"🏷️ Extracted Named Entities ({len(ents)})", expanded=False):
                        ent_df = pd.DataFrame(ents)
                        st.dataframe(ent_df, use_container_width=True)

                # Download Research Brief Button
                md_report = generate_markdown_report(m_state)
                st.download_button(
                    label="📥 Download Research Brief (.md)",
                    data=md_report,
                    file_name=f"Research_Brief_{int(time.time())}.md",
                    mime="text/markdown",
                    key=f"dl_{time.time()}"
                )


# =========================================================================
# TAB 2: NLP Preprocessing & Vectorization Lab
# =========================================================================

with tab2:
    st.markdown("### 🔬 NLP Preprocessing & Vectorization Lab")
    st.markdown("Evaluates **NLTK Morphological Normalization** and compares **Sparse TF-IDF** with **Dense Sentence Embeddings** (MiniLM-L6-v2).")

    if st.button("🚀 Run Preprocessing & Vectorization Benchmark"):
        with st.spinner("Executing morphological extraction and vector space models..."):
            bench = run_morphological_benchmarks("data/sample_book.pdf")
            models = build_tfidf_and_dense_models(pdf_path="data/sample_book.pdf")

        st.success("Benchmarks executed successfully! Empirical artifacts saved in `outputs/`.")

        # Metric summary cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Raw Token Count", f"{bench['raw_token_count']:,}")
        m2.metric("Cleaned Token Count", f"{bench['clean_token_count']:,}")
        m3.metric("Token Reduction", f"{bench['token_reduction_pct']:.2f}%")
        m4.metric("Vocab Reduction", f"{bench['vocab_reduction_pct']:.2f}%")

        st.markdown("---")
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### 📊 Morphological Pruning Table")
            summary_table = pd.DataFrame({
                "Metric": ["Character Count", "Total Tokens", "Unique Vocabulary"],
                "Raw Corpus": [bench["raw_char_count"], bench["raw_token_count"], bench["raw_vocab_size"]],
                "Cleaned (Stemmed)": [bench["clean_char_count"], bench["clean_token_count"], bench["clean_vocab_size"]],
                "Reduction (%)": [
                    f"{((bench['raw_char_count'] - bench['clean_char_count']) / bench['raw_char_count']) * 100:.2f}%",
                    f"{bench['token_reduction_pct']:.2f}%",
                    f"{bench['vocab_reduction_pct']:.2f}%"
                ]
            })
            st.table(summary_table)

            st.markdown("#### ✂️ Sample Porter Stemmer Transformations")
            for item in bench["sample_transformations"][:8]:
                st.markdown(f"- {item}")

        with col_right:
            st.markdown("#### 📐 Sparse vs Dense Representation")
            st.markdown(f"- **TF-IDF Matrix Shape:** `{models['tfidf_matrix_shape']}` (15 Documents × 2500 Max Features)")
            st.markdown(f"- **Dense SBERT Embeddings Shape:** `{models['dense_embeddings_shape']}` (15 Documents × 384 Dimensions)")

            st.markdown("#### 🔍 Pairwise Cosine Similarity Explorer")
            p1 = st.selectbox("Select Document Page A", range(1, 16), index=6)  # Page 7: Transformers
            p2 = st.selectbox("Select Document Page B", range(1, 16), index=7)  # Page 8: BERT

            tfidf_sim = float(models["tfidf_cosine_sim"][p1-1][p2-1])
            dense_sim = float(models["dense_cosine_sim"][p1-1][p2-1])

            c_sim1, c_sim2 = st.columns(2)
            c_sim1.metric("Sparse TF-IDF Cosine Sim", f"{tfidf_sim:.4f}")
            c_sim2.metric("Dense MiniLM Cosine Sim", f"{dense_sim:.4f}")

            st.caption("Dense embeddings capture higher conceptual similarity across semantically related chapters (Transformers & BERT) where exact keyword overlap is sparse.")


# =========================================================================
# TAB 3: NER & Topic Modeling Explorer
# =========================================================================

with tab3:
    st.markdown("### 🏷️ NER & Topic Modeling Explorer")
    st.markdown("Explore **spaCy Named Entity Recognition** and **Gensim LDA Topic Modeling** across the corpus.")

    col_ner_btn, col_lda_btn = st.columns(2)
    with col_ner_btn:
        run_ner = st.button("🏷️ Run spaCy NER Extraction")
    with col_lda_btn:
        run_lda = st.button("📑 Train Gensim LDA Model")

    # Load or run NER
    pages = extract_text_from_pdf("data/sample_book.pdf")
    all_texts = [p["raw_text"] for p in pages]

    if run_ner or os.path.exists("outputs/ner_results.json"):
        if run_ner:
            with st.spinner("Extracting entities across 15 pages with spaCy..."):
                ner_data = run_ner_corpus_benchmark(pages)
        else:
            with open("outputs/ner_results.json", "r", encoding="utf-8") as f:
                ner_data = json.load(f)

        st.markdown("#### 🏷️ spaCy Entity Distribution")
        lbl_dist = ner_data.get("label_distribution", {})
        if lbl_dist:
            dist_df = pd.DataFrame(list(lbl_dist.items()), columns=["Entity Label", "Count"])
            st.bar_chart(dist_df.set_index("Entity Label"))

        st.markdown("#### Top Detected Entities in Corpus")
        top_ents = ner_data.get("top_entities", [])
        if top_ents:
            st.table(pd.DataFrame(top_ents, columns=["Entity Text", "Frequency"]))

    st.markdown("---")

    # Load or run LDA
    if run_lda or os.path.exists("outputs/topic_model_results.txt"):
        if run_lda:
            with st.spinner("Training Gensim LDA (5 topics, 10 passes)..."):
                lda_res = train_lda_topic_model(all_texts, num_topics=5, passes=10)
                topics = lda_res["topics"]
        else:
            # Re-train quickly or display
            with st.spinner("Generating topic distributions..."):
                lda_res = train_lda_topic_model(all_texts, num_topics=5, passes=10)
                topics = lda_res["topics"]

        st.markdown("#### 📑 Gensim LDA Dominant Topics (5 Topics Discovered)")
        for t in topics:
            t_id = t["topic_id"] + 1
            words = ", ".join([f"**{w}** ({weight:.3f})" for w, weight in t["weighted_keywords"]])
            st.markdown(f"- **Topic #{t_id}:** {words}")


# =========================================================================
# TAB 4: Architecture & Project Info
# =========================================================================

with tab4:
    st.markdown("### 📐 System Architecture & Project Specifications")

    st.markdown("""
```mermaid
flowchart TD
    User([User Query]) --> MultilingualGateway[Multilingual Gateway: langdetect]
    MultilingualGateway --> IntentRouter{Intent Classification}
    
    IntentRouter -->|Conversational Greeting| WritingAgent
    IntentRouter -->|Academic Research| ResearchAgent[Research Agent]
    
    subgraph MultiAgentCore [Cyclic LangGraph Core]
        ResearchAgent --> HybridRetrieval[(ChromaDB + TF-IDF)]
        HybridRetrieval --> RelevanceCheck{Relevance Score >= 0.15?}
        RelevanceCheck -->|No / Low Score| ArxivTool[Live arXiv Paper Ingestion]
        ArxivTool --> HybridRetrieval
        RelevanceCheck -->|Yes| AnalysisAgent[Analysis Agent]
        
        AnalysisAgent --> SpacyNER[spaCy NER Tool]
        AnalysisAgent --> DistilBERTTone[DistilBERT Sentiment / Tone Tool]
        SpacyNER --> WritingAgent[Writing Agent]
        DistilBERTTone --> WritingAgent
        
        WritingAgent --> ProofreaderAgent[Proofreader Agent]
        ProofreaderAgent --> CitationAudit{Citation Verification}
        CitationAudit --> ProvenanceFormatter[Provenance & Attribution]
        ProvenanceFormatter --> MultilingualOutput[Translation to Target Language]
    end
    
    MultilingualOutput --> Answer([Verified Cited Academic Response])
```
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📋 NLP Practicals Syllabus Mapping")

    syllabus_data = [
        {"Practical": "Practical 01", "Topic": "Text Extraction & Preprocessing", "Implementation": "PyMuPDF text extraction, NLTK tokenization, stopword pruning, Porter stemming", "Module": "src/preprocessing.py"},
        {"Practical": "Practical 02", "Topic": "Sparse & Dense Vectorization", "Implementation": "Scikit-Learn TF-IDF Matrix (2500 features) & SentenceTransformers all-MiniLM-L6-v2 cosine sim", "Module": "src/preprocessing.py"},
        {"Practical": "Practical 03", "Topic": "Sentiment & Rhetorical Tone", "Implementation": "Hugging Face DistilBERT fine-tuned on SST-2 classifying research evidence chunks", "Module": "src/nlp_analyzers.py"},
        {"Practical": "Practical 04", "Topic": "Named Entity Recognition (NER)", "Implementation": "spaCy en_core_web_sm extracting PERSON, ORG, GPE, DATE, LAW entities", "Module": "src/nlp_analyzers.py"},
        {"Practical": "Practical 05", "Topic": "Topic Modeling", "Implementation": "Gensim Latent Dirichlet Allocation (LDA) with 5 topics and Gibbs sampling", "Module": "src/nlp_analyzers.py"},
        {"Practical": "Practical 06", "Topic": "Vector DB & Agentic RAG", "Implementation": "Persistent ChromaDB vector store, live arXiv ingestion & 4-Agent LangGraph workflow", "Module": "src/vector_db.py & src/graph.py"}
    ]
    st.table(pd.DataFrame(syllabus_data))

    st.markdown("---")
    st.markdown("#### 👥 Team Details")
    st.markdown("- **Sameer Sirsath** (PRN: `202401110023`)")
    st.markdown("- **Rameshwar Sanap** (PRN: `202401110021`)")
    st.markdown("- **Institution:** Department of Artificial Intelligence & Data Science")
    st.markdown("- **Course:** Natural Language Processing (NLP)")
