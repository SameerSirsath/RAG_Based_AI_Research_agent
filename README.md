# 🏛️ Agentic RAG-Based AI Research Assistant
> **Academic Mini Project for Natural Language Processing (NLP)**  
> **Authors / Team Members:**
> - **Sameer Sirsath** — PRN: `202401110023`
> - **Rameshwar Sanap** — PRN: `202401110021`  
> **Course:** Natural Language Processing (NLP)  
> **Live Interface:** Deployable on Streamlit Cloud & Hugging Face Spaces

---

## 📌 Executive Summary

The **Agentic RAG-Based AI Research Assistant** is an autonomous multi-agent Natural Language Processing system that bridges classical computational linguistics and modern agentic Retrieval-Augmented Generation (RAG). Built for scholars and researchers, the assistant ingests academic literature (PDF textbooks and live arXiv papers), runs active classical and deep NLP analysis pipelines as executable agent tools, and synthesizes verifiable, citation-grounded scholarly briefs through an interactive multi-tab web interface.

Unlike rigid linear RAG pipelines, our system implements a **cyclic 4-agent LangGraph workflow** featuring dynamic intent classification, multilingual translation gateways, self-correcting retrieval fallbacks, rhetorical stance analysis, and automated citation auditing.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    User([User Query - Any Language]) --> MLG[Multilingual Gateway: langdetect]
    MLG --> IntentDecider{Intent Classification}
    
    IntentDecider -->|Conversational Greeting| WritingAgent
    IntentDecider -->|Academic Inquiry| ResearchAgent[Research Agent]
    
    subgraph MultiAgentEngine [Cyclic LangGraph 4-Agent Swarm]
        ResearchAgent --> HybridSearch[(ChromaDB + TF-IDF Hybrid Search)]
        HybridSearch --> RelevanceCheck{Score >= 0.15?}
        
        RelevanceCheck -->|Insufficient Context| ArxivIngest[Live arXiv Fetcher & Ingestion]
        ArxivIngest --> HybridSearch
        
        RelevanceCheck -->|Sufficient Context| AnalysisAgent[Analysis Agent]
        
        AnalysisAgent --> SpacyNER[spaCy NER Tool: PERSON, ORG, DATE, LAW]
        AnalysisAgent --> DistilBERTTone[DistilBERT Rhetorical Stance Tool]
        
        SpacyNER --> WritingAgent[Writing Agent]
        DistilBERTTone --> WritingAgent
        
        WritingAgent --> ProofreaderAgent[Proofreader Agent]
        ProofreaderAgent --> CitationAudit{Verify [Page X] / [arXiv: ID]}
        CitationAudit --> ProvenanceBuilder[Build Provenance & Entity Badges]
        ProvenanceBuilder --> TargetTranslator[Translate Back to Native Language]
    end
    
    TargetTranslator --> FinalAnswer([Verified Scholarly Response + Research Brief])
```

---

## 📋 Syllabus Mapping & NLP Practical Matrix

Every component of this project directly implements and satisfies the requirements of the academic Natural Language Processing syllabus:

| Practical | Module / Pipeline | Scientific Methodology | Empirical Artifacts Generated |
| :--- | :--- | :--- | :--- |
| **Practical 01** | `src/preprocessing.py` | Page-by-page text extraction (`PyMuPDF`), NLTK word tokenization, alphanumeric filtering, stopword pruning, and `PorterStemmer` rule-based stemming. | `outputs/preprocessing_analysis.txt` |
| **Practical 02** | `src/preprocessing.py` | High-dimensional sparse TF-IDF matrix generation (`TfidfVectorizer`, 2500 features) vs. dense continuous embeddings (`all-MiniLM-L6-v2`) with pairwise cosine similarity. | `outputs/tfidf_matrix.pkl` |
| **Practical 03** | `src/nlp_analyzers.py` | Transformer fine-tuning via Hugging Face `pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")` for academic stance classification. | `outputs/sentiment_results.csv` |
| **Practical 04** | `src/nlp_analyzers.py` | Information extraction with `spaCy` (`en_core_web_sm`) extracting named entities (`PERSON`, `ORG`, `GPE`, `DATE`, `LAW`) across corpus chapters. | `outputs/ner_results.json` |
| **Practical 05** | `src/nlp_analyzers.py` | Unsupervised probabilistic topic modeling using `Gensim` `LdaModel` (5 topics, 10 passes, Dirichlet priors) with Gibbs sampling keyword weights. | `outputs/topic_model_results.txt` |
| **Practical 06** | `src/vector_db.py` & `src/graph.py` | Persistent vector store (`ChromaDB`, HNSW index), hybrid dense/sparse retrieval, live arXiv ingestion, and cyclic 4-agent LangGraph state machine. | Interactive Web App & Executed Benchmark Notebook |

---

## 🤖 4-Agent LangGraph Swarm Specification

1. **Research Agent**:
   - Detects query language via `langdetect` and routes to `deep-translator` if non-English.
   - Evaluates user intent (`conversational` vs. `academic_research`).
   - Queries ChromaDB using **Hybrid Retrieval** (dense cosine distance + sparse TF-IDF keyword overlap).
   - **Self-Correction Trigger:** If local context is missing or relevance score falls below threshold (`0.15`), it dynamically invokes `arxiv` API, downloads the relevant papers into `data/arxiv_cache/`, chunks and embeds them into ChromaDB, and re-queries.

2. **Analysis Agent**:
   - Executes `spaCy` NER to detect researchers, institutions, dates, and legal frameworks across retrieved chunks.
   - Runs `DistilBERT` sentiment/rhetorical stance analysis on each retrieved passage, categorizing claims as affirmative empirical evidence or critical methodological limitations.

3. **Writing Agent**:
   - Operates under strict academic prompt engineering.
   - Grounds syntheses exclusively in retrieved context passages.
   - Explicitly cites `[Page X]` or `[arXiv: ID]` for all technical assertions.

4. **Proofreader Agent**:
   - Audits draft text to ensure no ungrounded claims or missing citations exist.
   - Formats the **Academic Provenance Block** with entity badges and rhetorical stance indicators.
   - Automatically translates the finalized scholarly answer back into the user's detected native language.

---

## 📁 Repository Directory Structure

```text
agentic-rag-research-assistant/
├── app.py                                # Main Streamlit web application (cloud + local deployable)
├── ingest_references.py                  # CLI utility to ingest reference academic PDFs into ChromaDB
├── requirements.txt                      # Pinned Python dependencies
├── .env.example                          # Configuration template for cloud LLM API keys
├── README.md                             # Comprehensive project documentation
├── data/
│   ├── sample_book.pdf                   # 15-page academic NLP textbook corpus (auto-generated if missing)
│   ├── references/                       # Directory for custom scholarly reference PDFs
│   └── arxiv_cache/                      # Cache directory for dynamically downloaded arXiv PDFs
├── outputs/                              # Empirical artifacts generated by NLP pipelines
│   ├── preprocessing_analysis.txt       # NLTK tokenization & stemming reduction report
│   ├── tfidf_matrix.pkl                 # Persisted Scikit-Learn TF-IDF model & matrix
│   ├── sentiment_results.csv            # DistilBERT rhetorical stance classifications
│   ├── ner_results.json                 # spaCy named entity extractions & label counts
│   └── topic_model_results.txt          # Gensim LDA 5-topic keyword distributions
├── chroma_db/                            # Persistent ChromaDB vector store
├── src/
│   ├── __init__.py                       # Package initializer with project metadata
│   ├── preprocessing.py                  # PyMuPDF text extraction, NLTK preprocessing & TF-IDF
│   ├── nlp_analyzers.py                  # DistilBERT Sentiment, spaCy NER, Gensim LDA, Multilingual
│   ├── vector_db.py                      # Recursive chunking, ChromaDB indexing, hybrid search & ingestion
│   ├── agents.py                         # 4 LangGraph agents & hybrid LLM factory (Groq/Gemini/Ollama)
│   └── graph.py                          # Cyclic LangGraph StateGraph workflow & Markdown generator
└── notebooks/
    └── NLP_Practicals_Benchmark.ipynb    # Clean Jupyter notebook executing Practicals 01-06
```

---

## 🔐 Backend API Key Configuration

To ensure maximum security and academic presentation standards, API keys are **never exposed in the web interface**. They are read securely on the backend via environment variables or secret vaults.

### Option A: Local `.env` File (Recommended)
1. Copy the example configuration template:
   ```bash
   # Windows PowerShell
   copy .env.example .env

   # Linux / macOS
   cp .env.example .env
   ```
2. Open `.env` in any text editor and paste your credentials:
   ```ini
   # For ultra-fast Groq LLaMA 3.1 inference:
   GROQ_API_KEY=gsk_your_groq_api_key_here

   # OR for Google Gemini 2.5 Flash:
   GOOGLE_API_KEY=AIzaSy_your_gemini_api_key_here

   # Optional: customize local Ollama model (default: llama3.2:1b)
   OLLAMA_MODEL=llama3.2:1b
   ```
3. When you run `streamlit run app.py`, the backend automatically loads these keys via `python-dotenv` and switches to the fastest available cloud LLM!

### Option B: Terminal / Environment Variable Export
- **Windows (PowerShell):**
  ```powershell
  $env:GROQ_API_KEY="gsk_your_groq_api_key_here"
  # or
  $env:GOOGLE_API_KEY="AIzaSy_your_gemini_api_key_here"
  streamlit run app.py
  ```
- **Linux / macOS (Bash):**
  ```bash
  export GROQ_API_KEY="gsk_your_groq_api_key_here"
  # or
  export GOOGLE_API_KEY="AIzaSy_your_gemini_api_key_here"
  streamlit run app.py
  ```

### Option C: Streamlit Community Cloud Deployment
In your Streamlit App dashboard:
1. Go to **Settings** -> **Secrets**.
2. Add:
   ```toml
   GROQ_API_KEY = "gsk_your_groq_api_key_here"
   # OR
   GOOGLE_API_KEY = "AIzaSy_your_gemini_api_key_here"
   ```
3. Save; the application accesses them via `st.secrets` without exposing keys to users.

### Option D: Hugging Face Spaces Deployment
In your Space repository:
1. Go to **Settings** -> **Variables and secrets**.
2. Click **New secret** and add `GROQ_API_KEY` or `GOOGLE_API_KEY`.

---

## 📚 Adding Reference PDFs into the Vector Store

To supply additional context, literature surveys, textbooks, or research papers, use the included ingestion script:

### 1. Ingest All PDFs from `data/references/`
Simply drop any academic PDFs into `data/references/` and run:
```bash
python ingest_references.py
```

### 2. Ingest a Specific PDF File
```bash
python ingest_references.py --file path/to/any_paper.pdf
```

### 3. Ingest All PDFs from a Custom Directory
```bash
python ingest_references.py --dir D:/my_research_papers/
```

### 4. List All Indexed Sources & Database Statistics
```bash
python ingest_references.py --list
```

### 5. Generate and Ingest a Sample Literature Survey PDF
```bash
python ingest_references.py --create-sample
```
*(Extracts, chunks, embeds into ChromaDB, and updates the TF-IDF hybrid lexical index immediately without restarting the application!)*

---

## 🚀 Installation & Local Execution

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/your-username/agentic-rag-research-assistant.git
cd agentic-rag-research-assistant

# Create virtual environment
python -m venv venv
# Activate on Windows:
venv\Scripts\activate
# Activate on Linux/macOS:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure LLM Backend (Hybrid Cloud + Local)
The system automatically adapts to available compute:
- **Local (Ollama - Zero Cost, Fully Offline):**
  Ensure [Ollama](https://ollama.ai) is running locally with LLaMA 3.2:
  ```bash
  ollama run llama3.2:1b
  ```
- **Cloud (Groq or Google Gemini - Ultra Fast):**
  Create a `.env` file based on `.env.example`:
  ```bash
  copy .env.example .env
  ```
  Add your API key:
  ```ini
  GROQ_API_KEY=gsk_your_groq_api_key_here
  # OR
  GOOGLE_API_KEY=AIzaSy_your_gemini_api_key_here
  ```

### 3. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ Deployment Guide (Hugging Face Spaces / Streamlit Cloud)

### Deploying to Streamlit Community Cloud
1. Push repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io).
3. Select your repository, branch `main`, and main file `app.py`.
4. In **Advanced Settings -> Secrets**, add:
   ```toml
   GROQ_API_KEY = "gsk_..."
   # OR
   GOOGLE_API_KEY = "AIzaSy_..."
   ```
5. Click **Deploy!**

### Deploying to Hugging Face Spaces
1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces) with SDK: **Streamlit**.
2. Push your files to the Space repository.
3. In Space **Settings -> Variables and secrets**, add `GROQ_API_KEY` or `GOOGLE_API_KEY`.
4. The application will start and build the ChromaDB index automatically using CPU-friendly `all-MiniLM-L6-v2` embeddings.

---

## 📊 Sample Empirical Outputs

### 1. Morphological Preprocessing Report (`outputs/preprocessing_analysis.txt`)
```text
METRIC                   | RAW CORPUS       | CLEANED (STEMMED) | REDUCTION (%)
------------------------------------------------------------------------
Character Count          | 20813            | 13019             | 37.45%
Token Count              | 3535             | 1918              | 45.74%
Unique Vocabulary Size   | 1297             | 852               | 34.31%
```

### 2. Scholarly Query Execution Sample
**Query:** *"What is the Self-Attention mechanism in Transformers?"*
```markdown
**Executive Summary**
The Self-Attention mechanism in Transformers replaces sequential recurrence with a parallelized mathematical projection [Page 7]. 

**Key Technical Principles**
1. Scaled Dot-Product Attention:
   Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V
   The scaling factor sqrt(d_k) prevents dot products from pushing the softmax function into regions with vanishing gradients [Page 7].
2. Multi-Head Projections:
   Queries, Keys, and Values are projected h times into distinct representation subspaces [Page 7].

---
### 🏛️ Academic Provenance & Verification Block
- Source Documents Cited: `sample_book.pdf` (Page 7)
- Citation Audit: ✅ Verified (Explicit Page/arXiv Citations Present)
- Key Extracted Entities (spaCy): Vaswani et al. (PERSON), Transformer (ORG), Multi-Head Attention (ORG)
- Rhetorical Stance Distribution (DistilBERT): Positive/Affirmative (4), Critical/Hedged (0)
- System Attribution: Sameer Sirsath & Rameshwar Sanap (NLP Mini Project)
```

---

## 👥 Contributors & Academic Integrity
- **Sameer Sirsath** (PRN: `202401110023`)
- **Rameshwar Sanap** (PRN: `202401110021`)
- **Department:** Artificial Intelligence & Data Science  
- **Course:** Natural Language Processing (NLP) Mini Project
