# 📚 Academic Literature & Reference Corpus Index
## Foundational Books & Papers in AI/ML, Deep Learning, and Natural Language Processing

> **Project:** Agentic RAG-Based AI Research Assistant  
> **Course:** Natural Language Processing (NLP)  
> **Authors:** Sameer Sirsath (PRN: `202401110023`) & Rameshwar Sanap (PRN: `202401110021`)  
> **Generated:** 2026-11-23 22:26:36  

---

### 📌 Overview
This repository incorporates a curated collection of **15 seminal academic foundation papers and textbooks** downloaded directly from official open-access academic repositories (arXiv / Open Science). All references are indexed into the persistent **ChromaDB vector store** using `sentence-transformers/all-MiniLM-L6-v2` embeddings and balanced with sparse **TF-IDF lexical search**, enabling the multi-agent swarm to answer questions grounded in verified scientific literature.

---

## 📑 Curated Literature Matrix

| # | Paper / Book Title | Domain | Year | Authors | Direct PDF Link |
| :-: | :--- | :--- | :-: | :--- | :--- |
| 1 | **Attention Is All You Need** | `NLP / Deep Learning` | 2017 | Ashish Vaswani, Noam Shazeer, Niki Parmar, Ja... | [Download PDF](https://arxiv.org/pdf/1706.03762.pdf) |
| 2 | **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** | `NLP / Foundation Models` | 2018 | Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kri... | [Download PDF](https://arxiv.org/pdf/1810.04805.pdf) |
| 3 | **Language Models are Few-Shot Learners (GPT-3)** | `LLMs / Generative AI` | 2020 | Tom B. Brown et al. (OpenAI)... | [Download PDF](https://arxiv.org/pdf/2005.14165.pdf) |
| 4 | **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG)** | `Agentic NLP / RAG` | 2020 | Patrick Lewis, Ethan Perez, Aleksandara Piktu... | [Download PDF](https://arxiv.org/pdf/2005.11401.pdf) |
| 5 | **Deep Residual Learning for Image Recognition (ResNet)** | `Deep Learning / Computer Vision` | 2015 | Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian... | [Download PDF](https://arxiv.org/pdf/1512.03385.pdf) |
| 6 | **LoRA: Low-Rank Adaptation of Large Language Models** | `PEFT / Efficient Deep Learning` | 2021 | Edward J. Hu, Yelong Shen, Phillip Wallis, Ze... | [Download PDF](https://arxiv.org/pdf/2106.09685.pdf) |
| 7 | **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** | `Prompting / LLM Reasoning` | 2022 | Jason Wei, Xuezhi Wang, Dale Schuurmans, Maar... | [Download PDF](https://arxiv.org/pdf/2201.11903.pdf) |
| 8 | **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** | `Hardware Acceleration / Deep Learning` | 2022 | Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Ru... | [Download PDF](https://arxiv.org/pdf/2205.14135.pdf) |
| 9 | **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks** | `Dense Semantic Embeddings` | 2019 | Nils Reimers, Iryna Gurevych (UKP Lab, TU Dar... | [Download PDF](https://arxiv.org/pdf/1908.10084.pdf) |
| 10 | **Direct Preference Optimization: Your Language Model is Secretly a Reward Model** | `LLM Alignment / RLHF` | 2023 | Rafael Rafailov, Archit Sharma, Eric Mitchell... | [Download PDF](https://arxiv.org/pdf/2305.18290.pdf) |
| 11 | **DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter** | `Model Compression / NLP` | 2019 | Victor Sanh, Lysandre Debut, Julien Chaumond,... | [Download PDF](https://arxiv.org/pdf/1910.01108.pdf) |
| 12 | **LLaMA: Open and Efficient Foundation Language Models** | `Foundation Models / Open LLMs` | 2023 | Hugo Touvron et al. (Meta AI)... | [Download PDF](https://arxiv.org/pdf/2302.13971.pdf) |
| 13 | **Adam: A Method for Stochastic Optimization** | `Optimization / Machine Learning` | 2014 | Diederik P. Kingma, Jimmy Ba (University of A... | [Download PDF](https://arxiv.org/pdf/1412.6980.pdf) |
| 14 | **A Survey of Large Language Models** | `AI/ML & NLP Comprehensive Survey` | 2023 | Wayne Xin Zhao et al. (Renmin University / Un... | [Download PDF](https://arxiv.org/pdf/2303.18223.pdf) |
| 15 | **Deep Learning: An Introduction for Applied Mathematicians** | `Foundations of AI/ML & Deep Learning` | 2018 | Catherine F. Higham, Desmond J. Higham (Unive... | [Download PDF](https://arxiv.org/pdf/1801.05894.pdf) |

---

## 📖 Detailed Annotations & Scientific Summaries

### 1. Attention Is All You Need (2017)
- **Domain:** `NLP / Deep Learning`
- **Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
- **Direct Download Link:** [https://arxiv.org/pdf/1706.03762.pdf](https://arxiv.org/pdf/1706.03762.pdf)
- **Local Filename:** `vaswani2017_attention_is_all_you_need.pdf`
- **Significance & Abstract Summary:**
  > The seminal paper that introduced the Transformer architecture, replacing recurrent connections with Scaled Dot-Product and Multi-Head Self-Attention mechanisms.

### 2. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018)
- **Domain:** `NLP / Foundation Models`
- **Authors:** Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova (Google AI)
- **Direct Download Link:** [https://arxiv.org/pdf/1810.04805.pdf](https://arxiv.org/pdf/1810.04805.pdf)
- **Local Filename:** `devlin2018_bert.pdf`
- **Significance & Abstract Summary:**
  > Introduced bidirectional contextual representations via Masked Language Modeling (MLM) and Next Sentence Prediction (NSP), transforming modern NLP.

### 3. Language Models are Few-Shot Learners (GPT-3) (2020)
- **Domain:** `LLMs / Generative AI`
- **Authors:** Tom B. Brown et al. (OpenAI)
- **Direct Download Link:** [https://arxiv.org/pdf/2005.14165.pdf](https://arxiv.org/pdf/2005.14165.pdf)
- **Local Filename:** `brown2020_gpt3_few_shot_learners.pdf`
- **Significance & Abstract Summary:**
  > Demonstrated that scaling autoregressive transformers to 175 billion parameters yields emergent in-context few-shot learning without gradient updates.

### 4. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG) (2020)
- **Domain:** `Agentic NLP / RAG`
- **Authors:** Patrick Lewis, Ethan Perez, Aleksandara Piktus, Fabio Petroni, Vladimir Karpukhin et al. (Meta AI / UCL / NYU)
- **Direct Download Link:** [https://arxiv.org/pdf/2005.11401.pdf](https://arxiv.org/pdf/2005.11401.pdf)
- **Local Filename:** `lewis2020_rag.pdf`
- **Significance & Abstract Summary:**
  > Foundational paper introducing the Retrieval-Augmented Generation paradigm, marrying dense neural retrieval with sequence-to-sequence generators.

### 5. Deep Residual Learning for Image Recognition (ResNet) (2015)
- **Domain:** `Deep Learning / Computer Vision`
- **Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft Research)
- **Direct Download Link:** [https://arxiv.org/pdf/1512.03385.pdf](https://arxiv.org/pdf/1512.03385.pdf)
- **Local Filename:** `he2015_resnet_deep_residual_learning.pdf`
- **Significance & Abstract Summary:**
  > Introduced residual skip connections to resolve the degradation problem in extremely deep neural networks (152 layers).

### 6. LoRA: Low-Rank Adaptation of Large Language Models (2021)
- **Domain:** `PEFT / Efficient Deep Learning`
- **Authors:** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (Microsoft)
- **Direct Download Link:** [https://arxiv.org/pdf/2106.09685.pdf](https://arxiv.org/pdf/2106.09685.pdf)
- **Local Filename:** `hu2021_lora.pdf`
- **Significance & Abstract Summary:**
  > Parameter-efficient fine-tuning technique freezing pre-trained model weights and injecting trainable rank decomposition matrices into attention layers.

### 7. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (2022)
- **Domain:** `Prompting / LLM Reasoning`
- **Authors:** Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou (Google Research)
- **Direct Download Link:** [https://arxiv.org/pdf/2201.11903.pdf](https://arxiv.org/pdf/2201.11903.pdf)
- **Local Filename:** `wei2022_chain_of_thought.pdf`
- **Significance & Abstract Summary:**
  > Demonstrated that generating a series of intermediate reasoning steps dramatically enhances multi-step problem solving in arithmetic and symbolic tasks.

### 8. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (2022)
- **Domain:** `Hardware Acceleration / Deep Learning`
- **Authors:** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré (Stanford University)
- **Direct Download Link:** [https://arxiv.org/pdf/2205.14135.pdf](https://arxiv.org/pdf/2205.14135.pdf)
- **Local Filename:** `dao2022_flashattention.pdf`
- **Significance & Abstract Summary:**
  > IO-aware exact attention algorithm leveraging GPU SRAM memory tiling to achieve 2x-4x speedups without materializing large attention matrices in HBM.

### 9. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks (2019)
- **Domain:** `Dense Semantic Embeddings`
- **Authors:** Nils Reimers, Iryna Gurevych (UKP Lab, TU Darmstadt)
- **Direct Download Link:** [https://arxiv.org/pdf/1908.10084.pdf](https://arxiv.org/pdf/1908.10084.pdf)
- **Local Filename:** `reimers2019_sentence_bert.pdf`
- **Significance & Abstract Summary:**
  > Modification of pre-trained BERT using Siamese and triplet network structures to derive semantically meaningful sentence embeddings compared via cosine distance.

### 10. Direct Preference Optimization: Your Language Model is Secretly a Reward Model (2023)
- **Domain:** `LLM Alignment / RLHF`
- **Authors:** Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn (Stanford University)
- **Direct Download Link:** [https://arxiv.org/pdf/2305.18290.pdf](https://arxiv.org/pdf/2305.18290.pdf)
- **Local Filename:** `rafailov2023_direct_preference_optimization.pdf`
- **Significance & Abstract Summary:**
  > Replaces complex reinforcement learning from human feedback (RLHF) with a closed-form implicit reward loss function for preference alignment.

### 11. DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter (2019)
- **Domain:** `Model Compression / NLP`
- **Authors:** Victor Sanh, Lysandre Debut, Julien Chaumond, Thomas Wolf (Hugging Face)
- **Direct Download Link:** [https://arxiv.org/pdf/1910.01108.pdf](https://arxiv.org/pdf/1910.01108.pdf)
- **Local Filename:** `sanh2019_distilbert.pdf`
- **Significance & Abstract Summary:**
  > Applies knowledge distillation during pre-training to compress BERT by 40% while preserving 97% of language understanding and accelerating inference by 60%.

### 12. LLaMA: Open and Efficient Foundation Language Models (2023)
- **Domain:** `Foundation Models / Open LLMs`
- **Authors:** Hugo Touvron et al. (Meta AI)
- **Direct Download Link:** [https://arxiv.org/pdf/2302.13971.pdf](https://arxiv.org/pdf/2302.13971.pdf)
- **Local Filename:** `touvron2023_llama.pdf`
- **Significance & Abstract Summary:**
  > Seminal open foundation model suite trained exclusively on publicly available datasets using RMSNorm, SwiGLU activations, and Rotary Positional Embeddings.

### 13. Adam: A Method for Stochastic Optimization (2014)
- **Domain:** `Optimization / Machine Learning`
- **Authors:** Diederik P. Kingma, Jimmy Ba (University of Amsterdam / University of Toronto)
- **Direct Download Link:** [https://arxiv.org/pdf/1412.6980.pdf](https://arxiv.org/pdf/1412.6980.pdf)
- **Local Filename:** `kingma2014_adam_optimizer.pdf`
- **Significance & Abstract Summary:**
  > The dominant first-order gradient-based optimization algorithm computing individual adaptive learning rates from estimates of first and second moments of gradients.

### 14. A Survey of Large Language Models (2023)
- **Domain:** `AI/ML & NLP Comprehensive Survey`
- **Authors:** Wayne Xin Zhao et al. (Renmin University / University of Montreal / AI Box)
- **Direct Download Link:** [https://arxiv.org/pdf/2303.18223.pdf](https://arxiv.org/pdf/2303.18223.pdf)
- **Local Filename:** `zhao2023_survey_large_language_models.pdf`
- **Significance & Abstract Summary:**
  > Exhaustive survey exploring the background, evolutionary tree, pre-training, adaptation, utilization, and emergent capabilities of modern Large Language Models.

### 15. Deep Learning: An Introduction for Applied Mathematicians (2018)
- **Domain:** `Foundations of AI/ML & Deep Learning`
- **Authors:** Catherine F. Higham, Desmond J. Higham (University of Edinburgh)
- **Direct Download Link:** [https://arxiv.org/pdf/1801.05894.pdf](https://arxiv.org/pdf/1801.05894.pdf)
- **Local Filename:** `higham2018_deep_learning_primer.pdf`
- **Significance & Abstract Summary:**
  > Rigorous academic introduction to deep neural networks, backpropagation calculus, stochastic gradient descent, cross-entropy, and regularization mathematics.

---

## 🛠️ How to Ingest Additional References
To index additional PDF books or papers into the ChromaDB vector database:
```powershell
# 1. Place any academic PDF into data/references/
# 2. Run the ingestion utility:
python ingest_references.py

# 3. Or inspect the updated database:
python ingest_references.py --list
```

---
*Curated for NLP Academic Research by Sameer Sirsath & Rameshwar Sanap.*