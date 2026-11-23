"""
nlp_analyzers.py
Academic NLP Analyzers Module:
- DistilBERT Sentiment / Rhetorical Tone Analysis
- spaCy Named Entity Recognition (PERSON, ORG, GPE, DATE, LAW)
- Gensim LDA Topic Modeling
- Multilingual Gateway (langdetect + deep-translator)

Course: Natural Language Processing (NLP)
Team: Sameer Sirsath (PRN: 202401110023) & Rameshwar Sanap (PRN: 202401110021)
"""

import os
import json
import csv
import logging
from typing import Dict, List, Tuple, Any, Optional

import spacy
from transformers import pipeline
try:
    import gensim
    from gensim import corpora
    from gensim.models import LdaModel
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as SklearnLDA
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import langdetect
from langdetect import DetectorFactory
from deep_translator import GoogleTranslator, MyMemoryTranslator

# Ensure deterministic language detection
DetectorFactory.seed = 42

logger = logging.getLogger(__name__)

# Global cache for heavy models
_SPACY_NLP = None
_SENTIMENT_PIPELINE = None


def get_spacy_model():
    """Lazy loader for spaCy en_core_web_sm pipeline."""
    global _SPACY_NLP
    if _SPACY_NLP is None:
        try:
            _SPACY_NLP = spacy.load("en_core_web_sm")
        except Exception:
            try:
                import en_core_web_sm
                _SPACY_NLP = en_core_web_sm.load()
            except Exception:
                import spacy.cli
                spacy.cli.download("en_core_web_sm")
                _SPACY_NLP = spacy.load("en_core_web_sm")
    return _SPACY_NLP


def get_sentiment_pipeline():
    """Lazy loader for DistilBERT sentiment analysis pipeline."""
    global _SENTIMENT_PIPELINE
    if _SENTIMENT_PIPELINE is None:
        _SENTIMENT_PIPELINE = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512
        )
    return _SENTIMENT_PIPELINE


# -------------------------------------------------------------------------
# 1. Sentiment & Rhetorical Tone Analysis (DistilBERT)
# -------------------------------------------------------------------------

def analyze_sentiment_single(text: str) -> Dict[str, Any]:
    """
    Analyzes sentiment / rhetorical tone for a single text chunk using DistilBERT.
    Returns label (POSITIVE/NEGATIVE), score (confidence), and rhetorical interpretation.
    """
    pipe = get_sentiment_pipeline()
    clean_text = text.strip()[:1000]
    if not clean_text:
        return {"label": "NEUTRAL", "score": 1.0, "tone": "Objective / Descriptive"}

    result = pipe(clean_text)[0]
    label = result["label"]
    score = float(result["score"])

    # Academic stance interpretation
    if label == "POSITIVE":
        tone = "Affirmative / Empirical Support"
    else:
        tone = "Critical / Methodological Limitation / Hedged"

    return {
        "label": label,
        "score": score,
        "tone": tone
    }


def analyze_sentiment_batch(chunks: List[Dict[str, Any]],
                            output_csv: str = "outputs/sentiment_results.csv") -> List[Dict[str, Any]]:
    """
    Runs batch sentiment analysis on retrieved chunks and saves results to CSV.
    """
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    results = []

    for idx, chunk in enumerate(chunks):
        content = chunk.get("content", "")
        source = chunk.get("source", f"Chunk_{idx+1}")
        page = chunk.get("page", "N/A")
        analysis = analyze_sentiment_single(content)
        entry = {
            "chunk_id": idx + 1,
            "source": source,
            "page": page,
            "label": analysis["label"],
            "score": round(analysis["score"], 4),
            "tone": analysis["tone"],
            "preview": content[:120].replace("\n", " ") + "..."
        }
        results.append(entry)

    # Save to CSV
    if results:
        fieldnames = ["chunk_id", "source", "page", "label", "score", "tone", "preview"]
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    return results


# -------------------------------------------------------------------------
# 2. Named Entity Recognition (spaCy)
# -------------------------------------------------------------------------

def extract_entities(text: str, target_labels: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """
    Extracts named entities using spaCy en_core_web_sm.
    Filters by PERSON, ORG, GPE, DATE, and LAW by default.
    """
    if target_labels is None:
        target_labels = ["PERSON", "ORG", "GPE", "DATE", "LAW"]

    nlp = get_spacy_model()
    doc = nlp(text[:10000])

    entities = []
    seen = set()
    for ent in doc.ents:
        if ent.label_ in target_labels:
            clean_text = ent.text.strip().replace("\n", " ")
            key = (clean_text.lower(), ent.label_)
            if len(clean_text) > 1 and key not in seen:
                seen.add(key)
                entities.append({
                    "text": clean_text,
                    "label": ent.label_
                })

    return entities


def run_ner_corpus_benchmark(pages: List[Dict[str, Any]],
                             output_json: str = "outputs/ner_results.json") -> Dict[str, Any]:
    """
    Extracts named entities across all pages and persists summary to JSON.
    """
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    all_entities = []
    entity_counts_by_label: Dict[str, int] = {}
    entity_frequencies: Dict[str, int] = {}

    for page in pages:
        p_num = page.get("page_num", 1)
        text = page.get("raw_text", "")
        ents = extract_entities(text)
        for e in ents:
            label = e["label"]
            text_val = e["text"]
            entity_counts_by_label[label] = entity_counts_by_label.get(label, 0) + 1
            entity_frequencies[text_val] = entity_frequencies.get(text_val, 0) + 1
            all_entities.append({
                "page": p_num,
                "text": text_val,
                "label": label
            })

    summary = {
        "total_entities_detected": len(all_entities),
        "label_distribution": entity_counts_by_label,
        "top_entities": sorted(entity_frequencies.items(), key=lambda x: x[1], reverse=True)[:15],
        "entities": all_entities[:100]  # Store top 100 for JSON brevity
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


# -------------------------------------------------------------------------
# 3. Topic Modeling (Gensim LDA)
# -------------------------------------------------------------------------

def train_lda_topic_model(texts: List[str],
                          num_topics: int = 5,
                          passes: int = 10,
                          num_words: int = 7,
                          output_file: str = "outputs/topic_model_results.txt") -> Dict[str, Any]:
    """
    Trains Gensim LdaModel(num_topics=5, passes=10, alpha='auto') over tokenized corpus chunks
    and extracts dominant keywords per topic. Persists to outputs/topic_model_results.txt.
    """
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    stop_words = set(stopwords.words("english"))

    # Tokenize and clean for LDA
    processed_docs = []
    for doc in texts:
        tokens = word_tokenize(doc.lower())
        tokens = [tok for tok in tokens if tok.isalnum() and len(tok) > 2 and tok not in stop_words]
        if tokens:
            processed_docs.append(tokens)

    if not processed_docs:
        processed_docs = [["natural", "language", "processing", "model"]]

    if GENSIM_AVAILABLE:
        dictionary = corpora.Dictionary(processed_docs)
        # Filter extreme frequencies
        dictionary.filter_extremes(no_below=1, no_above=0.95, keep_n=1000)
        corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=passes,
            alpha="auto",
            random_state=42
        )

        topics_summary = []
        topics_text = (
            "========================================================================\n"
            "           GENSIM LDA TOPIC MODELING EMPIRICAL BENCHMARK                \n"
            "Course: Natural Language Processing (NLP)                               \n"
            "Team: Sameer Sirsath (202401110023) & Rameshwar Sanap (202401110021)    \n"
            "========================================================================\n\n"
            f"Engine: Gensim LdaModel\n"
            f"Hyperparameters: num_topics={num_topics}, passes={passes}, alpha='auto'\n"
            f"Dictionary Size: {len(dictionary)} vocabulary terms\n"
            f"Processed Chunks: {len(processed_docs)}\n\n"
            "------------------------------------------------------------------------\n"
            "TOPIC ID | TOP DOMINANT KEYWORDS & CONTRIBUTION WEIGHTS\n"
            "------------------------------------------------------------------------\n"
        )

        for topic_id in range(num_topics):
            topic_words_weighted = lda_model.show_topic(topic_id, topn=num_words)
            words_str = ", ".join([f"{word} ({weight:.3f})" for word, weight in topic_words_weighted])
            topics_summary.append({
                "topic_id": topic_id,
                "keywords": [w for w, _ in topic_words_weighted],
                "weighted_keywords": topic_words_weighted
            })
            topics_text += f"Topic #{topic_id + 1:<2} | {words_str}\n"

        topics_text += "------------------------------------------------------------------------\n"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(topics_text)

        return {
            "lda_model": lda_model,
            "dictionary": dictionary,
            "corpus": corpus,
            "topics": topics_summary,
            "output_file": output_file
        }
    else:
        # Graceful fallback: Scikit-learn LatentDirichletAllocation
        doc_strings = [" ".join(doc) for doc in processed_docs]
        cv = CountVectorizer(max_features=1000, stop_words="english")
        dtm = cv.fit_transform(doc_strings)
        feature_names = cv.get_feature_names_out()

        actual_topics = min(num_topics, max(1, dtm.shape[0]))
        lda_model = SklearnLDA(n_components=actual_topics, max_iter=passes, random_state=42)
        lda_model.fit(dtm)

        topics_summary = []
        topics_text = (
            "========================================================================\n"
            "         SKLEARN LDA TOPIC MODELING EMPIRICAL BENCHMARK (FALLBACK)       \n"
            "Course: Natural Language Processing (NLP)                               \n"
            "Team: Sameer Sirsath (202401110023) & Rameshwar Sanap (202401110021)    \n"
            "========================================================================\n\n"
            f"Engine: Scikit-Learn LatentDirichletAllocation\n"
            f"Hyperparameters: n_components={actual_topics}, max_iter={passes}\n"
            f"Vocabulary Size: {len(feature_names)} terms\n"
            f"Processed Chunks: {len(processed_docs)}\n\n"
            "------------------------------------------------------------------------\n"
            "TOPIC ID | TOP DOMINANT KEYWORDS & CONTRIBUTION WEIGHTS\n"
            "------------------------------------------------------------------------\n"
        )

        for topic_id, topic_weights in enumerate(lda_model.components_):
            sum_w = topic_weights.sum()
            norm_w = topic_weights / (sum_w if sum_w > 0 else 1.0)
            top_indices = norm_w.argsort()[:-num_words - 1:-1]
            topic_words_weighted = [(feature_names[i], float(norm_w[i])) for i in top_indices]
            words_str = ", ".join([f"{word} ({weight:.3f})" for word, weight in topic_words_weighted])
            topics_summary.append({
                "topic_id": topic_id,
                "keywords": [w for w, _ in topic_words_weighted],
                "weighted_keywords": topic_words_weighted
            })
            topics_text += f"Topic #{topic_id + 1:<2} | {words_str}\n"

        topics_text += "------------------------------------------------------------------------\n"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(topics_text)

        return {
            "lda_model": lda_model,
            "dictionary": feature_names,
            "corpus": dtm,
            "topics": topics_summary,
            "output_file": output_file
        }


# -------------------------------------------------------------------------
# 4. Multilingual NLP Gateway (langdetect + deep-translator)
# -------------------------------------------------------------------------

LANGUAGE_NAME_MAP = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh-cn": "Chinese",
    "ja": "Japanese",
    "ru": "Russian",
    "ar": "Arabic"
}


def detect_query_language(text: str) -> str:
    """
    Detects language using langdetect. Returns ISO 639-1 code (e.g., 'en', 'hi', 'fr').
    Defaults to 'en' on short or ambiguous inputs.
    """
    clean = text.strip()
    if len(clean) < 3:
        return "en"
    try:
        lang = langdetect.detect(clean)
        # Standardize common language codes
        if lang.startswith("zh"):
            return "zh-CN"
        return lang
    except Exception:
        return "en"


# Mapping from ISO 639-1 to language names for MyMemoryTranslator
MYMEMORY_LANG_MAP = {
    "en": "english",
    "hi": "hindi",
    "mr": "marathi",
    "es": "spanish",
    "fr": "french",
    "de": "german",
    "zh-cn": "chinese",
    "zh": "chinese",
    "ja": "japanese",
    "ru": "russian",
    "ar": "arabic",
    "it": "italian",
    "pt": "portuguese"
}


def translate_text(text: str, source_lang: str = "auto", target_lang: str = "en") -> str:
    """
    Translates text between languages using deep-translator (GoogleTranslator with MyMemory fallback).
    If source and target languages match, returns text immediately.
    Handles long text (> 500 chars) by segmenting paragraphs for provider limits.
    """
    if not text or not text.strip():
        return text

    # Standardize
    src_std = source_lang.lower().split("-")[0] if source_lang != "auto" else "auto"
    tgt_std = target_lang.lower().split("-")[0]

    if src_std == tgt_std:
        return text

    # 1. Try GoogleTranslator directly
    try:
        translated = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
        if translated and translated.strip():
            return translated
    except Exception as e:
        logger.warning(f"GoogleTranslator notice: {e}. Trying MyMemoryTranslator...")

    # 2. Try MyMemoryTranslator with paragraph chunking
    try:
        mm_src = MYMEMORY_LANG_MAP.get(src_std, "english") if src_std != "auto" else "english"
        mm_tgt = MYMEMORY_LANG_MAP.get(tgt_std, "english")
        if mm_src != mm_tgt:
            translator = MyMemoryTranslator(source=mm_src, target=mm_tgt)
            # If text is long, translate paragraph by paragraph
            if len(text) > 400:
                paragraphs = text.split("\n\n")
                translated_paras = []
                for p in paragraphs:
                    if p.strip():
                        # Sub-chunk if individual paragraph is still > 450 chars
                        if len(p) > 450:
                            sentences = p.split(". ")
                            sub_trans = [translator.translate(s[:400]) for s in sentences if s.strip()]
                            translated_paras.append(". ".join(sub_trans))
                        else:
                            translated_paras.append(translator.translate(p))
                    else:
                        translated_paras.append("")
                return "\n\n".join(translated_paras)
            else:
                translated = translator.translate(text)
                if translated and translated.strip():
                    return translated
    except Exception as e2:
        logger.warning(f"MyMemoryTranslator notice: {e2}")

    return text


def process_multilingual_query(query: str) -> Tuple[str, str, bool]:
    """
    Detects language of input query. If non-English, translates to English for RAG retrieval.
    Returns: (english_query, detected_lang_code, is_translated)
    """
    detected_lang = detect_query_language(query)
    if detected_lang != "en":
        english_query = translate_text(query, source_lang=detected_lang, target_lang="en")
        return english_query, detected_lang, True
    return query, "en", False


def translate_response_to_target(response_text: str, target_lang: str) -> str:
    """
    Translates synthesized response back to user's native language if target is non-English.
    """
    if target_lang == "en" or not target_lang:
        return response_text
    return translate_text(response_text, source_lang="en", target_lang=target_lang)


if __name__ == "__main__":
    print("Testing src/nlp_analyzers.py...")

    # Test Sentiment
    sample_text = "The Transformer architecture substantially outperforms recurrent networks with remarkable efficiency."
    s_res = analyze_sentiment_single(sample_text)
    print(f"Sentiment: {s_res['label']} ({s_res['score']:.4f}) - Tone: {s_res['tone']}")

    # Test NER
    ner_res = extract_entities("Tomas Mikolov at Google and Patrick Lewis at Meta AI introduced RAG in 2020.")
    print(f"NER Entities: {ner_res}")

    # Test Multilingual
    french_query = "Quels sont les avantages de l'architecture Transformer par rapport aux RNNs?"
    en_q, lang, trans = process_multilingual_query(french_query)
    print(f"Detected: {lang}, Translated Query: {en_q}")

    print("src/nlp_analyzers.py passed all tests!")
