"""
Módulo central de Procesamiento de Lenguaje Natural (NLP) para Similarity Lab.
Incluye preprocesamiento de texto, similitud de Jaccard, similitud del Coseno (TF-IDF),
similitud por Embeddings Semánticos Híbridos Locales (LSA TruncatedSVD + Contexto + FastText, 0 Tokens),
escaneo profundo de palabras repetidas y contexto oración a oración,
extracción de archivos (PDF, DOCX, TXT) y análisis estadístico.
"""

from .preprocessing import (
    SPANISH_STOPWORDS,
    normalize_text,
    tokenize,
    identify_stopwords,
    remove_stopwords_from_tokens,
    preprocess_text,
    preprocess_document,
)
from .jaccard import calculate_jaccard_similarity
from .cosine import calculate_cosine_similarity, calculate_manual_cosine_similarity
from .embeddings import (
    calculate_embedding_similarity,
    analyze_sentence_context_alignment,
    extract_kwic_contextual_windows,
)
from .file_reader import extract_text_from_file, ALLOWED_EXTENSIONS
from .profiles import (
    COMPARISON_PROFILES,
    compare_profiles_side_by_side,
    get_profile_config,
    normalize_profile_name,
    preprocess_reference_minimal,
    run_profile_comparison,
)
from .statistics import (
    generate_statistical_analysis,
    generate_interpretation,
    analyze_word_and_phrase_repetitions,
    analyze_context_domains_and_discourse,
)

__all__ = [
    "SPANISH_STOPWORDS",
    "normalize_text",
    "tokenize",
    "identify_stopwords",
    "remove_stopwords_from_tokens",
    "preprocess_text",
    "preprocess_document",
    "calculate_jaccard_similarity",
    "calculate_cosine_similarity",
    "calculate_manual_cosine_similarity",
    "calculate_embedding_similarity",
    "analyze_sentence_context_alignment",
    "extract_kwic_contextual_windows",
    "extract_text_from_file",
    "ALLOWED_EXTENSIONS",
    "COMPARISON_PROFILES",
    "get_profile_config",
    "normalize_profile_name",
    "preprocess_reference_minimal",
    "run_profile_comparison",
    "compare_profiles_side_by_side",
    "generate_statistical_analysis",
    "generate_interpretation",
    "analyze_word_and_phrase_repetitions",
    "analyze_context_domains_and_discourse",
]
