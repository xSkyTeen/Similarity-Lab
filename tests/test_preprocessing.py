"""
Pruebas unitarias para el módulo de preprocesamiento NLP (core/preprocessing.py).
"""

import os
import sys
import unicodedata

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.preprocessing import (
    SPANISH_STOPWORDS,
    clean_pdf_and_web_artifacts,
    identify_stopwords,
    normalize_text,
    preprocess_document,
    preprocess_semantic,
    preprocess_text,
    remove_stopwords_from_tokens,
    split_paragraphs,
    split_sentences,
    stem_word,
    tokenize,
)


def test_lowercase_and_whitespace_normalization():
    raw = "   HOLA    MUNDO\n\tEsto   ES   Una   PRUEBA   "
    assert normalize_text(raw) == "hola mundo esto es una prueba"


def test_unicode_nfc_normalization_and_spanish_characters():
    # Texto en forma descompuesta NFD (tildes y virgulilla separadas)
    raw_nfd = unicodedata.normalize(
        "NFD",
        "¡ Qué mañana tan lingüística! El niño estudió programación, álgebra y análisis.",
    )
    normalized = normalize_text(raw_nfd)
    assert unicodedata.is_normalized("NFC", normalized)
    assert normalized == "qué mañana tan lingüística el niño estudió programación álgebra y análisis"
    tokens = tokenize(raw_nfd)
    assert "mañana" in tokens
    assert "lingüística" in tokens
    assert "niño" in tokens
    assert "estudió" in tokens
    assert "programación" in tokens


def test_unify_accents_option_preserving_enie():
    raw = "Esta mañana el niño está programando rápido en España"
    norm_with_accents = normalize_text(raw, unify_accents=False)
    norm_unified = normalize_text(raw, unify_accents=True)

    assert "está" in norm_with_accents
    assert "rápido" in norm_with_accents
    assert "está" not in norm_unified
    assert "esta" in norm_unified
    assert "rapido" in norm_unified
    # La ñ siempre se conserva intacta
    assert "mañana" in norm_unified
    assert "niño" in norm_unified
    assert "españa" in norm_unified


def test_pdf_hyphenation_urls_and_emails_cleanup():
    raw_pdf = (
        "El proce-\n"
        "samiento de docu-\n"
        "mentos en https://ejemplo.org/test y contacto@dominio.es\n"
        "Página 1 de 10\n"
        "permite tokeni-\n"
        "zación limpia."
    )
    cleaned = clean_pdf_and_web_artifacts(raw_pdf)
    tokens = tokenize(cleaned)
    assert "procesamiento" in tokens
    assert "documentos" in tokens
    assert "tokenización" in tokens
    assert "https" not in tokens
    assert "dominio" not in tokens


def test_morphological_stemming_in_lexical_pipeline():
    assert stem_word("tokenización") == stem_word("tokenizar")
    assert stem_word("analiza") == stem_word("analizar")
    assert stem_word("analizar") == stem_word("análisis")
    assert stem_word("procesamiento") == stem_word("procesar")


def test_dual_pipelines_lexical_vs_semantic():
    text = "El estudiante analiza un documento extenso. La inteligencia artificial avanza rápido."
    doc = preprocess_document(text, remove_stopwords=True, use_stemming=True)
    sem = preprocess_semantic(text)

    # El pipeline semántico conserva stopwords, mayúsculas/estructura y divide en oraciones
    assert len(sem["sentences"]) == 2
    assert "El estudiante" in sem["clean_text"]
    assert len(doc["sentences"]) == 2
    assert len(doc["paragraphs"]) == 1
    assert len(doc["stemmed_tokens"]) == len(doc["tokens"])


def test_preserve_original_text_immutability():
    original = "Texto Original Con MAYÚSCULAS y signos: ¡Intacto!"
    result = preprocess_document(original, remove_stopwords=True)
    assert result["original_text"] == original
    assert result["normalized_text"] != original
    assert "mayúsculas" in result["tokens"]


def test_configurable_stopwords_removal():
    text = "El procesamiento del lenguaje natural es fascinante"
    tokens_with_sw = preprocess_text(text, remove_stopwords=False)
    tokens_without_sw = preprocess_text(text, remove_stopwords=True)

    assert "el" in tokens_with_sw
    assert "del" in tokens_with_sw
    assert "es" in tokens_with_sw

    assert "el" not in tokens_without_sw
    assert "del" not in tokens_without_sw
    assert "es" not in tokens_without_sw
    assert tokens_without_sw == ["procesamiento", "lenguaje", "natural", "fascinante"]


def test_edge_cases_empty_stopwords_single_char_emojis_numbers():
    assert normalize_text("") == ""
    assert tokenize("") == []
    assert tokenize("   ... ,,, ¡¡¡ ??? --- 🚀🔥✨ 😊   ") == []

    doc_empty = preprocess_document("🚀🔥✨", remove_stopwords=True)
    assert doc_empty["tokens"] == []
    assert doc_empty["validation"]["edge_case"] == "empty_text"

    doc_sw = preprocess_document("el la los las de del en y o pero sin para por", remove_stopwords=True)
    assert doc_sw["tokens"] == []
    assert doc_sw["validation"]["edge_case"] == "only_stopwords"

    doc_num = preprocess_document("12345 67890 2026", remove_stopwords=True)
    assert doc_num["validation"]["edge_case"] == "only_numbers"

    doc_single = preprocess_document("a", remove_stopwords=False)
    assert doc_single["tokens"] == ["a"]


def test_identify_stopwords():
    tokens = ["el", "algoritmo", "de", "inteligencia", "artificial", "es", "veloz"]
    found = identify_stopwords(tokens)
    assert found == ["el", "de", "es"]
    filtered = remove_stopwords_from_tokens(tokens)
    assert filtered == ["algoritmo", "inteligencia", "artificial", "veloz"]
    assert "de" in SPANISH_STOPWORDS
