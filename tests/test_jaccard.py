"""
Pruebas unitarias para el módulo de Similitud de Jaccard (core/jaccard.py).
"""

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from core.jaccard import calculate_jaccard_similarity


def test_identical_texts_jaccard():
    text = "La inteligencia artificial transforma la ciencia de datos moderna."
    res = calculate_jaccard_similarity(text, text, remove_stopwords=True)
    assert res["similarity_score"] == 1.0
    assert res["similarity_percentage"] == 100.0
    assert res["dice_score"] == 1.0
    assert res["containment_score"] == 1.0
    assert res["weighted_jaccard_score"] == 1.0
    assert res["bigram_jaccard_score"] == 1.0
    assert res["jaccard_distance"] == 0.0
    assert res["intersection_size"] == res["union_size"]
    assert res["exclusive_a_count"] == 0
    assert res["exclusive_b_count"] == 0


def test_completely_different_texts_jaccard():
    text_a = "astronomía planetas galaxias telescopio universo estrellas"
    text_b = "gastronomía cocina ingredientes receta hornear postres"
    res = calculate_jaccard_similarity(text_a, text_b, remove_stopwords=True)
    assert res["similarity_score"] == 0.0
    assert res["similarity_percentage"] == 0.0
    assert res["dice_score"] == 0.0
    assert res["containment_score"] == 0.0
    assert res["jaccard_distance"] == 1.0
    assert res["intersection_size"] == 0
    assert res["union_size"] == 12


def test_partially_similar_academic_reference_example():
    """
    Verifica el ejemplo manual de la Guía Completa de Similitud NLP (Sección 1.2):
    Oración A: "El procesamiento del lenguaje natural es fascinante"
    Oración B: "El procesamiento de lenguaje y texto es genial"
    |A ∩ B| = 2, |A ∪ B| = 6 -> J(A,B) = 2/6 = 0.333333 (33.33%)
    """
    doc_a = "El procesamiento del lenguaje natural es fascinante"
    doc_b = "El procesamiento de lenguaje y texto es genial"
    res = calculate_jaccard_similarity(doc_a, doc_b, remove_stopwords=True)

    assert res["unique_a_count"] == 4
    assert res["unique_b_count"] == 4
    assert res["intersection_size"] == 2
    assert res["union_size"] == 6
    assert res["shared_words"] == ["lenguaje", "procesamiento"]
    assert res["similarity_score"] == pytest.approx(2.0 / 6.0, rel=1e-4)
    assert res["similarity_percentage"] == 33.33
    assert res["dice_score"] == pytest.approx(4.0 / 8.0, rel=1e-4)
    assert res["containment_score"] == pytest.approx(2.0 / 4.0, rel=1e-4)


def test_morphological_variations_count_as_matching_in_jaccard():
    """
    Verifica que variaciones morfológicas como 'tokenización' vs 'tokenizar' y
    'analiza' vs 'analizar' cuentan como coincidentes en el pipeline léxico.
    """
    res_1 = calculate_jaccard_similarity("tokenización", "tokenizar", remove_stopwords=True)
    assert res_1["similarity_score"] == 1.0
    assert res_1["intersection_size"] == 1
    assert len(res_1["morphological_matches"]) == 1

    res_2 = calculate_jaccard_similarity("analiza", "analizar", remove_stopwords=True)
    assert res_2["similarity_score"] == 1.0
    assert res_2["intersection_size"] == 1

    res_pair = calculate_jaccard_similarity(
        "La tokenización analiza el texto",
        "Tokenizar permite analizar textos",
        remove_stopwords=True,
    )
    assert res_pair["intersection_size"] >= 3
    assert res_pair["similarity_score"] >= 0.75


def test_repeated_words_and_containment_jaccard():
    """
    Si el texto B repite 3 veces el texto A:
    - Jaccard clásico = 1.0
    - Containment = 1.0
    - Jaccard ponderado refleja la proporción de frecuencias (1/3 = 0.3333).
    """
    doc_a = "algoritmo datos modelo"
    doc_b = " ".join([doc_a, doc_a, doc_a])
    res = calculate_jaccard_similarity(doc_a, doc_b)
    assert res["similarity_score"] == 1.0
    assert res["containment_score"] == 1.0
    assert res["weighted_jaccard_score"] == pytest.approx(1.0 / 3.0, rel=1e-4)


def test_different_lengths_containment_is_one():
    doc_a = "redes neuronales profundas"
    doc_b = "redes neuronales profundas aplicadas al reconocimiento óptico de imágenes médicas complejas"
    res = calculate_jaccard_similarity(doc_a, doc_b, remove_stopwords=True)
    assert 0.0 < res["similarity_percentage"] < 100.0
    assert res["containment_score"] == 1.0
    assert res["containment_percentage"] == 100.0
    assert res["bigram_jaccard_score"] > 0.0


def test_empty_stopwords_and_emojis_jaccard():
    for a, b in [("", ""), ("texto con palabras", ""), ("el la los", "de del en"), ("🚀🔥", "✨🎉")]:
        res = calculate_jaccard_similarity(a, b, remove_stopwords=True)
        assert res["similarity_score"] == 0.0
        assert not math.isnan(res["similarity_score"])
        assert not math.isnan(res["dice_score"])
        assert not math.isnan(res["containment_score"])
        assert not math.isnan(res["weighted_jaccard_score"])


def test_spanish_special_characters_jaccard():
    doc_a = "El niño aprendió lingüística y programación en España."
    doc_b = "La lingüística y la programación se enseñan al niño en España."
    res = calculate_jaccard_similarity(doc_a, doc_b, remove_stopwords=True)
    assert "niño" in res["shared_words"]
    assert "lingüística" in res["shared_words"]
    assert "programación" in res["shared_words"]
    assert "españa" in res["shared_words"]
    assert 0.0 <= res["similarity_percentage"] <= 100.0


def test_analytical_jaccard_numeric_sets_half():
    """
    Valor analítico verificado:
    Jaccard({1, 2, 3, 4, 5}, {0, ..., 9}) = 5 / 10 = 0.5
    Containment/Overlap = 5 / min(5, 10) = 1.0 (A está incluido en B)
    Dice = 2 * 5 / (5 + 10) = 10 / 15 = 2/3
    """
    set_a = {1, 2, 3, 4, 5}
    set_b = set(range(10))
    res = calculate_jaccard_similarity(set_a, set_b, use_stemming=False, remove_stopwords=False)
    assert res["intersection_size"] == 5
    assert res["union_size"] == 10
    assert res["similarity_score"] == pytest.approx(0.5, abs=1e-9)
    assert res["similarity_percentage"] == 50.0
    assert res["containment_score"] == pytest.approx(1.0, abs=1e-9)
    assert res["overlap_score"] == pytest.approx(1.0, abs=1e-9)
    assert res["dice_score"] == pytest.approx(2.0 / 3.0, abs=1e-6)


def test_analytical_minhash_3_shared_of_10_unique_is_0_3():
    """
    Ejemplo clásico de MinHash:
    3 elementos compartidos de 10 elementos únicos en la unión -> Jaccard = 3/10 = 0.3.
    """
    # Conjunto A tiene 6 elementos: {t1, t2, t3, t4, t5, t6}
    # Conjunto B tiene 7 elementos: {t4, t5, t6, t7, t8, t9, t10}
    # Intersección = {t4, t5, t6} (3 elementos), Unión = {t1..t10} (10 elementos)
    doc_a = ["t1", "t2", "t3", "t4", "t5", "t6"]
    doc_b = ["t4", "t5", "t6", "t7", "t8", "t9", "t10"]
    res = calculate_jaccard_similarity(doc_a, doc_b, use_stemming=False, remove_stopwords=False)
    assert res["intersection_size"] == 3
    assert res["union_size"] == 10
    assert res["similarity_score"] == pytest.approx(0.3, abs=1e-9)
    assert res["similarity_percentage"] == 30.0


def test_shingles_k2_k3_k5_and_variants_with_without_stemming_stopwords():
    """
    Verifica:
    - Cálculo de Shingles para k=2, k=3 y k=5.
    - Exposición de variantes con y sin stemming/stopwords en `res['variants']`.
    """
    doc_a = "el algoritmo procesa grandes volumenes de datos complejos en memoria"
    doc_b = "el algoritmo procesa grandes volumenes de datos simples en disco"
    res = calculate_jaccard_similarity(doc_a, doc_b, remove_stopwords=False, use_stemming=False)

    assert "k2" in res["shingles"]
    assert "k3" in res["shingles"]
    assert "k5" in res["shingles"]
    assert res["bigram_jaccard_score"] > 0.0
    assert res["trigram_jaccard_score"] > 0.0
    assert res["pentagram_jaccard_score"] > 0.0
    assert res["shingle_5_jaccard_score"] == res["pentagram_jaccard_score"]

    assert "variants" in res
    variants = res["variants"]
    assert "without_stemming_without_stopwords" in variants
    assert "without_stemming_with_stopwords_removed" in variants
    assert "with_stemming_without_stopwords" in variants
    assert "with_stemming_with_stopwords_removed" in variants
    assert "usage_guide" in res
    assert "jaccard" in res["usage_guide"]
    assert "overlap" in res["usage_guide"]
    assert "dice" in res["usage_guide"]
