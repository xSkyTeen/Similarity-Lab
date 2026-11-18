"""
Módulo de Perfiles de Comparación (`reference` vs `enhanced`) para Similarity Lab.

Define y ejecuta dos perfiles de análisis en paralelo para permitir tanto la
validación exacta contra herramientas externas canónicas (`reference`) como el
análisis lingüístico profundo en español (`enhanced`):

1. Perfil `reference` (Comportamiento Canónico / Estándar de Librería):
   - Preprocesamiento mínimo: únicamente conversión a minúsculas y tokenización por palabras.
     Sin stemming, sin eliminación de stopwords, sin lematización.
   - Jaccard / Dice / Overlap (Containment): calculados sobre el SET de tokens crudos
     (además de shingles k=2, 3 y 5).
   - Coseno TF-IDF: `TfidfVectorizer()` con los parámetros por defecto de Scikit-learn
     (`smooth_idf=True, sublinear_tf=False, norm='l2', stop_words=None`), ejecutando
     `fit_transform([texto_a, texto_b])` y `cosine_similarity` (diferencia < 1e-9 respecto
     a `sklearn` directo).
   - Embeddings: coseno CRUDO (sin calibración) y sobre el documento completo como valor
     principal (`use_calibrated_as_primary=False, embed_full_document=True, bertscore_use_idf=False`).

2. Perfil `enhanced` (Comportamiento Enriquecido de Similarity Lab):
   - Preprocesamiento léxico completo: normalización NFC, eliminación de stopwords del español,
     stemming morfológico Snowball español (o spaCy si disponible) con unificación de tildes.
   - Jaccard / Dice / Containment / Ponderado / Shingles (k=2, 3, 5) sobre raíces morfológicas.
   - Coseno TF-IDF: `sublinear_tf=True, smooth_idf=True, norm='l2'` con IDF ajustado sobre
     los segmentos/oraciones de ambos documentos (`idf_mode="segments"`) y corpus de respaldo.
   - Embeddings: valor calibrado como principal (`use_calibrated_as_primary=True`), combinando
     vectores contextuales (±3), BERTScore con IDF y LSA (`TruncatedSVD`).
"""

from typing import Any, Dict, List, Optional, Set

from .cosine import calculate_cosine_similarity
from .embeddings import calculate_embedding_similarity
from .jaccard import calculate_jaccard_similarity
from .preprocessing import normalize_text, tokenize


COMPARISON_PROFILES: Dict[str, Dict[str, Any]] = {
    "reference": {
        "id": "reference",
        "name": "Modo Referencia (Canónico / Sklearn Estándar)",
        "short_label": "Referencia (Canónico)",
        "description": (
            "Preprocesado mínimo (solo minúsculas y palabras, sin stemming ni stopwords), "
            "Jaccard/Dice/Overlap sobre el SET de tokens crudos, Coseno TF-IDF con TfidfVectorizer() "
            "por defecto de Scikit-learn sobre 2 documentos, y Embeddings con coseno CRUDO sin calibrar."
        ),
        "preprocessing": {
            "lowercase": True,
            "remove_stopwords": False,
            "use_stemming": False,
            "lemmatization": False,
        },
        "jaccard_params": {
            "remove_stopwords": False,
            "use_stemming": False,
            "tokenizer_mode": "standard",
        },
        "cosine_params": {
            "remove_stopwords": False,
            "use_stemming": False,
            "sublinear_tf": False,
            "smooth_idf": True,
            "idf_mode": "two_docs",
            "tokenizer_mode": "sklearn_default",
        },
        "embeddings_params": {
            "remove_stopwords": False,
            "use_calibrated_as_primary": False,
            "embed_full_document": True,
            "bertscore_use_idf": False,
        },
    },
    "enhanced": {
        "id": "enhanced",
        "name": "Modo Enriquecido (NLP Español / Segment IDF / Calibrado)",
        "short_label": "Enriquecido (NLP Español)",
        "description": (
            "Pipeline NLP en español con eliminación de stopwords, stemming Snowball morfológico, "
            "IDF ajustado por segmentos con sublinear_tf=True, y Embeddings contextuales calibrados."
        ),
        "preprocessing": {
            "lowercase": True,
            "remove_stopwords": True,
            "use_stemming": True,
            "lemmatization": True,
        },
        "jaccard_params": {
            "remove_stopwords": True,
            "use_stemming": True,
            "tokenizer_mode": "standard",
        },
        "cosine_params": {
            "remove_stopwords": True,
            "use_stemming": True,
            "sublinear_tf": True,
            "smooth_idf": True,
            "idf_mode": "segments",
            "tokenizer_mode": "preprocessed",
        },
        "embeddings_params": {
            "remove_stopwords": True,
            "use_calibrated_as_primary": True,
            "embed_full_document": False,
            "bertscore_use_idf": True,
        },
    },
}


def normalize_profile_name(profile: Optional[str]) -> str:
    """Normaliza el identificador del perfil ('reference' o 'enhanced')."""
    if not profile:
        return "enhanced"
    clean = str(profile).strip().lower()
    if clean in ("reference", "ref", "canonico", "canónico", "standard", "sklearn"):
        return "reference"
    return "enhanced"


def get_profile_config(profile: str = "enhanced") -> Dict[str, Any]:
    """Devuelve el diccionario de configuración del perfil solicitado."""
    norm_name = normalize_profile_name(profile)
    return COMPARISON_PROFILES[norm_name]


def preprocess_reference_minimal(text: str) -> List[str]:
    """
    Preprocesado mínimo del perfil `reference`:
    Solo minúsculas y tokenización por palabras. Sin stemming, sin stopwords, sin lematización.
    """
    if not text or not isinstance(text, str):
        return []
    return tokenize(normalize_text(text))


def run_profile_comparison(
    text_a: str,
    text_b: str,
    profile: str = "enhanced",
    remove_stopwords: Optional[bool] = None,
    custom_stopwords: Optional[Set[str]] = None,
    gemini_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ejecuta Jaccard (+ Dice, Overlap, Shingles k=2,3,5), Coseno TF-IDF y Embeddings
    bajo las reglas del perfil indicado (`"reference"` o `"enhanced"`).
    """
    profile_id = normalize_profile_name(profile)
    cfg = get_profile_config(profile_id)

    if profile_id == "reference":
        jaccard_res = calculate_jaccard_similarity(
            text_a,
            text_b,
            remove_stopwords=False,
            custom_stopwords=None,
            use_stemming=False,
            tokenizer_mode="standard",
            compute_variants=True,
        )
        cosine_res = calculate_cosine_similarity(
            text_a,
            text_b,
            remove_stopwords=False,
            custom_stopwords=None,
            use_stemming=False,
            sublinear_tf=False,
            smooth_idf=True,
            idf_mode="two_docs",
            tokenizer_mode="sklearn_default",
        )
        embeddings_res = calculate_embedding_similarity(
            text_a,
            text_b,
            remove_stopwords=False,
            custom_stopwords=None,
            gemini_api_key=gemini_api_key,
            use_calibrated_as_primary=False,
            embed_full_document=True,
            bertscore_use_idf=False,
        )
    else:
        eff_stopwords = remove_stopwords if remove_stopwords is not None else True
        jaccard_res = calculate_jaccard_similarity(
            text_a,
            text_b,
            remove_stopwords=eff_stopwords,
            custom_stopwords=custom_stopwords,
            use_stemming=True,
            tokenizer_mode="standard",
            compute_variants=True,
        )
        cosine_res = calculate_cosine_similarity(
            text_a,
            text_b,
            remove_stopwords=eff_stopwords,
            custom_stopwords=custom_stopwords,
            use_stemming=True,
            sublinear_tf=True,
            smooth_idf=True,
            idf_mode="segments",
            tokenizer_mode="preprocessed",
        )
        embeddings_res = calculate_embedding_similarity(
            text_a,
            text_b,
            remove_stopwords=eff_stopwords,
            custom_stopwords=custom_stopwords,
            gemini_api_key=gemini_api_key,
            use_calibrated_as_primary=True,
            embed_full_document=False,
            bertscore_use_idf=True,
        )

    avg_pct = round(
        (
            jaccard_res["similarity_percentage"]
            + cosine_res["similarity_percentage"]
            + embeddings_res["similarity_percentage"]
        )
        / 3.0,
        2,
    )

    return {
        "profile_id": profile_id,
        "profile_name": cfg["name"],
        "short_label": cfg["short_label"],
        "description": cfg["description"],
        "config": cfg,
        "jaccard": jaccard_res,
        "cosine": cosine_res,
        "embeddings": embeddings_res,
        "summary": {
            "average_similarity_percentage": avg_pct,
            "jaccard_percentage": jaccard_res["similarity_percentage"],
            "jaccard_score": jaccard_res["similarity_score"],
            "jaccard_bruto_percentage": jaccard_res.get("raw_alpha_jaccard_percentage", jaccard_res["similarity_percentage"]),
            "jaccard_bruto_score": jaccard_res.get("raw_alpha_jaccard_score", jaccard_res["similarity_score"]),
            "jaccard_filtrado_percentage": jaccard_res.get("filtered_surface_jaccard_percentage", jaccard_res["similarity_percentage"]),
            "jaccard_filtrado_score": jaccard_res.get("filtered_surface_jaccard_score", jaccard_res["similarity_score"]),
            "dice_percentage": jaccard_res["dice_percentage"],
            "dice_score": jaccard_res["dice_score"],
            "overlap_percentage": jaccard_res["overlap_percentage"],
            "overlap_score": jaccard_res["overlap_score"],
            "weighted_jaccard_percentage": jaccard_res["weighted_jaccard_percentage"],
            "shingles_k2_percentage": jaccard_res["bigram_jaccard_percentage"],
            "shingles_k3_percentage": jaccard_res["trigram_jaccard_percentage"],
            "shingles_k5_percentage": jaccard_res["pentagram_jaccard_percentage"],
            "cosine_tfidf_percentage": cosine_res["similarity_percentage"],
            "cosine_tfidf_score": cosine_res["similarity_score"],
            "bow_count_cosine_percentage": cosine_res.get("bow_count_cosine_percentage", 0.0),
            "bow_count_cosine_score": cosine_res.get("bow_count_cosine_score", 0.0),
            "tfidf_unigram_cosine_percentage": cosine_res.get("tfidf_unigram_cosine_percentage", 0.0),
            "tfidf_unigram_cosine_score": cosine_res.get("tfidf_unigram_cosine_score", 0.0),
            "tfidf_bigram_cosine_percentage": cosine_res.get("tfidf_bigram_cosine_percentage", 0.0),
            "tfidf_bigram_cosine_score": cosine_res.get("tfidf_bigram_cosine_score", 0.0),
            "embeddings_primary_percentage": embeddings_res["similarity_percentage"],
            "embeddings_primary_score": embeddings_res["similarity_score"],
            "embeddings_primary_type": embeddings_res["primary_score_type"],
            "embeddings_estimated_range": embeddings_res.get("estimated_range", "0.00 – 0.00"),
            "embeddings_raw_percentage": embeddings_res["raw_similarity_percentage"],
            "embeddings_raw_score": embeddings_res["raw_similarity_score"],
            "embeddings_calibrated_percentage": embeddings_res["calibrated_similarity_percentage"],
            "embeddings_calibrated_score": embeddings_res["calibrated_similarity_score"],
            "bertscore_f1_percentage": embeddings_res["bertscore"]["f1_percentage"],
        },
    }


def compare_profiles_side_by_side(
    text_a: str,
    text_b: str,
    primary_profile: str = "enhanced",
    remove_stopwords: bool = True,
    custom_stopwords: Optional[Set[str]] = None,
    gemini_api_key: Optional[str] = None,
    precomputed_primary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Ejecuta tanto el perfil `reference` como el perfil `enhanced` sobre `(text_a, text_b)`
    y construye una tabla comparativa en paralelo indicando cuál es el perfil principal.
    """
    primary_id = normalize_profile_name(primary_profile)

    if primary_id == "enhanced" and precomputed_primary is not None:
        enhanced_res = precomputed_primary
    else:
        enhanced_res = run_profile_comparison(
            text_a,
            text_b,
            profile="enhanced",
            remove_stopwords=remove_stopwords,
            custom_stopwords=custom_stopwords,
            gemini_api_key=gemini_api_key,
        )

    if primary_id == "reference" and precomputed_primary is not None:
        reference_res = precomputed_primary
    else:
        reference_res = run_profile_comparison(
            text_a,
            text_b,
            profile="reference",
            remove_stopwords=False,
            custom_stopwords=None,
            gemini_api_key=gemini_api_key,
        )

    ref_sum = reference_res["summary"]
    enh_sum = enhanced_res["summary"]

    metrics_table: List[Dict[str, Any]] = [
        {
            "metric_id": "jaccard",
            "metric_label": "Índice de Jaccard (Set)",
            "reference_percentage": ref_sum["jaccard_percentage"],
            "reference_score": ref_sum["jaccard_score"],
            "enhanced_percentage": enh_sum["jaccard_percentage"],
            "enhanced_score": enh_sum["jaccard_score"],
            "delta_percentage": round(enh_sum["jaccard_percentage"] - ref_sum["jaccard_percentage"], 2),
            "reference_config": "Set tokens crudos (sin stemming/stopwords)",
            "enhanced_config": "Set raíces Snowball (sin stopwords)",
        },
        {
            "metric_id": "dice",
            "metric_label": "Coeficiente de Sørensen-Dice",
            "reference_percentage": ref_sum["dice_percentage"],
            "reference_score": ref_sum["dice_score"],
            "enhanced_percentage": enh_sum["dice_percentage"],
            "enhanced_score": enh_sum["dice_score"],
            "delta_percentage": round(enh_sum["dice_percentage"] - ref_sum["dice_percentage"], 2),
            "reference_config": "2|A∩B|/(|A|+|B|) crudo",
            "enhanced_config": "2|A∩B|/(|A|+|B|) con stemming",
        },
        {
            "metric_id": "overlap",
            "metric_label": "Solapamiento / Containment (Overlap)",
            "reference_percentage": ref_sum["overlap_percentage"],
            "reference_score": ref_sum["overlap_score"],
            "enhanced_percentage": enh_sum["overlap_percentage"],
            "enhanced_score": enh_sum["overlap_score"],
            "delta_percentage": round(enh_sum["overlap_percentage"] - ref_sum["overlap_percentage"], 2),
            "reference_config": "|A∩B|/min(|A|,|B|) crudo",
            "enhanced_config": "|A∩B|/min(|A|,|B|) con stemming",
        },
        {
            "metric_id": "shingles_k3",
            "metric_label": "Shingles k=3 (Trigramas)",
            "reference_percentage": ref_sum["shingles_k3_percentage"],
            "reference_score": reference_res["jaccard"]["trigram_jaccard_score"],
            "enhanced_percentage": enh_sum["shingles_k3_percentage"],
            "enhanced_score": enhanced_res["jaccard"]["trigram_jaccard_score"],
            "delta_percentage": round(enh_sum["shingles_k3_percentage"] - ref_sum["shingles_k3_percentage"], 2),
            "reference_config": "k=3 palabras crudas",
            "enhanced_config": "k=3 raíces normalizadas",
        },
        {
            "metric_id": "shingles_k5",
            "metric_label": "Shingles k=5 (Pentagramas)",
            "reference_percentage": ref_sum["shingles_k5_percentage"],
            "reference_score": reference_res["jaccard"]["pentagram_jaccard_score"],
            "enhanced_percentage": enh_sum["shingles_k5_percentage"],
            "enhanced_score": enhanced_res["jaccard"]["pentagram_jaccard_score"],
            "delta_percentage": round(enh_sum["shingles_k5_percentage"] - ref_sum["shingles_k5_percentage"], 2),
            "reference_config": "k=5 palabras crudas",
            "enhanced_config": "k=5 raíces normalizadas",
        },
        {
            "metric_id": "cosine_tfidf",
            "metric_label": "Coseno TF-IDF",
            "reference_percentage": ref_sum["cosine_tfidf_percentage"],
            "reference_score": ref_sum["cosine_tfidf_score"],
            "enhanced_percentage": enh_sum["cosine_tfidf_percentage"],
            "enhanced_score": enh_sum["cosine_tfidf_score"],
            "delta_percentage": round(enh_sum["cosine_tfidf_percentage"] - ref_sum["cosine_tfidf_percentage"], 2),
            "reference_config": "Sklearn default (2 docs, sublinear_tf=False, smooth_idf=True)",
            "enhanced_config": "IDF por segmentos + sublinear_tf=True + stemming",
        },
        {
            "metric_id": "embeddings",
            "metric_label": "Embeddings Semánticos (Valor Principal)",
            "reference_percentage": ref_sum["embeddings_primary_percentage"],
            "reference_score": ref_sum["embeddings_primary_score"],
            "enhanced_percentage": enh_sum["embeddings_primary_percentage"],
            "enhanced_score": enh_sum["embeddings_primary_score"],
            "delta_percentage": round(enh_sum["embeddings_primary_percentage"] - ref_sum["embeddings_primary_percentage"], 2),
            "reference_config": f"Coseno CRUDO ({ref_sum['embeddings_raw_percentage']}%) · Doc Completo",
            "enhanced_config": f"Calibrado ({enh_sum['embeddings_calibrated_percentage']}%, Crudo: {enh_sum['embeddings_raw_percentage']}%)",
        },
        {
            "metric_id": "bertscore_f1",
            "metric_label": "BERTScore F1 (Alineamiento Token-a-Token)",
            "reference_percentage": ref_sum["bertscore_f1_percentage"],
            "reference_score": reference_res["embeddings"]["bertscore"]["f1"],
            "enhanced_percentage": enh_sum["bertscore_f1_percentage"],
            "enhanced_score": enhanced_res["embeddings"]["bertscore"]["f1"],
            "delta_percentage": round(enh_sum["bertscore_f1_percentage"] - ref_sum["bertscore_f1_percentage"], 2),
            "reference_config": "Sin ponderación IDF (use_idf=False)",
            "enhanced_config": "Ponderado por IDF (use_idf=True)",
        },
    ]

    return {
        "primary_profile": primary_id,
        "primary_profile_label": COMPARISON_PROFILES[primary_id]["name"],
        "reference": {
            "profile_id": "reference",
            "is_primary": primary_id == "reference",
            "name": reference_res["profile_name"],
            "short_label": reference_res["short_label"],
            "description": reference_res["description"],
            "summary": ref_sum,
            "jaccard": reference_res["jaccard"],
            "cosine": reference_res["cosine"],
            "embeddings": reference_res["embeddings"],
        },
        "enhanced": {
            "profile_id": "enhanced",
            "is_primary": primary_id == "enhanced",
            "name": enhanced_res["profile_name"],
            "short_label": enhanced_res["short_label"],
            "description": enhanced_res["description"],
            "summary": enh_sum,
            "jaccard": enhanced_res["jaccard"],
            "cosine": enhanced_res["cosine"],
            "embeddings": enhanced_res["embeddings"],
        },
        "comparison_table": metrics_table,
    }
