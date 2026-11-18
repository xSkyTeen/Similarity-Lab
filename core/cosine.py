"""
Módulo de Similitud del Coseno mediante TF-IDF para Similarity Lab.

Implementa tanto el modo enriquecido (`enhanced`, con IDF por segmentos) como el
modo canónico de referencia (`reference`, idéntico a `TfidfVectorizer()` por defecto
de Scikit-learn con diferencia < 1e-9):

1. Modos de ajuste de IDF (`idf_mode`):
   - `"segments"` (Modo Enriquecido por defecto):
     Ajusta el IDF sobre el corpus de segmentos (oraciones/bloques de ambos textos),
     con respaldo de un corpus de referencia en español cuando hay pocos segmentos.
     NOTA TÉCNICA: Con solo N=2 documentos (`two_docs`), el IDF casi no discrimina:
     según la fórmula de Scikit-learn con `smooth_idf=True`:
         idf(t) = ln((1 + n) / (1 + df(t))) + 1
     para n=2 solo existen dos valores posibles de IDF:
       • Si df=2 (término compartido en A y B): idf = ln(3/3) + 1 = 1.0000
       • Si df=1 (término exclusivo de A o B):  idf = ln(3/2) + 1 ≈ 1.4055
     Todos los términos compartidos reciben exactamente el mismo peso IDF (1.0),
     anulando la capacidad de distinguir stopwords de términos técnicos específicos.
     Por eso `"segments"` es el modo enriquecido y `"two_docs"` es el modo referencia.
   - `"two_docs"` (Modo Referencia canónico):
     Ajusta y transforma directamente los 2 documentos completos (`fit_transform([a, b])`).
   - `"fit_a_transform_b"`:
     Ajusta el vocabulario e IDF únicamente sobre el Documento A (`fit([a])`) y
     transforma ambos documentos (`transform([a, b])`), útil para calibrar contra
     herramientas asimétricas que tratan a A como documento fuente/referencia.

2. Parámetros explícitos configurables:
   - `sublinear_tf`: `True` aplica `1 + ln(tf)`; `False` usa frecuencia cruda `tf` (default de sklearn).
   - `smooth_idf`: `True` aplica `idf(t) = ln((1 + n) / (1 + df(t))) + 1` (default de sklearn);
                   `False` aplica `idf(t) = ln(n / df(t)) + 1`.
   - `tokenizer_mode`:
       • `"preprocessed"`: usa los tokens léxicos preprocesados de Similarity Lab.
       • `"sklearn_default"`: usa el analizador/regex nativo de `TfidfVectorizer()` de
         Scikit-learn (`lowercase=True, token_pattern=r'(?u)\\b\\w\\w+\\b', stop_words=None`).
       • `"whitespace"`: tokenización por espacios simples (`text.lower().split()`).
"""

import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .params_store import load_trained_params
from .preprocessing import SPANISH_STOPWORDS, preprocess_document, preprocess_text, stem_word


# Corpus de referencia en español (diversos dominios) para estabilizar los pesos IDF
# cuando los dos documentos comparados tienen muy pocos segmentos/oraciones en modo "segments".
SPANISH_REFERENCE_CORPUS_SENTENCES: List[str] = [
    "El desarrollo de software moderno requiere arquitectura modular pruebas automatizadas y control de versiones.",
    "La inteligencia artificial y el aprendizaje automático procesan grandes volúmenes de datos complejos.",
    "Los algoritmos estadísticos analizan distribuciones de probabilidad y correlación entre variables.",
    "El sistema educativo universitario fomenta la investigación científica y la redacción académica rigurosa.",
    "La medicina preventiva y el diagnóstico clínico mejoran la salud de los pacientes en el hospital.",
    "El crecimiento económico depende de la inversión financiera el comercio internacional y el empleo.",
    "Las energías renovables como la solar y la eólica reducen las emisiones contaminantes del planeta.",
    "El procesamiento del lenguaje natural permite tokenizar textos extraer entidades y medir similitud.",
    "Las redes neuronales profundas aprenden representaciones vectoriales continuas en múltiples capas.",
    "La administración pública gestiona recursos infraestructura transporte y servicios para los ciudadanos.",
    "El análisis químico de laboratorio determina la composición molecular de sustancias orgánicas.",
    "La literatura hispanoamericana destaca por su riqueza léxica narrativa histórica y expresión poética.",
    "Los sistemas operativos administran la memoria principal el procesador y los dispositivos de almacenamiento.",
    "La seguridad informática protege las redes corporativas mediante cifrado criptográfico y autenticación.",
    "El cambio climático altera las precipitaciones la temperatura global y los ecosistemas marinos.",
    "La ingeniería civil diseña puentes estructuras antisísmicas carreteras y obras hidráulicas.",
]

_PRECOMPUTED_REFERENCE_STEMS: Optional[List[List[str]]] = None
_PRECOMPUTED_REFERENCE_SURFACES: Optional[List[List[str]]] = None


def compute_sklearn_idf_formula(n_docs: int, df: int, smooth_idf: bool = True) -> float:
    """
    Calcula el peso IDF exacto según la documentación oficial de Scikit-learn (`TfidfTransformer`):
    - Con `smooth_idf=True` (por defecto en sklearn):
          idf(t) = ln((1 + n) / (1 + df(t))) + 1
    - Con `smooth_idf=False`:
          idf(t) = ln(n / df(t)) + 1
    donde `n` es el número total de documentos/segmentos y `df(t)` es la frecuencia
    documental del término `t`.
    """
    if smooth_idf:
        return math.log((1.0 + float(n_docs)) / (1.0 + float(df))) + 1.0
    if df <= 0:
        return 1.0
    return math.log(float(n_docs) / float(df)) + 1.0


def _get_reference_corpus_tokens(use_stemming: bool = True, remove_stopwords: bool = True) -> List[List[str]]:
    """Devuelve las oraciones del corpus de referencia preprocesadas."""
    global _PRECOMPUTED_REFERENCE_STEMS, _PRECOMPUTED_REFERENCE_SURFACES
    if use_stemming and remove_stopwords:
        if _PRECOMPUTED_REFERENCE_STEMS is None:
            _PRECOMPUTED_REFERENCE_STEMS = [
                preprocess_text(sent, remove_stopwords=True, use_stemming=True)
                for sent in SPANISH_REFERENCE_CORPUS_SENTENCES
            ]
        return _PRECOMPUTED_REFERENCE_STEMS
    if not use_stemming and remove_stopwords:
        if _PRECOMPUTED_REFERENCE_SURFACES is None:
            _PRECOMPUTED_REFERENCE_SURFACES = [
                preprocess_text(sent, remove_stopwords=True, use_stemming=False)
                for sent in SPANISH_REFERENCE_CORPUS_SENTENCES
            ]
        return _PRECOMPUTED_REFERENCE_SURFACES
    return [
        preprocess_text(sent, remove_stopwords=remove_stopwords, use_stemming=use_stemming)
        for sent in SPANISH_REFERENCE_CORPUS_SENTENCES
    ]


def _identity_tokenizer(tokens: List[str]) -> List[str]:
    """Tokenizador identidad para que TfidfVectorizer no altere ni descarte tokens de 1 carácter."""
    return tokens


def _whitespace_tokenizer(text: str) -> List[str]:
    """Tokenizador por espacios en blanco en minúsculas."""
    if not isinstance(text, str):
        return [str(t).lower() for t in text if str(t).strip()]
    return [tok for tok in text.lower().split() if tok]


def _extract_tokens_and_segments(
    source: Union[str, Iterable[str]],
    remove_stopwords: bool = False,
    custom_stopwords: Optional[Set[str]] = None,
    use_stemming: bool = True,
    tokenizer_mode: str = "preprocessed",
) -> Tuple[List[str], List[str], Dict[str, Counter], List[List[str]], str]:
    """
    Extrae:
    - Lista de tokens normalizados (con stemming si `use_stemming=True`).
    - Lista de tokens de superficie.
    - Mapa de stem -> Counter de palabras superficiales.
    - Segmentos (oraciones) tokenizados para alimentar el ajuste de IDF.
    - Texto limpio para el vectorizador de n-gramas de caracteres.
    """
    if isinstance(source, str):
        if tokenizer_mode == "whitespace":
            raw_words = [w for w in source.lower().split() if w]
            if remove_stopwords:
                from .preprocessing import SPANISH_STOPWORDS
                sw = set(SPANISH_STOPWORDS)
                if custom_stopwords:
                    sw |= {w.lower() for w in custom_stopwords}
                raw_words = [w for w in raw_words if w not in sw]
            surface_tokens = raw_words
            clean_text = source.lower().strip()
            segments = [
                [
                    stem_word(w, unify_accents=True) if use_stemming else w
                    for w in s.lower().split()
                    if w
                ]
                for s in source.split(".")
                if s.strip()
            ]
        elif tokenizer_mode == "sklearn_default":
            import re
            raw_words = re.findall(r"(?u)\b\w\w+\b", source.lower())
            if remove_stopwords:
                from .preprocessing import SPANISH_STOPWORDS
                sw = set(SPANISH_STOPWORDS)
                if custom_stopwords:
                    sw |= {w.lower() for w in custom_stopwords}
                raw_words = [w for w in raw_words if w not in sw]
            surface_tokens = raw_words
            clean_text = source.lower().strip()
            segments = [
                [
                    stem_word(w, unify_accents=True) if use_stemming else w
                    for w in re.findall(r"(?u)\b\w\w+\b", s.lower())
                ]
                for s in source.split(".")
                if s.strip()
            ]
        else:
            doc_info = preprocess_document(
                source,
                remove_stopwords=remove_stopwords,
                custom_stopwords=custom_stopwords,
                use_stemming=use_stemming,
            )
            surface_tokens = doc_info["tokens"]
            segments = doc_info["sentence_lexical_segments"]
            clean_text = doc_info["normalized_text"]
    elif source is None:
        surface_tokens = []
        segments = []
        clean_text = ""
    else:
        surface_tokens = [str(t).strip().lower() for t in source if str(t).strip()]
        segments = []
        clean_text = " ".join(surface_tokens)

    norm_tokens: List[str] = []
    stem_to_surfaces: Dict[str, Counter] = {}
    for surf in surface_tokens:
        norm = stem_word(surf, unify_accents=True) if use_stemming else surf
        if not norm:
            continue
        norm_tokens.append(norm)
        if norm not in stem_to_surfaces:
            stem_to_surfaces[norm] = Counter()
        stem_to_surfaces[norm][surf] += 1

    # Si no había múltiples oraciones pero el documento es largo, dividir en ventanas de 25 tokens
    if len(segments) <= 1 and len(norm_tokens) >= 30:
        chunk_size = 25
        segments = [
            norm_tokens[i : i + chunk_size]
            for i in range(0, len(norm_tokens), chunk_size)
            if norm_tokens[i : i + chunk_size]
        ]
    elif not segments and norm_tokens:
        segments = [norm_tokens]

    return norm_tokens, surface_tokens, stem_to_surfaces, segments, clean_text


def calculate_manual_cosine_similarity(
    doc_a: Union[str, Iterable[str]],
    doc_b: Union[str, Iterable[str]],
    remove_stopwords: bool = False,
    custom_stopwords: Optional[Set[str]] = None,
    use_stemming: bool = True,
) -> Dict[str, Any]:
    """
    Calcula la similitud del coseno clásica basada en frecuencia de términos (Bag of Words)
    sobre tokens normalizados.
    """
    tokens_a, _, _, _, _ = _extract_tokens_and_segments(
        doc_a,
        remove_stopwords=remove_stopwords,
        custom_stopwords=custom_stopwords,
        use_stemming=use_stemming,
    )
    tokens_b, _, _, _, _ = _extract_tokens_and_segments(
        doc_b,
        remove_stopwords=remove_stopwords,
        custom_stopwords=custom_stopwords,
        use_stemming=use_stemming,
    )

    vocab = sorted(list(set(tokens_a + tokens_b)))
    if not vocab or not tokens_a or not tokens_b:
        return {
            "similarity_score": 0.0,
            "similarity_percentage": 0.0,
            "dot_product": 0.0,
            "norm_a": 0.0,
            "norm_b": 0.0,
            "vocabulary_size": len(vocab),
        }

    vec_a = [tokens_a.count(word) for word in vocab]
    vec_b = [tokens_b.count(word) for word in vocab]

    dot_product = float(sum(v1 * v2 for v1, v2 in zip(vec_a, vec_b)))
    norm_a = math.sqrt(sum(v1 ** 2 for v1 in vec_a))
    norm_b = math.sqrt(sum(v2 ** 2 for v2 in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        score = 0.0
    else:
        score = dot_product / (norm_a * norm_b)

    score = max(0.0, min(1.0, float(score)))
    return {
        "similarity_score": round(score, 6),
        "similarity_percentage": round(score * 100.0, 2),
        "dot_product": round(dot_product, 4),
        "norm_a": round(norm_a, 4),
        "norm_b": round(norm_b, 4),
        "vocabulary_size": len(vocab),
    }


def _calculate_char_ngram_cosine(clean_text_a: str, clean_text_b: str) -> float:
    """
    Calcula la similitud del coseno sobre n-gramas de caracteres (`analyzer='char_wb'`,
    rango 3 a 5), altamente tolerante a variaciones morfológicas y errores tipográficos.
    """
    if not clean_text_a.strip() or not clean_text_b.strip():
        return 0.0
    try:
        char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            norm="l2",
            lowercase=True,
        )
        mat = char_vec.fit_transform([clean_text_a, clean_text_b])
        sim = float(cosine_similarity(mat[0:1], mat[1:2])[0][0])
        if math.isnan(sim):
            return 0.0
        return max(0.0, min(1.0, sim))
    except ValueError:
        return 0.0


def calculate_cosine_similarity(
    doc_a: Union[str, Iterable[str]],
    doc_b: Union[str, Iterable[str]],
    remove_stopwords: bool = False,
    custom_stopwords: Optional[Set[str]] = None,
    use_stemming: bool = True,
    ngram_range: Tuple[int, int] = (1, 2),
    sublinear_tf: bool = True,
    smooth_idf: bool = True,
    idf_mode: str = "segments",
    tokenizer_mode: str = "preprocessed",
    params_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Calcula la similitud del coseno mediante vectorización TF-IDF con Scikit-learn.

    Parámetros de calibración y modo de referencia:
    - `sublinear_tf`: Si `True` (modo enriquecido), reemplaza tf por `1 + log(tf)`.
      Si `False` (modo referencia sklearn por defecto), usa frecuencia lineal `tf`.
    - `smooth_idf`: Si `True` (por defecto en sklearn), calcula `idf = ln((1+n)/(1+df)) + 1`.
      Si `False`, calcula `idf = ln(n/df) + 1`.
    - `idf_mode`:
      • `"segments"` (Modo Enriquecido): ajusta el IDF sobre los segmentos (oraciones/bloques)
        de ambos textos + corpus de referencia si hay < 6 segmentos. Con solo 2 documentos,
        el IDF apenas discrimina (`ln(3/3)+1=1.0` vs `ln(3/2)+1≈1.4055`), por lo que `"segments"`
        ofrece una discriminación mucho más rica.
      • `"two_docs"` (Modo Referencia): ajusta el IDF únicamente con los 2 documentos `[A, B]`
        (`fit_transform([A, B])`), reproduciendo el comportamiento estándar de Scikit-learn.
      • `"fit_a_transform_b"`: ajusta el IDF solo en el documento A (`fit([A])`) y transforma
        ambos documentos (`transform([A, B])`).
    - `tokenizer_mode`:
      • `"preprocessed"`: usa la tubería léxica de Similarity Lab.
      • `"sklearn_default"`: usa `TfidfVectorizer` con el tokenizador regex nativo de sklearn
        (`(?u)\\b\\w\\w+\\b`), garantizando diferencia < 1e-9 frente a sklearn directo cuando se
        combina con `idf_mode="two_docs"`, `sublinear_tf=False`, `smooth_idf=True`,
        `use_stemming=False`, `remove_stopwords=False`.
      • `"whitespace"`: separa tokens por espacios en blanco (`split()`).
    """
    if idf_mode not in ("segments", "two_docs", "fit_a_transform_b"):
        idf_mode = "segments"

    str_a = doc_a if isinstance(doc_a, str) else " ".join(str(x) for x in (doc_a or []))
    str_b = doc_b if isinstance(doc_b, str) else " ".join(str(x) for x in (doc_b or []))

    tokens_a, _, map_a, segments_a, clean_a = _extract_tokens_and_segments(
        doc_a,
        remove_stopwords=remove_stopwords,
        custom_stopwords=custom_stopwords,
        use_stemming=use_stemming,
        tokenizer_mode=tokenizer_mode,
    )
    tokens_b, _, map_b, segments_b, clean_b = _extract_tokens_and_segments(
        doc_b,
        remove_stopwords=remove_stopwords,
        custom_stopwords=custom_stopwords,
        use_stemming=use_stemming,
        tokenizer_mode=tokenizer_mode,
    )

    warnings: List[str] = []

    def _compute_filtered_sparse_variants() -> Dict[str, float]:
        """
        Calcula las 3 variantes vectoriales dispersas canónicas sobre palabras de superficie
        sin stopwords ni números (`re.findall(r'\\b[a-záéíóúüñ]+\\b', text.lower())`):
        - Coseno (Count / BoW): `CountVectorizer()`
        - Coseno (TF-IDF Unigramas): `TfidfVectorizer(ngram_range=(1, 1), ...)`
        - Coseno (TF-IDF Bigramas): `TfidfVectorizer(ngram_range=(1, 2), ...)`
        """
        trained_lex = (params_override or load_trained_params()).get("lexical", {})
        min_tok_len = int(trained_lex.get("min_token_len", 3))
        stop_set = set(SPANISH_STOPWORDS)
        if custom_stopwords:
            stop_set |= {str(w).strip().lower() for w in custom_stopwords}
        alpha_re = re.compile(r"\b[a-záéíóúüñ]+\b")
        raw_a = alpha_re.findall(str_a.lower())
        raw_b = alpha_re.findall(str_b.lower())
        if not custom_stopwords and (len(raw_a) >= 40 or len(raw_b) >= 40):
            stop_set |= {str(w).strip().lower() for w in trained_lex.get("extra_stopwords", [])}
        toks_fa = [w for w in raw_a if w not in stop_set and len(w) >= min_tok_len]
        toks_fb = [w for w in raw_b if w not in stop_set and len(w) >= min_tok_len]
        if not toks_fa or not toks_fb:
            return {"bow_count": 0.0, "tfidf_unigram": 0.0, "tfidf_bigram": 0.0}
        fa_str = " ".join(toks_fa)
        fb_str = " ".join(toks_fb)
        try:
            mat_cnt = CountVectorizer(binary=bool(trained_lex.get("bow_binary", False))).fit_transform([fa_str, fb_str])
            bow_cnt = float(cosine_similarity(mat_cnt[0:1], mat_cnt[1:2])[0][0])
        except ValueError:
            bow_cnt = 0.0
        try:
            mat_u = TfidfVectorizer(
                ngram_range=(1, 1),
                sublinear_tf=bool(trained_lex.get("tfidf_unigram_sublinear_tf", False)),
                smooth_idf=bool(trained_lex.get("tfidf_unigram_smooth_idf", True)),
            ).fit_transform([fa_str, fb_str])
            tf_u = float(cosine_similarity(mat_u[0:1], mat_u[1:2])[0][0])
        except ValueError:
            tf_u = 0.0
        try:
            mat_b = TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=bool(trained_lex.get("tfidf_bigram_sublinear_tf", False)),
                smooth_idf=bool(trained_lex.get("tfidf_bigram_smooth_idf", True)),
            ).fit_transform([fa_str, fb_str])
            tf_b = float(cosine_similarity(mat_b[0:1], mat_b[1:2])[0][0])
        except ValueError:
            tf_b = 0.0
        return {
            "bow_count": 0.0 if math.isnan(bow_cnt) else max(0.0, min(1.0, bow_cnt)),
            "tfidf_unigram": 0.0 if math.isnan(tf_u) else max(0.0, min(1.0, tf_u)),
            "tfidf_bigram": 0.0 if math.isnan(tf_b) else max(0.0, min(1.0, tf_b)),
        }

    def _empty_response(warn_list: List[str], idf_segs: int = 0, used_ref: bool = False) -> Dict[str, Any]:
        return {
            "similarity_score": 0.0,
            "similarity_percentage": 0.0,
            "raw_similarity_score": 0.0,
            "bow_count_cosine_score": 0.0,
            "bow_count_cosine_percentage": 0.0,
            "tfidf_unigram_cosine_score": 0.0,
            "tfidf_unigram_cosine_percentage": 0.0,
            "tfidf_bigram_cosine_score": 0.0,
            "tfidf_bigram_cosine_percentage": 0.0,
            "char_ngram_cosine_score": 0.0,
            "char_ngram_cosine_percentage": 0.0,
            "ngram_1_2_cosine_score": 0.0,
            "ngram_1_2_cosine_percentage": 0.0,
            "vocabulary_size": 0,
            "dimensionality": 0,
            "is_empty_vocabulary": True,
            "idf_mode": idf_mode,
            "sublinear_tf": sublinear_tf,
            "smooth_idf": smooth_idf,
            "tokenizer_mode": tokenizer_mode,
            "warnings": warn_list,
            "formula": "CosSim(A, B) = (A · B) / (||A|| × ||B||)",
            "idf_formula": (
                "idf(t) = ln((1 + n) / (1 + df(t))) + 1"
                if smooth_idf
                else "idf(t) = ln(n / df(t)) + 1"
            ),
            "vectorization_info": {
                "vectorizer": f"Scikit-learn TfidfVectorizer (idf_mode={idf_mode}, sublinear_tf={sublinear_tf}, smooth_idf={smooth_idf})",
                "idf_mode": idf_mode,
                "sublinear_tf": sublinear_tf,
                "smooth_idf": smooth_idf,
                "tokenizer_mode": tokenizer_mode,
                "matrix_shape": [2, 0],
                "idf_corpus_segments": idf_segs,
                "used_reference_corpus": used_ref,
                "non_zero_a": 0,
                "non_zero_b": 0,
                "dot_product": 0.0,
                "norm_a": 0.0,
                "norm_b": 0.0,
                "bow_cosine_percentage": 0.0,
                "bow_cosine_score": 0.0,
                "bow_count_cosine_score": 0.0,
                "bow_count_cosine_percentage": 0.0,
                "tfidf_unigram_cosine_score": 0.0,
                "tfidf_unigram_cosine_percentage": 0.0,
                "tfidf_bigram_cosine_score": 0.0,
                "tfidf_bigram_cosine_percentage": 0.0,
                "char_ngram_cosine_percentage": 0.0,
                "ngram_1_2_cosine_percentage": 0.0,
                "top_weighted_terms": [],
                "top_terms_a": [],
                "top_terms_b": [],
                "top_contributing_terms": [],
                "term_weights": {},
            },
        }

    if not tokens_a and not tokens_b:
        return _empty_response(["Ambos documentos carecen de términos válidos para vectorizar."])

    # Función para traducir cada stem a su palabra legible más frecuente
    def _surface_for_stem(stem_or_ngram: str) -> str:
        parts = stem_or_ngram.split(" ")
        readable_parts: List[str] = []
        for st in parts:
            combined = Counter()
            if st in map_a:
                combined.update(map_a[st])
            if st in map_b:
                combined.update(map_b[st])
            if combined:
                readable_parts.append(sorted(combined.items(), key=lambda x: (-x[1], x[0]))[0][0])
            else:
                readable_parts.append(st)
        return " ".join(readable_parts)

    unique_stems_ab = sorted(list(set(tokens_a + tokens_b)))
    vocabulary_size = len(unique_stems_ab)
    used_reference_corpus = False
    fit_segments: List[Any] = []

    try:
        # Caso especial canónico: sklearn directo sobre strings cuando no hay stemming ni stopwords
        # y tokenizer_mode == "sklearn_default" con idf_mode in ("two_docs", "fit_a_transform_b")
        if (
            tokenizer_mode == "sklearn_default"
            and not use_stemming
            and not remove_stopwords
            and isinstance(doc_a, str)
            and isinstance(doc_b, str)
            and idf_mode in ("two_docs", "fit_a_transform_b")
        ):
            vectorizer = TfidfVectorizer(
                smooth_idf=smooth_idf,
                sublinear_tf=sublinear_tf,
                norm="l2",
                stop_words=None,
            )
            if idf_mode == "two_docs":
                tfidf_matrix = vectorizer.fit_transform([str_a, str_b])
                fit_segments = [str_a, str_b]
            else:
                vectorizer.fit([str_a])
                tfidf_matrix = vectorizer.transform([str_a, str_b])
                fit_segments = [str_a]
            vocabulary_size = len(vectorizer.get_feature_names_out())
        else:
            # Vectorizador usando lista de tokens ya tokenizados/normalizados según configuración
            effective_sublinear_tf = sublinear_tf
            if idf_mode == "segments":
                if len(tokens_a) >= 60 or len(tokens_b) >= 60:
                    # En documentos extensos, dividir A y B en decenas de oraciones penaliza los
                    # términos temáticos centrales (que se repiten en muchas oraciones de su propio
                    # documento) y sobre-pondera palabras incidentales raras (df=2). Ajustar a nivel
                    # documental respaldado por el corpus de referencia y TF lineal conserva la
                    # dominancia de las palabras clave de cada dominio.
                    used_reference_corpus = True
                    fit_segments = [tokens_a, tokens_b] + _get_reference_corpus_tokens(
                        use_stemming=use_stemming,
                        remove_stopwords=remove_stopwords,
                    )
                    effective_sublinear_tf = False
                else:
                    fit_segments = [seg for seg in (segments_a + segments_b) if seg]
                    if len(fit_segments) < 6:
                        used_reference_corpus = True
                        fit_segments = fit_segments + _get_reference_corpus_tokens(
                            use_stemming=use_stemming,
                            remove_stopwords=remove_stopwords,
                        )
                        warnings.append(
                            "Los documentos tienen pocos segmentos/oraciones; se utilizó el corpus de referencia en español para estabilizar la ponderación IDF."
                        )
                vocab_param: Optional[List[str]] = unique_stems_ab
            elif idf_mode == "two_docs":
                fit_segments = [tokens_a, tokens_b]
                vocab_param = unique_stems_ab
            else:  # "fit_a_transform_b"
                if not tokens_a:
                    return _empty_response(
                        ["El Documento A está vacío; no se puede ajustar IDF en modo fit_a_transform_b."],
                        idf_segs=0,
                    )
                fit_segments = [tokens_a]
                vocab_param = sorted(list(set(tokens_a)))

            vectorizer = TfidfVectorizer(
                tokenizer=_identity_tokenizer,
                preprocessor=None,
                lowercase=False,
                token_pattern=None,
                vocabulary=vocab_param,
                sublinear_tf=effective_sublinear_tf,
                norm="l2",
                use_idf=True,
                smooth_idf=smooth_idf,
            )
            vectorizer.fit(fit_segments)
            tfidf_matrix = vectorizer.transform([tokens_a, tokens_b])
            vocabulary_size = len(vectorizer.get_feature_names_out())
    except ValueError:
        return _empty_response(warnings, idf_segs=len(fit_segments), used_ref=used_reference_corpus)

    cos_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    df_sim = pd.DataFrame(
        cos_sim_matrix,
        index=["Texto A", "Texto B"],
        columns=["Texto A", "Texto B"],
    )

    raw_score = float(df_sim.loc["Texto A", "Texto B"])
    if math.isnan(raw_score):
        raw_score = 0.0
    similarity_score = max(0.0, min(1.0, raw_score))
    similarity_percentage = round(similarity_score * 100.0, 2)

    # 2. Calcular también TF-IDF con ngram_range=(1, 2) sobre los documentos
    ngram_1_2_score = similarity_score
    if ngram_range == (1, 2) and tokens_a and tokens_b:
        try:
            vec_1_2 = TfidfVectorizer(
                tokenizer=_identity_tokenizer,
                preprocessor=None,
                lowercase=False,
                token_pattern=None,
                ngram_range=(1, 2),
                sublinear_tf=sublinear_tf,
                norm="l2",
                use_idf=True,
                smooth_idf=smooth_idf,
            )
            if idf_mode == "segments":
                vec_1_2.fit([s for s in fit_segments if isinstance(s, list)] + [tokens_a, tokens_b])
            elif idf_mode == "fit_a_transform_b":
                vec_1_2.fit([tokens_a])
            else:
                vec_1_2.fit([tokens_a, tokens_b])
            mat_1_2 = vec_1_2.transform([tokens_a, tokens_b])
            s12 = float(cosine_similarity(mat_1_2[0:1], mat_1_2[1:2])[0][0])
            if not math.isnan(s12):
                ngram_1_2_score = max(0.0, min(1.0, s12))
        except ValueError:
            ngram_1_2_score = similarity_score

    # 3. Calcular Coseno con n-gramas de caracteres (char_wb, 3-5)
    char_ngram_score = _calculate_char_ngram_cosine(clean_a, clean_b)

    feature_names = vectorizer.get_feature_names_out()
    dense_matrix = tfidf_matrix.toarray()
    vec_a = dense_matrix[0]
    vec_b = dense_matrix[1]

    dot_product = float(np.dot(vec_a, vec_b))
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    non_zero_a = int(np.count_nonzero(vec_a))
    non_zero_b = int(np.count_nonzero(vec_b))

    term_weights: Dict[str, Dict[str, float]] = {}
    weighted_terms_list: List[Dict[str, Any]] = []
    top_terms_a: List[Dict[str, Any]] = []
    top_terms_b: List[Dict[str, Any]] = []
    top_contributing_terms: List[Dict[str, Any]] = []

    idf_values = vectorizer.idf_

    for idx, stem_term in enumerate(feature_names):
        w_a = float(vec_a[idx])
        w_b = float(vec_b[idx])
        idf_val = float(idf_values[idx])
        contrib = w_a * w_b
        contrib_pct = (contrib / dot_product * 100.0) if dot_product > 1e-9 else 0.0

        readable_term = _surface_for_stem(str(stem_term))
        entry = {
            "term": readable_term,
            "stem": str(stem_term),
            "tfidf_a": round(w_a, 4),
            "tfidf_b": round(w_b, 4),
            "idf": round(idf_val, 4),
            "combined_weight": round(w_a + w_b, 4),
            "contribution": round(contrib, 6),
            "contribution_percentage": round(contrib_pct, 2),
        }

        term_weights[readable_term] = {
            "tfidf_a": round(w_a, 4),
            "tfidf_b": round(w_b, 4),
            "idf": round(idf_val, 4),
            "combined_weight": round(w_a + w_b, 4),
            "contribution": round(contrib, 6),
        }
        if str(stem_term) not in term_weights:
            term_weights[str(stem_term)] = term_weights[readable_term]

        weighted_terms_list.append(entry)
        if w_a > 0:
            top_terms_a.append(entry)
        if w_b > 0:
            top_terms_b.append(entry)
        if contrib > 0:
            top_contributing_terms.append(entry)

    weighted_terms_list.sort(key=lambda item: (-item["combined_weight"], item["term"]))
    top_terms_a.sort(key=lambda item: (-item["tfidf_a"], item["term"]))
    top_terms_b.sort(key=lambda item: (-item["tfidf_b"], item["term"]))
    top_contributing_terms.sort(key=lambda item: (-item["contribution"], item["term"]))

    bow_cosine = calculate_manual_cosine_similarity(
        doc_a,
        doc_b,
        remove_stopwords=remove_stopwords,
        custom_stopwords=custom_stopwords,
        use_stemming=use_stemming,
    )
    sparse_variants = _compute_filtered_sparse_variants()

    vectorizer_desc = (
        "Scikit-learn TfidfVectorizer (Canónico 2-Docs, Linear TF, Smooth IDF, L2 Norm)"
        if idf_mode == "two_docs" and not sublinear_tf
        else f"Scikit-learn TfidfVectorizer (idf_mode={idf_mode}, sublinear_tf={sublinear_tf}, smooth_idf={smooth_idf})"
    )

    return {
        "similarity_score": round(similarity_score, 6),
        "raw_similarity_score": similarity_score,
        "similarity_percentage": similarity_percentage,
        "bow_count_cosine_score": round(sparse_variants["bow_count"], 6),
        "bow_count_cosine_percentage": round(sparse_variants["bow_count"] * 100.0, 2),
        "tfidf_unigram_cosine_score": round(sparse_variants["tfidf_unigram"], 6),
        "tfidf_unigram_cosine_percentage": round(sparse_variants["tfidf_unigram"] * 100.0, 2),
        "tfidf_bigram_cosine_score": round(sparse_variants["tfidf_bigram"], 6),
        "tfidf_bigram_cosine_percentage": round(sparse_variants["tfidf_bigram"] * 100.0, 2),
        "char_ngram_cosine_score": round(char_ngram_score, 6),
        "char_ngram_cosine_percentage": round(char_ngram_score * 100.0, 2),
        "ngram_1_2_cosine_score": round(ngram_1_2_score, 6),
        "ngram_1_2_cosine_percentage": round(ngram_1_2_score * 100.0, 2),
        "vocabulary_size": vocabulary_size,
        "dimensionality": vocabulary_size,
        "is_empty_vocabulary": False,
        "idf_mode": idf_mode,
        "sublinear_tf": sublinear_tf,
        "smooth_idf": smooth_idf,
        "tokenizer_mode": tokenizer_mode,
        "warnings": warnings,
        "formula": "CosSim(A, B) = (A · B) / (||A|| × ||B||)",
        "idf_formula": (
            "idf(t) = ln((1 + n) / (1 + df(t))) + 1"
            if smooth_idf
            else "idf(t) = ln(n / df(t)) + 1"
        ),
        "vectorization_info": {
            "vectorizer": vectorizer_desc,
            "idf_mode": idf_mode,
            "sublinear_tf": sublinear_tf,
            "smooth_idf": smooth_idf,
            "tokenizer_mode": tokenizer_mode,
            "matrix_shape": [int(tfidf_matrix.shape[0]), int(tfidf_matrix.shape[1])],
            "idf_corpus_segments": len(fit_segments),
            "used_reference_corpus": used_reference_corpus,
            "non_zero_a": non_zero_a,
            "non_zero_b": non_zero_b,
            "dot_product": round(dot_product, 6),
            "norm_a": round(norm_a, 4),
            "norm_b": round(norm_b, 4),
            "bow_cosine_score": bow_cosine["similarity_score"],
            "bow_cosine_percentage": bow_cosine["similarity_percentage"],
            "bow_count_cosine_score": round(sparse_variants["bow_count"], 6),
            "bow_count_cosine_percentage": round(sparse_variants["bow_count"] * 100.0, 2),
            "tfidf_unigram_cosine_score": round(sparse_variants["tfidf_unigram"], 6),
            "tfidf_unigram_cosine_percentage": round(sparse_variants["tfidf_unigram"] * 100.0, 2),
            "tfidf_bigram_cosine_score": round(sparse_variants["tfidf_bigram"], 6),
            "tfidf_bigram_cosine_percentage": round(sparse_variants["tfidf_bigram"] * 100.0, 2),
            "char_ngram_cosine_score": round(char_ngram_score, 6),
            "char_ngram_cosine_percentage": round(char_ngram_score * 100.0, 2),
            "ngram_1_2_cosine_score": round(ngram_1_2_score, 6),
            "ngram_1_2_cosine_percentage": round(ngram_1_2_score * 100.0, 2),
            "top_weighted_terms": weighted_terms_list[:15],
            "top_terms_a": top_terms_a[:10],
            "top_terms_b": top_terms_b[:10],
            "top_contributing_terms": top_contributing_terms[:15],
            "term_weights": term_weights,
        },
    }
