"""
Módulo de Similitud de Jaccard, Coeficientes de Conjuntos y Shingles (k=2, 3, 5) para Similarity Lab.

Guía de uso de cada métrica de conjuntos (basada en literatura de IR y detección de plagio):
- **Jaccard (`J(A, B) = |A ∩ B| / |A ∪ B|`)**:
  Ideal para detectar **copia literal** o superposición global entre dos documentos de longitud
  comparable. Penaliza si uno de los documentos añade mucho texto adicional.
- **Overlap / Szymkiewicz-Simpson (`Overlap(A, B) = |A ∩ B| / min(|A|, |B|)`) y
  Containment (`Cont(A, B) = |A ∩ B| / |A|`)**:
  Ideal cuando un **texto corto está contenido dentro de uno largo** (por ejemplo, un párrafo
  plagiado insertado dentro de un ensayo extenso). Cuando A ⊆ B, Overlap = 1.0 aunque Jaccard sea bajo.
- **Sørensen-Dice (`Dice(A, B) = 2|A ∩ B| / (|A| + |B|)`)**:
  **Pondera más los términos compartidos** (el doble en el numerador respecto a la suma de cardinalidades).
  Satisface `Dice = 2J / (1 + J)` y suaviza la caída de puntaje en textos breves.
- **Jaccard Ponderado / Multiconjunto (`∑ min(f_A, f_B) / ∑ max(f_A, f_B)`)**:
  Evalúa coincidencias considerando cuántas veces se repite cada término en cada documento.
- **Shingles de palabras (`k=2`, `k=3`, `k=5`)**:
  Aplica Jaccard sobre n-gramas contiguos de `k` palabras. `k=3` y `k=5` son los valores canónicos
  en algoritmos de MinHash y detección de copia literal de frases.

Soporta ejecución con y sin stemming/stopwords, además de tokenización estándar, por espacios
o mediante la expresión regular por defecto de scikit-learn (`(?u)\\b\\w\\w+\\b`).
"""

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
from .params_store import load_trained_params
from .preprocessing import (
    SPANISH_STOPWORDS,
    preprocess_text,
    remove_stopwords_from_tokens,
    stem_word,
    tokenize,
)

SKLEARN_DEFAULT_TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")

METRIC_USAGE_GUIDE: Dict[str, str] = {
    "jaccard": "Copia literal y superposición global entre documentos de longitud comparable (|A ∩ B| / |A ∪ B|).",
    "overlap": "Texto corto contenido dentro de uno largo (|A ∩ B| / min(|A|, |B|)). Da 1.0 si un conjunto es subconjunto del otro.",
    "containment_a_in_b": "Proporción del vocabulario de A que está incluida dentro de B (|A ∩ B| / |A|).",
    "dice": "Pondera el doble los términos compartidos frente a la suma de tamaños individuales (2|A ∩ B| / (|A| + |B|)).",
    "weighted_jaccard": "Jaccard sobre multiconjuntos ponderado por la frecuencia de aparición de cada término.",
    "shingles_k2_k3_k5": "Detección de copia literal de frases mediante n-gramas contiguos de k=2, k=3 y k=5 palabras (estándar MinHash).",
}


def _tokenize_raw_reference(text: str, tokenizer_mode: str = "standard") -> List[str]:
    """
    Tokenización mínima para el Modo Referencia o el arnés de calibración:
    - 'sklearn_regex': usa el patrón por defecto de Scikit-learn `(?u)\\b\\w\\w+\\b` en minúsculas.
    - 'whitespace': convierte a minúsculas y separa por espacios (`text.lower().split()`).
    - 'standard': minúsculas y tokenización alfanumérica por palabras sin stemming ni stopwords.
    """
    if not text or not isinstance(text, str):
        return []
    lowered = text.lower().strip()
    if not lowered:
        return []
    if tokenizer_mode == "sklearn_regex":
        return SKLEARN_DEFAULT_TOKEN_PATTERN.findall(lowered)
    if tokenizer_mode == "whitespace":
        # Limpiar puntuación básica pegada a bordes o split directo por espacios
        cleaned = re.sub(r"[^\w\sáéíóúüñ]", " ", lowered)
        return [tok for tok in cleaned.split() if tok]
    return tokenize(lowered, unify_accents=False)


def _extract_normalized_tokens_and_surfaces(
    source: Union[str, Iterable[Any]],
    remove_stopwords: bool = False,
    custom_stopwords: Optional[Set[str]] = None,
    use_stemming: bool = True,
    tokenizer_mode: str = "standard",
) -> Tuple[List[str], List[str], Dict[str, Counter]]:
    """
    Extrae la lista secuencial de tokens normalizados (stems o crudos), la lista de tokens
    de superficie legibles y un mapa de frecuencias de superficie por cada token normalizado.
    Acepta tanto cadenas de texto como conjuntos/listas arbitrarias (ej. `{1, 2, 3, 4, 5}`).
    """
    if isinstance(source, str):
        if tokenizer_mode in ("sklearn_regex", "whitespace"):
            surface_tokens = _tokenize_raw_reference(source, tokenizer_mode=tokenizer_mode)
            if remove_stopwords:
                surface_tokens = remove_stopwords_from_tokens(
                    surface_tokens, custom_stopwords=custom_stopwords
                )
        else:
            surface_tokens = preprocess_text(
                source,
                remove_stopwords=remove_stopwords,
                custom_stopwords=custom_stopwords,
                use_stemming=False,
            )
    elif source is None:
        surface_tokens = []
    else:
        surface_tokens = [str(token).strip().lower() for token in source if str(token).strip() != ""]
        if remove_stopwords:
            stop_set = custom_stopwords if custom_stopwords is not None else SPANISH_STOPWORDS
            surface_tokens = [t for t in surface_tokens if t not in stop_set]

    norm_tokens: List[str] = []
    stem_to_surfaces: Dict[str, Counter] = {}

    for surf in surface_tokens:
        norm = stem_word(surf, unify_accents=True) if use_stemming else surf
        if norm == "":
            continue
        norm_tokens.append(norm)
        if norm not in stem_to_surfaces:
            stem_to_surfaces[norm] = Counter()
        stem_to_surfaces[norm][surf] += 1

    return norm_tokens, surface_tokens, stem_to_surfaces


def _build_word_shingles(tokens: List[str], k: int) -> Set[str]:
    """
    Construye el conjunto de k-shingles (n-gramas contiguos de palabras) de longitud `k`.
    """
    if len(tokens) < k or k <= 0:
        return set()
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def _shingle_jaccard(tokens_a: List[str], tokens_b: List[str], k: int) -> Dict[str, Any]:
    """
    Calcula el índice de Jaccard, Dice y Overlap entre conjuntos de k-shingles de longitud `k`.
    """
    shingles_a = _build_word_shingles(tokens_a, k)
    shingles_b = _build_word_shingles(tokens_b, k)
    if not shingles_a and not shingles_b:
        return {
            "k": k,
            "score": 0.0,
            "percentage": 0.0,
            "intersection_size": 0,
            "union_size": 0,
            "count_a": 0,
            "count_b": 0,
            "shared": [],
        }
    union = shingles_a.union(shingles_b)
    intersection = shingles_a.intersection(shingles_b)
    score = max(0.0, min(1.0, float(len(intersection)) / float(len(union)))) if union else 0.0
    return {
        "k": k,
        "score": round(score, 6),
        "percentage": round(score * 100.0, 2),
        "intersection_size": len(intersection),
        "union_size": len(union),
        "count_a": len(shingles_a),
        "count_b": len(shingles_b),
        "shared": sorted(list(intersection)),
    }


def _compute_core_set_metrics(
    norm_a: List[str],
    norm_b: List[str],
    surf_a: List[str],
    surf_b: List[str],
    map_a: Dict[str, Counter],
    map_b: Dict[str, Counter],
) -> Dict[str, Any]:
    """
    Calcula todas las métricas de conjuntos (Jaccard, Dice, Overlap/Containment,
    Jaccard ponderado y Shingles k=2, 3, 5) para un par de secuencias de tokens.
    """
    set_a = set(norm_a)
    set_b = set(norm_b)
    freq_a = Counter(norm_a)
    freq_b = Counter(norm_b)

    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    exclusive_a = set_a.difference(set_b)
    exclusive_b = set_b.difference(set_a)

    intersection_size = len(intersection)
    union_size = len(union)
    unique_a_count = len(set_a)
    unique_b_count = len(set_b)

    # 1. Índice de Jaccard clásico: |A ∩ B| / |A ∪ B|
    if union_size == 0:
        similarity_score = 0.0
        jaccard_distance = 1.0
    else:
        similarity_score = max(0.0, min(1.0, float(intersection_size) / float(union_size)))
        jaccard_distance = max(0.0, min(1.0, 1.0 - similarity_score))

    # 2. Coeficiente de Sørensen-Dice: 2|A ∩ B| / (|A| + |B|)
    denom_dice = unique_a_count + unique_b_count
    dice_score = (
        max(0.0, min(1.0, (2.0 * intersection_size) / float(denom_dice)))
        if denom_dice > 0
        else 0.0
    )

    # 3. Coeficiente de Solapamiento / Overlap: |A ∩ B| / min(|A|, |B|)
    #    y Containment direccional: |A ∩ B| / |A| y |A ∩ B| / |B|
    min_card = min(unique_a_count, unique_b_count)
    containment_score = (
        max(0.0, min(1.0, float(intersection_size) / float(min_card)))
        if min_card > 0
        else 0.0
    )
    containment_a_in_b = (
        max(0.0, min(1.0, float(intersection_size) / float(unique_a_count)))
        if unique_a_count > 0
        else 0.0
    )
    containment_b_in_a = (
        max(0.0, min(1.0, float(intersection_size) / float(unique_b_count)))
        if unique_b_count > 0
        else 0.0
    )

    # 4. Jaccard Ponderado (Multiconjunto por frecuencia): ∑ min(f_A, f_B) / ∑ max(f_A, f_B)
    if union_size == 0:
        weighted_jaccard_score = 0.0
    else:
        min_sum = sum(min(freq_a[t], freq_b[t]) for t in union)
        max_sum = sum(max(freq_a[t], freq_b[t]) for t in union)
        weighted_jaccard_score = (
            max(0.0, min(1.0, float(min_sum) / float(max_sum))) if max_sum > 0 else 0.0
        )

    # 5. Shingles k=2, k=3 y k=5 (sobre tokens normalizados y sobre superficie)
    shingle_k2_norm = _shingle_jaccard(norm_a, norm_b, 2)
    shingle_k3_norm = _shingle_jaccard(norm_a, norm_b, 3)
    shingle_k5_norm = _shingle_jaccard(norm_a, norm_b, 5)

    shingle_k2_surf = _shingle_jaccard(surf_a, surf_b, 2)
    shingle_k3_surf = _shingle_jaccard(surf_a, surf_b, 3)
    shingle_k5_surf = _shingle_jaccard(surf_a, surf_b, 5)

    def _surface_for_stem(stem: str) -> str:
        combined = Counter()
        if stem in map_a:
            combined.update(map_a[stem])
        if stem in map_b:
            combined.update(map_b[stem])
        if not combined:
            return stem
        return sorted(combined.items(), key=lambda x: (-x[1], x[0]))[0][0]

    shared_stems_sorted = sorted(
        list(intersection),
        key=lambda st: (-(freq_a[st] + freq_b[st]), _surface_for_stem(st)),
    )
    shared_words = [_surface_for_stem(st) for st in shared_stems_sorted]

    exclusive_a_stems_sorted = sorted(
        list(exclusive_a),
        key=lambda st: (-freq_a[st], _surface_for_stem(st)),
    )
    exclusive_a_words = [_surface_for_stem(st) for st in exclusive_a_stems_sorted]

    exclusive_b_stems_sorted = sorted(
        list(exclusive_b),
        key=lambda st: (-freq_b[st], _surface_for_stem(st)),
    )
    exclusive_b_words = [_surface_for_stem(st) for st in exclusive_b_stems_sorted]

    union_stems_sorted = sorted(
        list(union),
        key=lambda st: (-(freq_a[st] + freq_b[st]), _surface_for_stem(st)),
    )
    union_words = [_surface_for_stem(st) for st in union_stems_sorted]

    unique_a_words = sorted([_surface_for_stem(st) for st in set_a])
    unique_b_words = sorted([_surface_for_stem(st) for st in set_b])

    morphological_matches: List[Dict[str, Any]] = []
    for st in shared_stems_sorted:
        surf_in_a = sorted(map_a[st].keys())[0] if st in map_a and map_a[st] else st
        surf_in_b = sorted(map_b[st].keys())[0] if st in map_b and map_b[st] else st
        if surf_in_a != surf_in_b:
            morphological_matches.append({
                "stem": st,
                "word_a": surf_in_a,
                "word_b": surf_in_b,
                "freq_a": int(freq_a[st]),
                "freq_b": int(freq_b[st]),
            })

    return {
        "similarity_score": round(similarity_score, 6),
        "similarity_percentage": round(similarity_score * 100.0, 2),
        "jaccard_distance": round(jaccard_distance, 6),
        "dice_score": round(dice_score, 6),
        "dice_percentage": round(dice_score * 100.0, 2),
        "overlap_score": round(containment_score, 6),
        "overlap_percentage": round(containment_score * 100.0, 2),
        "containment_score": round(containment_score, 6),
        "containment_percentage": round(containment_score * 100.0, 2),
        "containment_a_in_b": round(containment_a_in_b, 6),
        "containment_b_in_a": round(containment_b_in_a, 6),
        "weighted_jaccard_score": round(weighted_jaccard_score, 6),
        "weighted_jaccard_percentage": round(weighted_jaccard_score * 100.0, 2),
        # Shingles k=2, k=3 y k=5
        "bigram_jaccard_score": shingle_k2_norm["score"],
        "bigram_jaccard_percentage": shingle_k2_norm["percentage"],
        "trigram_jaccard_score": shingle_k3_norm["score"],
        "trigram_jaccard_percentage": shingle_k3_norm["percentage"],
        "pentagram_jaccard_score": shingle_k5_norm["score"],
        "pentagram_jaccard_percentage": shingle_k5_norm["percentage"],
        "shingle_5_jaccard_score": shingle_k5_norm["score"],
        "shingle_5_jaccard_percentage": shingle_k5_norm["percentage"],
        "shared_bigrams": shingle_k2_surf["shared"] or shingle_k2_norm["shared"],
        "shared_trigrams": shingle_k3_surf["shared"] or shingle_k3_norm["shared"],
        "shared_pentagrams": shingle_k5_surf["shared"] or shingle_k5_norm["shared"],
        "shingles": {
            "k2": shingle_k2_norm,
            "k3": shingle_k3_norm,
            "k5": shingle_k5_norm,
        },
        "shared_count": intersection_size,
        "intersection_size": intersection_size,
        "union_size": union_size,
        "unique_a_count": unique_a_count,
        "unique_b_count": unique_b_count,
        "exclusive_a_count": len(exclusive_a),
        "exclusive_b_count": len(exclusive_b),
        "shared_words": shared_words,
        "union_words": union_words,
        "exclusive_a_words": exclusive_a_words,
        "exclusive_b_words": exclusive_b_words,
        "unique_a_words": unique_a_words,
        "unique_b_words": unique_b_words,
        "morphological_matches": morphological_matches,
        "stem_frequencies_a": {_surface_for_stem(st): int(cnt) for st, cnt in freq_a.items()},
        "stem_frequencies_b": {_surface_for_stem(st): int(cnt) for st, cnt in freq_b.items()},
        "formula": "J(A, B) = |A ∩ B| / |A ∪ B|",
        "usage_guide": METRIC_USAGE_GUIDE,
        "is_empty_union": union_size == 0,
    }


def calculate_jaccard_similarity(
    doc_a: Union[str, Iterable[Any]],
    doc_b: Union[str, Iterable[Any]],
    remove_stopwords: bool = False,
    custom_stopwords: Optional[Set[str]] = None,
    use_stemming: bool = True,
    tokenizer_mode: str = "standard",
    include_variants: bool = True,
    compute_variants: Optional[bool] = None,
    params_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Calcula la familia completa de métricas de similitud de conjuntos:
    - Jaccard clásico (`similarity_score`)
    - Sørensen-Dice (`dice_score`)
    - Overlap / Containment (`containment_score`, `containment_a_in_b`, `containment_b_in_a`)
    - Jaccard Ponderado sobre multiconjunto (`weighted_jaccard_score`)
    - Shingles para k=2, k=3 y k=5 (`shingles['k2']`, `shingles['k3']`, `shingles['k5']`)

    Expone además en `variants` los resultados **con** y **sin** stemming/stopwords
    para permitir comparación directa entre el Modo Referencia y el Modo Enriquecido.
    """
    norm_a, surf_a, map_a = _extract_normalized_tokens_and_surfaces(
        doc_a,
        remove_stopwords=remove_stopwords,
        custom_stopwords=custom_stopwords,
        use_stemming=use_stemming,
        tokenizer_mode=tokenizer_mode,
    )
    norm_b, surf_b, map_b = _extract_normalized_tokens_and_surfaces(
        doc_b,
        remove_stopwords=remove_stopwords,
        custom_stopwords=custom_stopwords,
        use_stemming=use_stemming,
        tokenizer_mode=tokenizer_mode,
    )

    result = _compute_core_set_metrics(norm_a, norm_b, surf_a, surf_b, map_a, map_b)
    result["config"] = {
        "remove_stopwords": bool(remove_stopwords),
        "use_stemming": bool(use_stemming),
        "tokenizer_mode": tokenizer_mode,
    }

    # Jaccard (Bruto): sets de palabras alfabéticas sin filtro de stopwords ni stemming
    alpha_pattern = re.compile(r"\b[a-záéíóúüñ]+\b")
    if isinstance(doc_a, str):
        raw_alpha_a = alpha_pattern.findall(doc_a.lower())
    elif doc_a is None:
        raw_alpha_a = []
    else:
        raw_alpha_a = [str(t).strip().lower() for t in doc_a if str(t).strip()]

    if isinstance(doc_b, str):
        raw_alpha_b = alpha_pattern.findall(doc_b.lower())
    elif doc_b is None:
        raw_alpha_b = []
    else:
        raw_alpha_b = [str(t).strip().lower() for t in doc_b if str(t).strip()]

    set_raw_alpha_a = set(raw_alpha_a)
    set_raw_alpha_b = set(raw_alpha_b)
    inter_raw_alpha = set_raw_alpha_a.intersection(set_raw_alpha_b)
    union_raw_alpha = set_raw_alpha_a.union(set_raw_alpha_b)
    raw_alpha_score = (
        max(0.0, min(1.0, float(len(inter_raw_alpha)) / float(len(union_raw_alpha))))
        if union_raw_alpha
        else 0.0
    )

    # Jaccard (Filtrado): sets de palabras de superficie sin stopwords ni números
    trained_lex = (params_override or load_trained_params()).get("lexical", {})
    min_tok_len = int(trained_lex.get("min_token_len", 3))
    stop_filter = set(SPANISH_STOPWORDS)
    if custom_stopwords:
        stop_filter |= {str(w).strip().lower() for w in custom_stopwords}
    elif isinstance(doc_a, str) and isinstance(doc_b, str) and (len(raw_alpha_a) >= 40 or len(raw_alpha_b) >= 40):
        stop_filter |= {str(w).strip().lower() for w in trained_lex.get("extra_stopwords", [])}
    filt_surf_a = [w for w in raw_alpha_a if w not in stop_filter and (not isinstance(doc_a, str) or len(w) >= min_tok_len)]
    filt_surf_b = [w for w in raw_alpha_b if w not in stop_filter and (not isinstance(doc_b, str) or len(w) >= min_tok_len)]
    set_filt_a = set(filt_surf_a)
    set_filt_b = set(filt_surf_b)
    inter_filt = set_filt_a.intersection(set_filt_b)
    union_filt = set_filt_a.union(set_filt_b)
    filt_surf_score = (
        max(0.0, min(1.0, float(len(inter_filt)) / float(len(union_filt))))
        if union_filt
        else 0.0
    )

    result["raw_alpha_jaccard_score"] = round(raw_alpha_score, 6)
    result["raw_alpha_jaccard_percentage"] = round(raw_alpha_score * 100.0, 2)
    result["raw_alpha_intersection_size"] = len(inter_raw_alpha)
    result["raw_alpha_union_size"] = len(union_raw_alpha)

    result["filtered_surface_jaccard_score"] = round(filt_surf_score, 6)
    result["filtered_surface_jaccard_percentage"] = round(filt_surf_score * 100.0, 2)
    result["filtered_surface_intersection_size"] = len(inter_filt)
    result["filtered_surface_union_size"] = len(union_filt)
    result["filtered_surface_shared_words"] = sorted(list(inter_filt))

    if compute_variants is not None:
        include_variants = bool(compute_variants)
    if include_variants:
        # Variante 1: Modo Referencia canónico (sin stopwords, sin stemming, set de palabras crudas)
        raw_a, raw_surf_a, raw_map_a = _extract_normalized_tokens_and_surfaces(
            doc_a,
            remove_stopwords=False,
            custom_stopwords=custom_stopwords,
            use_stemming=False,
            tokenizer_mode=tokenizer_mode,
        )
        raw_b, raw_surf_b, raw_map_b = _extract_normalized_tokens_and_surfaces(
            doc_b,
            remove_stopwords=False,
            custom_stopwords=custom_stopwords,
            use_stemming=False,
            tokenizer_mode=tokenizer_mode,
        )
        ref_metrics = _compute_core_set_metrics(raw_a, raw_b, raw_surf_a, raw_surf_b, raw_map_a, raw_map_b)

        # Variante 2: Sin stopwords pero sin stemming (superficie filtrada)
        nostop_a, nostop_surf_a, nostop_map_a = _extract_normalized_tokens_and_surfaces(
            doc_a,
            remove_stopwords=True,
            custom_stopwords=custom_stopwords,
            use_stemming=False,
            tokenizer_mode=tokenizer_mode,
        )
        nostop_b, nostop_surf_b, nostop_map_b = _extract_normalized_tokens_and_surfaces(
            doc_b,
            remove_stopwords=True,
            custom_stopwords=custom_stopwords,
            use_stemming=False,
            tokenizer_mode=tokenizer_mode,
        )
        nostop_metrics = _compute_core_set_metrics(
            nostop_a, nostop_surf_b if False else nostop_b, nostop_surf_a, nostop_surf_b, nostop_map_a, nostop_map_b
        )

        # Variante 3: Modo Enriquecido (con eliminación de stopwords y stemming Snowball)
        enh_a, enh_surf_a, enh_map_a = _extract_normalized_tokens_and_surfaces(
            doc_a,
            remove_stopwords=True,
            custom_stopwords=custom_stopwords,
            use_stemming=True,
            tokenizer_mode="standard",
        )
        enh_b, enh_surf_b, enh_map_b = _extract_normalized_tokens_and_surfaces(
            doc_b,
            remove_stopwords=True,
            custom_stopwords=custom_stopwords,
            use_stemming=True,
            tokenizer_mode="standard",
        )
        enh_metrics = _compute_core_set_metrics(enh_a, enh_b, enh_surf_a, enh_surf_b, enh_map_a, enh_map_b)

        # Variante 4: Con stemming pero conservando stopwords
        stem_sw_a, stem_sw_surf_a, stem_sw_map_a = _extract_normalized_tokens_and_surfaces(
            doc_a,
            remove_stopwords=False,
            custom_stopwords=custom_stopwords,
            use_stemming=True,
            tokenizer_mode="standard",
        )
        stem_sw_b, stem_sw_surf_b, stem_sw_map_b = _extract_normalized_tokens_and_surfaces(
            doc_b,
            remove_stopwords=False,
            custom_stopwords=custom_stopwords,
            use_stemming=True,
            tokenizer_mode="standard",
        )
        stem_sw_metrics = _compute_core_set_metrics(
            stem_sw_a, stem_sw_b, stem_sw_surf_a, stem_sw_surf_b, stem_sw_map_a, stem_sw_map_b
        )

        result["variants"] = {
            "without_stemming_without_stopwords": {
                "jaccard_score": ref_metrics["similarity_score"],
                "jaccard_percentage": ref_metrics["similarity_percentage"],
                "dice_score": ref_metrics["dice_score"],
                "overlap_score": ref_metrics["overlap_score"],
                "weighted_jaccard_score": ref_metrics["weighted_jaccard_score"],
                "shingles_k2": ref_metrics["bigram_jaccard_score"],
                "shingles_k3": ref_metrics["trigram_jaccard_score"],
                "shingles_k5": ref_metrics["pentagram_jaccard_score"],
            },
            "without_stemming_with_stopwords_removed": {
                "jaccard_score": nostop_metrics["similarity_score"],
                "jaccard_percentage": nostop_metrics["similarity_percentage"],
                "dice_score": nostop_metrics["dice_score"],
                "overlap_score": nostop_metrics["overlap_score"],
                "weighted_jaccard_score": nostop_metrics["weighted_jaccard_score"],
                "shingles_k2": nostop_metrics["bigram_jaccard_score"],
                "shingles_k3": nostop_metrics["trigram_jaccard_score"],
                "shingles_k5": nostop_metrics["pentagram_jaccard_score"],
            },
            "with_stemming_without_stopwords": {
                "jaccard_score": stem_sw_metrics["similarity_score"],
                "jaccard_percentage": stem_sw_metrics["similarity_percentage"],
                "dice_score": stem_sw_metrics["dice_score"],
                "overlap_score": stem_sw_metrics["overlap_score"],
                "weighted_jaccard_score": stem_sw_metrics["weighted_jaccard_score"],
                "shingles_k2": stem_sw_metrics["bigram_jaccard_score"],
                "shingles_k3": stem_sw_metrics["trigram_jaccard_score"],
                "shingles_k5": stem_sw_metrics["pentagram_jaccard_score"],
            },
            "with_stemming_with_stopwords_removed": {
                "jaccard_score": enh_metrics["similarity_score"],
                "jaccard_percentage": enh_metrics["similarity_percentage"],
                "dice_score": enh_metrics["dice_score"],
                "overlap_score": enh_metrics["overlap_score"],
                "weighted_jaccard_score": enh_metrics["weighted_jaccard_score"],
                "shingles_k2": enh_metrics["bigram_jaccard_score"],
                "shingles_k3": enh_metrics["trigram_jaccard_score"],
                "shingles_k5": enh_metrics["pentagram_jaccard_score"],
            },
        }

    return result
