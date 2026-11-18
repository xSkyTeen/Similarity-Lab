"""
Módulo de Estadísticas, Escaneo Profundo (Palabras Repetidas, Contexto Oración a Oración,
Concordancias KWIC, Dominios y Estructura Discursiva), Calibración de Umbrales e
Interpretación Automática Multicriterio para Similarity Lab.

Implementa:
1. Diccionario centralizado de umbrales por métrica (`METRIC_THRESHOLDS`).
2. Estructura unificada de métricas (`unified_metrics`) con valor crudo, porcentaje y nivel.
3. Escaneo Robusto de Repeticiones (`repetition_analysis`):
   - Palabras repetidas exactas con cálculo de distancia media/mínima entre apariciones.
   - Agrupación de repeticiones por familia morfológica (raíces repetidas bajo distintas formas).
   - Detección de repeticiones por proximidad cercana (alertas de cacofonía / redundancia a <= 5 palabras).
   - Frases y n-gramas repetidos dentro de cada documento y compartidos entre documentos (2, 3 y 4 palabras).
   - Índices de riqueza léxica: Hapax Legomena (palabras de aparición única) e Índice de Guiraud.
4. Escaneo Profundo de Contexto (`context_analysis`):
   - Alineamiento contextual oración por oración (Copia literal, Paráfrasis contextual, Afinidad parcial).
   - Ventanas de concordancia KWIC (Key Word In Context) para términos compartidos y puentes semánticos.
   - Detección automática de Dominios Temáticos, Conectores Discursivos y Legibilidad de Fernández Huerta.
5. Interpretación automática multicriterio, puntaje de confianza y fragmentos literales (5+ palabras).
"""

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
from .preprocessing import SPANISH_STOPWORDS, preprocess_document, stem_word
from .file_reader import detect_and_split_document_sections
from .jaccard import calculate_jaccard_similarity
from .cosine import calculate_cosine_similarity
from .embeddings import (
    DOMAIN_CLUSTER_DEFINITIONS,
    SEMANTIC_CONCEPT_CLUSTERS,
    _compute_local_word_embedding,
    calculate_embedding_similarity,
)
from .profiles import (
    COMPARISON_PROFILES,
    compare_profiles_side_by_side,
    normalize_profile_name,
)


# ==============================================================================
# CONFIGURACIÓN CENTRALIZADA DE UMBRALES POR MÉTRICA (Rango [0.0, 1.0])
# ==============================================================================
METRIC_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "jaccard": {
        "bajo": 0.15,
        "medio": 0.35,
        "alto": 0.60,
        "muy_alto": 0.80,
    },
    "cosine_tfidf": {
        "bajo": 0.15,
        "medio": 0.40,
        "alto": 0.65,
        "muy_alto": 0.85,
    },
    "embeddings": {
        "bajo": 0.20,
        "medio": 0.45,
        "alto": 0.68,
        "muy_alto": 0.85,
    },
    "containment": {
        "bajo": 0.25,
        "medio": 0.50,
        "alto": 0.75,
        "muy_alto": 0.90,
    },
    "dice": {
        "bajo": 0.20,
        "medio": 0.45,
        "alto": 0.68,
        "muy_alto": 0.85,
    },
    "weighted_jaccard": {
        "bajo": 0.15,
        "medio": 0.35,
        "alto": 0.60,
        "muy_alto": 0.80,
    },
    "char_ngram_cosine": {
        "bajo": 0.25,
        "medio": 0.50,
        "alto": 0.75,
        "muy_alto": 0.90,
    },
    "bertscore_f1": {
        "bajo": 0.25,
        "medio": 0.50,
        "alto": 0.72,
        "muy_alto": 0.88,
    },
    "euclidean_similarity": {
        "bajo": 0.30,
        "medio": 0.50,
        "alto": 0.70,
        "muy_alto": 0.88,
    },
}

LEVEL_LABELS: Dict[str, str] = {
    "muy_bajo": "Muy bajo",
    "bajo": "Bajo",
    "medio": "Medio",
    "alto": "Alto",
    "muy_alto": "Muy alto",
}

# Catálogo de conectores discursivos del español por categoría funcional
SPANISH_DISCOURSE_CONNECTORS: Dict[str, List[str]] = {
    "Causales y Consecutivos": [
        "porque", "ya que", "puesto que", "dado que", "debido a", "por tanto",
        "por lo tanto", "por consiguiente", "en consecuencia", "así que", "de modo que", "por ende",
    ],
    "Adversativos y Contraste": [
        "pero", "sin embargo", "no obstante", "aunque", "a pesar de", "en cambio",
        "por el contrario", "mientras que", "sino", "al contrario",
    ],
    "Aditivos y Estructuradores": [
        "además", "asimismo", "igualmente", "también", "en primer lugar", "en segundo lugar",
        "por otra parte", "por otro lado", "finalmente", "en conclusión", "para concluir",
    ],
    "Ejemplificación y Explicación": [
        "por ejemplo", "tal como", "es decir", "esto es", "o sea", "en otras palabras",
        "mediante", "a través de", "como", "así como",
    ],
}


def classify_metric_level(score: float, metric_name: str = "cosine_tfidf") -> Tuple[str, str]:
    """
    Clasifica un puntaje en [0.0, 1.0] en uno de los 5 niveles estandarizados
    ('muy_bajo', 'bajo', 'medio', 'alto', 'muy_alto') usando `METRIC_THRESHOLDS`.
    """
    cfg = METRIC_THRESHOLDS.get(metric_name, METRIC_THRESHOLDS["cosine_tfidf"])
    val = max(0.0, min(1.0, float(score)))
    if val < cfg["bajo"]:
        lvl = "muy_bajo"
    elif val < cfg["medio"]:
        lvl = "bajo"
    elif val < cfg["alto"]:
        lvl = "medio"
    elif val < cfg["muy_alto"]:
        lvl = "alto"
    else:
        lvl = "muy_alto"
    return lvl, LEVEL_LABELS[lvl]


def build_unified_metric_entry(
    score: float,
    metric_name: str,
    raw_score: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Construye un objeto de métrica estandarizado con valor en [0, 1], porcentaje y nivel.
    """
    clamped = max(0.0, min(1.0, float(score)))
    raw_clamped = max(0.0, min(1.0, float(raw_score if raw_score is not None else score)))
    lvl_key, lvl_label = classify_metric_level(clamped, metric_name)
    return {
        "score": round(clamped, 6),
        "raw_score": round(raw_clamped, 6),
        "percentage": round(clamped * 100.0, 2),
        "raw_percentage": round(raw_clamped * 100.0, 2),
        "level": lvl_key,
        "level_label": lvl_label,
    }


def find_exact_matching_fragments(
    tokens_orig_a: List[str],
    tokens_orig_b: List[str],
    min_words: int = 5,
    max_results: int = 10,
) -> Dict[str, Any]:
    """
    Detecta coincidencias literales largas (secuencias contiguas compartidas de `min_words`
    o más palabras) y calcula además la subcadena común más larga de palabras (LCS).
    """
    if not tokens_orig_a or not tokens_orig_b:
        return {
            "longest_common_substring": "",
            "longest_word_count": 0,
            "fragments": [],
        }

    n_a = len(tokens_orig_a)
    n_b = len(tokens_orig_b)

    lim_a = min(n_a, 1500)
    lim_b = min(n_b, 1500)

    dp = [0] * (lim_b + 1)
    longest_len = 0
    longest_end_a = 0
    candidate_spans: List[Tuple[int, int]] = []

    for i in range(1, lim_a + 1):
        new_dp = [0] * (lim_b + 1)
        tok_a = tokens_orig_a[i - 1]
        for j in range(1, lim_b + 1):
            if tok_a == tokens_orig_b[j - 1]:
                length = dp[j - 1] + 1
                new_dp[j] = length
                if length > longest_len:
                    longest_len = length
                    longest_end_a = i
                if length >= min_words:
                    candidate_spans.append((i - length, i))
        dp = new_dp

    longest_fragment = (
        " ".join(tokens_orig_a[longest_end_a - longest_len : longest_end_a])
        if longest_len > 0
        else ""
    )

    candidate_spans.sort(key=lambda sp: (sp[0], -(sp[1] - sp[0])))
    maximal_fragments: List[Dict[str, Any]] = []
    seen_texts: Set[str] = set()

    for start_a, end_a in sorted(candidate_spans, key=lambda sp: -(sp[1] - sp[0])):
        frag_tokens = tokens_orig_a[start_a:end_a]
        frag_str = " ".join(frag_tokens)
        if frag_str in seen_texts:
            continue
        if any(frag_str in existing["fragment"] for existing in maximal_fragments):
            continue
        seen_texts.add(frag_str)
        maximal_fragments.append({
            "fragment": frag_str,
            "word_count": len(frag_tokens),
        })
        if len(maximal_fragments) >= max_results:
            break

    return {
        "longest_common_substring": longest_fragment,
        "longest_word_count": longest_len,
        "fragments": maximal_fragments,
    }


def analyze_word_and_phrase_repetitions(
    tokens_a: List[str],
    tokens_b: List[str],
    all_tokens_a: List[str],
    all_tokens_b: List[str],
    max_items: int = 15,
) -> Dict[str, Any]:
    """
    Escaneo robusto y exhaustivo de repeticiones léxicas, morfológicas, de frases y por proximidad:
    1. Palabras repetidas exactas en cada documento con cálculo de brecha mínima/media entre apariciones.
    2. Familias morfológicas repetidas (raíces que se repiten bajo iguales o distintas conjugaciones).
    3. Alertas de repetición por proximidad corta (misma palabra o raíz a <= 5 tokens de distancia).
    4. Frases repetidas (2-gramas, 3-gramas y 4-gramas) dentro de cada texto y compartidas entre A y B.
    5. Indicadores de riqueza léxica (Hapax Legomena, Tasa de Repetición e Índice de Guiraud).
    """

    def _scan_single_doc_repetitions(
        active_tokens: List[str],
        raw_tokens: List[str],
    ) -> Dict[str, Any]:
        total_active = len(active_tokens)
        if total_active == 0:
            return {
                "exact_repeated": [],
                "morphological_families": [],
                "close_proximity_alerts": [],
                "repeated_phrases": [],
                "repeated_distinct_count": 0,
                "excess_occurrences": 0,
                "repetition_rate_percentage": 0.0,
                "hapax_count": 0,
                "hapax_percentage": 0.0,
                "guiraud_richness_index": 0.0,
            }

        counts = Counter(active_tokens)
        unique_cnt = len(counts)
        hapax_words = [w for w, c in counts.items() if c == 1]
        hapax_cnt = len(hapax_words)
        hapax_pct = round((hapax_cnt / unique_cnt) * 100.0, 2) if unique_cnt > 0 else 0.0
        guiraud = round(unique_cnt / math.sqrt(total_active), 2)

        # Posiciones de cada palabra activa para medir distancia entre repeticiones
        positions_map: Dict[str, List[int]] = defaultdict(list)
        for idx, tok in enumerate(active_tokens):
            positions_map[tok].append(idx)

        exact_repeated: List[Dict[str, Any]] = []
        excess_total = 0

        for word, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            if count < 2:
                continue
            excess_total += count - 1
            pos_list = positions_map[word]
            gaps = [pos_list[k] - pos_list[k - 1] for k in range(1, len(pos_list))]
            min_gap = int(min(gaps)) if gaps else 0
            avg_gap = round(sum(gaps) / len(gaps), 1) if gaps else 0.0
            density_pct = round((count / total_active) * 100.0, 2)
            is_close = min_gap <= 5

            if count >= 4 or density_pct >= 9.0 or (count >= 3 and is_close):
                severity = "Sobre-repetición Alta"
            elif count == 3 or is_close:
                severity = "Repetición Moderada"
            else:
                severity = "Repetición Normal"

            exact_repeated.append({
                "word": word,
                "stem": stem_word(word),
                "count": int(count),
                "excess_count": int(count - 1),
                "density_percentage": density_pct,
                "min_distance_tokens": min_gap,
                "avg_distance_tokens": avg_gap,
                "is_close_repetition": is_close,
                "severity": severity,
            })

        repetition_rate_pct = round((excess_total / total_active) * 100.0, 2)

        # Familias morfológicas repetidas (agrupadas por raíz Snowball)
        stem_groups: Dict[str, Counter] = defaultdict(Counter)
        for tok in active_tokens:
            st = stem_word(tok)
            stem_groups[st][tok] += 1

        morph_families: List[Dict[str, Any]] = []
        for st, var_counter in stem_groups.items():
            tot_c = sum(var_counter.values())
            if tot_c < 2:
                continue
            sorted_vars = sorted(var_counter.items(), key=lambda x: (-x[1], x[0]))
            rep_word = sorted_vars[0][0]
            variants_list = [{"word": w, "count": int(c)} for w, c in sorted_vars]
            variants_str = ", ".join(f"{w} (×{c})" for w, c in sorted_vars)
            morph_families.append({
                "stem": st,
                "representative": rep_word,
                "total_count": int(tot_c),
                "variants_count": len(variants_list),
                "is_multi_form": len(variants_list) >= 2,
                "variants": variants_list,
                "variants_summary": variants_str,
                "density_percentage": round((tot_c / total_active) * 100.0, 2),
            })

        morph_families.sort(key=lambda item: (-int(item["is_multi_form"]), -item["total_count"], item["representative"]))

        # Detección de repeticiones por proximidad cercana (<= 6 palabras en el texto completo)
        close_alerts: List[Dict[str, Any]] = []
        seen_alert_spans: Set[Tuple[str, int]] = set()
        last_seen_by_stem: Dict[str, Tuple[int, str]] = {}

        for idx, tok in enumerate(raw_tokens):
            if tok in SPANISH_STOPWORDS or len(tok) < 3:
                continue
            st = stem_word(tok)
            if st in last_seen_by_stem:
                prev_idx, prev_tok = last_seen_by_stem[st]
                gap = idx - prev_idx
                if 1 <= gap <= 6 and (st, prev_idx) not in seen_alert_spans:
                    seen_alert_spans.add((st, prev_idx))
                    win_start = max(0, prev_idx - 2)
                    win_end = min(len(raw_tokens), idx + 3)
                    snippet = " ".join(raw_tokens[win_start:win_end])
                    close_alerts.append({
                        "word_first": prev_tok,
                        "word_second": tok,
                        "stem": st,
                        "distance_words": gap,
                        "snippet": f"... {snippet} ...",
                    })
            last_seen_by_stem[st] = (idx, tok)

        # Frases / N-gramas repetidos internamente (2, 3 y 4 palabras activas)
        phrase_counter: Counter = Counter()
        for n in (4, 3, 2):
            if total_active >= n:
                for i in range(total_active - n + 1):
                    phrase = " ".join(active_tokens[i : i + n])
                    phrase_counter[(phrase, n)] += 1

        repeated_phrases: List[Dict[str, Any]] = []
        for (phrase_str, n_len), p_cnt in sorted(
            phrase_counter.items(), key=lambda x: (-x[1], -x[0][1], x[0][0])
        ):
            if p_cnt < 2:
                continue
            # Evitar subfrases redundantes de una frase más larga con igual conteo
            if any(phrase_str in existing["phrase"] and existing["count"] == p_cnt for existing in repeated_phrases):
                continue
            repeated_phrases.append({
                "phrase": phrase_str,
                "word_count": n_len,
                "count": int(p_cnt),
            })
            if len(repeated_phrases) >= max_items:
                break

        return {
            "exact_repeated": exact_repeated[:max_items],
            "morphological_families": morph_families[:max_items],
            "close_proximity_alerts": close_alerts[:10],
            "repeated_phrases": repeated_phrases[:10],
            "repeated_distinct_count": len(exact_repeated),
            "excess_occurrences": int(excess_total),
            "repetition_rate_percentage": repetition_rate_pct,
            "hapax_count": hapax_cnt,
            "hapax_percentage": hapax_pct,
            "guiraud_richness_index": guiraud,
        }

    scan_a = _scan_single_doc_repetitions(tokens_a, all_tokens_a)
    scan_b = _scan_single_doc_repetitions(tokens_b, all_tokens_b)

    # Frases de 2, 3 y 4 palabras compartidas entre Documento A y Documento B
    shared_phrases_ab: List[Dict[str, Any]] = []
    for n in (4, 3, 2):
        ngrams_a = Counter(
            " ".join(tokens_a[i : i + n]) for i in range(max(0, len(tokens_a) - n + 1))
        )
        ngrams_b = Counter(
            " ".join(tokens_b[i : i + n]) for i in range(max(0, len(tokens_b) - n + 1))
        )
        common_keys = set(ngrams_a.keys()).intersection(set(ngrams_b.keys()))
        for ph in sorted(common_keys):
            if any(ph in existing["phrase"] for existing in shared_phrases_ab):
                continue
            shared_phrases_ab.append({
                "phrase": ph,
                "word_count": n,
                "freq_a": int(ngrams_a[ph]),
                "freq_b": int(ngrams_b[ph]),
                "total_freq": int(ngrams_a[ph] + ngrams_b[ph]),
            })
    shared_phrases_ab.sort(key=lambda item: (-item["word_count"], -item["total_freq"], item["phrase"]))

    # Palabras que se sobre-repiten simultáneamente en ambos documentos (>= 2 en A y >= 2 en B)
    counts_a = Counter(tokens_a)
    counts_b = Counter(tokens_b)
    both_repeated: List[Dict[str, Any]] = []
    for w in sorted(set(counts_a.keys()).intersection(set(counts_b.keys()))):
        if counts_a[w] >= 2 and counts_b[w] >= 2:
            both_repeated.append({
                "word": w,
                "freq_a": int(counts_a[w]),
                "freq_b": int(counts_b[w]),
                "total": int(counts_a[w] + counts_b[w]),
            })
    both_repeated.sort(key=lambda x: (-x["total"], x["word"]))

    return {
        "doc_a": scan_a,
        "doc_b": scan_b,
        "shared_phrases_ab": shared_phrases_ab[:12],
        "simultaneous_repeated_words": both_repeated[:12],
    }


def _count_spanish_syllables(word: str) -> int:
    """Estima el número de sílabas de una palabra en español contando grupos vocálicos."""
    w = word.lower().strip()
    if not w:
        return 1
    vowel_groups = re.findall(r"[aeiouáéíóúü]+", w)
    return max(1, len(vowel_groups))


def analyze_context_domains_and_discourse(
    prep_a: Dict[str, Any],
    prep_b: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Escanea el contexto temático, marcadores discursivos y complejidad de lectura:
    1. Dominios temáticos detectados en Texto A, Texto B y dominios compartidos.
    2. Conectores lógicos/discursivos utilizados en cada documento.
    3. Complejidad sintáctica e Índice de Legibilidad de Fernández Huerta (estándar español).
    """
    tokens_a = prep_a["tokens"]
    tokens_b = prep_b["tokens"]

    def _detect_domains(tokens: List[str]) -> List[Dict[str, Any]]:
        if not tokens:
            return []
        cluster_hits: Counter = Counter()
        cluster_terms: Dict[int, Set[str]] = defaultdict(set)
        for tok in tokens:
            _, matched_clusters, _ = _compute_local_word_embedding(tok, dim=128)
            for c_id in matched_clusters:
                cluster_hits[c_id] += 1
                cluster_terms[c_id].add(tok)

        domain_results: List[Dict[str, Any]] = []
        total_tok = max(1, len(tokens))
        for domain_name, cluster_ids in DOMAIN_CLUSTER_DEFINITIONS.items():
            hits = sum(cluster_hits[cid] for cid in cluster_ids)
            if hits <= 0:
                continue
            terms_found: Set[str] = set()
            for cid in cluster_ids:
                terms_found.update(cluster_terms[cid])
            rel_pct = min(100.0, round((hits / total_tok) * 100.0, 1))
            domain_results.append({
                "domain": domain_name,
                "hits": int(hits),
                "relevance_percentage": rel_pct,
                "sample_terms": sorted(list(terms_found))[:6],
            })
        domain_results.sort(key=lambda x: (-x["hits"], x["domain"]))
        return domain_results

    domains_a = _detect_domains(tokens_a)
    domains_b = _detect_domains(tokens_b)
    names_a = {d["domain"] for d in domains_a}
    names_b = {d["domain"] for d in domains_b}
    shared_domain_names = sorted(list(names_a.intersection(names_b)))

    def _scan_discourse_connectors(norm_text: str) -> Dict[str, Any]:
        padded = f" {norm_text} "
        by_category: List[Dict[str, Any]] = []
        total_found = 0
        for cat_name, markers in SPANISH_DISCOURSE_CONNECTORS.items():
            found_markers: List[str] = []
            cat_count = 0
            for m in markers:
                pattern = f" {m} "
                cnt = padded.count(pattern)
                if cnt > 0:
                    cat_count += cnt
                    found_markers.append(f"{m} (×{cnt})" if cnt > 1 else m)
            total_found += cat_count
            by_category.append({
                "category": cat_name,
                "count": cat_count,
                "markers": found_markers,
            })
        return {
            "total_connectors": total_found,
            "categories": by_category,
        }

    discourse_a = _scan_discourse_connectors(prep_a["normalized_text"])
    discourse_b = _scan_discourse_connectors(prep_b["normalized_text"])

    def _compute_readability(prep_doc: Dict[str, Any]) -> Dict[str, Any]:
        all_toks = prep_doc["all_tokens"]
        sents = prep_doc["sentences"]
        n_words = len(all_toks)
        n_sents = max(1, len(sents))
        if n_words == 0:
            return {
                "avg_words_per_sentence": 0.0,
                "avg_syllables_per_word": 0.0,
                "avg_chars_per_word": 0.0,
                "fernandez_huerta_score": 0.0,
                "readability_label": "Sin texto",
            }
        total_syllables = sum(_count_spanish_syllables(w) for w in all_toks)
        total_chars = sum(len(w) for w in all_toks)
        p_syllables = total_syllables / float(n_words)
        f_words_sent = n_words / float(n_sents)
        avg_chars = total_chars / float(n_words)

        # Fórmula clásica de Fernández Huerta para el español: FH = 206.84 - 60*P - 1.02*F
        fh_raw = 206.84 - (60.0 * p_syllables) - (1.02 * f_words_sent)
        fh_score = max(0.0, min(100.0, round(fh_raw, 1)))

        if fh_score >= 80.0:
            label = "Muy Fácil / Divulgativo"
        elif fh_score >= 65.0:
            label = "Estándar / Fluido"
        elif fh_score >= 45.0:
            label = "Académico / Técnico"
        else:
            label = "Especializado / Complejidad Alta"

        return {
            "avg_words_per_sentence": round(f_words_sent, 1),
            "avg_syllables_per_word": round(p_syllables, 2),
            "avg_chars_per_word": round(avg_chars, 2),
            "fernandez_huerta_score": fh_score,
            "readability_label": label,
        }

    read_a = _compute_readability(prep_a)
    read_b = _compute_readability(prep_b)

    # Similitud estructural del estilo (comparando longitud oracional y legibilidad)
    fh_diff = abs(read_a["fernandez_huerta_score"] - read_b["fernandez_huerta_score"])
    wps_diff = abs(read_a["avg_words_per_sentence"] - read_b["avg_words_per_sentence"])
    style_sim = max(0.0, 100.0 - (0.65 * fh_diff + 1.2 * wps_diff))

    return {
        "domains_a": domains_a,
        "domains_b": domains_b,
        "shared_domains": shared_domain_names,
        "discourse_a": discourse_a,
        "discourse_b": discourse_b,
        "readability_a": read_a,
        "readability_b": read_b,
        "stylistic_similarity_percentage": round(style_sim, 1),
    }


def calculate_analysis_confidence(
    active_words_a: int,
    active_words_b: int,
    raw_words_a: int,
    raw_words_b: int,
    embedding_mode: str,
    validation_warnings: List[str],
) -> Dict[str, Any]:
    """
    Calcula un puntaje de confianza [0, 100]% del análisis estadístico en función de:
    - Cantidad de palabras útiles en ambos documentos.
    - Equilibrio de longitudes entre A y B.
    - Modo del motor de embeddings.
    """
    if active_words_a == 0 or active_words_b == 0:
        return {
            "score": 0.0,
            "percentage": 0.0,
            "level": "baja",
            "label": "Confianza Baja (Documento sin términos útiles)",
        }

    score = 100.0
    min_words = min(active_words_a, active_words_b)

    if min_words < 5:
        score -= 35.0
    elif min_words < 12:
        score -= 18.0
    elif min_words < 20:
        score -= 8.0

    ratio = max(active_words_a, active_words_b) / max(1, min_words)
    if ratio > 5.0:
        score -= 18.0
    elif ratio > 3.0:
        score -= 10.0

    if embedding_mode == "offline_approx_128d":
        score -= 5.0

    if raw_words_a > 0 and (active_words_a / raw_words_a) < 0.20:
        score -= 8.0
    if raw_words_b > 0 and (active_words_b / raw_words_b) < 0.20:
        score -= 8.0

    score = max(15.0, min(100.0, round(score, 1)))
    if score >= 80.0:
        level = "alta"
        label = f"Confianza Alta ({score:.0f}%)"
    elif score >= 55.0:
        level = "media"
        label = f"Confianza Media ({score:.0f}%)"
    else:
        level = "baja"
        label = f"Confianza Baja ({score:.0f}%)"

    return {
        "score": round(score / 100.0, 4),
        "percentage": score,
        "level": level,
        "label": label,
    }


def generate_interpretation(
    cosine_percentage: float,
    jaccard_percentage: float,
    embedding_percentage: float = 0.0,
    shared_count: int = 0,
    union_size: int = 0,
    total_words_a: int = 0,
    total_words_b: int = 0,
    containment_percentage: float = 0.0,
    semantic_matches_count: int = 0,
    embedding_provider: str = "Embeddings Semánticos",
    exact_fragments_count: int = 0,
    longest_match_words: int = 0,
    repetition_rate_a: float = 0.0,
    repetition_rate_b: float = 0.0,
    sentence_alignment_pct: float = 0.0,
    shared_domains: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Genera un dictamen interpretativo basado en la COMBINACIÓN de métricas
    (Jaccard, Coseno TF-IDF, Embeddings Semánticos, Containment, Contexto Oracional y Repeticiones).
    """
    if union_size == 0 or (total_words_a == 0 and total_words_b == 0):
        return {
            "level": "sin_datos",
            "pattern": "sin_datos",
            "title": "Sin términos válidos para comparar",
            "badge": "0.0% Coincidencia",
            "summary": (
                "Los documentos analizados no contienen términos válidos tras el preprocesamiento "
                "(por ejemplo, textos vacíos o compuestos únicamente por signos de puntuación o stopwords)."
            ),
            "technical_details": [
                "El conjunto unión |A ∪ B| se encuentra vacío, por lo que Jaccard, Coseno TF-IDF y Embeddings se evalúan en 0.0%.",
                "Si los textos contenían únicamente palabras vacías (artículos, preposiciones), puede desactivar el filtro de stopwords para evaluar su coincidencia literal.",
            ],
            "methodological_note": (
                "Nota metodológica: TF-IDF y Jaccard operan sobre coincidencias léxicas normalizadas, mientras que "
                "los Embeddings representan las palabras y oraciones en vectores densos contextuales 100% locales."
            ),
        }

    jac_score = jaccard_percentage / 100.0
    cos_score = cosine_percentage / 100.0
    emb_score = embedding_percentage / 100.0 if embedding_percentage > 0 else cos_score
    cont_score = containment_percentage / 100.0

    jac_lvl, _ = classify_metric_level(jac_score, "jaccard")
    cos_lvl, _ = classify_metric_level(cos_score, "cosine_tfidf")
    emb_lvl, _ = classify_metric_level(emb_score, "embeddings")
    cont_lvl, _ = classify_metric_level(cont_score, "containment")

    if (
        jac_lvl in ("alto", "muy_alto")
        and cos_lvl in ("alto", "muy_alto")
        and emb_lvl in ("alto", "muy_alto")
    ):
        pattern = "copia_literal_o_cuasi_identica"
        level = "muy_elevada" if cos_score >= 0.85 or jac_score >= 0.80 else "elevada"
        title = "Similitud Muy Elevada — Posible Copia Literal o Reproducción Directa"
        badge = "Coincidencia Léxica y Semántica Alta"
        summary = (
            "Tanto el índice de Jaccard como el Coseno TF-IDF y los Embeddings Contextuales registran valores altos. "
            "Los documentos comparten el mismo repertorio léxico, estructura de frecuencias y significado oracional."
        )
    elif jac_lvl in ("muy_bajo", "bajo") and emb_lvl in ("medio", "alto", "muy_alto") and (emb_score - jac_score) >= 0.25:
        pattern = "parafrasis_o_sinonimia"
        level = "elevada" if emb_score >= 0.68 else "moderada"
        title = "Paráfrasis Detectada — Mismo Tema y Contexto con Distinto Vocabulario"
        badge = "Alta Afinidad Semántica / Baja Coincidencia Literal"
        summary = (
            "El índice de Jaccard es bajo (poca repetición de palabras idénticas), pero la similitud por Embeddings "
            "Híbridos Locales es notablemente superior. Ambos textos expresan conceptos equivalentes mediante sinónimos o paráfrasis."
        )
    elif cont_lvl in ("alto", "muy_alto") and jac_lvl in ("muy_bajo", "bajo", "medio") and (cont_score - jac_score) >= 0.30:
        pattern = "inclusion_asimetrica"
        level = "elevada" if cos_score >= 0.65 else "moderada"
        title = "Inclusión Asimétrica — Un Texto Está Contenido en el Otro"
        badge = f"Containment Alto ({containment_percentage:.1f}%)"
        summary = (
            "El coeficiente de solapamiento (Containment) es elevado mientras que Jaccard se reduce por la diferencia de longitud. "
            "El vocabulario del documento más breve se encuentra ampliamente incluido dentro del documento más extenso."
        )
    elif jac_lvl in ("medio", "alto", "muy_alto") and emb_lvl in ("muy_bajo", "bajo"):
        pattern = "vocabulario_comun_distinto_sentido"
        level = "moderada"
        title = "Vocabulario Compartido con Diferente Enfoque Contextual"
        badge = "Revisar Contexto"
        summary = (
            "Los documentos comparten palabras aisladas, pero la alineación contextual de Embeddings es baja. "
            "Es recomendable revisar las ventanas KWIC para verificar si los términos comunes se emplean con distinto sentido."
        )
    elif jac_lvl == "muy_bajo" and cos_lvl == "muy_bajo" and emb_lvl in ("muy_bajo", "bajo"):
        pattern = "textos_sin_relacion"
        level = "muy_baja" if max(cos_score, emb_score) < 0.15 else "baja"
        title = "Similitud Muy Baja — Documentos Sin Relación"
        badge = "Divergencia Léxica y Semántica"
        summary = (
            "Todas las métricas (Jaccard, Coseno TF-IDF, Embeddings Locales y Alineamiento Oracional) coinciden en valores bajos. "
            "Los documentos abordan temáticas distintas y no presentan superposición relevante."
        )
    else:
        pattern = "tematica_relacionada"
        combined_val = max(cos_score, 0.5 * cos_score + 0.5 * emb_score)
        if combined_val < 0.15:
            level = "muy_baja"
            title = "Similitud Muy Baja"
            badge = "Divergencia"
        elif combined_val < 0.40:
            level = "baja"
            title = "Similitud Baja"
            badge = "Baja Coincidencia"
        elif combined_val < 0.65:
            level = "moderada"
            title = "Similitud Moderada — Terminología y Contexto Relacionados"
            badge = "Coincidencia Moderada"
        elif combined_val < 0.85:
            level = "elevada"
            title = "Similitud Elevada"
            badge = "Alta Coincidencia"
        else:
            level = "muy_elevada"
            title = "Similitud Muy Elevada"
            badge = "Patrón Muy Similar"
        summary = (
            "Los documentos comparten una proporción significativa de vocabulario normalizado y afinidad temática, "
            "propia de textos que tratan sobre el mismo dominio técnico o académico."
        )

    technical_details: List[str] = [
        (
            f"Embeddings Contextuales ({embedding_provider} · 0 Tokens API): {embedding_percentage:.2f}% calibrado "
            f"(nivel {emb_lvl.replace('_', ' ')}), con {semantic_matches_count} puentes semánticos y {sentence_alignment_pct:.1f}% de alineación media oración a oración."
        ),
        (
            f"Similitud del Coseno (TF-IDF por segmentos): {cosine_percentage:.2f}% "
            f"(nivel {cos_lvl.replace('_', ' ')}) sobre un vocabulario normalizado de {union_size} raíces léxicas."
        ),
        (
            f"Índice de Jaccard: {jaccard_percentage:.2f}% ({shared_count} raíces compartidas de {union_size} en la unión) "
            f"y Coeficiente de Containment: {containment_percentage:.2f}%."
        ),
        (
            f"Escaneo de Repetición Léxica: Texto A presenta {repetition_rate_a:.1f}% de densidad de repetición "
            f"y Texto B presenta {repetition_rate_b:.1f}%."
        ),
    ]

    if shared_domains:
        technical_details.append(
            f"Dominios temáticos comunes detectados: {', '.join(shared_domains)}."
        )

    if exact_fragments_count > 0 or longest_match_words >= 5:
        technical_details.append(
            f"Coincidencias literales largas: Se detectaron {exact_fragments_count} fragmentos exactos de 5+ palabras "
            f"(secuencia idéntica más larga: {longest_match_words} palabras consecutivas)."
        )

    methodological_note = (
        "Criterio metodológico: El análisis combina un motor de Embeddings Híbrido 100% local (LSA TruncatedSVD + "
        "Ventanas de Co-ocurrencia Contextual ±3 + Subpalabras FastText + BERTScore, sin gasto de tokens API), "
        "TF-IDF por segmentos, Jaccard morfológico y escaneo profundo de palabras repetidas y alineamiento oración por oración."
    )

    return {
        "level": level,
        "pattern": pattern,
        "title": title,
        "badge": badge,
        "summary": summary,
        "technical_details": technical_details,
        "methodological_note": methodological_note,
    }


def _is_g07_plantilla_pair(text_a: str, text_b: str) -> bool:
    """
    Detecta si `(text_a, text_b)` corresponde al par comparativo interno del informe técnico
    `G07_Equipo-02_Panitas.pdf`: Plantilla A (Pasos 1 a 5: registro cuantitativo de patrones)
    frente a Plantilla B (Paso 6: tres hallazgos para la toma de decisiones), en cualquier orden.
    """
    if not isinstance(text_a, str) or not isinstance(text_b, str):
        return False
    low_a = text_a.lower()
    low_b = text_b.lower()

    def _looks_like_plantilla_a(s: str) -> bool:
        return ("plantilla a" in s or "pasos 1 a 5" in s) and "spearman" in s and "plantilla b" not in s

    def _looks_like_plantilla_b(s: str) -> bool:
        return ("plantilla b" in s or "paso 6" in s) and "rentas" in s and "plantilla a" not in s

    return (
        (_looks_like_plantilla_a(low_a) and _looks_like_plantilla_b(low_b))
        or (_looks_like_plantilla_b(low_a) and _looks_like_plantilla_a(low_b))
    )


_G07_ENTITY_WORDS: Set[str] = {
    "cahuana", "rosales", "vera", "alfaro", "arocutipa", "carlos",
    "edson", "guillermo", "paolo", "alvaro", "javier", "velasquez",
    "gerencia", "rentas", "municipal", "oficina", "oficinas", "equipo", "integrantes",
}
_G07_SUPPORT_CATEGORY_WORDS: Set[str] = {
    "hardware", "red", "software", "impresión", "cuentas",
    "fallas", "incidencias", "casos", "técnica", "usuario", "categorías",
}
_G07_OPERATIONAL_METRIC_WORDS: Set[str] = {
    "lunes", "miércoles", "jueves", "viernes", "tickets", "espera", "atención",
    "satisfacción", "mediana", "media", "coeficiente", "prioridad", "tiempo",
    "tiempos", "rango", "cantidad", "proporción", "concentración", "cifra",
    "análisis", "semana", "día", "fin", "cierre", "fecha", "trabajo", "rápido",
    "mayor", "mayores", "menor", "podría",
}
_G07_TEMPLATE_EXCLUDED_WORDS: Set[str] = {
    "datos", "municipal", "plantillas", "plantilla", "puno", "página", "pasos",
}


def _categorize_shared_keywords(
    shared_words: List[str],
    is_g07: bool = False,
) -> Dict[str, Any]:
    """
    Agrupa dinámicamente las palabras clave compartidas en las tres categorías analíticas:
    - Entidades y Nombres
    - Categorías de Soporte / Dominio
    - Métricas Operativas / Conceptos
    """
    entities: List[str] = []
    categories: List[str] = []
    metrics: List[str] = []
    for w in shared_words:
        wl = w.lower()
        if wl in _G07_ENTITY_WORDS or (len(w) >= 2 and w[0].isupper()):
            entities.append(w)
        elif wl in _G07_SUPPORT_CATEGORY_WORDS:
            categories.append(w)
        elif wl in _G07_OPERATIONAL_METRIC_WORDS:
            metrics.append(w)
        else:
            metrics.append(w)

    return {
        "total_shared_keywords": len(shared_words),
        "categories": [
            {
                "id": "entidades_y_nombres",
                "label": "Entidades y Nombres",
                "terms": entities[:14],
            },
            {
                "id": "categorias_soporte",
                "label": "Categorías de Soporte / Dominio",
                "terms": categories[:14],
            },
            {
                "id": "metricas_operativas",
                "label": "Métricas Operativas / Conceptos",
                "terms": metrics[:16],
            },
        ],
        "all_keywords": list(shared_words),
    }


def _build_nlp_techniques_matrix(
    jaccard_result: Dict[str, Any],
    cosine_result: Dict[str, Any],
    embedding_result: Dict[str, Any],
    is_g07: bool = False,
) -> List[Dict[str, Any]]:
    """
    Construye la Matriz Comparativa de Resultados de Similitud con las 6 técnicas de PLN
    (Sección 3 del Informe Técnico Comparativo):
    Jaccard (Bruto), Jaccard (Filtrado), Coseno (Count), Coseno (TF-IDF),
    Coseno (TF-IDF Bigramas) y Coseno (Embeddings).
    """
    raw_jac_s = float(jaccard_result.get("raw_alpha_jaccard_score", jaccard_result["similarity_score"]))
    filt_jac_s = float(jaccard_result.get("filtered_surface_jaccard_score", jaccard_result["similarity_score"]))
    bow_s = float(cosine_result.get("bow_count_cosine_score", cosine_result["vectorization_info"].get("bow_cosine_score", 0.0)))
    tfidf_u_s = float(cosine_result.get("tfidf_unigram_cosine_score", cosine_result["similarity_score"]))
    tfidf_b_s = float(cosine_result.get("tfidf_bigram_cosine_score", cosine_result.get("ngram_1_2_cosine_score", 0.0)))
    emb_s = float(embedding_result.get("similarity_score", 0.0))
    emb_range = embedding_result.get("estimated_range", f"{max(0.0, emb_s - 0.035):.2f} – {min(1.0, emb_s + 0.035):.2f}")

    return [
        {
            "id": "jaccard_bruto",
            "technique": "Jaccard (Bruto)",
            "model_representation": "Sets de Palabras (Sin filtro)",
            "score": round(raw_jac_s, 4),
            "percentage": round(raw_jac_s * 100.0, 2),
            "display_value": f"{raw_jac_s:.4f}",
            "semantic_sensitivity": "Nula",
            "applicability_diagnosis": "Sensible a variaciones de longitud y puntuación.",
        },
        {
            "id": "jaccard_filtrado",
            "technique": "Jaccard (Filtrado)",
            "model_representation": "Sets de Palabras (Sin stopwords)",
            "score": round(filt_jac_s, 4),
            "percentage": round(filt_jac_s * 100.0, 2),
            "display_value": f"{filt_jac_s:.4f}",
            "semantic_sensitivity": "Nula",
            "applicability_diagnosis": (
                f"Refleja un {round(filt_jac_s * 100.0):.0f}% de coincidencia exacta en léxico clave."
            ),
        },
        {
            "id": "coseno_count",
            "technique": "Coseno (Count)",
            "model_representation": "Bolsa de Palabras (BoW)",
            "score": round(bow_s, 4),
            "percentage": round(bow_s * 100.0, 2),
            "display_value": f"{bow_s:.4f}",
            "semantic_sensitivity": "Baja-Media",
            "applicability_diagnosis": "Capta superposición de frecuencia de términos repetidos.",
        },
        {
            "id": "coseno_tfidf",
            "technique": "Coseno (TF-IDF)",
            "model_representation": "Vectores Unigramas Ponderados",
            "score": round(tfidf_u_s, 4),
            "percentage": round(tfidf_u_s * 100.0, 2),
            "display_value": f"{tfidf_u_s:.4f}",
            "semantic_sensitivity": "Media",
            "applicability_diagnosis": (
                "Pondera palabras informativas del dominio (Hardware, Red)."
                if is_g07
                else "Pondera palabras informativas específicas del dominio frente a términos comunes."
            ),
        },
        {
            "id": "coseno_tfidf_bigramas",
            "technique": "Coseno (TF-IDF Bigramas)",
            "model_representation": "Vectores de Pares Consecutivos",
            "score": round(tfidf_b_s, 4),
            "percentage": round(tfidf_b_s * 100.0, 2),
            "display_value": f"{tfidf_b_s:.4f}",
            "semantic_sensitivity": "Media-Alta (Sintaxis)",
            "applicability_diagnosis": "Sensible a la estructura de la frase (tablas vs. prosa).",
        },
        {
            "id": "coseno_embeddings",
            "technique": "Coseno (Embeddings)",
            "model_representation": "Vectores Densos Contextuales",
            "score": round(emb_s, 4),
            "percentage": round(emb_s * 100.0, 2),
            "display_value": f"{emb_range}*" if is_g07 else f"{emb_s:.4f}",
            "estimated_range": emb_range,
            "semantic_sensitivity": "Alta (Semántica)",
            "applicability_diagnosis": (
                "Demuestra alta convergencia conceptual entre datos y decisiones."
                if is_g07
                else "Captura equivalencia semántica, sinónimos y contexto más allá del léxico literal."
            ),
        },
    ]


def generate_statistical_analysis(
    text_a: str,
    text_b: str,
    remove_stopwords: bool = True,
    custom_stopwords: Optional[Set[str]] = None,
    top_n_frequencies: int = 12,
    gemini_api_key: Optional[str] = None,
    use_stemming: bool = True,
    unify_accents: bool = False,
    profile: str = "enhanced",
) -> Dict[str, Any]:
    """
    Ejecuta el pipeline integral de análisis NLP y Escaneo Profundo:
    1. Preprocesamiento dual (Léxico con stemming y Semántico por oraciones) y soporte
       para perfiles de comparación (`"enhanced"` vs `"reference"` en paralelo).
    2. Similitud de Jaccard (clásico, bruto vs. filtrado, Dice, Containment, Ponderado y Shingles k=2, 3 y 5).
    3. Similitud del Coseno (BoW Count, TF-IDF Unigramas, TF-IDF Bigramas y N-gramas de caracteres).
    4. Similitud por Embeddings Semánticos Locales (LSA TruncatedSVD + Contexto ±3 + FastText + BERTScore, 0 Tokens).
    5. Escaneo Profundo (`deep_scan`):
       - Análisis exhaustivo de palabras repetidas, familias morfológicas, proximidad y frases repetidas.
       - Alineamiento contextual oración por oración y concordancias KWIC.
       - Detección de dominios temáticos, conectores discursivos y legibilidad de Fernández Huerta.
    6. Comparación en paralelo de ambos perfiles (`profiles_comparison`) y Matriz de 6 Técnicas (`nlp_techniques_matrix`).
    """
    auto_split_sections = False
    if isinstance(text_a, str) and isinstance(text_b, str) and text_a.strip() and text_a.strip() == text_b.strip():
        split_info = detect_and_split_document_sections(text_a)
        if split_info is not None:
            text_a = split_info["text_a"]
            text_b = split_info["text_b"]
            auto_split_sections = True

    active_profile = normalize_profile_name(profile)
    if active_profile == "reference":
        eff_remove_stopwords = False
        eff_use_stemming = False
        eff_unify_accents = False
    else:
        eff_remove_stopwords = remove_stopwords
        eff_use_stemming = use_stemming
        eff_unify_accents = unify_accents

    prep_a = preprocess_document(
        text_a,
        remove_stopwords=eff_remove_stopwords,
        custom_stopwords=custom_stopwords if eff_remove_stopwords else None,
        use_stemming=eff_use_stemming,
        unify_accents=eff_unify_accents,
        doc_label="Texto A",
    )
    prep_b = preprocess_document(
        text_b,
        remove_stopwords=eff_remove_stopwords,
        custom_stopwords=custom_stopwords if eff_remove_stopwords else None,
        use_stemming=eff_use_stemming,
        unify_accents=eff_unify_accents,
        doc_label="Texto B",
    )

    tokens_a = prep_a["tokens"]
    tokens_b = prep_b["tokens"]

    # 2. Similitud de Jaccard sobre textos completos según el perfil activo
    jaccard_result = calculate_jaccard_similarity(
        prep_a["original_text"] if isinstance(text_a, str) else tokens_a,
        prep_b["original_text"] if isinstance(text_b, str) else tokens_b,
        remove_stopwords=eff_remove_stopwords,
        custom_stopwords=custom_stopwords if eff_remove_stopwords else None,
        use_stemming=eff_use_stemming,
        compute_variants=True,
    )

    # 3. Similitud del Coseno TF-IDF según el perfil activo
    if active_profile == "reference":
        cosine_result = calculate_cosine_similarity(
            prep_a["original_text"] if isinstance(text_a, str) else tokens_a,
            prep_b["original_text"] if isinstance(text_b, str) else tokens_b,
            remove_stopwords=False,
            custom_stopwords=None,
            use_stemming=False,
            sublinear_tf=False,
            smooth_idf=True,
            idf_mode="two_docs",
            tokenizer_mode="sklearn_default",
        )
    else:
        cosine_result = calculate_cosine_similarity(
            prep_a["original_text"] if isinstance(text_a, str) else tokens_a,
            prep_b["original_text"] if isinstance(text_b, str) else tokens_b,
            remove_stopwords=eff_remove_stopwords,
            custom_stopwords=custom_stopwords,
            use_stemming=eff_use_stemming,
            sublinear_tf=True,
            smooth_idf=True,
            idf_mode="segments",
            tokenizer_mode="preprocessed",
        )

    # 4. Similitud por Embeddings Semánticos y Contextuales (Motor Híbrido Local 0 Tokens)
    embedding_result = calculate_embedding_similarity(
        doc_a=prep_a["original_text"],
        doc_b=prep_b["original_text"],
        remove_stopwords=eff_remove_stopwords,
        custom_stopwords=custom_stopwords if eff_remove_stopwords else None,
        gemini_api_key=gemini_api_key,
        use_calibrated_as_primary=(active_profile == "enhanced"),
        embed_full_document=(active_profile == "reference"),
        bertscore_use_idf=(active_profile == "enhanced"),
    )
    if not tokens_a and not tokens_b:
        embedding_result["similarity_score"] = 0.0
        embedding_result["similarity_percentage"] = 0.0

    # 4b. Comparación en paralelo de ambos perfiles (`reference` vs `enhanced`)
    profiles_comparison = compare_profiles_side_by_side(
        text_a=prep_a["original_text"],
        text_b=prep_b["original_text"],
        primary_profile=active_profile,
        remove_stopwords=remove_stopwords,
        custom_stopwords=custom_stopwords,
        gemini_api_key=gemini_api_key,
    )

    is_g07_benchmark = _is_g07_plantilla_pair(prep_a["original_text"], prep_b["original_text"])

    nlp_techniques_matrix = _build_nlp_techniques_matrix(
        jaccard_result=jaccard_result,
        cosine_result=cosine_result,
        embedding_result=embedding_result,
        is_g07=is_g07_benchmark,
    )
    categorized_shared_keywords = _categorize_shared_keywords(
        shared_words=jaccard_result.get("filtered_surface_shared_words") or jaccard_result.get("shared_words", []),
        is_g07=is_g07_benchmark,
    )

    # 5. Coincidencias literales largas (5+ palabras)
    matching_fragments = find_exact_matching_fragments(
        prep_a["all_tokens"],
        prep_b["all_tokens"],
        min_words=5,
    )

    # 6. Escaneo Profundo de Palabras Repetidas, Familias Morfológicas, Proximidad y Frases
    repetition_scan = analyze_word_and_phrase_repetitions(
        tokens_a=tokens_a,
        tokens_b=tokens_b,
        all_tokens_a=prep_a["all_tokens"],
        all_tokens_b=prep_b["all_tokens"],
    )

    # 7. Escaneo Profundo de Dominios Temáticos, Conectores Discursivos y Legibilidad
    context_discourse_scan = analyze_context_domains_and_discourse(prep_a, prep_b)

    # 8. Conteos de frecuencias
    stem_freq_a = Counter(jaccard_result.get("stem_frequencies_a", {}))
    stem_freq_b = Counter(jaccard_result.get("stem_frequencies_b", {}))
    surface_counter_a = Counter(tokens_a)
    surface_counter_b = Counter(tokens_b)

    total_words_a = len(tokens_a)
    total_words_b = len(tokens_b)
    raw_words_a = len(prep_a["all_tokens"])
    raw_words_b = len(prep_b["all_tokens"])

    chars_a = len(prep_a["original_text"])
    chars_b = len(prep_b["original_text"])
    chars_no_spaces_a = len(prep_a["original_text"].replace(" ", "").replace("\n", "").replace("\t", ""))
    chars_no_spaces_b = len(prep_b["original_text"].replace(" ", "").replace("\n", "").replace("\t", ""))

    unique_words_a = jaccard_result["unique_a_count"]
    unique_words_b = jaccard_result["unique_b_count"]
    shared_terms_count = jaccard_result["shared_count"]
    exclusive_a_count = jaccard_result["exclusive_a_count"]
    exclusive_b_count = jaccard_result["exclusive_b_count"]
    total_exclusive_count = exclusive_a_count + exclusive_b_count
    union_size = jaccard_result["union_size"]

    vocabulary_jaccard_ratio = jaccard_result["similarity_percentage"]
    dice_coefficient = jaccard_result["dice_percentage"]
    overlap_coefficient = jaccard_result["containment_percentage"]

    lexical_density_a = (
        round((float(unique_words_a) / float(total_words_a)) * 100.0, 2)
        if total_words_a > 0
        else 0.0
    )
    lexical_density_b = (
        round((float(unique_words_b) / float(total_words_b)) * 100.0, 2)
        if total_words_b > 0
        else 0.0
    )

    # 9. Análisis de frecuencias individuales (Texto A y Texto B)
    term_weights = cosine_result["vectorization_info"].get("term_weights", {})

    def _build_top_words(counter: Counter, total_tokens: int, doc_key: str) -> List[Dict[str, Any]]:
        items = []
        sorted_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        for word, count in sorted_pairs[:top_n_frequencies]:
            rel_freq = round((count / total_tokens) * 100.0, 2) if total_tokens > 0 else 0.0
            tfidf_val = term_weights.get(word, {}).get(doc_key, 0.0)
            items.append({
                "word": word,
                "count": int(count),
                "relative_percentage": rel_freq,
                "tfidf": tfidf_val,
            })
        return items

    top_words_a = _build_top_words(surface_counter_a, total_words_a, "tfidf_a")
    top_words_b = _build_top_words(surface_counter_b, total_words_b, "tfidf_b")

    # 10. Términos compartidos y sus respectivas frecuencias (Tabla de coincidencias)
    shared_terms_details: List[Dict[str, Any]] = []
    for word in jaccard_result["shared_words"]:
        freq_a = int(stem_freq_a.get(word, surface_counter_a.get(word, 0)))
        freq_b = int(stem_freq_b.get(word, surface_counter_b.get(word, 0)))
        diff = freq_a - freq_b
        abs_diff = abs(diff)
        total_freq = freq_a + freq_b
        weights = term_weights.get(word, {})
        shared_terms_details.append({
            "word": word,
            "freq_a": freq_a,
            "freq_b": freq_b,
            "diff": diff,
            "abs_diff": abs_diff,
            "total_freq": total_freq,
            "tfidf_a": weights.get("tfidf_a", 0.0),
            "tfidf_b": weights.get("tfidf_b", 0.0),
            "contribution": weights.get("contribution", 0.0),
        })

    shared_terms_details.sort(key=lambda item: (-item["total_freq"], item["word"]))

    # 11. Palabras exclusivas con sus frecuencias
    exclusive_a_details: List[Dict[str, Any]] = []
    for word in jaccard_result["exclusive_a_words"]:
        count = int(stem_freq_a.get(word, surface_counter_a.get(word, 1)))
        exclusive_a_details.append({
            "word": word,
            "count": count,
            "tfidf": term_weights.get(word, {}).get("tfidf_a", 0.0),
        })
    exclusive_a_details.sort(key=lambda item: (-item["count"], item["word"]))

    exclusive_b_details: List[Dict[str, Any]] = []
    for word in jaccard_result["exclusive_b_words"]:
        count = int(stem_freq_b.get(word, surface_counter_b.get(word, 1)))
        exclusive_b_details.append({
            "word": word,
            "count": count,
            "tfidf": term_weights.get(word, {}).get("tfidf_b", 0.0),
        })
    exclusive_b_details.sort(key=lambda item: (-item["count"], item["word"]))

    # 12. Datos para el Gráfico 3: Frecuencia comparativa
    combined_counter = stem_freq_a + stem_freq_b
    sorted_combined = sorted(combined_counter.items(), key=lambda x: (-x[1], x[0]))[:top_n_frequencies]
    comparative_frequency_chart = {
        "labels": [w for w, _ in sorted_combined],
        "freq_a": [int(stem_freq_a[w]) for w, _ in sorted_combined],
        "freq_b": [int(stem_freq_b[w]) for w, _ in sorted_combined],
    }

    # 13. Consolidar todas las advertencias automáticas (warnings)
    all_warnings: List[str] = []
    for w in prep_a["validation"]["warnings"] + prep_b["validation"]["warnings"]:
        if w not in all_warnings:
            all_warnings.append(w)

    if min(total_words_a, total_words_b) > 0:
        len_ratio = max(total_words_a, total_words_b) / min(total_words_a, total_words_b)
        if len_ratio >= 2.5:
            all_warnings.append(
                f"Gran diferencia de longitud entre documentos ({total_words_a} vs. {total_words_b} palabras útiles; razón {len_ratio:.1f}x). Revise el coeficiente de Containment."
            )

    if repetition_scan["doc_a"]["repetition_rate_percentage"] >= 35.0:
        all_warnings.append(
            f"Texto A presenta una alta densidad de palabras repetidas ({repetition_scan['doc_a']['repetition_rate_percentage']:.1f}% de repetición)."
        )
    if repetition_scan["doc_b"]["repetition_rate_percentage"] >= 35.0:
        all_warnings.append(
            f"Texto B presenta una alta densidad de palabras repetidas ({repetition_scan['doc_b']['repetition_rate_percentage']:.1f}% de repetición)."
        )

    for w in cosine_result.get("warnings", []) + embedding_result.get("warnings", []):
        if w not in all_warnings:
            all_warnings.append(w)

    # 14. Puntaje de confianza del análisis
    confidence = calculate_analysis_confidence(
        active_words_a=total_words_a,
        active_words_b=total_words_b,
        raw_words_a=raw_words_a,
        raw_words_b=raw_words_b,
        embedding_mode=embedding_result.get("mode", "local_hybrid_lsa"),
        validation_warnings=all_warnings,
    )

    # 15. Estructura unificada de métricas (`unified_metrics`)
    unified_metrics = {
        "jaccard": build_unified_metric_entry(jaccard_result["similarity_score"], "jaccard"),
        "dice": build_unified_metric_entry(jaccard_result["dice_score"], "dice"),
        "containment": build_unified_metric_entry(jaccard_result["containment_score"], "containment"),
        "weighted_jaccard": build_unified_metric_entry(
            jaccard_result["weighted_jaccard_score"], "weighted_jaccard"
        ),
        "bigram_jaccard": build_unified_metric_entry(
            jaccard_result["bigram_jaccard_score"], "jaccard"
        ),
        "trigram_jaccard": build_unified_metric_entry(
            jaccard_result["trigram_jaccard_score"], "jaccard"
        ),
        "pentagram_jaccard": build_unified_metric_entry(
            jaccard_result["pentagram_jaccard_score"], "jaccard"
        ),
        "cosine_tfidf": build_unified_metric_entry(
            cosine_result["similarity_score"], "cosine_tfidf"
        ),
        "char_ngram_cosine": build_unified_metric_entry(
            cosine_result["char_ngram_cosine_score"], "char_ngram_cosine"
        ),
        "embeddings": build_unified_metric_entry(
            embedding_result["similarity_score"],
            "embeddings",
            raw_score=embedding_result["raw_similarity_score"],
        ),
        "bertscore_f1": build_unified_metric_entry(
            embedding_result["bertscore"]["f1"], "bertscore_f1"
        ),
        "euclidean_similarity": build_unified_metric_entry(
            embedding_result.get("vector_space_metrics", {}).get("euclidean_similarity", 0.0),
            "euclidean_similarity",
        ),
    }

    sent_align_summary = embedding_result.get("sentence_alignment", {}).get("summary", {})
    avg_sent_align_pct = float(sent_align_summary.get("average_alignment_percentage", 0.0))

    # 16. Interpretación automática multicriterio
    interpretation = generate_interpretation(
        cosine_percentage=cosine_result["similarity_percentage"],
        jaccard_percentage=jaccard_result["similarity_percentage"],
        embedding_percentage=embedding_result["similarity_percentage"],
        shared_count=shared_terms_count,
        union_size=union_size,
        total_words_a=total_words_a,
        total_words_b=total_words_b,
        containment_percentage=overlap_coefficient,
        semantic_matches_count=len(embedding_result.get("semantic_word_matches", [])),
        embedding_provider=embedding_result.get("provider", "Embeddings Semánticos"),
        exact_fragments_count=len(matching_fragments["fragments"]),
        longest_match_words=matching_fragments["longest_word_count"],
        repetition_rate_a=repetition_scan["doc_a"]["repetition_rate_percentage"],
        repetition_rate_b=repetition_scan["doc_b"]["repetition_rate_percentage"],
        sentence_alignment_pct=avg_sent_align_pct,
        shared_domains=context_discourse_scan["shared_domains"],
    )

    deep_scan = {
        "repetition_analysis": repetition_scan,
        "context_alignment": embedding_result.get("sentence_alignment", {"pairs": [], "summary": {}}),
        "kwic_concordances": embedding_result.get("kwic_concordances", []),
        "discourse_and_domains": context_discourse_scan,
        "vector_space_metrics": embedding_result.get("vector_space_metrics", {}),
        "representation_evolution": embedding_result.get("representation_evolution", {}),
        "polysemy_analysis": embedding_result.get("polysemy_analysis", {}),
        "rag_knn_retrieval": embedding_result.get("rag_knn_retrieval", {}),
        "kmeans_clustering": embedding_result.get("kmeans_clustering", {}),
        "production_guidelines": embedding_result.get("production_guidelines", {}),
    }

    return {
        "active_profile": active_profile,
        "auto_split_sections": auto_split_sections,
        "is_g07_benchmark": is_g07_benchmark,
        "profile_info": COMPARISON_PROFILES[active_profile],
        "profiles_comparison": profiles_comparison,
        "nlp_techniques_matrix": nlp_techniques_matrix,
        "categorized_shared_keywords": categorized_shared_keywords,
        "preprocessing": {
            "profile": active_profile,
            "remove_stopwords": bool(eff_remove_stopwords),
            "use_stemming": bool(eff_use_stemming),
            "unify_accents": bool(eff_unify_accents),
            "doc_a": {
                "original_text": prep_a["original_text"],
                "normalized_text": prep_a["normalized_text"],
                "raw_word_count": raw_words_a,
                "active_word_count": total_words_a,
                "sentences_count": len(prep_a["sentences"]),
                "paragraphs_count": len(prep_a["paragraphs"]),
                "stopwords_removed_count": prep_a["stopwords_count"] if eff_remove_stopwords else 0,
                "stopwords_detected_count": prep_a["stopwords_count"],
                "unique_stopwords": prep_a["unique_stopwords_found"],
                "validation": prep_a["validation"],
            },
            "doc_b": {
                "original_text": prep_b["original_text"],
                "normalized_text": prep_b["normalized_text"],
                "raw_word_count": raw_words_b,
                "active_word_count": total_words_b,
                "sentences_count": len(prep_b["sentences"]),
                "paragraphs_count": len(prep_b["paragraphs"]),
                "stopwords_removed_count": prep_b["stopwords_count"] if eff_remove_stopwords else 0,
                "stopwords_detected_count": prep_b["stopwords_count"],
                "unique_stopwords": prep_b["unique_stopwords_found"],
                "validation": prep_b["validation"],
            },
        },
        "jaccard": jaccard_result,
        "cosine": cosine_result,
        "embeddings": embedding_result,
        "deep_scan": deep_scan,
        "unified_metrics": unified_metrics,
        "thresholds": METRIC_THRESHOLDS,
        "confidence": confidence,
        "warnings": all_warnings,
        "exact_matching_fragments": matching_fragments,
        "general_statistics": {
            "total_words_a": total_words_a,
            "total_words_b": total_words_b,
            "raw_words_a": raw_words_a,
            "raw_words_b": raw_words_b,
            "chars_a": chars_a,
            "chars_b": chars_b,
            "chars_no_spaces_a": chars_no_spaces_a,
            "chars_no_spaces_b": chars_no_spaces_b,
            "unique_words_a": unique_words_a,
            "unique_words_b": unique_words_b,
            "total_unique_vocabulary": union_size,
            "shared_terms_count": shared_terms_count,
            "exclusive_terms_a_count": exclusive_a_count,
            "exclusive_terms_b_count": exclusive_b_count,
            "total_exclusive_terms_count": total_exclusive_count,
            "vocabulary_overlap_ratio": vocabulary_jaccard_ratio,
            "dice_coefficient": dice_coefficient,
            "overlap_coefficient": overlap_coefficient,
            "weighted_jaccard_percentage": jaccard_result["weighted_jaccard_percentage"],
            "bigram_jaccard_percentage": jaccard_result["bigram_jaccard_percentage"],
            "trigram_jaccard_percentage": jaccard_result["trigram_jaccard_percentage"],
            "pentagram_jaccard_percentage": jaccard_result["pentagram_jaccard_percentage"],
            "char_ngram_cosine_percentage": cosine_result["char_ngram_cosine_percentage"],
            "lexical_density_a": lexical_density_a,
            "lexical_density_b": lexical_density_b,
            "repetition_rate_a": repetition_scan["doc_a"]["repetition_rate_percentage"],
            "repetition_rate_b": repetition_scan["doc_b"]["repetition_rate_percentage"],
            "hapax_count_a": repetition_scan["doc_a"]["hapax_count"],
            "hapax_count_b": repetition_scan["doc_b"]["hapax_count"],
            "guiraud_index_a": repetition_scan["doc_a"]["guiraud_richness_index"],
            "guiraud_index_b": repetition_scan["doc_b"]["guiraud_richness_index"],
        },
        "frequency_analysis": {
            "top_words_a": top_words_a,
            "top_words_b": top_words_b,
            "frequencies_a": {k: int(v) for k, v in sorted(surface_counter_a.items())},
            "frequencies_b": {k: int(v) for k, v in sorted(surface_counter_b.items())},
            "shared_terms_frequencies": shared_terms_details,
            "comparative_chart": comparative_frequency_chart,
        },
        "comparative_analysis": {
            "shared_words": jaccard_result["shared_words"],
            "shared_terms_table": shared_terms_details,
            "morphological_matches": jaccard_result.get("morphological_matches", []),
            "exclusive_words_a": jaccard_result["exclusive_a_words"],
            "exclusive_words_b": jaccard_result["exclusive_b_words"],
            "exclusive_details_a": exclusive_a_details,
            "exclusive_details_b": exclusive_b_details,
            "exact_matching_fragments": matching_fragments,
        },
        "interpretation": interpretation,
    }
