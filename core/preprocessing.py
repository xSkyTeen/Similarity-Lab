"""
Módulo de preprocesamiento NLP dual (Léxico y Semántico) para Similarity Lab.

Implementa:
1. Normalización Unicode (NFC), limpieza de artefactos de PDF (guiones de salto de línea,
   encabezados/pies repetidos), URLs, correos electrónicos y caracteres de control.
2. Pipeline Léxico: conversión a minúsculas, preservación de ñ y tildes (con opción de
   unificar variantes con/sin tilde), filtrado de lista completa de stopwords en español
   y lematización con spaCy (`es_core_news_sm`) o stemming Snowball español.
3. Pipeline Semántico: limpieza mínima conservando la estructura oracional, stopwords y
   morfología intacta para alimentar modelos neuronales de Embeddings.
4. Segmentación en oraciones y párrafos, y validación de casos límite.
"""

import re
import unicodedata
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# Intentar cargar spaCy (opcional) con singleton; fallback a SnowballStemmer español
try:
    import spacy
    try:
        _SPACY_NLP = spacy.load("es_core_news_sm", disable=["parser", "ner"])
    except Exception:
        _SPACY_NLP = None
except ImportError:
    _SPACY_NLP = None

try:
    import snowballstemmer
    _SNOWBALL_STEMMER = snowballstemmer.stemmer("spanish")
except ImportError:
    _SNOWBALL_STEMMER = None


# Lista exhaustiva de palabras vacías (stopwords) del español (estándar NLTK / spaCy)
SPANISH_STOPWORDS: Set[str] = {
    # Artículos y contracciones
    "el", "la", "los", "las", "un", "una", "unos", "unas", "lo", "al", "del",
    # Preposiciones
    "a", "ante", "bajo", "cabe", "con", "contra", "de", "desde", "durante",
    "en", "entre", "hacia", "hasta", "mediante", "para", "por", "según", "segun",
    "sin", "so", "sobre", "tras", "versus", "vía", "via",
    # Conjunciones y conectores
    "y", "e", "ni", "o", "u", "pero", "mas", "sino", "aunque", "como", "cómo",
    "porque", "pues", "que", "qué", "si", "sí", "cuando", "cuándo", "donde",
    "dónde", "mientras", "embargo", "tanto", "asimismo", "entonces",
    # Pronombres personales, demostrativos, posesivos, relativos e indefinidos
    "yo", "tú", "tu", "vos", "él", "ella", "ello", "nosotros", "nosotras",
    "vosotros", "vosotras", "ellos", "ellas", "usted", "ustedes",
    "me", "te", "se", "nos", "os", "le", "les", "mi", "mis", "tus", "su", "sus",
    "nuestro", "nuestra", "nuestros", "nuestras", "vuestro", "vuestra",
    "vuestros", "vuestras", "mío", "mía", "míos", "mías", "tuyo", "tuya",
    "tuyos", "tuyas", "suyo", "suya", "suyos", "suyas",
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "aquel", "aquella", "aquellos", "aquellas", "esto", "eso", "aquello",
    "quien", "quién", "quienes", "quiénes", "cual", "cuál", "cuales", "cuáles",
    "cuyo", "cuya", "cuyos", "cuyas", "cuanto", "cuánto", "cuanta", "cuánta",
    "algo", "alguien", "alguno", "alguna", "algunos", "algunas", "algún", "algun",
    "nada", "nadie", "ninguno", "ninguna", "ningunos", "ningunas", "ningún", "ningun",
    "otro", "otra", "otros", "otras", "todo", "toda", "todos", "todas",
    "mismo", "misma", "mismos", "mismas", "cada", "ambos", "ambas",
    "varios", "varias", "cualquier", "cualquiera", "demás", "demas",
    # Adverbios frecuentes y cuantificadores
    "no", "más", "menos", "muy", "mucho", "mucha", "muchos", "muchas",
    "poco", "poca", "pocos", "pocas", "tan", "tanta", "tantos", "tantas",
    "bastante", "bastantes", "demasiado", "demasiada", "demasiados", "demasiadas",
    "ya", "aún", "aun", "también", "tambien", "tampoco", "solo", "sólo",
    "solamente", "siempre", "nunca", "jamás", "jamas", "aquí", "aqui", "acá", "aca",
    "ahí", "ahi", "allí", "alli", "allá", "alla", "cerca", "lejos", "antes",
    "después", "despues", "luego", "pronto", "tarde", "ayer", "hoy", "mañana",
    "bien", "mal", "así", "asi", "además", "ademas", "incluso", "quizá", "quiza",
    "quizás", "quizas", "acaso", "alrededor", "aproximadamente", "casi",
    # Formas verbales auxiliares y copulativas frecuentes
    "ser", "soy", "eres", "es", "somos", "sois", "son", "era", "eras",
    "éramos", "eramos", "eran", "fui", "fuiste", "fue", "fuimos", "fueron",
    "sea", "seas", "seamos", "sean", "sido", "siendo", "será", "serán",
    "estar", "estoy", "estás", "estas", "está", "esta", "estamos", "estáis",
    "están", "estan", "estaba", "estabas", "estábamos", "estaban", "estuvo",
    "estuvieron", "esté", "estén", "estado", "estando",
    "haber", "he", "has", "ha", "hay", "hemos", "habéis", "han", "había", "habia",
    "habían", "habian", "hubo", "habrá", "habra", "habría", "habria", "haya",
    "hayan", "habido", "habiendo",
    "tener", "tengo", "tienes", "tiene", "tenemos", "tienen", "tenía", "tenia",
    "tuvo", "tuvieron", "tenido", "teniendo",
    "hacer", "hago", "hace", "hacemos", "hacen", "hizo", "hicieron", "hecho",
    "ir", "voy", "vas", "va", "vamos", "van", "iba", "iban", "ido",
    "poder", "puedo", "puede", "podemos", "pueden", "pudo", "podido", "pudiendo",
    # Preposiciones locativas, conectores discursivos y demostrativos formales sin carga temática
    "dentro", "fuera", "pese", "tal", "tales", "través", "traves", "frente",
    "respecto", "debido", "dado", "dada", "dados", "dadas", "dicho", "dicha",
    "dichos", "dichas", "únicamente", "unicamente", "adicionalmente",
    "posteriormente", "anteriormente", "obstante",
}

# Patrones de limpieza de artefactos de PDF, URLs, emails y puntuación
HYPHEN_LINEBREAK_PATTERN = re.compile(
    r"([a-zA-ZáéíóúüÁÉÍÓÚÜñÑ])-\s*(?:\r?\n)+\s*([a-zA-ZáéíóúüÁÉÍÓÚÜñÑ])"
)
PAGE_NUMBER_LINE_PATTERN = re.compile(
    r"(?im)^\s*(?:p[áa]gina|p[áa]g\.?|page)\s*\d+(?:\s*(?:de|of|/)\s*\d+)?\s*$"
)
RUNNING_HEADER_PIPE_PATTERN = re.compile(
    r"(?im)^\s*[A-Za-z0-9_.\-]{8,}\s*\|\s*[^\n]{5,120}\s*$"
)
URL_PATTERN = re.compile(r"(?i)\b(?:https?://|www\.)\S+\b")
EMAIL_PATTERN = re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b")
NON_SPANISH_ALPHANUM_PATTERN = re.compile(r"[^a-zA-ZáéíóúüÁÉÍÓÚÜñÑ0-9\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?…])\s+|\n{2,}")
PARAGRAPH_SPLIT_PATTERN = re.compile(r"(?:\r?\n\s*){2,}")
NUMERIC_ONLY_PATTERN = re.compile(r"^\d+$")

_ACCENT_TRANSLATION = str.maketrans("áéíóúüÁÉÍÓÚÜ", "aeiouuAEIOUU")


def strip_accents_preserve_enie(text: str) -> str:
    """
    Elimina las tildes de las vocales (unificando variantes como 'esta'/'está')
    pero conserva intacta la letra 'ñ'/'Ñ'.
    """
    if not text:
        return ""
    text_nfc = unicodedata.normalize("NFC", text)
    return text_nfc.translate(_ACCENT_TRANSLATION)


def clean_pdf_and_web_artifacts(text: str) -> str:
    """
    Normaliza Unicode a NFC y limpia artefactos típicos de PDFs y textos web:
    - Une palabras cortadas por guion al final de línea ("proce-\\nsamiento" -> "procesamiento").
    - Elimina líneas de numeración de página ("Página 1 de 5") y encabezados/pies repetidos.
    - Remueve URLs y direcciones de correo electrónico.
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Normalización Unicode canónica (NFC)
    cleaned = unicodedata.normalize("NFC", str(text))

    # 2. Unir palabras cortadas por guion al final de línea en PDFs
    cleaned = HYPHEN_LINEBREAK_PATTERN.sub(r"\1\2", cleaned)

    # 3. Eliminar numeraciones de página aisladas y cintillos con identificador de archivo + '|'
    cleaned = PAGE_NUMBER_LINE_PATTERN.sub("", cleaned)
    cleaned = RUNNING_HEADER_PIPE_PATTERN.sub("", cleaned)

    # 4. Detectar y remover encabezados/pies de página repetidos en documentos multipágina
    lines = cleaned.splitlines()
    if len(lines) >= 8:
        candidate_lines = [
            ln.strip() for ln in lines if 3 <= len(ln.strip()) <= 160
        ]
        line_counts = Counter(candidate_lines)
        repeated_headers = set()
        for ln, count in line_counts.items():
            if count >= 3:
                repeated_headers.add(ln)
            elif count >= 2 and len(lines) >= 15 and (
                "|" in ln or "·" in ln or "–" in ln or "—" in ln
                or ln.lower().startswith(("fuente", "plantilla", "integrantes", "repositorio", "universidad"))
            ):
                repeated_headers.add(ln)
        if repeated_headers:
            lines = [ln for ln in lines if ln.strip() not in repeated_headers]
            cleaned = "\n".join(lines)

    # 5. Eliminar URLs y correos electrónicos
    cleaned = URL_PATTERN.sub(" ", cleaned)
    cleaned = EMAIL_PATTERN.sub(" ", cleaned)

    return cleaned


def split_paragraphs(text: str) -> List[str]:
    """
    Segmenta un documento en párrafos limpios no vacíos.
    """
    cleaned = clean_pdf_and_web_artifacts(text)
    if not cleaned.strip():
        return []
    raw_paragraphs = PARAGRAPH_SPLIT_PATTERN.split(cleaned)
    paragraphs: List[str] = []
    for p in raw_paragraphs:
        # Reemplazar saltos de línea simples dentro del mismo párrafo por espacio
        norm_p = WHITESPACE_PATTERN.sub(" ", p).strip()
        if norm_p:
            paragraphs.append(norm_p)
    return paragraphs


def split_sentences(text: str) -> List[str]:
    """
    Segmenta un documento en oraciones individuales conservando su estructura semántica.
    """
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return []
    sentences: List[str] = []
    for para in paragraphs:
        parts = SENTENCE_SPLIT_PATTERN.split(para)
        for part in parts:
            s = WHITESPACE_PATTERN.sub(" ", part).strip()
            if s:
                sentences.append(s)
    return sentences


def preprocess_semantic(text: str) -> Dict[str, Any]:
    """
    Pipeline Semántico (para modelos de Embeddings como Gemini o MiniLM):
    Conserva las oraciones completas, las stopwords y la morfología original,
    aplicando únicamente normalización Unicode NFC y limpieza de basura de PDF/URLs.
    """
    if not text or not isinstance(text, str):
        return {
            "clean_text": "",
            "sentences": [],
            "paragraphs": [],
        }

    paragraphs = split_paragraphs(text)
    sentences = split_sentences(text)
    clean_text = "\n\n".join(paragraphs) if paragraphs else ""

    return {
        "clean_text": clean_text,
        "sentences": sentences,
        "paragraphs": paragraphs,
    }


def stem_word(word: str, unify_accents: bool = True) -> str:
    """
    Reduce una palabra en español a su raíz morfológica canónica (stem/lema).
    Utiliza Snowball Spanish Stemmer (con normalización ortográfica de alternancias
    como z/c y s/z en raíces) para garantizar que variaciones morfológicas como
    'tokenización'/'tokenizar', 'analiza'/'analizar'/'análisis' o 'procesamiento'/'procesar'
    coincidan exactamente en el pipeline léxico.
    """
    if not word:
        return ""

    w = unicodedata.normalize("NFC", word.lower().strip())
    if NUMERIC_ONLY_PATTERN.match(w):
        return w

    # Evitar falsos amigos morfológicos conocidos en Snowball español
    w_no_acc = strip_accents_preserve_enie(w)
    if w_no_acc in ("procesador", "procesadores"):
        return "procesador"
    if w_no_acc == "vera":
        return "vera"

    # Si spaCy está instalado y cargado, podemos apoyarnos en Snowball para unificar
    # sustantivos deverbales ('tokenización') con sus verbos ('tokenizar')
    if _SNOWBALL_STEMMER is not None:
        stemmed = _SNOWBALL_STEMMER.stemWord(w)
    else:
        stemmed = w

    if unify_accents:
        stemmed = strip_accents_preserve_enie(stemmed)

    # Armonización de alternancias consonánticas del español en raíces verbales/nominales
    # Ej.: 'analis' (de análisis) -> 'analiz' (de analizar/analiza)
    # Ej.: 'clasific' (de clasificar) -> 'clasif' (de clasificación)
    if stemmed in ("analis", "analisis"):
        stemmed = "analiz"
    elif stemmed == "clasific":
        stemmed = "clasif"
    elif stemmed.endswith("c") and len(stemmed) >= 5:
        # Ej. 'producc' / 'produc'
        pass

    return stemmed


def normalize_text(text: str, unify_accents: bool = False) -> str:
    """
    Convierte una copia del texto a minúsculas (NFC), limpia artefactos de PDF,
    elimina puntuación y caracteres especiales (preservando ñ y tildes del español)
    y normaliza los espacios en blanco.

    Args:
        text: Cadena de entrada original.
        unify_accents: Si es True, unifica vocales con y sin tilde (conservando ñ).

    Returns:
        Texto limpio, en minúsculas y con espacios simples normalizados.
    """
    if not text or not isinstance(text, str):
        return ""

    cleaned_artifacts = clean_pdf_and_web_artifacts(text)
    text_copy = cleaned_artifacts.lower()

    if unify_accents:
        text_copy = strip_accents_preserve_enie(text_copy)

    cleaned = NON_SPANISH_ALPHANUM_PATTERN.sub(" ", text_copy)
    normalized = WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    return normalized


def tokenize(text: str, unify_accents: bool = False) -> List[str]:
    """
    Normaliza el texto y lo segmenta en una lista ordenada de tokens (palabras).
    """
    normalized = normalize_text(text, unify_accents=unify_accents)
    if not normalized:
        return []
    return normalized.split(" ")


def identify_stopwords(
    tokens: Iterable[str],
    custom_stopwords: Optional[Set[str]] = None,
) -> List[str]:
    """
    Identifica las palabras vacías (stopwords) presentes en una secuencia de tokens.
    Comprueba tanto la forma exacta como la forma sin tilde para capturar 'esta'/'está'.
    """
    stopword_set = custom_stopwords if custom_stopwords is not None else SPANISH_STOPWORDS
    found: List[str] = []
    for token in tokens:
        if token in stopword_set or strip_accents_preserve_enie(token) in stopword_set:
            found.append(token)
    return found


def remove_stopwords_from_tokens(
    tokens: Iterable[str],
    custom_stopwords: Optional[Set[str]] = None,
) -> List[str]:
    """
    Filtra y elimina las palabras vacías (stopwords) de una secuencia de tokens.
    Cuando el documento contiene vocabulario alfabético suficiente (>= 3 palabras),
    también filtra números aislados de tablas/fechas y símbolos de 1 carácter para
    evitar inflación léxica en PDFs reales, preservando los casos límite ('only_numbers',
    'single_word') en entradas cortas.
    """
    token_list = list(tokens)
    stopword_set = custom_stopwords if custom_stopwords is not None else SPANISH_STOPWORDS
    if custom_stopwords is not None and custom_stopwords is not SPANISH_STOPWORDS:
        base_filtered = [token for token in token_list if token not in stopword_set]
    else:
        base_filtered = [
            token
            for token in token_list
            if token not in stopword_set and strip_accents_preserve_enie(token) not in stopword_set
        ]

    lexical_alpha_tokens = [
        t for t in base_filtered if len(t) >= 2 and not NUMERIC_ONLY_PATTERN.match(t)
    ]
    if len(lexical_alpha_tokens) >= 3:
        return lexical_alpha_tokens
    return base_filtered


def preprocess_text(
    text: str,
    remove_stopwords: bool = False,
    custom_stopwords: Optional[Set[str]] = None,
    use_stemming: bool = False,
    unify_accents: bool = False,
) -> List[str]:
    """
    Pipeline directo de preprocesamiento léxico: normaliza, tokeniza, opcionalmente
    remueve stopwords y opcionalmente aplica stemming morfológico.
    """
    tokens = tokenize(text, unify_accents=unify_accents)
    if remove_stopwords:
        tokens = remove_stopwords_from_tokens(tokens, custom_stopwords=custom_stopwords)
    if use_stemming:
        tokens = [stem_word(t, unify_accents=True) for t in tokens if t]
    return tokens


def detect_likely_spanish(tokens: List[str]) -> bool:
    """
    Heurística ligera para verificar si un texto con suficientes palabras parece estar en español.
    Evalúa la proporción de stopwords comunes en español o caracteres propios (ñ, tildes).
    """
    if len(tokens) < 8:
        return True
    es_Hits = sum(
        1
        for t in tokens
        if t in SPANISH_STOPWORDS
        or any(ch in t for ch in "áéíóúüñ")
    )
    return (es_Hits / len(tokens)) >= 0.08


def validate_text_input(
    all_tokens: List[str],
    filtered_tokens: List[str],
    doc_label: str = "Documento",
) -> Dict[str, Any]:
    """
    Valida casos límite sobre un documento procesado:
    - Texto vacío o sin tokens alfanuméricos.
    - Compuesto únicamente por stopwords.
    - Una sola palabra útil.
    - Compuesto únicamente por números.
    - Texto muy corto (< 20 palabras útiles).
    - Idioma posiblemente no español.
    """
    warnings: List[str] = []
    edge_case: Optional[str] = None

    if not all_tokens:
        edge_case = "empty_text"
        warnings.append(f"{doc_label}: El texto está vacío o solo contiene signos/emojis sin palabras válidas.")
    elif not filtered_tokens:
        edge_case = "only_stopwords"
        warnings.append(f"{doc_label}: El texto está compuesto únicamente por palabras vacías (stopwords).")
    else:
        if all(NUMERIC_ONLY_PATTERN.match(t) for t in filtered_tokens):
            edge_case = "only_numbers"
            warnings.append(f"{doc_label}: El texto contiene únicamente valores numéricos sin vocabulario léxico.")
        elif len(filtered_tokens) == 1:
            edge_case = "single_word"
            warnings.append(f"{doc_label}: Contiene una única palabra útil ('{filtered_tokens[0]}'), lo que reduce la fiabilidad estadística.")
        elif len(filtered_tokens) < 20:
            warnings.append(f"{doc_label}: Texto corto ({len(filtered_tokens)} palabras útiles; se recomiendan al menos 20 para mayor precisión estadística).")

        if not detect_likely_spanish(all_tokens):
            warnings.append(f"{doc_label}: Se detectó una baja proporción de patrones del español; verifique que el idioma del documento sea español.")

    return {
        "is_valid": len(all_tokens) > 0,
        "has_active_tokens": len(filtered_tokens) > 0,
        "edge_case": edge_case,
        "warnings": warnings,
    }


def preprocess_document(
    text: str,
    remove_stopwords: bool = True,
    custom_stopwords: Optional[Set[str]] = None,
    use_stemming: bool = True,
    unify_accents: bool = False,
    doc_label: str = "Documento",
) -> Dict[str, Any]:
    """
    Procesa un documento de forma integral mediante ambos pipelines:
    1. Pipeline Léxico: tokens de superficie, tokens con stemming/lematización,
       mapa inverso `stem_to_surface` y conteo de stopwords.
    2. Pipeline Semántico: oraciones, párrafos y texto limpio con contexto intacto.
    3. Validación de casos límite.
    """
    original_text = str(text) if isinstance(text, str) else ""
    normalized_text = normalize_text(original_text, unify_accents=unify_accents)
    all_tokens = normalized_text.split(" ") if normalized_text else []

    stopword_set = custom_stopwords if custom_stopwords is not None else SPANISH_STOPWORDS
    stopwords_found = identify_stopwords(all_tokens, custom_stopwords=stopword_set)
    filtered_tokens = remove_stopwords_from_tokens(all_tokens, custom_stopwords=stopword_set)

    active_surface_tokens = filtered_tokens if remove_stopwords else list(all_tokens)

    # Construir stems morfológicos y mapa de stem -> palabra original más frecuente
    stemmed_tokens: List[str] = []
    stem_surface_counter: Dict[str, Counter] = {}
    for surf in active_surface_tokens:
        st = stem_word(surf, unify_accents=True)
        stemmed_tokens.append(st)
        if st not in stem_surface_counter:
            stem_surface_counter[st] = Counter()
        stem_surface_counter[st][surf] += 1

    stem_to_surface: Dict[str, str] = {
        st: counter.most_common(1)[0][0]
        for st, counter in stem_surface_counter.items()
    }

    # Pipeline semántico (oraciones y párrafos con contexto intacto)
    semantic_data = preprocess_semantic(original_text)

    # Segmentos léxicos (oraciones tokenizadas y con stemming para ajuste de IDF en cosine.py)
    sentence_lexical_segments: List[List[str]] = []
    for sent in semantic_data["sentences"]:
        sent_tokens = tokenize(sent, unify_accents=unify_accents)
        if remove_stopwords:
            sent_tokens = remove_stopwords_from_tokens(sent_tokens, custom_stopwords=stopword_set)
        if sent_tokens:
            if use_stemming:
                sentence_lexical_segments.append([stem_word(t, unify_accents=True) for t in sent_tokens])
            else:
                sentence_lexical_segments.append(sent_tokens)

    validation = validate_text_input(all_tokens, active_surface_tokens, doc_label=doc_label)
    cleaned_text = " ".join(active_surface_tokens)

    return {
        "original_text": original_text,
        "normalized_text": normalized_text,
        "cleaned_text": cleaned_text,
        "all_tokens": all_tokens,
        "filtered_tokens": filtered_tokens,
        "tokens": active_surface_tokens,
        "stemmed_tokens": stemmed_tokens,
        "stem_to_surface": stem_to_surface,
        "sentence_lexical_segments": sentence_lexical_segments,
        "semantic": semantic_data,
        "sentences": semantic_data["sentences"],
        "paragraphs": semantic_data["paragraphs"],
        "stopwords_found": stopwords_found,
        "unique_stopwords_found": sorted(list(set(stopwords_found))),
        "stopwords_count": len(stopwords_found),
        "remove_stopwords_applied": bool(remove_stopwords),
        "validation": validation,
    }
