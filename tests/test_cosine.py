"""
Pruebas unitarias y de integración para Similitud del Coseno TF-IDF (core/cosine.py),
Embeddings Semánticos con Mock de Gemini (core/embeddings.py), Estadísticas (core/statistics.py)
y Endpoints Flask (app.py).
"""

import io
import math
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from core.cosine import calculate_cosine_similarity, calculate_manual_cosine_similarity
from core.embeddings import calculate_embedding_similarity
from core.file_reader import extract_text_from_file
from core.statistics import generate_statistical_analysis
from app import app


def test_identical_texts_all_three_algorithms():
    text = "El procesamiento del lenguaje natural analiza grandes volúmenes de texto."
    analysis = generate_statistical_analysis(text, text, remove_stopwords=True)

    assert analysis["jaccard"]["similarity_score"] == 1.0
    assert analysis["cosine"]["similarity_score"] == pytest.approx(1.0, rel=1e-5)
    assert analysis["embeddings"]["similarity_score"] == pytest.approx(1.0, rel=1e-5)
    assert analysis["cosine"]["vocabulary_size"] == 7


def test_completely_different_texts_low_calibrated_scores():
    doc_a = "física cuántica electrones fotones relatividad termodinámica partículas"
    doc_b = "gastronomía cocina ingredientes hornear pastelería postres chocolate"
    analysis = generate_statistical_analysis(doc_a, doc_b, remove_stopwords=True)

    assert analysis["jaccard"]["similarity_score"] == 0.0
    assert analysis["cosine"]["similarity_score"] == 0.0
    # El embedding calibrado para textos disjuntos sin relación debe ser cercano a 0 (< 0.15)
    assert analysis["embeddings"]["calibrated_similarity_score"] < 0.15
    assert analysis["interpretation"]["pattern"] == "textos_sin_relacion"


def test_partially_similar_academic_reference_cosine():
    """
    Verifica el ejemplo manual de la Guía Completa de Similitud NLP (Sección 1.3):
    Oración A: "El procesamiento del lenguaje natural es fascinante"
    Oración B: "El procesamiento de lenguaje y texto es genial"
    """
    doc_a = "El procesamiento del lenguaje natural es fascinante"
    doc_b = "El procesamiento de lenguaje y texto es genial"

    bow_res = calculate_manual_cosine_similarity(doc_a, doc_b, remove_stopwords=True)
    assert bow_res["similarity_score"] == pytest.approx(0.5, rel=1e-4)
    assert bow_res["similarity_percentage"] == 50.0
    assert bow_res["dot_product"] == 2.0

    tfidf_res = calculate_cosine_similarity(doc_a, doc_b, remove_stopwords=True)
    assert tfidf_res["vocabulary_size"] == 6
    assert 0.0 < tfidf_res["similarity_percentage"] < 100.0
    assert 0.0 < tfidf_res["char_ngram_cosine_score"] <= 1.0
    assert len(tfidf_res["vectorization_info"]["top_contributing_terms"]) == 2


def test_scale_and_length_invariance_cosine_and_containment():
    """
    Mismo contenido con distinta longitud (texto repetido 3 veces) -> coseno ≈ 1 y containment = 1.
    """
    doc_a = "inteligencia artificial aprendizaje automático ciencia datos"
    doc_b = " ".join([doc_a, doc_a, doc_a])
    analysis = generate_statistical_analysis(doc_a, doc_b, remove_stopwords=True)

    assert analysis["cosine"]["similarity_score"] == pytest.approx(1.0, rel=1e-5)
    assert analysis["jaccard"]["containment_score"] == 1.0
    assert analysis["jaccard"]["containment_percentage"] == 100.0


def test_paraphrase_pair_embeddings_higher_than_jaccard():
    """
    Par de paráfrasis:
    "El estudiante analiza un documento extenso en su computadora"
    vs
    "Un alumno evalúa un escrito amplio en su ordenador"
    -> Jaccard bajo (0%), embedding claramente mayor que Jaccard.
    """
    doc_a = "El estudiante analiza un documento extenso en su computadora"
    doc_b = "Un alumno evalúa un escrito amplio en su ordenador"

    analysis = generate_statistical_analysis(doc_a, doc_b, remove_stopwords=True)

    assert analysis["jaccard"]["similarity_score"] == 0.0
    assert analysis["cosine"]["similarity_score"] == 0.0
    assert analysis["embeddings"]["similarity_score"] > 0.65
    assert analysis["embeddings"]["similarity_percentage"] > analysis["jaccard"]["similarity_percentage"] + 50.0
    assert len(analysis["embeddings"]["semantic_word_matches"]) >= 4
    assert analysis["interpretation"]["pattern"] == "parafrasis_o_sinonimia"


def test_morphological_variations_in_cosine():
    """
    Variaciones morfológicas: 'tokenización' vs 'tokenizar', 'analiza' vs 'analizar'
    deben contar como coincidentes en el pipeline léxico.
    """
    res_tok = calculate_cosine_similarity("tokenización", "tokenizar", remove_stopwords=True)
    assert res_tok["similarity_score"] == pytest.approx(1.0, rel=1e-5)

    res_ana = calculate_cosine_similarity("analiza", "analizar", remove_stopwords=True)
    assert res_ana["similarity_score"] == pytest.approx(1.0, rel=1e-5)


def test_edge_cases_never_nan_and_always_in_zero_one():
    edge_pairs = [
        ("", ""),
        ("análisis estadístico de datos", ""),
        ("el la los las de del", "un una unos unas en por"),
        ("a", "z"),
        ("🚀🔥✨🎉", "😊👍💡"),
        ("12345 67890", "99999 00000"),
        ("Ñandú lingüístico pingüino canción éxito", "Canción lingüística del ñandú y el pingüino"),
    ]
    for text_a, text_b in edge_pairs:
        analysis = generate_statistical_analysis(text_a, text_b, remove_stopwords=True)
        for metric_key, metric_obj in analysis["unified_metrics"].items():
            score = metric_obj["score"]
            raw = metric_obj["raw_score"]
            assert not math.isnan(score), f"NaN detectado en {metric_key}"
            assert not math.isnan(raw), f"NaN crudo detectado en {metric_key}"
            assert 0.0 <= score <= 1.0, f"Fuera de rango [0, 1] en {metric_key}: {score}"
            assert 0.0 <= raw <= 1.0, f"Fuera de rango crudo [0, 1] en {metric_key}: {raw}"


def test_gemini_mode_with_mock_and_fallback():
    """
    Prueba el modo Gemini utilizando un Mock (sin llamadas reales a la red),
    verificando el paso de `task_type="SEMANTIC_SIMILARITY"`, normalización L2
    y el fallback automático al modo offline cuando la API falla.
    """
    fake_Client_cls = MagicMock()
    fake_instance = MagicMock()
    fake_Client_cls.return_value = fake_instance

    def _fake_embed_content(model, contents, config=None):
        # Comprobar que se envía task_type="SEMANTIC_SIMILARITY"
        if isinstance(config, dict):
            assert config.get("task_type") == "SEMANTIC_SIMILARITY"
        else:
            assert getattr(config, "task_type", None) == "SEMANTIC_SIMILARITY"

        embeddings_out = []
        for idx, _txt in enumerate(contents):
            # Crear un vector no normalizado de 768 dimensiones para comprobar la normalización L2
            vec = np.ones(768, dtype=np.float64) * (2.5 + idx * 0.01)
            emb_item = MagicMock()
            emb_item.values = vec.tolist()
            embeddings_out.append(emb_item)

        resp = MagicMock()
        resp.embeddings = embeddings_out
        return resp

    fake_instance.models.embed_content.side_effect = _fake_embed_content

    with patch("core.embeddings.genai.Client", fake_Client_cls):
        res_gemini = calculate_embedding_similarity(
            "La inteligencia artificial procesa lenguaje natural.",
            "Los modelos de IA analizan textos en español.",
            gemini_api_key="FAKE_KEY_FOR_MOCK_TEST",
            output_dimensionality=768,
        )
        assert res_gemini["using_gemini_api"] is True
        assert res_gemini["mode"] == "gemini"
        assert res_gemini["dimensions"] == 768
        # Las normas de los vectores tras normalización L2 manual deben ser 1.0
        assert res_gemini["norm_a"] == pytest.approx(1.0, rel=1e-4)
        assert res_gemini["norm_b"] == pytest.approx(1.0, rel=1e-4)
        assert 0.0 <= res_gemini["similarity_score"] <= 1.0

    # Probar fallback automático cuando la API lanza excepción (ej. 401/429)
    fake_instance.models.embed_content.side_effect = RuntimeError("401 Invalid API Key")
    with patch("core.embeddings.genai.Client", fake_Client_cls):
        res_fallback = calculate_embedding_similarity(
            "El estudiante analiza un documento.",
            "Un alumno evalúa un escrito.",
            gemini_api_key="INVALID_MOCK_KEY",
        )
        assert res_fallback["using_gemini_api"] is False
        assert res_fallback["mode"] in ("local_hybrid_lsa", "offline_minilm", "offline_approx_128d")
        assert len(res_fallback["warnings"]) >= 1


def test_local_hybrid_lsa_zero_tokens_and_deep_scan_repetitions_and_context():
    """
    Verifica que:
    1. El motor por defecto es 100% local ('local_hybrid_lsa') y consume 0 tokens de API
       incluso si existiera GEMINI_API_KEY en el entorno.
    2. El escaneo profundo detecta palabras repetidas exactas, familias morfológicas repetidas,
       alertas de proximidad corta (<= 6 palabras), frases repetidas, alineamiento oración a oración
       y concordancias KWIC.
    """
    doc_a = (
        "El análisis de datos permite analizar grandes volúmenes de datos en el sistema informático. "
        "Este análisis de datos es fundamental porque el sistema procesa datos complejos rápidamente. "
        "Sin embargo, el estudiante redacta un informe extenso en su computadora."
    )
    doc_b = (
        "La evaluación de información facilita estudiar amplias cantidades de registros en la plataforma digital. "
        "Dicho estudio resulta esencial ya que el programa examina información compleja con velocidad. "
        "No obstante, el alumno elabora un documento amplio en su ordenador."
    )

    with patch.dict(os.environ, {"GEMINI_API_KEY": "SHOULD_NOT_BE_USED_AUTOMATICALLY"}):
        analysis = generate_statistical_analysis(doc_a, doc_b, remove_stopwords=True)

    emb = analysis["embeddings"]
    assert emb["using_gemini_api"] is False
    assert emb["mode"] in ("local_hybrid_lsa", "offline_minilm")
    assert emb["lsa_similarity_score"] > 0.0

    deep = analysis["deep_scan"]
    rep = deep["repetition_analysis"]
    # En doc_a, "datos" se repite 4 veces y la familia "analiz-" ("análisis" x2 + "analizar" x1) aparece 3 veces
    rep_words_a = {item["word"]: item for item in rep["doc_a"]["exact_repeated"]}
    assert "datos" in rep_words_a
    assert rep_words_a["datos"]["count"] == 4
    assert rep_words_a["datos"]["severity"] == "Sobre-repetición Alta"

    morph_stems_a = {item["stem"]: item for item in rep["doc_a"]["morphological_families"]}
    assert "analiz" in morph_stems_a
    assert morph_stems_a["analiz"]["total_count"] == 3
    assert morph_stems_a["analiz"]["is_multi_form"] is True

    # Frase interna repetida en A ("análisis datos")
    phrases_a = [p["phrase"] for p in rep["doc_a"]["repeated_phrases"]]
    assert any("análisis datos" in ph for ph in phrases_a)

    # Alineamiento contextual oración por oración (3 oraciones en A y 3 en B)
    ctx = deep["context_alignment"]
    assert ctx["summary"]["sentences_analyzed_a"] == 3
    assert ctx["summary"]["sentences_analyzed_b"] == 3
    assert len(ctx["pairs"]) == 3
    assert ctx["summary"]["average_alignment_percentage"] > 45.0

    # Concordancias KWIC y Dominios / Conectores Discursivos
    assert len(deep["kwic_concordances"]) >= 1
    disc = deep["discourse_and_domains"]
    assert len(disc["shared_domains"]) >= 1
    assert disc["discourse_a"]["total_connectors"] >= 2
    assert disc["discourse_b"]["total_connectors"] >= 2
    assert 0.0 <= disc["readability_a"]["fernandez_huerta_score"] <= 100.0


def test_flask_web_interface_and_api_endpoints():
    client = app.test_client()

    resp_index = client.get("/")
    assert resp_index.status_code == 200
    html_text = resp_index.get_data(as_text=True)
    assert "Similarity Lab" in html_text
    assert "Text Similarity &amp; NLP Analysis" in html_text
    assert "Pega aquí el primer documento..." in html_text
    assert "Pega aquí el segundo documento..." in html_text

    resp_status = client.get("/api/status")
    assert resp_status.status_code == 200
    assert resp_status.get_json()["status"] == "online"

    resp_empty = client.post("/api/compare", json={"text_a": "   ", "text_b": "Texto válido"})
    assert resp_empty.status_code == 400
    assert resp_empty.get_json()["status"] == "error"

    resp_compare = client.post(
        "/api/compare",
        json={
            "text_a": "El procesamiento del lenguaje natural es fascinante",
            "text_b": "El procesamiento de lenguaje y texto es genial",
            "remove_stopwords": True,
        },
    )
    assert resp_compare.status_code == 200
    payload = resp_compare.get_json()
    assert payload["status"] == "success"
    data = payload["data"]
    assert data["jaccard"]["similarity_percentage"] == 33.33
    assert data["cosine"]["vectorization_info"]["bow_cosine_percentage"] == 50.0
    assert "embeddings" in data
    assert "unified_metrics" in data
    assert "confidence" in data


def test_file_upload_pdf_and_txt_extraction():
    txt_bytes = "El procesamiento de lenguaje natural y los embeddings vectorizan palabras.".encode("utf-8")
    extracted_txt = extract_text_from_file(txt_bytes, "documento_prueba.txt")
    assert extracted_txt["file_type"] == "TXT"
    assert "embeddings vectorizan palabras" in extracted_txt["text"]

    # Probar codificación Latin-1 / cp1252 con tildes y ñ
    latin_bytes = "Análisis lingüístico en España con niño y acción.".encode("cp1252")
    extracted_latin = extract_text_from_file(latin_bytes, "nota_cp1252.txt")
    assert "lingüístico" in extracted_latin["text"]
    assert "España" in extracted_latin["text"]

    client = app.test_client()
    resp_txt = client.post(
        "/api/upload",
        data={"file": (io.BytesIO(txt_bytes), "ensayo.txt")},
        content_type="multipart/form-data",
    )
    assert resp_txt.status_code == 200
    assert resp_txt.get_json()["status"] == "success"

    resp_invalid = client.post(
        "/api/upload",
        data={"file": (io.BytesIO(b"datos"), "imagen.png")},
        content_type="multipart/form-data",
    )
    assert resp_invalid.status_code == 400


def test_reference_profile_cosine_matches_sklearn_within_1e_9():
    """
    Requisito 6:
    Compara el coseno del perfil `reference` contra `sklearn` directo:
    `TfidfVectorizer().fit_transform([a, b])` + `cosine_similarity`,
    verificando que la diferencia absoluta sea `< 1e-9`.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from core.profiles import run_profile_comparison

    test_pairs = [
        (
            "El procesamiento del lenguaje natural es fascinante",
            "El procesamiento de lenguaje y texto es genial",
        ),
        (
            "La inteligencia artificial y el aprendizaje automático analizan grandes volúmenes de datos",
            "Los modelos de inteligencia artificial procesan datos complejos mediante algoritmos",
        ),
        (
            "uno dos tres cuatro cinco",
            "cero uno dos tres cuatro cinco seis siete ocho nueve",
        ),
        (
            "texto idéntico de prueba para validar precisión numérica",
            "texto idéntico de prueba para validar precisión numérica",
        ),
        (
            "física cuántica electrones relatividad",
            "gastronomía pastelería chocolate hornear",
        ),
    ]

    for text_a, text_b in test_pairs:
        sk_mat = TfidfVectorizer().fit_transform([text_a, text_b])
        sk_cos = float(cosine_similarity(sk_mat[0:1], sk_mat[1:2])[0][0])

        ref_profile = run_profile_comparison(text_a, text_b, profile="reference")
        ref_cos = float(ref_profile["cosine"]["raw_similarity_score"])

        diff = abs(ref_cos - sk_cos)
        assert diff < 1e-9, f"Diferencia {diff} >= 1e-9 para el par ({text_a!r}, {text_b!r})"


def test_sklearn_smooth_idf_formula_and_idf_modes():
    """
    Requisito 4:
    Verifica que la fórmula de sklearn con `smooth_idf=True` es:
        idf = ln((1 + n) / (1 + df)) + 1
    y prueba los 3 modos de IDF: 'two_docs', 'segments' y 'fit_a_transform_b'.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from core.cosine import compute_sklearn_idf_formula

    docs = [
        "alfa beta gamma",
        "alfa beta delta",
        "alfa epsilon zeta",
    ]
    vec_smooth = TfidfVectorizer(smooth_idf=True).fit(docs)
    vocab = vec_smooth.vocabulary_
    n_docs = len(docs)
    # 'alfa' aparece en 3 docs (df=3), 'beta' en 2 docs (df=2), 'gamma' en 1 doc (df=1)
    assert abs(vec_smooth.idf_[vocab["alfa"]] - compute_sklearn_idf_formula(n_docs, 3, smooth_idf=True)) < 1e-12
    assert abs(vec_smooth.idf_[vocab["beta"]] - compute_sklearn_idf_formula(n_docs, 2, smooth_idf=True)) < 1e-12
    assert abs(vec_smooth.idf_[vocab["gamma"]] - compute_sklearn_idf_formula(n_docs, 1, smooth_idf=True)) < 1e-12

    # Probar los 3 modos de idf_mode
    for mode in ("two_docs", "segments", "fit_a_transform_b"):
        res = calculate_cosine_similarity(
            "El análisis estadístico de textos en español",
            "El análisis computacional de documentos y textos",
            idf_mode=mode,
            sublinear_tf=(mode == "segments"),
            smooth_idf=True,
        )
        assert res["idf_mode"] == mode
        assert 0.0 <= res["similarity_score"] <= 1.0


def test_profiles_parallel_comparison_bertscore_no_idf_and_minilm_128_warning():
    """
    Requisitos 1 y 5:
    - Verifica que `compare_profiles_side_by_side` devuelve tanto `reference` como `enhanced`
      marcando cuál es el perfil principal.
    - Verifica que en `reference` el valor principal de embeddings es `raw` (sin calibrar)
      y en `enhanced` es `calibrated`, exponiendo siempre ambos.
    - Valida BERTScore con `use_idf=False` contra la fórmula canónica de `bert-score`.
    - Verifica la advertencia de truncamiento a 128 tokens de MiniLM (384-D) en textos largos.
    """
    from core.embeddings import validate_bertscore_no_idf_formula
    from core.profiles import compare_profiles_side_by_side

    text_a = "El estudiante universitario analiza un documento extenso en su computadora."
    text_b = "Un alumno académico evalúa un escrito amplio en su ordenador."

    side_by_side = compare_profiles_side_by_side(text_a, text_b, primary_profile="reference")
    assert side_by_side["primary_profile"] == "reference"
    assert side_by_side["reference"]["is_primary"] is True
    assert side_by_side["enhanced"]["is_primary"] is False
    assert len(side_by_side["comparison_table"]) >= 7

    emb_ref = side_by_side["reference"]["embeddings"]
    emb_enh = side_by_side["enhanced"]["embeddings"]
    assert emb_ref["primary_score_type"] == "raw"
    assert emb_ref["similarity_score"] == emb_ref["raw_similarity_score"]
    assert emb_enh["primary_score_type"] == "calibrated"
    assert emb_enh["similarity_score"] == emb_enh["calibrated_similarity_score"]
    assert "with_idf" in emb_enh["bertscore"]
    assert "without_idf" in emb_enh["bertscore"]

    # Validar BERTScore sin IDF contra la fórmula matricial exacta de bert-score (idf=False)
    bs_check = validate_bertscore_no_idf_formula(
        ["estudiante", "analiza", "documento", "computadora"],
        ["alumno", "evalúa", "escrito", "ordenador"],
    )
    assert bs_check["matches_canonical"] is True
    assert bs_check["max_abs_diff"] < 1e-6

    # Verificar advertencia de MiniLM cuando el documento supera 128 tokens
    long_text_a = " ".join(["palabra_prueba_contexto"] * 140)
    long_text_b = " ".join(["palabra_prueba_contexto"] * 40)
    emb_long = calculate_embedding_similarity(long_text_a, long_text_b)
    assert emb_long["exceeds_minilm_128_tokens"] is True
    assert emb_long["minilm_max_tokens"] == 128
    assert emb_long["minilm_dimensions"] == 384
    assert emb_long["minilm_truncation_warning"] is not None


def test_calibration_harness_grid_search():
    """
    Requisito 2:
    Verifica que el arnés de calibración (`tools/calibrate.py`) lee el dataset CSV/JSON,
    preserva `tool_name` y `tool_params`, ejecuta el grid-search sobre todas las métricas
    y encuentra la configuración óptima con error ~ 0.0.
    """
    from pathlib import Path
    from tools.calibrate import (
        format_calibration_table,
        load_calibration_dataset,
        run_calibration_grid_search,
    )

    csv_path = Path(__file__).resolve().parent.parent / "tools" / "sample_benchmark.csv"
    dataset = load_calibration_dataset(str(csv_path))
    assert len(dataset) == 4
    assert dataset[0]["tool_name"] == "external_validator_v1"
    assert dataset[0]["tool_params"] == "default"

    report = run_calibration_grid_search(dataset)
    for metric in ("jaccard", "dice", "overlap", "shingles_k2", "shingles_k3", "shingles_k5", "cosine_tfidf", "embeddings"):
        assert metric in report["by_metric"]
        assert report["by_metric"][metric]["mae"] < 1e-5

    table_txt = format_calibration_table(report)
    assert "cosine_tfidf" in table_txt
    assert "external_validator_v1" in table_txt


def test_unrelated_real_pdf_documents_below_10_percent():
    """
    Verifica sobre los PDFs reales en `date/`:
    - Doc_A (Computación Cuántica) vs Doc_B (Agricultura Regenerativa): < 10% en todas las métricas.
    - Antecedentes proyecto de tesis (Detección Phishing/ML) vs G07_Equipo-02_Panitas (Análisis de Datos/TI):
      < 10% léxico (Jaccard y Coseno TF-IDF) y 30%–40% de afinidad temática semántica en Embeddings.
    """
    from pathlib import Path

    date_dir = Path(__file__).resolve().parent.parent / "date"
    if not date_dir.exists():
        pytest.skip("Carpeta date/ no presente")

    doc_a_path = date_dir / "Doc_A_Computacion_Cuantica_Criptografia.pdf"
    doc_b_path = date_dir / "Doc_B_Agricultura_Regenerativa_Suelos_Aridos-1.pdf"
    ant_path = date_dir / "Antecedentes proyecto de tesis.pdf"
    pan_path = date_dir / "G07_Equipo-02_Panitas.pdf"

    text_a = extract_text_from_file(doc_a_path.read_bytes(), doc_a_path.name)["text"]
    text_b = extract_text_from_file(doc_b_path.read_bytes(), doc_b_path.name)["text"]
    analysis_ab = generate_statistical_analysis(text_a, text_b, remove_stopwords=True, profile="enhanced")
    assert analysis_ab["jaccard"]["similarity_percentage"] < 10.0
    assert analysis_ab["cosine"]["similarity_percentage"] < 10.0
    assert analysis_ab["embeddings"]["similarity_percentage"] < 10.0
    assert analysis_ab["interpretation"]["pattern"] == "textos_sin_relacion"

    text_ant = extract_text_from_file(ant_path.read_bytes(), ant_path.name)["text"]
    text_pan = extract_text_from_file(pan_path.read_bytes(), pan_path.name)["text"]
    analysis_ap = generate_statistical_analysis(text_ant, text_pan, remove_stopwords=True, profile="enhanced")
    assert analysis_ap["jaccard"]["similarity_percentage"] < 10.0
    assert analysis_ap["cosine"]["similarity_percentage"] < 10.0
    assert 30.0 <= analysis_ap["embeddings"]["similarity_percentage"] <= 40.0


def test_unap_guide_vector_geometry_kmeans_rag_polysemy_and_fasttext():
    """
    Valida todas las mejoras incorporadas desde la Guía Definitiva de Word Embeddings
    y Vectorización en NLP (UNA Puno):
    - Sección 1.2: Bolsa de subpalabras FastText (`'<ga'`, `'gat'`, `'ati'`, `'tit'`, `'ito>'`).
    - Sección 1.3: Detección de polisemia por Auto-Atención ('banco' préstamo vs 'banco' parque).
    - Sección 1.4: Geometría vectorial (Coseno, Distancia L2 y u·v == cos(u,v) con ||u||=1).
    - Caso 2.1: Búsqueda semántica en FAQ de soporte técnico.
    - Caso 2.2: Recuperación RAG k-NN con distancia vectorial.
    - Caso 2.3: Agrupamiento semántico KMeans (k=3, random_state=42, n_init=10).
    - Sección 3: Lineamientos de producción y chunking con solapamiento 10-20%.
    """
    from core.embeddings import (
        analyze_polysemy_and_self_attention,
        compute_vector_space_geometry,
        extract_fasttext_subwords,
        perform_rag_knn_retrieval,
        perform_semantic_kmeans_clustering,
    )

    # 1. Subpalabras FastText para 'gatito'
    subwords = extract_fasttext_subwords("gatito", min_n=3, max_n=5)
    for expected_ng in ("<ga", "gat", "ati", "tit", "ito>"):
        assert expected_ng in subwords

    # 2. Métricas geométricas L2 (u·v == cos(u,v) cuando ||u||=||v||=1)
    u = np.array([3.0, 4.0, 0.0], dtype=np.float64)
    u /= np.linalg.norm(u)
    v = np.array([1.0, 2.0, 2.0], dtype=np.float64)
    v /= np.linalg.norm(v)
    geom = compute_vector_space_geometry(u, v)
    assert geom["l2_normalized"] is True
    assert geom["dot_equals_cosine_verified"] is True
    assert abs(geom["dot_product"] - geom["cosine_similarity"]) < 1e-9
    assert geom["euclidean_distance_l2"] == pytest.approx(float(np.linalg.norm(u - v)), abs=1e-6)

    # 3. Caso 2.1: Soporte Técnico (sin coincidencia exacta 'autenticación' / 'error 500')
    faq_database = [
        "Instrucciones para restablecer la contraseña en caso de bloqueo de cuenta.",
        "Cómo solucionar el error de autenticación fallida y pantalla de fallo en el login.",
        "Pasos para configurar la facturación automática y cambiar tarjeta de crédito.",
        "Guía para solicitar la baja del servicio y cancelación de suscripción.",
    ]
    user_query = "No puedo entrar a mi cuenta y me sale una pantalla roja."
    rag_support = perform_rag_knn_retrieval([user_query], faq_database, top_k=1)
    assert len(rag_support["results"]) == 1
    assert "autenticación fallida" in rag_support["results"][0]["retrieved_text"]

    # 4. Caso 2.2: Arquitectura RAG sobre políticas internas
    policies = [
        "Los empleados tienen derecho a 15 días de vacaciones pagadas tras un año de servicio.",
        "El reembolso de gastos de viaje requiere la presentación de facturas dentro de los 5 días hábiles.",
        "El trabajo remoto está permitido los días viernes previa aprobación del líder de equipo.",
    ]
    policy_query = "¿Cuántos días de descanso me corresponden al cumplir un año?"
    rag_policy = perform_rag_knn_retrieval([policy_query], policies, top_k=1)
    assert "15 días de vacaciones pagadas" in rag_policy["results"][0]["retrieved_text"]

    # 5. Caso 2.3: Agrupamiento semántico KMeans (k=3) sobre feedback de clientes
    reviews_a = [
        "La aplicación se cierra sola cuando intento pagar.",
        "El envío tardó una semana más de lo prometido, pésimo servicio.",
        "Excelente atención del personal en la tienda física.",
    ]
    reviews_b = [
        "Cada vez que abro el módulo de pagos la app colapsa.",
        "El paquete llegó roto y la caja aplastada.",
        "Los empleados de soporte fueron muy amables y rápidos.",
    ]
    km_res = perform_semantic_kmeans_clustering(reviews_a, reviews_b, n_clusters=3)
    assert km_res["num_clusters"] == 3
    assert km_res["total_sentences_clustered"] == 6
    assert len(km_res["clusters"]) == 3

    # 6. Sección 1.3: Polisemia en 'Fui al banco a pedir un préstamo frente al banco del parque'
    poly_sentence = "Fui al banco a pedir un préstamo frente al banco del parque."
    poly_res = analyze_polysemy_and_self_attention([poly_sentence], [])
    banco_items = [t for t in poly_res["polysemous_terms"] if t["word"] == "banco"]
    assert len(banco_items) == 1
    assert banco_items[0]["polysemy_detected"] is True
    assert banco_items[0]["contextual_divergence"] > 0.25
    assert banco_items[0]["instance_a"]["inferred_sense"] != banco_items[0]["instance_b"]["inferred_sense"]

    # 7. Integración completa en pipeline y contrato JSON para frontend
    full = generate_statistical_analysis(
        poly_sentence + " " + " ".join(reviews_a),
        "\n".join(faq_database + reviews_b),
        remove_stopwords=True,
        profile="enhanced",
    )
    emb = full["embeddings"]
    assert "vector_space_metrics" in emb
    assert emb["vector_space_metrics"]["dot_equals_cosine_verified"] is True
    assert "representation_evolution" in emb
    assert "polysemy_analysis" in emb
    assert "rag_knn_retrieval" in emb
    assert "kmeans_clustering" in emb
    assert emb["production_guidelines"]["chunking_overlap_percentage"] == 15
    assert "vector_space_metrics" in full["deep_scan"]
    assert "euclidean_similarity" in full["unified_metrics"]


def test_g07_panitas_plantilla_a_vs_plantilla_b_matches_reference_report():
    """
    Valida que al separar y comparar Plantilla A (Pasos 1 a 5, Págs. 1-2) vs. Plantilla B
    (Paso 6, Pág. 3) de `date/G07_Equipo-02_Panitas.pdf` —desde las secciones separadas,
    desde `/api/samples` o subiendo el PDF completo— todas las métricas de la matriz PLN
    y el desglose de palabras clave se calculen de forma consistente y en rango válido [0, 1].
    """
    from pathlib import Path

    pdf_path = Path(__file__).resolve().parent.parent / "date" / "G07_Equipo-02_Panitas.pdf"
    if not pdf_path.exists():
        pytest.skip("Archivo G07_Equipo-02_Panitas.pdf no presente en date/")

    extracted = extract_text_from_file(pdf_path.read_bytes(), pdf_path.name)
    split = extracted.get("split_sections")
    assert split is not None
    assert split["has_internal_comparison_sections"] is True
    assert "Plantilla A" in split["text_a"]
    assert "Plantilla B" in split["text_b"]

    res = generate_statistical_analysis(
        split["text_a"], split["text_b"], remove_stopwords=True, profile="enhanced"
    )
    assert res["is_g07_benchmark"] is True

    jac = res["jaccard"]
    assert jac["raw_alpha_jaccard_score"] == pytest.approx(0.2707, abs=1e-4)
    assert jac["raw_alpha_intersection_size"] == 95
    assert jac["raw_alpha_union_size"] == 351
    assert 0.15 <= jac["filtered_surface_jaccard_score"] <= 0.30
    assert 0.15 <= jac["similarity_score"] <= 0.35

    cos = res["cosine"]
    assert 0.35 <= cos["bow_count_cosine_score"] <= 0.65
    assert 0.25 <= cos["tfidf_unigram_cosine_score"] <= 0.50
    assert 0.15 <= cos["tfidf_bigram_cosine_score"] <= 0.40

    emb = res["embeddings"]
    assert 0.65 <= emb["similarity_score"] <= 0.90

    matrix = res["nlp_techniques_matrix"]
    assert len(matrix) == 6
    by_id = {row["id"]: row for row in matrix}
    assert by_id["jaccard_bruto"]["score"] == pytest.approx(jac["raw_alpha_jaccard_score"], abs=1e-4)
    assert by_id["coseno_embeddings"]["score"] >= by_id["jaccard_filtrado"]["score"]

    cats = res["categorized_shared_keywords"]
    assert cats["total_shared_keywords"] > 0
    assert len(cats["categories"]) == 3

    # Al enviar el documento completo en ambos campos debe auto-separar Plantilla A vs Plantilla B
    res_auto = generate_statistical_analysis(
        extracted["text"], extracted["text"], remove_stopwords=True, profile="enhanced"
    )
    assert res_auto["auto_split_sections"] is True
    assert res_auto["jaccard"]["similarity_score"] == pytest.approx(jac["similarity_score"], abs=1e-4)
    assert res_auto["cosine"]["similarity_score"] == pytest.approx(cos["similarity_score"], abs=1e-4)

    # Verificar endpoint /api/samples
    client = app.test_client()
    resp_samples = client.get("/api/samples")
    assert resp_samples.status_code == 200
    assert "g07_panitas_plantilla_a_vs_b" in resp_samples.get_json()["samples"]
