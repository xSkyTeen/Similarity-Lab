#!/usr/bin/env python3
"""
Arnés de Calibración de Similarity Lab (`tools/calibrate.py`).

Compara los resultados de Similarity Lab contra los valores esperados entregados
por una herramienta externa de validación (sin asumir cuál herramienta usa el
ingeniero; soporta los campos opcionales `tool_name` y `tool_params` en CSV/JSON).

Realiza una búsqueda exhaustiva en rejilla (Grid-Search) sobre todas las opciones
algorítmicas para identificar qué combinación minimiza el Error Absoluto Medio (MAE)
por cada métrica:
  - `stemming`: on / off
  - `stopwords`: on / off
  - `tokenizer`: regex por defecto de sklearn (`(?u)\\b\\w\\w+\\b`) vs split por espacios (`whitespace`) vs `standard`
  - `jaccard_mode`: sobre `set` vs `multiset` (Jaccard ponderado por frecuencias)
  - `TF-IDF`: `smooth_idf` on/off, `sublinear_tf` on/off, `idf_mode` (`"two_docs"` vs `"segments"` vs `"fit_a_transform_b"`)
  - `shingles`: k = 2, 3, 5
  - `embeddings`: `raw` (crudo) vs `calibrated` (calibrado); `full_document` vs `chunking`

Uso por línea de comandos:
    python tools/calibrate.py
    python tools/calibrate.py tools/sample_benchmark.csv
    python tools/calibrate.py tools/sample_benchmark.json
"""

import argparse
import csv
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Asegurar que el directorio raíz del proyecto esté en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.cosine import calculate_cosine_similarity
from core.embeddings import calculate_embedding_similarity
from core.jaccard import calculate_jaccard_similarity


# Métricas soportadas en archivos CSV / JSON de calibración
SUPPORTED_METRICS = (
    "jaccard",
    "dice",
    "overlap",
    "shingles",
    "shingles_k2",
    "shingles_k3",
    "shingles_k5",
    "cosine_tfidf",
    "embeddings",
)

# Alias de columnas para facilitar el mapeo desde distintos formatos de CSV/JSON
METRIC_COLUMN_ALIASES: Dict[str, str] = {
    "jaccard": "jaccard",
    "expected_jaccard": "jaccard",
    "jaccard_expected": "jaccard",
    "dice": "dice",
    "expected_dice": "dice",
    "sorensen_dice": "dice",
    "overlap": "overlap",
    "containment": "overlap",
    "expected_overlap": "overlap",
    "expected_containment": "overlap",
    "shingles": "shingles",
    "expected_shingles": "shingles",
    "shingles_k2": "shingles_k2",
    "bigram_jaccard": "shingles_k2",
    "shingles_k3": "shingles_k3",
    "trigram_jaccard": "shingles_k3",
    "shingles_k5": "shingles_k5",
    "pentagram_jaccard": "shingles_k5",
    "cosine": "cosine_tfidf",
    "cosine_tfidf": "cosine_tfidf",
    "tfidf_cosine": "cosine_tfidf",
    "expected_cosine": "cosine_tfidf",
    "embeddings": "embeddings",
    "embedding": "embeddings",
    "expected_embeddings": "embeddings",
    "semantic": "embeddings",
}


def _parse_optional_float(val: Any) -> Optional[float]:
    """Convierte un valor a float en escala [0.0, 1.0], o devuelve None si está vacío."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        fval = float(val)
    else:
        sval = str(val).strip().replace("%", "")
        if not sval or sval.lower() in ("none", "null", "nan", "n/a", "-"):
            return None
        try:
            fval = float(sval)
        except ValueError:
            return None

    # Si el valor viene expresado como porcentaje (p. ej. 50.0 en vez de 0.50), normalizar a [0, 1]
    if fval > 1.000001 and fval <= 100.0:
        fval = fval / 100.0
    return fval


def load_calibration_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    Carga un dataset de calibración desde un archivo CSV o JSON.
    Conserva los campos `tool_name` y `tool_params` si están presentes.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de calibración: {file_path}")

    raw_rows: List[Dict[str, Any]] = []
    suffix = path.suffix.lower()

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "pairs" in data:
                default_tool_name = data.get("tool_name", "")
                default_tool_params = data.get("tool_params", "")
                for item in data["pairs"]:
                    row = dict(item)
                    row.setdefault("tool_name", default_tool_name)
                    row.setdefault("tool_params", default_tool_params)
                    raw_rows.append(row)
            elif isinstance(data, list):
                raw_rows = [dict(x) for x in data]
            else:
                raise ValueError("El archivo JSON debe contener una lista de pares o un objeto con la clave 'pairs'.")
    else:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_rows.append(dict(row))

    dataset: List[Dict[str, Any]] = []
    for idx, row in enumerate(raw_rows):
        text_a = str(row.get("text_a") or row.get("texto_a") or row.get("doc_a") or "")
        text_b = str(row.get("text_b") or row.get("texto_b") or row.get("doc_b") or "")
        pair_id = str(row.get("pair_id") or row.get("id") or f"par_{idx + 1}")
        tool_name = str(row.get("tool_name") or "").strip()
        tool_params = row.get("tool_params")
        if isinstance(tool_params, dict):
            tool_params_str = json.dumps(tool_params, ensure_ascii=False)
        else:
            tool_params_str = str(tool_params or "").strip()

        expected_metrics: Dict[str, float] = {}
        # Si hay un subdiccionario "expected" en JSON
        if isinstance(row.get("expected"), dict):
            for k, v in row["expected"].items():
                canon_k = METRIC_COLUMN_ALIASES.get(k.strip().lower())
                if canon_k:
                    parsed = _parse_optional_float(v)
                    if parsed is not None:
                        expected_metrics[canon_k] = parsed

        # Revisar columnas de nivel superior
        for col_name, col_val in row.items():
            if col_name in ("text_a", "texto_a", "doc_a", "text_b", "texto_b", "doc_b", "pair_id", "id", "tool_name", "tool_params", "expected"):
                continue
            canon_k = METRIC_COLUMN_ALIASES.get(col_name.strip().lower())
            if canon_k:
                parsed = _parse_optional_float(col_val)
                if parsed is not None:
                    expected_metrics[canon_k] = parsed

        dataset.append({
            "pair_id": pair_id,
            "text_a": text_a,
            "text_b": text_b,
            "tool_name": tool_name,
            "tool_params": tool_params_str,
            "expected": expected_metrics,
        })

    return dataset


def _evaluate_jaccard_family_grid(
    dataset: List[Dict[str, Any]],
    metric_key: str,
) -> Dict[str, Any]:
    """
    Ejecuta grid-search para `jaccard`, `dice` u `overlap` sobre:
    - `stemming`: [False, True]
    - `stopwords`: [False, True]
    - `tokenizer`: ["sklearn_regex", "whitespace", "standard"]
    - `jaccard_mode`: ["set", "multiset"] (para `jaccard`; `dice` y `overlap` operan sobre `set`)
    """
    relevant_pairs = [p for p in dataset if metric_key in p["expected"]]
    if not relevant_pairs:
        return {"metric": metric_key, "evaluated_pairs": 0, "best_config": None, "mae": 0.0, "rows": []}

    stemming_opts = [False, True]
    stopwords_opts = [False, True]
    tokenizer_opts = ["sklearn_regex", "whitespace", "standard"]
    mode_opts = ["set", "multiset"] if metric_key == "jaccard" else ["set"]

    best_mae = float("inf")
    best_config_dict: Dict[str, Any] = {}
    best_config_str = ""
    best_rows: List[Dict[str, Any]] = []

    for stem, sw, tok_mode, j_mode in itertools.product(stemming_opts, stopwords_opts, tokenizer_opts, mode_opts):
        errors: List[float] = []
        current_rows: List[Dict[str, Any]] = []
        cfg_str = (
            f"stem={ 'on' if stem else 'off' }, "
            f"stopwords={ 'on' if sw else 'off' }, "
            f"tok={tok_mode}, "
            f"mode={j_mode}"
        )

        for pair in relevant_pairs:
            res = calculate_jaccard_similarity(
                pair["text_a"],
                pair["text_b"],
                remove_stopwords=sw,
                use_stemming=stem,
                tokenizer_mode=tok_mode,
                compute_variants=False,
            )
            if metric_key == "jaccard":
                obtained = res["weighted_jaccard_score"] if j_mode == "multiset" else res["similarity_score"]
            elif metric_key == "dice":
                obtained = res["dice_score"]
            else:  # overlap / containment
                obtained = res["overlap_score"]

            expected = pair["expected"][metric_key]
            diff = obtained - expected
            abs_err = abs(diff)
            errors.append(abs_err)
            current_rows.append({
                "pair_id": pair["pair_id"],
                "metric": metric_key,
                "expected": expected,
                "obtained": obtained,
                "difference": diff,
                "abs_error": abs_err,
                "config": cfg_str,
                "tool_name": pair["tool_name"],
                "tool_params": pair["tool_params"],
            })

        mae = sum(errors) / len(errors)
        if mae < best_mae:
            best_mae = mae
            best_config_dict = {
                "stemming": stem,
                "stopwords": sw,
                "tokenizer": tok_mode,
                "jaccard_mode": j_mode,
            }
            best_config_str = cfg_str
            best_rows = current_rows

    return {
        "metric": metric_key,
        "evaluated_pairs": len(relevant_pairs),
        "best_config": best_config_dict,
        "best_config_str": best_config_str,
        "mae": round(best_mae, 6),
        "rows": best_rows,
    }


def _evaluate_shingles_grid(
    dataset: List[Dict[str, Any]],
    metric_key: str,
    fixed_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Ejecuta grid-search para `shingles` (probando k=2, 3, 5) o para un k específico (`shingles_k2`, `shingles_k3`, `shingles_k5`):
    - `k`: [2, 3, 5] (o `[fixed_k]`)
    - `stemming`: [False, True]
    - `stopwords`: [False, True]
    - `tokenizer`: ["sklearn_regex", "whitespace", "standard"]
    """
    relevant_pairs = [p for p in dataset if metric_key in p["expected"]]
    if not relevant_pairs:
        return {"metric": metric_key, "evaluated_pairs": 0, "best_config": None, "mae": 0.0, "rows": []}

    k_opts = [fixed_k] if fixed_k is not None else [2, 3, 5]
    stemming_opts = [False, True]
    stopwords_opts = [False, True]
    tokenizer_opts = ["sklearn_regex", "whitespace", "standard"]

    best_mae = float("inf")
    best_config_dict: Dict[str, Any] = {}
    best_config_str = ""
    best_rows: List[Dict[str, Any]] = []

    for k_val, stem, sw, tok_mode in itertools.product(k_opts, stemming_opts, stopwords_opts, tokenizer_opts):
        errors: List[float] = []
        current_rows: List[Dict[str, Any]] = []
        cfg_str = (
            f"k={k_val}, "
            f"stem={ 'on' if stem else 'off' }, "
            f"stopwords={ 'on' if sw else 'off' }, "
            f"tok={tok_mode}"
        )

        for pair in relevant_pairs:
            res = calculate_jaccard_similarity(
                pair["text_a"],
                pair["text_b"],
                remove_stopwords=sw,
                use_stemming=stem,
                tokenizer_mode=tok_mode,
                compute_variants=False,
            )
            if k_val == 2:
                obtained = res["bigram_jaccard_score"]
            elif k_val == 3:
                obtained = res["trigram_jaccard_score"]
            else:
                obtained = res["pentagram_jaccard_score"]

            expected = pair["expected"][metric_key]
            diff = obtained - expected
            abs_err = abs(diff)
            errors.append(abs_err)
            current_rows.append({
                "pair_id": pair["pair_id"],
                "metric": metric_key,
                "expected": expected,
                "obtained": obtained,
                "difference": diff,
                "abs_error": abs_err,
                "config": cfg_str,
                "tool_name": pair["tool_name"],
                "tool_params": pair["tool_params"],
            })

        mae = sum(errors) / len(errors)
        if mae < best_mae:
            best_mae = mae
            best_config_dict = {
                "k": k_val,
                "stemming": stem,
                "stopwords": sw,
                "tokenizer": tok_mode,
            }
            best_config_str = cfg_str
            best_rows = current_rows

    return {
        "metric": metric_key,
        "evaluated_pairs": len(relevant_pairs),
        "best_config": best_config_dict,
        "best_config_str": best_config_str,
        "mae": round(best_mae, 6),
        "rows": best_rows,
    }


def _evaluate_cosine_tfidf_grid(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Ejecuta grid-search para `cosine_tfidf` sobre:
    - `stemming`: [False, True]
    - `stopwords`: [False, True]
    - `tokenizer`: ["sklearn_default", "whitespace", "preprocessed"]
    - `smooth_idf`: [True, False]
    - `sublinear_tf`: [False, True]
    - `idf_mode`: ["two_docs", "segments", "fit_a_transform_b"]
    """
    metric_key = "cosine_tfidf"
    relevant_pairs = [p for p in dataset if metric_key in p["expected"]]
    if not relevant_pairs:
        return {"metric": metric_key, "evaluated_pairs": 0, "best_config": None, "mae": 0.0, "rows": []}

    stemming_opts = [False, True]
    stopwords_opts = [False, True]
    tokenizer_opts = ["sklearn_default", "whitespace", "preprocessed"]
    smooth_idf_opts = [True, False]
    sublinear_tf_opts = [False, True]
    idf_mode_opts = ["two_docs", "segments", "fit_a_transform_b"]

    best_mae = float("inf")
    best_config_dict: Dict[str, Any] = {}
    best_config_str = ""
    best_rows: List[Dict[str, Any]] = []

    for stem, sw, tok_mode, smooth, sublin, idf_m in itertools.product(
        stemming_opts, stopwords_opts, tokenizer_opts, smooth_idf_opts, sublinear_tf_opts, idf_mode_opts
    ):
        errors: List[float] = []
        current_rows: List[Dict[str, Any]] = []
        cfg_str = (
            f"idf={idf_m}, "
            f"smooth_idf={ 'on' if smooth else 'off' }, "
            f"sublinear_tf={ 'on' if sublin else 'off' }, "
            f"stem={ 'on' if stem else 'off' }, "
            f"stopwords={ 'on' if sw else 'off' }, "
            f"tok={tok_mode}"
        )

        for pair in relevant_pairs:
            res = calculate_cosine_similarity(
                pair["text_a"],
                pair["text_b"],
                remove_stopwords=sw,
                use_stemming=stem,
                sublinear_tf=sublin,
                smooth_idf=smooth,
                idf_mode=idf_m,
                tokenizer_mode=tok_mode,
            )
            obtained = res["raw_similarity_score"]
            expected = pair["expected"][metric_key]
            diff = obtained - expected
            abs_err = abs(diff)
            errors.append(abs_err)
            current_rows.append({
                "pair_id": pair["pair_id"],
                "metric": metric_key,
                "expected": expected,
                "obtained": obtained,
                "difference": diff,
                "abs_error": abs_err,
                "config": cfg_str,
                "tool_name": pair["tool_name"],
                "tool_params": pair["tool_params"],
            })

        mae = sum(errors) / len(errors)
        if mae < best_mae:
            best_mae = mae
            best_config_dict = {
                "idf_mode": idf_m,
                "smooth_idf": smooth,
                "sublinear_tf": sublin,
                "stemming": stem,
                "stopwords": sw,
                "tokenizer": tok_mode,
            }
            best_config_str = cfg_str
            best_rows = current_rows

    return {
        "metric": metric_key,
        "evaluated_pairs": len(relevant_pairs),
        "best_config": best_config_dict,
        "best_config_str": best_config_str,
        "mae": round(best_mae, 6),
        "rows": best_rows,
    }


def _evaluate_embeddings_grid(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Ejecuta grid-search para `embeddings` sobre:
    - `score_type`: ["raw", "calibrated"]
    - `aggregation`: ["full_document", "chunking"]
    - `stopwords`: [False, True]
    """
    metric_key = "embeddings"
    relevant_pairs = [p for p in dataset if metric_key in p["expected"]]
    if not relevant_pairs:
        return {"metric": metric_key, "evaluated_pairs": 0, "best_config": None, "mae": 0.0, "rows": []}

    score_type_opts = ["raw", "calibrated"]
    aggregation_opts = ["full_document", "chunking"]
    stopwords_opts = [False, True]

    best_mae = float("inf")
    best_config_dict: Dict[str, Any] = {}
    best_config_str = ""
    best_rows: List[Dict[str, Any]] = []

    for s_type, agg_mode, sw in itertools.product(score_type_opts, aggregation_opts, stopwords_opts):
        errors: List[float] = []
        current_rows: List[Dict[str, Any]] = []
        use_cal = s_type == "calibrated"
        full_doc = agg_mode == "full_document"
        cfg_str = (
            f"score={s_type}, "
            f"agg={agg_mode}, "
            f"stopwords={ 'on' if sw else 'off' }"
        )

        for pair in relevant_pairs:
            res = calculate_embedding_similarity(
                pair["text_a"],
                pair["text_b"],
                remove_stopwords=sw,
                use_calibrated_as_primary=use_cal,
                embed_full_document=full_doc,
                bertscore_use_idf=use_cal,
            )
            if full_doc and not use_cal:
                obtained = res["full_document_raw_cosine_score"]
            elif not full_doc and not use_cal:
                obtained = res["chunk_alignment_score"]
            else:
                obtained = res["similarity_score"]

            expected = pair["expected"][metric_key]
            diff = obtained - expected
            abs_err = abs(diff)
            errors.append(abs_err)
            current_rows.append({
                "pair_id": pair["pair_id"],
                "metric": metric_key,
                "expected": expected,
                "obtained": obtained,
                "difference": diff,
                "abs_error": abs_err,
                "config": cfg_str,
                "tool_name": pair["tool_name"],
                "tool_params": pair["tool_params"],
            })

        mae = sum(errors) / len(errors)
        if mae < best_mae:
            best_mae = mae
            best_config_dict = {
                "score_type": s_type,
                "aggregation": agg_mode,
                "stopwords": sw,
            }
            best_config_str = cfg_str
            best_rows = current_rows

    return {
        "metric": metric_key,
        "evaluated_pairs": len(relevant_pairs),
        "best_config": best_config_dict,
        "best_config_str": best_config_str,
        "mae": round(best_mae, 6),
        "rows": best_rows,
    }


def run_calibration_grid_search(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Ejecuta el grid-search completo sobre todas las métricas presentes en `dataset`
    y devuelve las mejores configuraciones por métrica junto con todas las filas de comparación.
    """
    results_by_metric: Dict[str, Dict[str, Any]] = {}
    all_table_rows: List[Dict[str, Any]] = []

    evaluators = [
        ("jaccard", lambda ds: _evaluate_jaccard_family_grid(ds, "jaccard")),
        ("dice", lambda ds: _evaluate_jaccard_family_grid(ds, "dice")),
        ("overlap", lambda ds: _evaluate_jaccard_family_grid(ds, "overlap")),
        ("shingles", lambda ds: _evaluate_shingles_grid(ds, "shingles", fixed_k=None)),
        ("shingles_k2", lambda ds: _evaluate_shingles_grid(ds, "shingles_k2", fixed_k=2)),
        ("shingles_k3", lambda ds: _evaluate_shingles_grid(ds, "shingles_k3", fixed_k=3)),
        ("shingles_k5", lambda ds: _evaluate_shingles_grid(ds, "shingles_k5", fixed_k=5)),
        ("cosine_tfidf", _evaluate_cosine_tfidf_grid),
        ("embeddings", _evaluate_embeddings_grid),
    ]

    for metric_name, eval_fn in evaluators:
        res = eval_fn(dataset)
        if res["evaluated_pairs"] > 0:
            results_by_metric[metric_name] = res
            all_table_rows.extend(res["rows"])

    return {
        "pairs_count": len(dataset),
        "metrics_calibrated": list(results_by_metric.keys()),
        "by_metric": results_by_metric,
        "table_rows": all_table_rows,
    }


def format_calibration_table(calibration_report: Dict[str, Any]) -> str:
    """
    Formatea una tabla legible para consola con las columnas requeridas:
    `métrica | valor esperado | valor obtenido | diferencia | configuración`
    (incluyendo además `pair_id` y `tool_name` / `tool_params` si están informados).
    """
    lines: List[str] = []
    lines.append("=" * 124)
    lines.append("ARNÉS DE CALIBRACIÓN DE SIMILARITY LAB — RESUMEN DE GRID-SEARCH POR MÉTRICA")
    lines.append("=" * 124)

    by_metric = calibration_report.get("by_metric", {})
    if not by_metric:
        lines.append("No se encontraron valores esperados de métricas en el archivo de entrada.")
        return "\n".join(lines)

    lines.append(
        f"{'MÉTRICA':<16} | {'PARES':<5} | {'MAE (ERROR MEDIO)':<18} | {'MEJOR CONFIGURACIÓN QUE MINIMIZA EL ERROR'}"
    )
    lines.append("-" * 124)
    for m_name, m_info in by_metric.items():
        lines.append(
            f"{m_name:<16} | {m_info['evaluated_pairs']:<5d} | {m_info['mae']:<18.6f} | {m_info['best_config_str']}"
        )

    lines.append("")
    lines.append("=" * 124)
    lines.append("TABLA DETALLADA: MÉTRICA | VALOR ESPERADO | VALOR OBTENIDO | DIFERENCIA | CONFIGURACIÓN")
    lines.append("=" * 124)
    header = (
        f"{'PAR':<10} | {'MÉTRICA':<14} | {'ESPERADO':>12} | {'OBTENIDO':>12} | {'DIFERENCIA':>12} | {'CONFIGURACIÓN (Y HERRAMIENTA EXTERNA)'}"
    )
    lines.append(header)
    lines.append("-" * 124)

    for row in calibration_report.get("table_rows", []):
        tool_suffix = ""
        if row.get("tool_name") or row.get("tool_params"):
            t_name = row.get("tool_name") or "externa"
            t_params = f" ({row['tool_params']})" if row.get("tool_params") else ""
            tool_suffix = f" [tool={t_name}{t_params}]"
        cfg_display = f"{row['config']}{tool_suffix}"
        lines.append(
            f"{row['pair_id']:<10} | "
            f"{row['metric']:<14} | "
            f"{row['expected']:>12.6f} | "
            f"{row['obtained']:>12.6f} | "
            f"{row['difference']:>+12.6f} | "
            f"{cfg_display}"
        )

    lines.append("=" * 124)
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Arnés de calibración (Grid-Search) de Similarity Lab contra herramienta externa."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=str(Path(__file__).resolve().parent / "sample_benchmark.csv"),
        help="Ruta al archivo CSV o JSON con pares de textos y valores esperados por métrica.",
    )
    parser.add_argument(
        "--json-out",
        dest="json_out",
        default=None,
        help="Ruta opcional para exportar el reporte completo en formato JSON.",
    )
    args = parser.parse_args(argv)

    dataset = load_calibration_dataset(args.input_file)
    report = run_calibration_grid_search(dataset)
    table_str = format_calibration_table(report)
    print(table_str)

    if args.json_out:
        out_path = Path(args.json_out)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nReporte JSON guardado en: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
