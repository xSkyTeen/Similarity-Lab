"""
Almacén de parámetros entrenables para Similarity Lab (`core/trained_params.json`).

Permite que `core/jaccard.py`, `core/cosine.py` y `core/embeddings.py` utilicen
parámetros ajustados mediante optimización iterativa (`tools/train_params.py`)
sin memorizar ni fijar resultados por documento.
"""

import json
from pathlib import Path
from typing import Any, Dict

TRAINED_PARAMS_PATH = Path(__file__).resolve().parent / "trained_params.json"

DEFAULT_TRAINED_PARAMS: Dict[str, Any] = {
    "meta": {
        "best_loss": None,
        "iterations": 0,
        "description": "Parámetros por defecto antes del entrenamiento iterativo.",
    },
    "lexical": {
        "min_token_len": 3,
        "extra_stopwords": [],
        "bow_sublinear_tf": False,
        "tfidf_unigram_sublinear_tf": False,
        "tfidf_unigram_smooth_idf": True,
        "tfidf_bigram_sublinear_tf": False,
        "tfidf_bigram_smooth_idf": True,
    },
    "embeddings": {
        "it_cluster_boost_weight": 0.48,
        "it_min_cluster_hits": 5,
        "calibration_knee": 0.16,
        "calibration_lower_slope": 0.56,
        "calibration_power": 0.72,
    },
}


def load_trained_params() -> Dict[str, Any]:
    """Carga los parámetros desde `core/trained_params.json` mezclados con los valores por defecto."""
    params = json.loads(json.dumps(DEFAULT_TRAINED_PARAMS))
    if TRAINED_PARAMS_PATH.exists():
        try:
            raw = json.loads(TRAINED_PARAMS_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for section in ("meta", "lexical", "embeddings"):
                    if isinstance(raw.get(section), dict):
                        params[section].update(raw[section])
        except Exception:
            pass
    return params


def save_trained_params(params: Dict[str, Any]) -> Path:
    """Guarda los parámetros entrenados en `core/trained_params.json`."""
    TRAINED_PARAMS_PATH.write_text(
        json.dumps(params, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return TRAINED_PARAMS_PATH
