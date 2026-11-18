"""
Similarity Lab — Aplicación Web para Análisis Estadístico de Similitud Textual mediante NLP.

Punto de entrada del servidor Flask.
Incluye:
- Motor de Embeddings Híbrido 100% Local (Scikit-Learn TruncatedSVD LSA + Ventanas de Co-ocurrencia
  Contextual ±3 + Subpalabras FastText + BERTScore) con 0 consumo de tokens de APIs externas.
- Escaneo profundo de palabras repetidas, familias morfológicas, proximidad, alineamiento
  contextual oración por oración, concordancias KWIC y dominios temáticos.
- Validación estricta de entradas y límites de longitud con códigos HTTP apropiados.
"""

import logging
import time
from typing import Any, Dict, Tuple
from flask import Flask, jsonify, render_template, request, Response
from werkzeug.exceptions import RequestEntityTooLarge

from core import (
    COMPARISON_PROFILES,
    SPANISH_STOPWORDS,
    extract_text_from_file,
    generate_statistical_analysis,
)
from core.embeddings import get_sentence_transformer_model
from core.file_reader import MAX_TEXT_CHARS

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
)
logger = logging.getLogger("similarity_lab")

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

# Precargar modelo local opcional una sola vez al arrancar el proceso
get_sentence_transformer_model()


ACADEMIC_SAMPLES = {
    "g07_panitas_plantilla_a_vs_b": {
        "title": "Informe Técnico G07 Panitas: Plantilla A (Patrones Estadísticos) vs. Plantilla B (Decisiones de Gestión)",
        "description": "Comparativo oficial de las 6 técnicas de PLN (Jaccard Bruto 0.2707, Jaccard Filtrado 0.2007, Count 0.4988, TF-IDF Unigramas 0.3577, TF-IDF Bigramas 0.2425 y Embeddings 0.78–0.85).",
        "text_a": (
            "Plantillas de trabajo · G-07 · Análisis de Datos (SIS323) · 2026-II UNA-Puno \n"
            "Página 1 de 3 \n"
            "Plantilla A · Pasos 1 a 5: registro de patrones \n"
            "Integrantes del equipo: Cahuana Alfaro Edson Paolo – Rosales Arocutipa Carlos Javier – Vera Velasquez Alvaro Guillermo    Fecha: 03/11/2026 \n"
            "Qué analizamos Técnica Resultado (con cifra) ¿Es un patrón? ¿Cuál? Interpretación \n"
            "Tiempo y satisfacción Spearman Spearman: r = -0.44 \n"
            "(Pearson: r = -0.62) \n"
            "Sí. Existe una relación negativa \n"
            "moderada entre ambas \n"
            "variables. \n"
            "A medida que aumenta el tiempo de atención, la \n"
            "satisfacción del usuario disminuye. Corresponde \n"
            "usar Spearman porque satisfacción es ordinal y el \n"
            "tiempo presenta valores atípicos. \n"
            "Prioridad y tiempo Spearman Spearman: r = -0.27 Sí. Se observa una relación \n"
            "negativa débil a moderada. \n"
            "A mayor nivel de prioridad codificada (Alta = 3, \n"
            "Media = 2, Baja = 1), menor es el tiempo de atención \n"
            "empleado, lo cual indica que los casos urgentes \n"
            "tienden a atenderse antes. \n"
            "Satisfacción por categoría groupby Hardware: mediana 4.0, media \n"
            "3.86 (n=125). \n"
            "Red: mediana 4.0, media 4.29 \n"
            "(n=118). \n"
            "Software: mediana 5.0, media \n"
            "4.54 (n=155). \n"
            "Impresión: mediana 5.0, media \n"
            "4.70 (n=126). \n"
            "Cuentas: mediana 5.0, media 4.77 \n"
            "(n=83). \n"
            "Sí. Las categorías con mayores \n"
            "tiempos de atención registran \n"
            "menor satisfacción. \n"
            "Al comparar con los tiempos de la G-06, Hardware \n"
            "(mediana 17.55 h) y Red (11.00 h) son las categorías \n"
            "que más demoran y también las que tienen las \n"
            "calificaciones más bajas. \n"
            "Tickets de red por oficina (número y \n"
            "proporción) \n"
            "pivot_table y crosstab En número: Rentas (30 de 115), \n"
            "Gerencia Municipal (16 de 68), \n"
            "Desarrollo Económico (15 de 68). \n"
            "En proporción: Rentas (26.1%), \n"
            "Gerencia Municipal (23.5%), \n"
            "Desarrollo Económico (22.1%), \n"
            "RR.HH. (20.8%). \n"
            "Sí. Hay concentración de fallas \n"
            "de red en oficinas específicas, \n"
            "tanto en cantidad como en \n"
            "porcentaje. \n"
            "Rentas no solo tiene más casos de red por su \n"
            "tamaño total (115 tickets), sino que \n"
            "proporcionalmente también lidera las incidencias de \n"
            "red (26.1%), seguida de cerca por Gerencia \n"
            "Municipal (23.5%). \n"
            "Tiempo por categoría y prioridad pivot_table Mediana de tiempo (Alta / Media \n"
            "/ Baja): \n"
            "Cuentas: 1.0 / 1.5 / 3.0 h \n"
            "Impresión: 2.5 / 3.9 / 4.8 h \n"
            "Software: 4.0 / 5.7 / 8.5 h \n"
            "Red: 6.5 / 11.0 / 14.1 h \n"
            "Hardware: 11.2 / 18.5 / 26.5 h \n"
            "Sí. El orden de atención según \n"
            "la prioridad se cumple de \n"
            "forma consistente en todas las \n"
            "categorías. \n"
            "En los cinco tipos de problema, los tickets de \n"
            "prioridad Alta se resuelven en menos tiempo \n"
            "mediano que los de prioridad Media, y estos antes \n"
            "que los de prioridad Baja.\n\n"
            "Plantillas de trabajo · G-07 · Análisis de Datos (SIS323) · 2026-II UNA-Puno \n"
            "Página 2 de 3 \n"
            "Qué analizamos Técnica Resultado (con cifra) ¿Es un patrón? ¿Cuál? Interpretación \n"
            "Tickets por mes resample Promedio mensual: 51.7 tickets. \n"
            "Rango general: 36 (ago.) a 59 \n"
            "(may.). \n"
            "Pico aislado en octubre: 71 \n"
            "tickets. \n"
            "No hay tendencia sostenida; se \n"
            "presenta un pico estacional o \n"
            "aislado en octubre. \n"
            "Los conteos mensuales oscilan alrededor del \n"
            "promedio (51.7) sin un crecimiento o caída continua. \n"
            "El alza de octubre (71) puede ser una variación \n"
            "puntual; se requerirían datos de varios años o \n"
            "registro de eventos para confirmarlo. \n"
            "Tickets por día de la semana value_counts Lunes: 157 tickets. \n"
            "Viernes: 126, Martes: 124, \n"
            "Miércoles: 111, Jueves: 102. \n"
            "Sí. Existe una concentración \n"
            "marcada de carga de trabajo al \n"
            "inicio de la semana. \n"
            "Los lunes acumulan la mayor cantidad de reportes \n"
            "(157 frente a 102-126 el resto de la semana), \n"
            "posiblemente por incidencias ocurridas o retenidas \n"
            "desde el cierre del viernes y fin de semana. \n"
            "Tiempo y satisfacción dentro de cada \n"
            "categoría \n"
            "Spearman por grupo Spearman por grupo: \n"
            "Hardware: r = -0.57 \n"
            "Red: r = -0.34 \n"
            "Software: r = -0.30 \n"
            "Impresión: r = -0.02 \n"
            "Cuentas de usuario: r = 0.16 \n"
            "Sí. La relación entre espera y \n"
            "satisfacción varía según la \n"
            "categoría del problema. \n"
            "En Hardware, Red y Software esperar más reduce la \n"
            "satisfacción, mientras que en Impresión y Cuentas \n"
            "casi no influye porque se resuelven rápido. La \n"
            "gravedad del fallo o la interrupción del trabajo \n"
            "podría afectar ambas variables a la vez."
        ),
        "text_b": (
            "Plantillas de trabajo · G-07 · Análisis de Datos (SIS323) · 2026-II UNA-Puno \n"
            "Página 3 de 3 \n"
            "Plantilla B · Paso 6: tres hallazgos para la toma de decisiones \n"
            "Integrantes del equipo: Cahuana Alfaro Edson Paolo – Rosales Arocutipa Carlos Javier – Vera Velasquez Alvaro Guillermo    Fecha: 03/11/2026 \n"
            "N° Hallazgo Evidencia (cifra y gráfico) ¿Prueba una causa? ¿Por qué? Decisión que podría apoyar Limitación \n"
            "1 Los lunes concentran la mayor cantidad \n"
            "de tickets de la semana. \n"
            "157 tickets registrados los lunes, frente \n"
            "a un rango de 102 a 126 en los demás \n"
            "días (conteo por día y serie temporal \n"
            "del Paso 4). \n"
            "No. Es habitual que las fallas \n"
            "del fin de semana o del cierre \n"
            "del viernes se reporten el lunes \n"
            "a primera hora, pero el análisis \n"
            "exploratorio solo plantea esta \n"
            "hipótesis. \n"
            "Reforzar el turno de soporte técnico \n"
            "los lunes por la mañana o redistribuir \n"
            "las tareas administrativas de los \n"
            "técnicos hacia los miércoles y jueves. \n"
            "Datos correspondientes a un solo \n"
            "año (2025) y de carácter \n"
            "simulado; no se cuenta con la \n"
            "fecha exacta en que inició la falla \n"
            "real. \n"
            "2 Las categorías de Hardware y Red \n"
            "presentan los mayores tiempos de \n"
            "resolución y la menor satisfacción, \n"
            "especialmente cuando el tiempo se \n"
            "extiende. \n"
            "Hardware tiene mediana de 17.55 h \n"
            "(media de satisfacción 3.86 y r = -0.57) y \n"
            "Red mediana de 11.00 h (media 4.29 y r \n"
            "= -0.34), visibles en el mapa de calor y \n"
            "tablas de los Pasos 1, 2 y 5. \n"
            "No. Además del tiempo de \n"
            "espera, la complejidad técnica \n"
            "o la falta de repuestos puede \n"
            "demorar la solución y frustrar \n"
            "al usuario simultáneamente. \n"
            "Mantener un stock básico de \n"
            "repuestos de hardware críticos y \n"
            "establecer acuerdos de escalamiento \n"
            "rápido para incidencias complejas de \n"
            "hardware y red. \n"
            "Existen 13 registros con \n"
            "satisfacción faltante no \n"
            "imputados (607 válidos de 620) y \n"
            "no se dispone de datos sobre \n"
            "disponibilidad de repuestos. \n"
            "3 Rentas y Gerencia Municipal concentran \n"
            "la mayor proporción de problemas de red \n"
            "dentro de la municipalidad. \n"
            "Rentas registra 30 tickets de red (26.1% \n"
            "de sus 115 casos) y Gerencia Municipal \n"
            "16 tickets (23.5% de 68), según la tabla \n"
            "dinámica y tabla cruzada del Paso 3. \n"
            "No. La concentración puede \n"
            "deberse al estado del \n"
            "cableado, saturación de puntos \n"
            "de red o al uso intensivo de \n"
            "sistemas en línea en esas \n"
            "oficinas, lo cual requiere \n"
            "verificación en sitio. \n"
            "Priorizar una revisión preventiva de \n"
            "los equipos de conectividad, puntos \n"
            "de red y switches en las oficinas de \n"
            "Rentas y Gerencia Municipal. \n"
            "Los datos provienen de un único \n"
            "año de operación y no incluyen el \n"
            "inventario de equipos ni la \n"
            "cantidad de usuarios conectados \n"
            "por oficina."
        ),
    },
    "escaneo_repeticion_contexto": {
        "title": "Ejemplo Escaneo Profundo: Palabras Repetidas y Contexto Oración a Oración",
        "description": "Caso multi-oración con sobre-repetición léxica/morfológica, paráfrasis contextual y conectores discursivos.",
        "text_a": (
            "El análisis de datos permite analizar grandes volúmenes de datos en el sistema informático. "
            "Este análisis de datos es fundamental porque el sistema procesa datos complejos rápidamente. "
            "Sin embargo, cuando un estudiante universitario redacta un informe extenso en su computadora, "
            "suele repetir palabras clave dentro del mismo párrafo del informe."
        ),
        "text_b": (
            "La evaluación de información facilita estudiar amplias cantidades de registros mediante la plataforma digital. "
            "Dicho estudio resulta esencial ya que el programa examina información compleja con gran velocidad. "
            "No obstante, cuando un alumno académico elabora un documento amplio en su ordenador, "
            "frecuentemente duplica términos importantes en la redacción del escrito."
        ),
    },
    "sinonimia_embeddings": {
        "title": "Ejemplo IA Local: Sinonimia Pura (Embeddings LSA/Contexto vs. TF-IDF / Jaccard)",
        "description": "Oraciones con significado equivalente pero vocabulario literal distinto (demuestra el motor híbrido local sin gastar tokens).",
        "text_a": (
            "El estudiante universitario analiza un documento extenso en su computadora "
            "mediante un algoritmo rápido de inteligencia artificial."
        ),
        "text_b": (
            "Un alumno académico evalúa un escrito amplio en su ordenador "
            "utilizando un método veloz de aprendizaje automático."
        ),
    },
    "ejemplo_teorico": {
        "title": "Ejemplo Teórico (Jaccard vs. Coseno TF-IDF)",
        "description": "Oraciones clásicas para contrastar superposición de conjuntos frente a vectores ponderados.",
        "text_a": "El procesamiento del lenguaje natural es fascinante",
        "text_b": "El procesamiento de lenguaje y texto es genial",
    },
    "paraphrase_plagio": {
        "title": "Caso de Estudio: Paráfrasis Académica (Est. A vs. Est. B)",
        "description": "Respuestas de examen sobre tokenización con alta similitud léxica y estructural.",
        "text_a": (
            "La tokenización es el proceso de dividir un texto en unidades menores "
            "llamadas tokens como palabras o subpalabras."
        ),
        "text_b": (
            "El proceso de tokenización consiste en segmentar un texto en unidades más "
            "pequeñas denominadas tokens, ya sean palabras o subpalabras."
        ),
    },
    "tematica_comun": {
        "title": "Caso de Estudio: Terminología Compartida (Est. A vs. Est. C)",
        "description": "Textos sobre el mismo dominio técnico redactados con enfoques complementarios.",
        "text_a": (
            "La tokenización es el proceso de dividir un texto en unidades menores "
            "llamadas tokens como palabras o subpalabras."
        ),
        "text_b": (
            "La tokenización divide el texto en tokens. Es fundamental para preparar "
            "los datos antes de ingresarlos a un modelo de NLP."
        ),
    },
    "temas_disjuntos": {
        "title": "Caso de Estudio: Documentos Disjuntos (Est. A vs. Est. D)",
        "description": "Comparación entre respuestas de temáticas distintas (Tokenización vs. Visión por Computadora).",
        "text_a": (
            "La tokenización es el proceso de dividir un texto en unidades menores "
            "llamadas tokens como palabras o subpalabras."
        ),
        "text_b": (
            "El aprendizaje profundo requiere redes neuronales convolucionales y "
            "recurrentes para la visión por computadora."
        ),
    },
    "unap_soporte_rag": {
        "title": "Guía NLP (Casos 2.1 y 2.2): Búsqueda Semántica en Soporte Técnico y RAG k-NN",
        "description": "Consultas coloquiales de usuario y preguntas de políticas frente a la base de conocimiento empresarial (recuperación k-NN y distancia vectorial).",
        "text_a": (
            "No puedo entrar a mi cuenta y me sale una pantalla roja.\n"
            "¿Cuántos días de descanso me corresponden al cumplir un año?"
        ),
        "text_b": (
            "Instrucciones para restablecer la contraseña en caso de bloqueo de cuenta.\n"
            "Cómo solucionar el error de autenticación fallida y pantalla de fallo en el login.\n"
            "Pasos para configurar la facturación automática y cambiar tarjeta de crédito.\n"
            "Guía para solicitar la baja del servicio y cancelación de suscripción.\n"
            "Los empleados tienen derecho a 15 días de vacaciones pagadas tras un año de servicio.\n"
            "El reembolso de gastos de viaje requiere la presentación de facturas dentro de los 5 días hábiles.\n"
            "El trabajo remoto está permitido los días viernes previa aprobación del líder de equipo."
        ),
    },
    "unap_polisemia_clustering": {
        "title": "Guía NLP (Sección 1.3 y Caso 2.3): Polisemia ('banco') y Clustering k-Means",
        "description": "Desambiguación contextual mediante Auto-Atención y agrupamiento semántico no supervisado KMeans (k=3) sobre feedback de clientes.",
        "text_a": (
            "Fui al banco a pedir un préstamo frente al banco del parque.\n"
            "La aplicación se cierra sola cuando intento pagar.\n"
            "El envío tardó una semana más de lo prometido, pésimo servicio.\n"
            "Excelente atención del personal en la tienda física."
        ),
        "text_b": (
            "Cada vez que abro el módulo de pagos la app colapsa.\n"
            "El paquete llegó roto y la caja aplastada.\n"
            "Los empleados de soporte fueron muy amables y rápidos."
        ),
    },
}


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(_err: RequestEntityTooLarge) -> Tuple[Response, int]:
    """Maneja archivos que exceden el límite de 16 MB con código HTTP 413."""
    return (
        jsonify({
            "status": "error",
            "message": "El archivo supera el tamaño máximo permitido de 16 MB.",
        }),
        413,
    )


@app.route("/", methods=["GET"])
def index() -> str:
    """Renderiza la vista principal de Similarity Lab."""
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def system_status() -> Response:
    """Devuelve el estado operativo del motor NLP local (0 consumo de tokens)."""
    has_minilm = get_sentence_transformer_model() is not None
    return jsonify({
        "status": "online",
        "application": "Similarity Lab",
        "subtitle": "Text Similarity & Deep Context NLP Analysis",
        "engine": "Motor Híbrido Local (LSA TruncatedSVD + Contexto ±3 + FastText + BERTScore) · 0 Tokens API",
        "stopwords_catalog_size": len(SPANISH_STOPWORDS),
        "token_cost": 0,
        "local_neural_model_loaded": has_minilm,
    })


@app.route("/api/samples", methods=["GET"])
def get_samples() -> Response:
    """Devuelve el catálogo de documentos de prueba de referencia académica."""
    return jsonify({
        "status": "success",
        "samples": ACADEMIC_SAMPLES,
    })


@app.route("/api/profiles", methods=["GET"])
def get_profiles() -> Response:
    """Devuelve la configuración de los perfiles de comparación ('reference' y 'enhanced')."""
    return jsonify({
        "status": "success",
        "profiles": COMPARISON_PROFILES,
    })


@app.route("/api/upload", methods=["POST"])
def upload_document() -> Tuple[Response, int]:
    """
    Extrae el texto de un archivo subido (PDF, DOCX, TXT, MD, CSV) en memoria.
    """
    t0 = time.perf_counter()
    if "file" not in request.files:
        return (
            jsonify({
                "status": "error",
                "message": "No se recibió ningún archivo en la petición.",
            }),
            400,
        )

    uploaded_file = request.files["file"]
    filename = uploaded_file.filename or ""
    if not filename.strip():
        return (
            jsonify({
                "status": "error",
                "message": "El archivo seleccionado no tiene un nombre válido.",
            }),
            400,
        )

    try:
        file_bytes = uploaded_file.read()
        extracted = extract_text_from_file(file_bytes=file_bytes, filename=filename)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        logger.info(
            "Archivo procesado: '%s' (%s, %d págs/bloques, %d chars) en %.2f ms",
            extracted["filename"],
            extracted["file_type"],
            extracted["pages_count"],
            extracted["char_count"],
            elapsed_ms,
        )
        return (
            jsonify({
                "status": "success",
                "execution_time_ms": elapsed_ms,
                "data": extracted,
            }),
            200,
        )
    except ValueError as err:
        return (
            jsonify({
                "status": "error",
                "message": str(err),
            }),
            400,
        )
    except Exception as exc:
        logger.error("Error inesperado al leer archivo '%s': %s", filename, exc)
        return (
            jsonify({
                "status": "error",
                "message": f"Error inesperado al procesar el archivo: {exc}",
            }),
            500,
        )


@app.route("/api/compare", methods=["POST"])
@app.route("/compare", methods=["POST"])
def compare_texts() -> Tuple[Response, int]:
    """
    Endpoint principal de comparación NLP y Escaneo Profundo.
    100% local en memoria sin llamadas a APIs externas ni gasto de tokens.
    """
    start_time = time.perf_counter()

    if not request.is_json:
        return (
            jsonify({
                "status": "error",
                "message": "La petición debe tener formato JSON ('Content-Type: application/json').",
            }),
            400,
        )

    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    text_a = payload.get("text_a", "")
    text_b = payload.get("text_b", "")
    remove_stopwords = payload.get("remove_stopwords", True)
    unify_accents = payload.get("unify_accents", False)
    profile = payload.get("profile", "enhanced")

    if not isinstance(text_a, str) or not isinstance(text_b, str):
        return (
            jsonify({
                "status": "error",
                "message": "Los documentos de entrada deben ser cadenas de texto válidas.",
            }),
            400,
        )

    if not text_a.strip() or not text_b.strip():
        missing = []
        if not text_a.strip():
            missing.append("Texto A")
        if not text_b.strip():
            missing.append("Texto B")
        missing_str = " y ".join(missing)
        return (
            jsonify({
                "status": "error",
                "message": f"Por favor, ingresa contenido válido en {missing_str} antes de iniciar el análisis.",
            }),
            400,
        )

    if len(text_a) > MAX_TEXT_CHARS or len(text_b) > MAX_TEXT_CHARS:
        return (
            jsonify({
                "status": "error",
                "message": f"Cada documento puede contener como máximo {MAX_TEXT_CHARS:,} caracteres.",
            }),
            400,
        )

    try:
        analysis_result = generate_statistical_analysis(
            text_a=text_a,
            text_b=text_b,
            remove_stopwords=bool(remove_stopwords),
            unify_accents=bool(unify_accents),
            gemini_api_key=None,  # 0 consumo de tokens externos
            profile=str(profile),
        )
    except Exception as exc:
        logger.error("Error durante el análisis NLP: %s", exc)
        return (
            jsonify({
                "status": "error",
                "message": "Se produjo un error interno al ejecutar el motor de análisis NLP.",
            }),
            500,
        )

    elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
    logger.info(
        "Comparación + Escaneo Profundo en %.2f ms | len_A=%d | len_B=%d | mode=%s | 0_tokens | Jaccard=%.2f%% | Cosine=%.2f%% | EmbCal=%.2f%%",
        elapsed_ms,
        len(text_a),
        len(text_b),
        analysis_result["embeddings"].get("mode", "local_hybrid_lsa"),
        analysis_result["jaccard"]["similarity_percentage"],
        analysis_result["cosine"]["similarity_percentage"],
        analysis_result["embeddings"]["similarity_percentage"],
    )

    return (
        jsonify({
            "status": "success",
            "execution_time_ms": elapsed_ms,
            "data": analysis_result,
        }),
        200,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
