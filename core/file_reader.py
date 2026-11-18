"""
Módulo de extracción de texto desde archivos (PDF, DOCX, TXT, MD, CSV) en memoria.

Implementa:
- PDF: detección de PDFs escaneados sin texto seleccionable, conteo de páginas útiles,
  normalización Unicode NFC y unión de palabras cortadas por guion al final de línea.
- DOCX: extracción ordenada de encabezados, párrafos, tablas y pies de página.
- TXT/MD/CSV: detección automática de codificación con `charset_normalizer`
  (UTF-8, UTF-8-BOM, Latin-1, cp1252).
- Validación de límites de tamaño de archivo (16 MB) y longitud máxima de caracteres.
"""

import io
import re
import unicodedata
from typing import Any, Dict, List, Optional
from pypdf import PdfReader
import docx

try:
    from charset_normalizer import from_bytes as detect_charset_from_bytes
    HAS_CHARSET_NORMALIZER = True
except ImportError:
    detect_charset_from_bytes = None
    HAS_CHARSET_NORMALIZER = False

from .preprocessing import HYPHEN_LINEBREAK_PATTERN

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}
MAX_FILE_SIZE_BYTES = 16 * 1024 * 1024  # 16 MB
MAX_TEXT_CHARS = 500_000  # 500,000 caracteres


def extract_text_from_file(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Extrae el contenido textual de un archivo binario en memoria validando formato,
    tamaño máximo y límites de caracteres.
    """
    if not filename or not filename.strip():
        raise ValueError("El archivo proporcionado no tiene un nombre válido.")

    lower_name = filename.lower().strip()
    ext = ""
    for candidate in ALLOWED_EXTENSIONS:
        if lower_name.endswith(candidate):
            ext = candidate
            break

    if not ext:
        raise ValueError(
            "Formato de archivo no compatible. Se admiten únicamente archivos .pdf, .docx, .txt, .md y .csv."
        )

    if not file_bytes:
        raise ValueError(f"El archivo '{filename}' está vacío (0 bytes).")

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        size_mb = len(file_bytes) / (1024 * 1024)
        raise ValueError(
            f"El archivo '{filename}' ({size_mb:.1f} MB) supera el límite máximo permitido de 16 MB."
        )

    if ext == ".pdf":
        result = _extract_from_pdf(file_bytes, filename)
    elif ext == ".docx":
        result = _extract_from_docx(file_bytes, filename)
    else:
        result = _extract_from_plain_text(file_bytes, filename, ext)

    text_out = result.get("text", "")
    if len(text_out) > MAX_TEXT_CHARS:
        result["text"] = text_out[:MAX_TEXT_CHARS]
        result["char_count"] = MAX_TEXT_CHARS
        result["truncated"] = True
        result["warning"] = (
            f"El documento superaba los {MAX_TEXT_CHARS:,} caracteres y fue truncado al límite máximo."
        )
    else:
        result["truncated"] = False

    split_info = detect_and_split_document_sections(result.get("text", ""))
    if split_info is not None:
        result["split_sections"] = split_info

    return result


def detect_and_split_document_sections(text: str) -> Optional[Dict[str, Any]]:
    """
    Detecta si un documento contiene dos secciones internas comparables explícitas
    (por ejemplo `Plantilla A · Pasos 1 a 5` y `Plantilla B · Paso 6: tres hallazgos para la toma de decisiones`)
    y retorna ambas partes separadas para comparación directa.
    """
    if not text or not isinstance(text, str):
        return None

    has_plantilla_a = bool(re.search(r"Plantilla\s+A\b", text, re.IGNORECASE))
    match_b = re.search(
        r"(?:Plantillas\s+de\s+trabajo\s*[·\-].*?P[áa]gina\s+\d+\s+de\s+\d+\s*\n+)?Plantilla\s+B\b",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not (has_plantilla_a and match_b):
        return None

    # Buscar el inicio del bloque de página donde aparece Plantilla B para conservar el encabezado de página
    plantilla_b_pos = re.search(r"Plantilla\s+B\b", text, re.IGNORECASE)
    if not plantilla_b_pos:
        return None

    pos_b = plantilla_b_pos.start()
    page_break = text.rfind("\n\n", 0, pos_b)
    split_idx = page_break + 2 if page_break != -1 and (pos_b - page_break) < 180 else pos_b

    section_a = text[:split_idx].strip()
    section_b = text[split_idx:].strip()
    if len(section_a) < 80 or len(section_b) < 80:
        return None

    return {
        "has_internal_comparison_sections": True,
        "section_a_title": "Plantilla A · Pasos 1 a 5: registro de patrones (Págs. 1-2)",
        "section_b_title": "Plantilla B · Paso 6: tres hallazgos para la toma de decisiones (Pág. 3)",
        "text_a": section_a,
        "text_b": section_b,
    }


def _clean_extracted_text(raw_text: str) -> str:
    """Normaliza a Unicode NFC y une palabras cortadas por guion al final de línea."""
    if not raw_text:
        return ""
    text_nfc = unicodedata.normalize("NFC", raw_text)
    return HYPHEN_LINEBREAK_PATTERN.sub(r"\1\2", text_nfc).strip()


def _extract_from_pdf(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Extrae el texto de todas las páginas no vacías de un PDF en memoria,
    limpiando guiones de fin de línea y detectando PDFs escaneados sin capa de texto.
    """
    try:
        stream = io.BytesIO(file_bytes)
        reader = PdfReader(stream)
        total_pages = len(reader.pages)
        if total_pages == 0:
            raise ValueError(f"El archivo PDF '{filename}' no contiene páginas.")

        pages_text: List[str] = []
        for page in reader.pages:
            page_content = page.extract_text() or ""
            cleaned_page = _clean_extracted_text(page_content)
            if cleaned_page:
                pages_text.append(cleaned_page)

        full_text = "\n\n".join(pages_text).strip()
        if not full_text:
            raise ValueError(
                f"No se encontró texto seleccionable en '{filename}'. "
                "Es posible que el PDF esté escaneado como imagen y requiera reconocimiento óptico (OCR)."
            )

        return {
            "filename": filename,
            "file_type": "PDF",
            "pages_count": total_pages,
            "non_empty_pages_count": len(pages_text),
            "text": full_text,
            "char_count": len(full_text),
        }
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"No se pudo leer el archivo PDF '{filename}': {exc}") from exc


def _extract_from_docx(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Extrae en orden encabezados, párrafos, tablas y pies de página de un archivo Word (.docx).
    """
    try:
        stream = io.BytesIO(file_bytes)
        document = docx.Document(stream)
        blocks: List[str] = []
        seen_headers_footers = set()

        # 1. Encabezados de sección
        for section in document.sections:
            if section.header:
                for p in section.header.paragraphs:
                    txt = _clean_extracted_text(p.text)
                    if txt and txt not in seen_headers_footers:
                        seen_headers_footers.add(txt)
                        blocks.append(txt)

        # 2. Párrafos principales
        for para in document.paragraphs:
            txt = _clean_extracted_text(para.text)
            if txt:
                blocks.append(txt)

        # 3. Tablas
        for table in document.tables:
            for row in table.rows:
                row_cells = [
                    _clean_extracted_text(cell.text)
                    for cell in row.cells
                    if cell.text and cell.text.strip()
                ]
                if row_cells:
                    blocks.append(" | ".join(row_cells))

        # 4. Pies de página de sección
        for section in document.sections:
            if section.footer:
                for p in section.footer.paragraphs:
                    txt = _clean_extracted_text(p.text)
                    if txt and txt not in seen_headers_footers:
                        seen_headers_footers.add(txt)
                        blocks.append(txt)

        full_text = "\n\n".join(blocks).strip()
        if not full_text:
            raise ValueError(f"El documento Word '{filename}' no contiene texto extraíble.")

        return {
            "filename": filename,
            "file_type": "DOCX",
            "pages_count": len(blocks),
            "text": full_text,
            "char_count": len(full_text),
        }
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"No se pudo leer el documento Word '{filename}': {exc}") from exc


def _extract_from_plain_text(file_bytes: bytes, filename: str, ext: str) -> Dict[str, Any]:
    """
    Decodifica archivos de texto plano (.txt, .md, .csv) utilizando detección de
    codificación con `charset_normalizer` y respaldo para UTF-8-BOM, UTF-8, cp1252 y Latin-1.
    """
    decoded: str = ""
    detected_encoding: str = "utf-8"

    # 1. Probar UTF-8 con BOM y UTF-8 estricto primero
    for enc in ("utf-8-sig", "utf-8"):
        try:
            decoded = file_bytes.decode(enc)
            detected_encoding = enc
            break
        except UnicodeDecodeError:
            continue

    # 2. Si no es UTF-8 puro, usar charset_normalizer priorizando codificaciones del español / occidente
    if not decoded and HAS_CHARSET_NORMALIZER and detect_charset_from_bytes is not None:
        best_match = detect_charset_from_bytes(
            file_bytes,
            cp_isolation=["utf_8", "cp1252", "latin_1", "iso8859_15", "utf_16"],
        ).best()
        if best_match is not None:
            decoded = str(best_match)
            detected_encoding = best_match.encoding or "detected"

    # 3. Fallback determinista a cp1252 y latin-1
    if not decoded:
        for enc in ("cp1252", "latin-1"):
            try:
                decoded = file_bytes.decode(enc)
                detected_encoding = enc
                break
            except UnicodeDecodeError:
                continue

    if not decoded:
        raise ValueError(f"No se pudo decodificar la codificación de caracteres de '{filename}'.")

    full_text = _clean_extracted_text(decoded)
    if not full_text:
        raise ValueError(f"El archivo '{filename}' no contiene texto válido.")

    return {
        "filename": filename,
        "file_type": ext.upper().lstrip("."),
        "encoding": detected_encoding,
        "pages_count": 1,
        "text": full_text,
        "char_count": len(full_text),
    }
