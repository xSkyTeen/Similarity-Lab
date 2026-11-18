# Similarity Lab — Text Similarity & NLP Analysis

**Similarity Lab** es una aplicación web diseñada para el análisis estadístico, léxico y semántico de similitud textual entre dos documentos mediante algoritmos de Procesamiento de Lenguaje Natural (NLP) e Inteligencia Artificial. Admite texto directo o carga de archivos **PDF (`.pdf`)**, **Word (`.docx`)** y **Texto (`.txt`, `.md`, `.csv`)**, y presenta resultados mediante tres métricas principales (**Embeddings Semánticos**, **Coseno TF-IDF** y **Jaccard**), tablas comparativas, gráficos interactivos e interpretación automática.

> Para una explicación didáctica paso a paso de cómo funciona cada algoritmo y cómo se vectorizan las palabras, consulta también [GUIA_COMO_FUNCIONA.md](file:///home/sky/PycharmProjects/comparacion/GUIA_COMO_FUNCIONA.md) o haz clic en el botón **"¿Cómo funciona?"** dentro de la aplicación web.

---

## 1. Descripción del Proyecto

El sistema compara dos documentos en memoria y cuantifica su similitud a través de tres enfoques complementarios:
1. **Similitud por Embeddings Semánticos (IA Gemini / Vectorización Contextual 128-D):** Transforma las palabras y documentos en vectores densos continuos capaces de capturar **contexto amplio**, sinónimos (*estudiante ↔ alumno*, *computadora ↔ ordenador*, *dividir ↔ segmentar*) y familias morfológicas (*tokenización ↔ tokenizar*). Funciona tanto con **Google Gemini AI (`gemini-embedding-001`)** como con el motor semántico contextual local integrado.
2. **Similitud Vectorial Ponderada (TF-IDF + Similitud del Coseno):** Representa los documentos en un espacio multidimensional ponderando la especificidad de cada palabra mediante *Term Frequency – Inverse Document Frequency* y calcula la cercanía angular normalizada (norma Euclídea L2).
3. **Similitud Léxica de Conjuntos (Índice de Jaccard):** Evalúa la superposición exacta de vocabulario único sin verse afectada por la repetición de términos.

---

## 2. Tecnologías Utilizadas

### Backend, IA y Procesamiento NLP
- **Python 3**
- **Flask** (Servidor web y API REST en memoria)
- **Google GenAI SDK (`google-genai`)** (Integración con `gemini-embedding-001` para embeddings en la nube)
- **Scikit-learn** (`TfidfVectorizer`, `cosine_similarity`)
- **NumPy & Pandas** (Álgebra lineal, vectores densos 128-D, producto escalar y matrices de similitud)
- **pypdf & python-docx** (Extracción de texto en memoria desde archivos `.pdf` y `.docx`)
- **Pytest** (Suite de pruebas unitarias y de integración)

### Frontend y Visualización
- **HTML5 Semántico**
- **CSS3 Puro** (Variables CSS, Grid/Flexbox responsive, paleta turquesa pastel)
- **JavaScript Vanilla (ES6+)** (Fetch API asíncrona, subida Drag & Drop de PDF/DOCX/TXT)
- **Chart.js** (Gráficos interactivos comparando Embeddings, Coseno TF-IDF y Jaccard, distribución de vocabulario y frecuencias)

---

## 3. Estructura de Carpetas

```text
comparacion/
│
├── app.py                     # Punto de entrada Flask y endpoints REST JSON (/api/compare, /api/upload)
├── requirements.txt           # Dependencias del proyecto
├── README.md                  # Documentación técnica y guía de ejecución
├── GUIA_COMO_FUNCIONA.md      # Explicación sencilla y detallada de cómo funciona todo el sistema
│
├── core/                      # Motor matemático, semántico y de lectura de archivos
│   ├── __init__.py
│   ├── preprocessing.py       # Limpieza, normalización, tokenización y stopwords ES
│   ├── file_reader.py         # Extracción en memoria de archivos PDF, DOCX, TXT, MD y CSV
│   ├── jaccard.py             # Índice y distancia de Jaccard sobre conjuntos
│   ├── cosine.py              # Vectorización TF-IDF y Similitud del Coseno
│   ├── embeddings.py          # Embeddings Semánticos (IA Gemini / Vectorización Contextual 128-D)
│   └── statistics.py          # Estadísticas descriptivas, frecuencias e interpretación
│
├── templates/
│   └── index.html             # Vista principal única de Similarity Lab
│
├── static/
│   ├── css/
│   │   └── style.css          # Sistema de diseño visual turquesa pastel
│   ├── js/
│   │   └── app.js             # Lógica de cliente, subida de PDF, validaciones y Chart.js
│   └── assets/
│       └── logo.svg           # Isotipo vectorial minimalista
│
└── tests/                     # Pruebas automatizadas
    ├── test_preprocessing.py  # Pruebas de normalización, tildes, ñ y stopwords
    ├── test_jaccard.py        # Pruebas matemáticas del índice de Jaccard
    └── test_cosine.py         # Pruebas de TF-IDF, Embeddings, lectura de PDF y endpoints Flask
```

---

## 4. Configuración del Entorno Virtual e Instalación

1. Abre una terminal en el directorio raíz del proyecto:
   ```bash
   cd /home/sky/PycharmProjects/comparacion
   ```

2. Crea y activa el entorno virtual de Python:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Instala las dependencias listadas en `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

---

## 5. Ejecución de la Aplicación

Con el entorno virtual activado, inicia el servidor ejecutando:

```bash
python app.py
```

La aplicación estará disponible en tu navegador web en la dirección local de Flask:

```text
http://127.0.0.1:5000
```

*(Opcional)* Para utilizar los embeddings de Google Gemini en la nube (`gemini-embedding-001`), puedes pegar tu clave `GEMINI_API_KEY` directamente en el campo superior de la interfaz web o definirla como variable de entorno antes de iniciar:
```bash
export GEMINI_API_KEY="tu_clave_aqui"
python app.py
```

Para ejecutar la suite de pruebas automatizadas:
```bash
pytest -v
```
