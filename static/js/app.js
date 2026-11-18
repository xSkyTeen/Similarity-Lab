/**
 * Similarity Lab — Lógica Frontend, Carga de Archivos (PDF/DOCX/TXT) y Visualizaciones NLP
 * Desarrollado con JavaScript puro (ES6+), Fetch API y Chart.js.
 */

document.addEventListener("DOMContentLoaded", () => {
    // Paleta de identidad visual oficial
    const PALETTE = {
        bgMain: "#F4FAF9",
        turquoisePrimary: "#58BEB0",
        turquoiseSecondary: "#8DD8CC",
        turquoisePastel: "#DDF3EF",
        textPrimary: "#244B47",
        textSecondary: "#78918D",
        cardBg: "#FFFFFF",
        border: "#DCECE9",
    };

    // Referencias a elementos del DOM
    const textA = document.getElementById("textA");
    const textB = document.getElementById("textB");
    const cardA = document.getElementById("cardA");
    const cardB = document.getElementById("cardB");
    const fileInputA = document.getElementById("fileInputA");
    const fileInputB = document.getElementById("fileInputB");
    const fileBadgeA = document.getElementById("fileBadgeA");
    const fileBadgeB = document.getElementById("fileBadgeB");

    const wordCountA = document.getElementById("wordCountA");
    const charCountA = document.getElementById("charCountA");
    const statusA = document.getElementById("statusA");
    const wordCountB = document.getElementById("wordCountB");
    const charCountB = document.getElementById("charCountB");
    const statusB = document.getElementById("statusB");

    const compareBtn = document.getElementById("compareBtn");
    const swapBtn = document.getElementById("swapBtn");
    const clearBtn = document.getElementById("clearBtn");
    const removeStopwordsToggle = document.getElementById("removeStopwordsToggle");
    const profileSelect = document.getElementById("profileSelect");
    const sampleSelect = document.getElementById("sampleSelect");
    const geminiApiKeyInput = document.getElementById("geminiApiKeyInput");

    const validationAlert = document.getElementById("validationAlert");
    const validationMessage = document.getElementById("validationMessage");
    const closeAlertBtn = document.getElementById("closeAlertBtn");
    const loadingIndicator = document.getElementById("loadingIndicator");
    const loadingTitle = document.getElementById("loadingTitle");
    const loadingSubtitle = document.getElementById("loadingSubtitle");
    const resultsPanel = document.getElementById("resultsPanel");
    const sharedSearchInput = document.getElementById("sharedSearchInput");

    const openGuideBtn = document.getElementById("openGuideBtn");
    const closeGuideBtn = document.getElementById("closeGuideBtn");
    const guideModal = document.getElementById("guideModal");

    // Estado local
    let currentSharedTerms = [];
    let academicSamplesCache = {};
    let chartInstances = {
        similarity: null,
        vocabulary: null,
        frequency: null,
    };

    // Configuración global tipográfica para Chart.js
    if (window.Chart) {
        Chart.defaults.font.family = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif";
        Chart.defaults.color = PALETTE.textSecondary;
    }

    /**
     * Cuenta las palabras y caracteres de un texto en tiempo real.
     */
    function computeTextMetrics(rawText) {
        const text = rawText || "";
        const charLength = text.length;
        const trimmed = text.trim();
        const words = trimmed === "" ? [] : trimmed.split(/\s+/);
        return {
            words: words.length,
            chars: charLength,
        };
    }

    /**
     * Actualiza los contadores dinámicos de las tarjetas A y B.
     */
    function updateCounters() {
        const metricsA = computeTextMetrics(textA.value);
        wordCountA.textContent = metricsA.words.toLocaleString("es-ES");
        charCountA.textContent = metricsA.chars.toLocaleString("es-ES");
        statusA.textContent =
            metricsA.words > 0 ? "Documento listo" : "Admite texto o archivos PDF";

        const metricsB = computeTextMetrics(textB.value);
        wordCountB.textContent = metricsB.words.toLocaleString("es-ES");
        charCountB.textContent = metricsB.chars.toLocaleString("es-ES");
        statusB.textContent =
            metricsB.words > 0 ? "Documento listo" : "Admite texto o archivos PDF";

        if (metricsA.words > 0) cardA.classList.remove("input-error");
        if (metricsB.words > 0) cardB.classList.remove("input-error");
    }

    /**
     * Muestra un mensaje de validación o error elegante.
     */
    function showValidationError(message, highlightA = false, highlightB = false) {
        validationMessage.textContent = message;
        validationAlert.classList.remove("hidden");
        cardA.classList.toggle("input-error", highlightA);
        cardB.classList.toggle("input-error", highlightB);
    }

    /**
     * Oculta el banner de validación.
     */
    function hideValidationError() {
        validationAlert.classList.add("hidden");
        cardA.classList.remove("input-error");
        cardB.classList.remove("input-error");
    }

    /**
     * Verifica el estado del servidor Flask y carga muestras académicas.
     */
    async function initializeSystem() {
        try {
            const statusResp = await fetch("/api/status");
            if (statusResp.ok) {
                const statusData = await statusResp.json();
                const statusTextEl = document.getElementById("statusText");
                if (statusTextEl && statusData.status === "online") {
                    statusTextEl.textContent = statusData.gemini_env_configured
                        ? "Motor NLP + Gemini AI Activo"
                        : `Motor NLP + Embeddings Activo`;
                }
            }

            const samplesResp = await fetch("/api/samples");
            if (samplesResp.ok) {
                const samplesData = await samplesResp.json();
                if (samplesData.samples) {
                    academicSamplesCache = samplesData.samples;
                }
            }
        } catch (err) {
            console.warn("No se pudo verificar el estado inicial:", err);
        }
    }

    /**
     * Sube y extrae el texto de un archivo (PDF, DOCX, TXT, MD, CSV) hacia el Documento A o B.
     */
    async function handleFileUpload(file, targetSlot) {
        if (!file) return;
        hideValidationError();

        const targetTextarea = targetSlot === "A" ? textA : textB;
        const targetBadge = targetSlot === "A" ? fileBadgeA : fileBadgeB;
        const targetStatus = targetSlot === "A" ? statusA : statusB;

        targetStatus.textContent = `Extrayendo texto de ${file.name}...`;
        loadingTitle.textContent = `Extrayendo texto del archivo (${file.name})...`;
        loadingSubtitle.textContent = "Procesando páginas en memoria (PDF / DOCX / TXT).";
        loadingIndicator.classList.remove("hidden");

        try {
            const formData = new FormData();
            formData.append("file", file);

            const response = await fetch("/api/upload", {
                method: "POST",
                body: formData,
            });
            const payload = await response.json();

            if (!response.ok || payload.status !== "success") {
                showValidationError(
                    payload.message || `No fue posible extraer el texto de '${file.name}'.`,
                    targetSlot === "A",
                    targetSlot === "B"
                );
                targetStatus.textContent = "Error al leer archivo";
                return;
            }

            const info = payload.data;
            const otherTextarea = targetSlot === "A" ? textB : textA;
            if (
                info.split_sections &&
                info.split_sections.has_internal_comparison_sections &&
                !otherTextarea.value.trim()
            ) {
                textA.value = info.split_sections.text_a;
                textB.value = info.split_sections.text_b;
                fileBadgeA.textContent = `${info.file_type}: ${info.filename} (Plantilla A · Págs. 1-2)`;
                fileBadgeA.title = info.split_sections.section_a_title;
                fileBadgeA.classList.remove("hidden");
                fileBadgeB.textContent = `${info.file_type}: ${info.filename} (Plantilla B · Pág. 3)`;
                fileBadgeB.title = info.split_sections.section_b_title;
                fileBadgeB.classList.remove("hidden");
            } else {
                targetTextarea.value = info.text;
                targetBadge.textContent = `${info.file_type}: ${info.filename} (${info.pages_count} pág.)`;
                targetBadge.title = `${info.filename} · ${info.char_count} caracteres extraídos`;
                targetBadge.classList.remove("hidden");
            }
            updateCounters();
        } catch (err) {
            console.error("Error subiendo archivo:", err);
            showValidationError(
                "Error de conexión al intentar procesar el archivo en el servidor."
            );
        } finally {
            loadingIndicator.classList.add("hidden");
            loadingTitle.textContent = "Procesando análisis NLP y Embeddings...";
            loadingSubtitle.textContent =
                "Tokenizando documentos, vectorizando en espacio denso, construyendo matriz TF-IDF y calculando Jaccard.";
        }
    }

    /**
     * Configura eventos de Drag & Drop sobre una tarjeta de documento.
     */
    function setupDragAndDrop(cardEl, slot) {
        ["dragenter", "dragover"].forEach((evtName) => {
            cardEl.addEventListener(evtName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                cardEl.classList.add("drag-over");
            });
        });

        ["dragleave", "drop"].forEach((evtName) => {
            cardEl.addEventListener(evtName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                cardEl.classList.remove("drag-over");
            });
        });

        cardEl.addEventListener("drop", (e) => {
            const files = e.dataTransfer?.files;
            if (files && files.length > 0) {
                handleFileUpload(files[0], slot);
            }
        });
    }

    /**
     * Ejecuta la petición POST hacia Flask para comparar los dos textos.
     */
    async function performComparison(scrollToResults = true) {
        hideValidationError();

        const valA = textA.value.trim();
        const valB = textB.value.trim();

        if (!valA && !valB) {
            showValidationError(
                "Ambos campos están vacíos. Por favor, pega texto o sube un archivo PDF/DOCX/TXT en Texto A y Texto B.",
                true,
                true
            );
            return;
        }
        if (!valA) {
            showValidationError(
                "El campo 'Texto A' está vacío. Ingresa o sube el primer documento antes de comparar.",
                true,
                false
            );
            textA.focus();
            return;
        }
        if (!valB) {
            showValidationError(
                "El campo 'Texto B' está vacío. Ingresa o sube el segundo documento antes de comparar.",
                false,
                true
            );
            textB.focus();
            return;
        }

        // Mostrar indicador de procesamiento
        loadingIndicator.classList.remove("hidden");
        compareBtn.disabled = true;
        swapBtn.disabled = true;

        try {
            const response = await fetch("/api/compare", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    text_a: textA.value,
                    text_b: textB.value,
                    remove_stopwords: removeStopwordsToggle.checked,
                    profile: profileSelect ? profileSelect.value : "enhanced",
                    gemini_api_key: geminiApiKeyInput ? geminiApiKeyInput.value.trim() : "",
                }),
            });

            const payload = await response.json();

            if (!response.ok || payload.status !== "success") {
                const errorMsg =
                    payload.message || "Ocurrió un error al procesar los documentos.";
                showValidationError(errorMsg);
                return;
            }

            renderResultsPanel(payload.data, payload.execution_time_ms);

            if (scrollToResults) {
                resultsPanel.scrollIntoView({ behavior: "smooth", block: "start" });
            }
        } catch (error) {
            console.error("Error en petición NLP:", error);
            showValidationError(
                "No fue posible conectar con el motor NLP del servidor. Verifica que la aplicación Flask esté en ejecución."
            );
        } finally {
            loadingIndicator.classList.add("hidden");
            compareBtn.disabled = false;
            swapBtn.disabled = false;
        }
    }

    /**
     * Renderiza todos los componentes del panel estadístico, tablas y gráficos.
     */
    function renderResultsPanel(data, execTimeMs) {
        resultsPanel.classList.remove("hidden");

        const {
            preprocessing,
            jaccard,
            cosine,
            embeddings,
            general_statistics: stats,
            frequency_analysis: freq,
            comparative_analysis: comp,
            interpretation,
        } = data;

        // Encabezado del panel de resultados
        const swApplied = preprocessing.remove_stopwords;
        document.getElementById("stopwordsStatusPill").textContent = swApplied
            ? "Stopwords ES filtradas"
            : "Stopwords incluidas";
        if (data.confidence && document.getElementById("confidencePill")) {
            document.getElementById("confidencePill").textContent = data.confidence.label;
        }
        const modeLabels = {
            local_hybrid_lsa: `Híbrido LSA + Contexto (${embeddings.dimensions}-D · 0 Tokens)`,
            gemini: `Gemini AI (${embeddings.dimensions}-D)`,
            offline_minilm: `MiniLM Local (${embeddings.dimensions}-D · 0 Tokens)`,
            offline_approx_128d: `Contextual Local (${embeddings.dimensions}-D · 0 Tokens)`,
        };
        document.getElementById("embeddingEnginePill").textContent =
            modeLabels[embeddings.mode] || `Embeddings (${embeddings.dimensions}-D · 0 Tokens)`;
        document.getElementById("execTimePill").textContent = `${execTimeMs.toFixed(2)} ms`;
        document.getElementById("analysisMetaSubtitle").textContent =
            `${embeddings.gemini_status} · Vocabulario unificado de ${stats.total_unique_vocabulary} raíces léxicas.`;

        // Advertencias automáticas (warnings)
        const warningsBox = document.getElementById("analysisWarningsBox");
        const warningsList = document.getElementById("analysisWarningsList");
        if (warningsBox && warningsList) {
            warningsList.innerHTML = "";
            const warns = data.warnings || [];
            if (warns.length > 0) {
                warningsBox.classList.remove("hidden");
                warns.forEach((wText) => {
                    const li = document.createElement("li");
                    li.textContent = wText;
                    warningsList.appendChild(li);
                });
            } else {
                warningsBox.classList.add("hidden");
            }
        }

        // 6.1 Indicadores principales (KPIs)
        // 1. Embeddings Semánticos (IA / Contextual)
        const embPct = embeddings.similarity_percentage;
        const embRawPct =
            embeddings.raw_similarity_percentage !== undefined
                ? embeddings.raw_similarity_percentage
                : embPct;
        const embCalPct =
            embeddings.calibrated_similarity_percentage !== undefined
                ? embeddings.calibrated_similarity_percentage
                : embPct;
        const isRawPrimary = embeddings.primary_score_type === "raw";
        document.getElementById("kpiEmbeddingPct").textContent = `${embPct.toFixed(2)}%`;
        document.getElementById("kpiEmbeddingDec").textContent = isRawPrimary
            ? `Principal (Crudo): ${embeddings.raw_similarity_score.toFixed(4)} · Calibrado: ${embCalPct.toFixed(2)}%`
            : `Principal (Cal): ${embeddings.calibrated_similarity_score.toFixed(4)} · Crudo: ${embRawPct.toFixed(2)}%`;
        const rawVsCalEl = document.getElementById("kpiEmbeddingRawVsCal");
        if (rawVsCalEl) {
            rawVsCalEl.textContent = isRawPrimary
                ? `[Crudo: ${embRawPct.toFixed(1)}%] / Cal: ${embCalPct.toFixed(1)}%`
                : `Crudo: ${embRawPct.toFixed(1)}% / [Cal: ${embCalPct.toFixed(1)}%]`;
        }
        document.getElementById("progressEmbedding").style.width =
            `${Math.min(100, Math.max(0, embPct))}%`;
        document.getElementById("kpiEmbeddingDim").textContent =
            `${embeddings.dimensions}-D`;
        const semanticMatches = embeddings.semantic_word_matches || [];
        document.getElementById("kpiSemanticBridges").textContent =
            `${semanticMatches.length} pares`;
        document.getElementById("kpiEmbeddingBadge").textContent = isRawPrimary
            ? "Modo Referencia (Crudo)"
            : embeddings.using_gemini_api
            ? "Gemini AI (Calibrado)"
            : embeddings.mode === "offline_minilm"
            ? "MiniLM Local (Calibrado)"
            : "LSA + Contexto (Calibrado)";

        // 2. Coseno TF-IDF
        const cosPct = cosine.similarity_percentage;
        document.getElementById("kpiCosinePct").textContent = `${cosPct.toFixed(2)}%`;
        document.getElementById("kpiCosineDec").textContent =
            `Valor decimal: ${cosine.similarity_score.toFixed(4)}`;
        document.getElementById("progressCosine").style.width =
            `${Math.min(100, Math.max(0, cosPct))}%`;
        document.getElementById("kpiVocabSize").textContent =
            `${cosine.vocabulary_size} términos`;
        document.getElementById("kpiBowCosine").textContent =
            `${cosine.vectorization_info.bow_cosine_percentage.toFixed(2)}%`;

        // 3. Jaccard
        const jacPct = jaccard.similarity_percentage;
        document.getElementById("kpiJaccardPct").textContent = `${jacPct.toFixed(2)}%`;
        document.getElementById("kpiJaccardDec").textContent =
            `Valor decimal: ${jaccard.similarity_score.toFixed(4)}`;
        document.getElementById("progressJaccard").style.width =
            `${Math.min(100, Math.max(0, jacPct))}%`;
        document.getElementById("kpiIntersection").textContent =
            `${jaccard.intersection_size} palabras`;
        document.getElementById("kpiUnion").textContent =
            `${jaccard.union_size} palabras`;

        // 4. Palabras coincidentes y exclusivas
        document.getElementById("kpiSharedCount").textContent = stats.shared_terms_count;
        document.getElementById("kpiExclusiveTotal").textContent =
            stats.total_exclusive_terms_count;
        document.getElementById("progressShared").style.width =
            `${Math.min(100, Math.max(0, jacPct))}%`;
        document.getElementById("kpiExclusiveA").textContent = stats.exclusive_terms_a_count;
        document.getElementById("kpiExclusiveB").textContent = stats.exclusive_terms_b_count;
        document.getElementById("kpiDice").textContent = `${stats.dice_coefficient.toFixed(1)}%`;
        document.getElementById("kpiOverlap").textContent =
            `${stats.overlap_coefficient.toFixed(1)}%`;

        // 7. Interpretación automática
        document.getElementById("interpTitle").textContent = interpretation.title;
        document.getElementById("interpBadge").textContent = interpretation.badge;
        document.getElementById("interpSummary").textContent = interpretation.summary;

        const interpList = document.getElementById("interpDetailsList");
        interpList.innerHTML = "";
        (interpretation.technical_details || []).forEach((detail) => {
            const li = document.createElement("li");
            li.textContent = detail;
            interpList.appendChild(li);
        });
        document.getElementById("interpMethodologicalNote").textContent =
            interpretation.methodological_note;

        // Sección de Embeddings: Puentes semánticos y vectores densos
        renderSemanticBridgesTable(semanticMatches);
        renderEmbeddingVectorsBox(embeddings);

        // 5. Estadísticas generales y vectorización
        document.getElementById("statActiveWordsA").textContent = stats.total_words_a;
        document.getElementById("statActiveWordsB").textContent = stats.total_words_b;
        document.getElementById("statRawWordsA").textContent = stats.raw_words_a;
        document.getElementById("statRawWordsB").textContent = stats.raw_words_b;
        document.getElementById("statCharsA").textContent = stats.chars_a;
        document.getElementById("statCharsB").textContent = stats.chars_b;
        document.getElementById("statCharsNoSpacesA").textContent = stats.chars_no_spaces_a;
        document.getElementById("statCharsNoSpacesB").textContent = stats.chars_no_spaces_b;
        document.getElementById("statUniqueA").textContent = stats.unique_words_a;
        document.getElementById("statUniqueB").textContent = stats.unique_words_b;
        document.getElementById("statExclA").textContent = stats.exclusive_terms_a_count;
        document.getElementById("statExclB").textContent = stats.exclusive_terms_b_count;
        document.getElementById("statStopwordsA").textContent =
            preprocessing.doc_a.stopwords_detected_count;
        document.getElementById("statStopwordsB").textContent =
            preprocessing.doc_b.stopwords_detected_count;
        document.getElementById("statDensityA").textContent = `${stats.lexical_density_a.toFixed(2)}%`;
        document.getElementById("statDensityB").textContent = `${stats.lexical_density_b.toFixed(2)}%`;

        const vecInfo = cosine.vectorization_info;
        document.getElementById("vecEmbeddingEngine").textContent =
            `${embeddings.provider}`;
        document.getElementById("vecEngineName").textContent = vecInfo.vectorizer;
        document.getElementById("vecDimensionality").textContent =
            `${cosine.dimensionality} términos (TF-IDF) / ${embeddings.dimensions}-D (Embeddings)`;
        document.getElementById("vecMatrixShape").textContent =
            `(${vecInfo.matrix_shape[0]} × ${vecInfo.matrix_shape[1]})`;
        document.getElementById("vecDotProduct").textContent =
            `${vecInfo.dot_product.toFixed(4)} (TF-IDF) / ${embeddings.dot_product.toFixed(4)} (Emb)`;
        document.getElementById("vecNorms").textContent =
            `||A|| = ${vecInfo.norm_a.toFixed(2)} | ||B|| = ${vecInfo.norm_b.toFixed(2)}`;
        document.getElementById("vecSharedCount").textContent =
            `${stats.shared_terms_count} palabras exactas + ${semanticMatches.length} puentes semánticos`;
        document.getElementById("vecJaccardDist").textContent =
            jaccard.jaccard_distance.toFixed(4);
        document.getElementById("vecOverlapRatio").textContent =
            `${stats.vocabulary_overlap_ratio.toFixed(2)}% (Jaccard) / ${stats.dice_coefficient.toFixed(2)}% (Dice)`;

        // Tablas de palabras más frecuentes individuales (A y B)
        renderTopWordsTable("topWordsTableA", freq.top_words_a);
        renderTopWordsTable("topWordsTableB", freq.top_words_b);

        // 6.3 Tabla de coincidencias
        currentSharedTerms = comp.shared_terms_table || [];
        sharedSearchInput.value = "";
        renderSharedTermsTable(currentSharedTerms);

        // 6.4 Palabras exclusivas (chips)
        renderExclusiveChips(
            "exclusiveAChips",
            "exclusiveACountBadge",
            comp.exclusive_details_a || [],
            "El Texto A no posee términos exclusivos respecto al Texto B."
        );
        renderExclusiveChips(
            "exclusiveBChips",
            "exclusiveBCountBadge",
            comp.exclusive_details_b || [],
            "El Texto B no posee términos exclusivos respecto al Texto A."
        );

        // Inspección de textos originales conservados vs copias normalizadas
        document.getElementById("inspectOrigA").textContent = preprocessing.doc_a.original_text;
        document.getElementById("inspectNormA").textContent =
            preprocessing.doc_a.normalized_text || "(sin términos)";
        document.getElementById("inspectOrigB").textContent = preprocessing.doc_b.original_text;
        document.getElementById("inspectNormB").textContent =
            preprocessing.doc_b.normalized_text || "(sin términos)";

        // Poblar Matriz Comparativa de 6 Técnicas de PLN y tabla de Comparación en Paralelo
        renderNlpTechniquesMatrix(data);
        renderProfilesComparison(data);

        // Poblar secciones de Escaneo Profundo (Contexto Oración a Oración, KWIC, Dominios y Repeticiones)
        renderDeepContextScan(data);
        renderDeepRepetitionScan(data);
        renderVectorSpaceAndGuideFeatures(data);

        // Poblar sección desplegable de Métricas Avanzadas y Fragmentos Literales
        renderAdvancedMetricsAndFragments(data);

        // 6.2 Renderizar los 3 gráficos con Chart.js
        renderSimilarityChart(embPct, cosPct, jacPct, data.nlp_techniques_matrix);
        renderVocabularyChart(
            stats.shared_terms_count,
            stats.exclusive_terms_a_count,
            stats.exclusive_terms_b_count
        );
        renderFrequencyChart(freq.comparative_chart);
    }

    /**
     * Renderiza la Matriz Comparativa de 6 Técnicas de PLN (Jaccard Bruto, Jaccard Filtrado,
     * Coseno Count/BoW, Coseno TF-IDF Unigramas, Coseno TF-IDF Bigramas y Coseno Embeddings)
     * junto con las categorías de palabras clave en la intersección exacta.
     */
    function renderNlpTechniquesMatrix(data) {
        const tbody = document.getElementById("nlpTechniquesMatrixBody");
        if (tbody) {
            tbody.innerHTML = "";
            const matrix = data.nlp_techniques_matrix || [];
            matrix.forEach((row) => {
                const tr = document.createElement("tr");

                const tdTech = document.createElement("td");
                const strongTech = document.createElement("strong");
                strongTech.textContent = row.technique;
                tdTech.appendChild(strongTech);

                const tdModel = document.createElement("td");
                tdModel.textContent = row.model_representation;

                const tdVal = document.createElement("td");
                tdVal.className = "num-col mono";
                tdVal.style.fontWeight = "700";
                tdVal.textContent = row.display_value || row.score.toFixed(4);

                const tdPct = document.createElement("td");
                tdPct.className = "num-col";
                const pill = document.createElement("span");
                pill.className = "diff-badge diff-badge-zero";
                pill.textContent = `${row.percentage.toFixed(2)}%`;
                tdPct.appendChild(pill);

                const tdSens = document.createElement("td");
                tdSens.textContent = row.semantic_sensitivity;

                const tdDiag = document.createElement("td");
                tdDiag.style.fontSize = "0.82rem";
                tdDiag.style.color = "var(--text-secondary)";
                tdDiag.textContent = row.applicability_diagnosis;

                tr.append(tdTech, tdModel, tdVal, tdPct, tdSens, tdDiag);
                tbody.appendChild(tr);
            });
        }

        const badgeEl = document.getElementById("nlpMatrixBadge");
        const catInfo = data.categorized_shared_keywords;
        if (badgeEl && catInfo) {
            badgeEl.textContent = `${catInfo.total_shared_keywords ?? 0} palabras clave compartidas`;
        }

        const listEl = document.getElementById("categorizedKeywordsList");
        if (listEl && catInfo) {
            listEl.innerHTML = "";
            (catInfo.categories || []).forEach((cat) => {
                const rowDiv = document.createElement("div");
                const strongCat = document.createElement("strong");
                strongCat.textContent = `${cat.label}: `;
                const spanTerms = document.createElement("span");
                spanTerms.textContent = (cat.terms && cat.terms.length > 0)
                    ? cat.terms.join(", ")
                    : "(sin coincidencias en esta categoría)";
                rowDiv.append(strongCat, spanTerms);
                listEl.appendChild(rowDiv);
            });
        }
    }

    /**
     * Renderiza la tabla comparativa en paralelo entre el Perfil Referencia (`reference`)
     * y el Perfil Enriquecido (`enhanced`), destacando cuál es el perfil principal activo.
     */
    function renderProfilesComparison(data) {
        const profComp = data.profiles_comparison;
        const tbody = document.getElementById("profilesComparisonTableBody");
        if (!tbody || !profComp) return;

        const isRefPrimary = (profComp.primary_profile || data.active_profile) === "reference";
        const badgeEl = document.getElementById("activeProfileBannerBadge");
        if (badgeEl) {
            badgeEl.textContent = isRefPrimary
                ? "Perfil Principal Activo: Referencia (Canónico)"
                : "Perfil Principal Activo: Enriquecido (NLP Español)";
        }

        const thRef = document.getElementById("thReferenceCol");
        const thEnh = document.getElementById("thEnhancedCol");
        if (thRef) {
            thRef.textContent = isRefPrimary
                ? "★ Modo Referencia (PRINCIPAL)"
                : "Modo Referencia (Canónico)";
        }
        if (thEnh) {
            thEnh.textContent = !isRefPrimary
                ? "★ Modo Enriquecido (PRINCIPAL)"
                : "Modo Enriquecido (NLP Español)";
        }

        tbody.innerHTML = "";
        const rows = profComp.comparison_table || [];
        rows.forEach((row) => {
            const tr = document.createElement("tr");

            const tdMetric = document.createElement("td");
            const strongM = document.createElement("strong");
            strongM.textContent = row.metric_label;
            tdMetric.appendChild(strongM);

            const tdRef = document.createElement("td");
            tdRef.className = "num-col";
            if (isRefPrimary) {
                tdRef.style.backgroundColor = "#EBF9F6";
                tdRef.style.fontWeight = "700";
            }
            tdRef.textContent = `${row.reference_percentage.toFixed(2)}% (${row.reference_score.toFixed(4)})`;

            const tdEnh = document.createElement("td");
            tdEnh.className = "num-col";
            if (!isRefPrimary) {
                tdEnh.style.backgroundColor = "#EBF9F6";
                tdEnh.style.fontWeight = "700";
            }
            tdEnh.textContent = `${row.enhanced_percentage.toFixed(2)}% (${row.enhanced_score.toFixed(4)})`;

            const tdDelta = document.createElement("td");
            tdDelta.className = "num-col";
            const deltaVal = row.delta_percentage || 0;
            const sign = deltaVal > 0 ? "+" : "";
            const deltaSpan = document.createElement("span");
            deltaSpan.className = "diff-badge diff-badge-zero";
            deltaSpan.textContent = `${sign}${deltaVal.toFixed(2)} pts`;
            tdDelta.appendChild(deltaSpan);

            const tdCfg = document.createElement("td");
            tdCfg.style.fontSize = "0.78rem";
            tdCfg.style.color = "var(--text-secondary)";
            tdCfg.textContent = `Ref: ${row.reference_config} | Enh: ${row.enhanced_config}`;

            tr.append(tdMetric, tdRef, tdEnh, tdDelta, tdCfg);
            tbody.appendChild(tr);
        });
    }

    /**
     * Renderiza el Escaneo Profundo de Contexto:
     * - Alineamiento contextual oración por oración (Texto A vs. Texto B)
     * - Concordancias KWIC (Key Word In Context)
     * - Dominios temáticos, conectores discursivos y legibilidad de Fernández Huerta
     */
    function renderDeepContextScan(data) {
        const deep = data.deep_scan || {};
        const ctxAlign = deep.context_alignment || data.embeddings?.sentence_alignment || { pairs: [], summary: {} };
        const kwicList = deep.kwic_concordances || data.embeddings?.kwic_concordances || [];
        const discDom = deep.discourse_and_domains || {};

        const summary = ctxAlign.summary || {};
        const readA = discDom.readability_a || {};
        const readB = discDom.readability_b || {};
        const lsaPct = data.embeddings?.lsa_similarity_percentage ?? 0;
        const stylePct = discDom.stylistic_similarity_percentage ?? 0;

        const setEl = (id, text) => {
            const el = document.getElementById(id);
            if (el) el.textContent = text;
        };

        setEl(
            "ctxSentencesCount",
            `${summary.sentences_analyzed_a ?? 0} en A / ${summary.sentences_analyzed_b ?? 0} en B`
        );
        setEl(
            "ctxAvgAlignment",
            `Afinidad media: ${(summary.average_alignment_percentage ?? 0).toFixed(1)}%`
        );
        setEl(
            "ctxMatchBreakdown",
            `${summary.literal_matches_count ?? 0} lit. · ${summary.paraphrase_matches_count ?? 0} paráf.`
        );
        setEl(
            "ctxPartialBreakdown",
            `${summary.partial_matches_count ?? 0} parciales · ${summary.divergent_count ?? 0} divergentes`
        );
        setEl(
            "ctxReadabilityScores",
            `${(readA.fernandez_huerta_score ?? 0).toFixed(1)} / ${(readB.fernandez_huerta_score ?? 0).toFixed(1)}`
        );
        setEl(
            "ctxReadabilityLabels",
            `${readA.readability_label || "N/A"} vs. ${readB.readability_label || "N/A"}`
        );
        setEl("ctxStyleAndLsa", `${stylePct.toFixed(1)}% / ${lsaPct.toFixed(1)}%`);

        // 1. Tabla de Alineamiento Oración por Oración
        const sentTbody = document.getElementById("sentenceAlignmentBody");
        if (sentTbody) {
            sentTbody.innerHTML = "";
            const pairs = ctxAlign.pairs || [];
            if (pairs.length === 0) {
                const tr = document.createElement("tr");
                tr.innerHTML = `<td colspan="4" class="empty-state-cell">No hay oraciones suficientes para construir la matriz de alineamiento contextual.</td>`;
                sentTbody.appendChild(tr);
            } else {
                pairs.forEach((p) => {
                    const tr = document.createElement("tr");

                    const tdA = document.createElement("td");
                    tdA.className = "sentence-cell";
                    const badgeA = document.createElement("span");
                    badgeA.className = "sent-idx-badge";
                    badgeA.textContent = `A#${p.index_a}`;
                    const txtA = document.createElement("span");
                    txtA.textContent = p.sentence_a;
                    tdA.append(badgeA, txtA);

                    const tdB = document.createElement("td");
                    tdB.className = "sentence-cell";
                    const badgeB = document.createElement("span");
                    badgeB.className = "sent-idx-badge";
                    badgeB.textContent = `B#${p.index_b}`;
                    const txtB = document.createElement("span");
                    txtB.textContent = p.sentence_b;
                    tdB.append(badgeB, txtB);

                    const tdRel = document.createElement("td");
                    const relPill = document.createElement("span");
                    relPill.className = `severity-pill ${
                        p.relation_level === "muy_alto"
                            ? "severity-pill-high"
                            : p.relation_level === "alto"
                            ? "severity-pill-mod"
                            : ""
                    }`;
                    relPill.textContent = p.relation_type;
                    tdRel.appendChild(relPill);

                    if (p.shared_clues && p.shared_clues.length > 0) {
                        const cluesDiv = document.createElement("div");
                        cluesDiv.style.fontSize = "0.75rem";
                        cluesDiv.style.color = "var(--text-secondary)";
                        cluesDiv.style.marginTop = "4px";
                        cluesDiv.textContent = `Pistas: ${p.shared_clues.join(", ")}`;
                        tdRel.appendChild(cluesDiv);
                    }

                    const tdSim = document.createElement("td");
                    tdSim.className = "num-col";
                    const simBadge = document.createElement("span");
                    simBadge.className = "diff-badge diff-badge-zero";
                    simBadge.textContent = `${p.similarity_percentage.toFixed(1)}%`;
                    const subBreakdown = document.createElement("div");
                    subBreakdown.style.fontSize = "0.72rem";
                    subBreakdown.style.color = "var(--text-secondary)";
                    subBreakdown.style.marginTop = "3px";
                    subBreakdown.textContent = `Sem: ${p.semantic_percentage.toFixed(0)}% | Léx: ${p.lexical_percentage.toFixed(0)}%`;
                    tdSim.append(simBadge, subBreakdown);

                    tr.append(tdA, tdB, tdRel, tdSim);
                    sentTbody.appendChild(tr);
                });
            }
        }

        // 2. Tabla de Concordancias KWIC
        const kwicTbody = document.getElementById("kwicConcordancesBody");
        if (kwicTbody) {
            kwicTbody.innerHTML = "";
            if (kwicList.length === 0) {
                const tr = document.createElement("tr");
                tr.innerHTML = `<td colspan="3" class="empty-state-cell">No se detectaron concordancias contextuales KWIC entre ambos textos.</td>`;
                kwicTbody.appendChild(tr);
            } else {
                kwicList.forEach((k) => {
                    const tr = document.createElement("tr");

                    const tdTerm = document.createElement("td");
                    tdTerm.className = "word-cell";
                    tdTerm.textContent =
                        k.term_a === k.term_b ? k.term_a : `${k.term_a} ↔ ${k.term_b}`;
                    const relSub = document.createElement("div");
                    relSub.style.fontSize = "0.72rem";
                    relSub.style.fontWeight = "400";
                    relSub.style.color = "var(--text-secondary)";
                    relSub.textContent = k.relation;
                    tdTerm.appendChild(relSub);

                    const tdCtx = document.createElement("td");
                    const divA = document.createElement("div");
                    divA.className = "kwic-snippet";
                    divA.style.marginBottom = "4px";
                    divA.textContent = `A: ${k.context_a}`;
                    const divB = document.createElement("div");
                    divB.className = "kwic-snippet";
                    divB.textContent = `B: ${k.context_b}`;
                    tdCtx.append(divA, divB);

                    const tdSim = document.createElement("td");
                    tdSim.className = "num-col";
                    const pill = document.createElement("span");
                    pill.className = "diff-badge diff-badge-zero";
                    pill.textContent = `${k.context_similarity_percentage.toFixed(1)}%`;
                    tdSim.appendChild(pill);

                    tr.append(tdTerm, tdCtx, tdSim);
                    kwicTbody.appendChild(tr);
                });
            }
        }

        // 3. Dominios Temáticos y Conectores Discursivos
        const sharedDoms = discDom.shared_domains || [];
        setEl(
            "sharedDomainsText",
            sharedDoms.length > 0
                ? sharedDoms.join(" · ")
                : "Sin dominio compartido dominante"
        );

        const domContainer = document.getElementById("domainsContainer");
        if (domContainer) {
            domContainer.innerHTML = "";
            const domsA = discDom.domains_a || [];
            const domsB = discDom.domains_b || [];
            if (domsA.length === 0 && domsB.length === 0) {
                const p = document.createElement("p");
                p.className = "empty-chips-msg";
                p.textContent = "No se identificaron dominios específicos del catálogo.";
                domContainer.appendChild(p);
            } else {
                domsA.slice(0, 3).forEach((d) => {
                    const chip = document.createElement("span");
                    chip.className = "term-chip";
                    chip.textContent = `A: ${d.domain} (${d.hits} térm.)`;
                    domContainer.appendChild(chip);
                });
                domsB.slice(0, 3).forEach((d) => {
                    const chip = document.createElement("span");
                    chip.className = "term-chip";
                    chip.textContent = `B: ${d.domain} (${d.hits} térm.)`;
                    domContainer.appendChild(chip);
                });
            }
        }

        const connContainer = document.getElementById("discourseConnectorsContainer");
        if (connContainer) {
            connContainer.innerHTML = "";
            const catsA = discDom.discourse_a?.categories || [];
            const catsB = discDom.discourse_b?.categories || [];
            let anyMarker = false;
            catsA.forEach((c) => {
                if (c.count > 0) {
                    anyMarker = true;
                    const chip = document.createElement("span");
                    chip.className = "term-chip";
                    chip.textContent = `A [${c.category}]: ${c.markers.join(", ")}`;
                    connContainer.appendChild(chip);
                }
            });
            catsB.forEach((c) => {
                if (c.count > 0) {
                    anyMarker = true;
                    const chip = document.createElement("span");
                    chip.className = "term-chip";
                    chip.textContent = `B [${c.category}]: ${c.markers.join(", ")}`;
                    connContainer.appendChild(chip);
                }
            });
            if (!anyMarker) {
                const p = document.createElement("p");
                p.className = "empty-chips-msg";
                p.textContent =
                    "Oraciones directas sin conectores argumentativos compuestos detectados.";
                connContainer.appendChild(p);
            }
        }
    }

    /**
     * Renderiza el Escaneo Robusto de Palabras Repetidas, Familias Morfológicas,
     * Proximidad Cercana y Frases Recurrentes.
     */
    function renderDeepRepetitionScan(data) {
        const rep = data.deep_scan?.repetition_analysis;
        if (!rep) return;

        const docA = rep.doc_a || {};
        const docB = rep.doc_b || {};
        const simWords = rep.simultaneous_repeated_words || [];
        const sharedPhrases = rep.shared_phrases_ab || [];

        const setEl = (id, text) => {
            const el = document.getElementById(id);
            if (el) el.textContent = text;
        };

        setEl(
            "repRateAB",
            `${(docA.repetition_rate_percentage ?? 0).toFixed(1)}% / ${(docB.repetition_rate_percentage ?? 0).toFixed(1)}%`
        );
        setEl(
            "repExcessAB",
            `${docA.excess_occurrences ?? 0} en A / ${docB.excess_occurrences ?? 0} en B apariciones extra`
        );
        setEl(
            "repDistinctAB",
            `${docA.repeated_distinct_count ?? 0} en A · ${docB.repeated_distinct_count ?? 0} en B`
        );
        setEl(
            "repSimultaneousCount",
            `${simWords.length} palabras repetidas en ambos textos`
        );
        setEl(
            "repHapaxAB",
            `${docA.hapax_count ?? 0} (${(docA.hapax_percentage ?? 0).toFixed(0)}%) / ${docB.hapax_count ?? 0} (${(docB.hapax_percentage ?? 0).toFixed(0)}%)`
        );
        setEl(
            "repGuiraudAB",
            `${(docA.guiraud_richness_index ?? 0).toFixed(2)} / ${(docB.guiraud_richness_index ?? 0).toFixed(2)}`
        );
        const proxCount =
            (docA.close_proximity_alerts?.length ?? 0) +
            (docB.close_proximity_alerts?.length ?? 0);
        setEl("repProximityAlertsCount", `${proxCount} alertas de proximidad corta (≤ 6 pal.)`);

        const populateRepTable = (tbodyId, items) => {
            const tbody = document.getElementById(tbodyId);
            if (!tbody) return;
            tbody.innerHTML = "";
            if (!items || items.length === 0) {
                const tr = document.createElement("tr");
                tr.innerHTML = `<td colspan="4" class="empty-state-cell">Sin palabras repetidas (todas las palabras activas aparecen 1 sola vez).</td>`;
                tbody.appendChild(tr);
                return;
            }
            items.forEach((item) => {
                const tr = document.createElement("tr");

                const tdWord = document.createElement("td");
                tdWord.className = "word-cell";
                tdWord.textContent = item.word;

                const tdCnt = document.createElement("td");
                tdCnt.className = "num-col mono";
                tdCnt.textContent = `×${item.count} (${item.density_percentage.toFixed(1)}%)`;

                const tdDist = document.createElement("td");
                tdDist.className = "num-col mono";
                tdDist.textContent = `mín: ${item.min_distance_tokens} / med: ${item.avg_distance_tokens}`;

                const tdSev = document.createElement("td");
                const pill = document.createElement("span");
                pill.className = `severity-pill ${
                    item.severity === "Sobre-repetición Alta"
                        ? "severity-pill-high"
                        : item.severity === "Repetición Moderada"
                        ? "severity-pill-mod"
                        : ""
                }`;
                pill.textContent = item.is_close_repetition
                    ? `${item.severity} (Cercana)`
                    : item.severity;
                tdSev.appendChild(pill);

                tr.append(tdWord, tdCnt, tdDist, tdSev);
                tbody.appendChild(tr);
            });
        };

        populateRepTable("repeatedWordsBodyA", docA.exact_repeated || []);
        populateRepTable("repeatedWordsBodyB", docB.exact_repeated || []);

        // Familias morfológicas repetidas
        const morphBox = document.getElementById("morphRepetitionContainer");
        if (morphBox) {
            morphBox.innerHTML = "";
            const famsA = (docA.morphological_families || []).slice(0, 6);
            const famsB = (docB.morphological_families || []).slice(0, 6);
            if (famsA.length === 0 && famsB.length === 0) {
                const p = document.createElement("p");
                p.className = "empty-chips-msg";
                p.textContent = "No se detectaron raíces morfológicas repetidas.";
                morphBox.appendChild(p);
            } else {
                famsA.forEach((f) => {
                    const chip = document.createElement("span");
                    chip.className = "term-chip";
                    chip.textContent = `A [${f.stem}- ×${f.total_count}]: ${f.variants_summary}`;
                    morphBox.appendChild(chip);
                });
                famsB.forEach((f) => {
                    const chip = document.createElement("span");
                    chip.className = "term-chip";
                    chip.textContent = `B [${f.stem}- ×${f.total_count}]: ${f.variants_summary}`;
                    morphBox.appendChild(chip);
                });
            }
        }

        // Alertas de repetición por proximidad cercana (<= 6 palabras)
        const proxBox = document.getElementById("closeProximityContainer");
        if (proxBox) {
            proxBox.innerHTML = "";
            const alertsA = docA.close_proximity_alerts || [];
            const alertsB = docB.close_proximity_alerts || [];
            if (alertsA.length === 0 && alertsB.length === 0) {
                const p = document.createElement("p");
                p.className = "empty-chips-msg";
                p.textContent =
                    "Excelente distribución: no se detectaron palabras repetidas a corta distancia (≤ 6 palabras).";
                proxBox.appendChild(p);
            } else {
                alertsA.forEach((a) => {
                    const chip = document.createElement("span");
                    chip.className = "term-chip";
                    chip.title = a.snippet;
                    chip.textContent = `A: "${a.word_first} ... ${a.word_second}" (a ${a.distance_words} pal.)`;
                    proxBox.appendChild(chip);
                });
                alertsB.forEach((b) => {
                    const chip = document.createElement("span");
                    chip.className = "term-chip";
                    chip.title = b.snippet;
                    chip.textContent = `B: "${b.word_first} ... ${b.word_second}" (a ${b.distance_words} pal.)`;
                    proxBox.appendChild(chip);
                });
            }
        }

        // Frases repetidas (2 a 4 palabras) internas y compartidas
        const phrasesBox = document.getElementById("repeatedPhrasesContainer");
        if (phrasesBox) {
            phrasesBox.innerHTML = "";
            const phA = docA.repeated_phrases || [];
            const phB = docB.repeated_phrases || [];
            if (sharedPhrases.length === 0 && phA.length === 0 && phB.length === 0) {
                const p = document.createElement("p");
                p.className = "empty-chips-msg";
                p.textContent = "No se encontraron frases compuestas (2-4 palabras) repetidas.";
                phrasesBox.appendChild(p);
            } else {
                sharedPhrases.slice(0, 6).forEach((sp) => {
                    const chip = document.createElement("span");
                    chip.className = "term-chip";
                    chip.textContent = `A∩B: "${sp.phrase}" (A:×${sp.freq_a}, B:×${sp.freq_b})`;
                    phrasesBox.appendChild(chip);
                });
                phA.slice(0, 5).forEach((pa) => {
                    const chip = document.createElement("span");
                    chip.className = "term-chip";
                    chip.textContent = `Interna A: "${pa.phrase}" (×${pa.count})`;
                    phrasesBox.appendChild(chip);
                });
                phB.slice(0, 5).forEach((pb) => {
                    const chip = document.createElement("span");
                    chip.className = "term-chip";
                    chip.textContent = `Interna B: "${pb.phrase}" (×${pb.count})`;
                    phrasesBox.appendChild(chip);
                });
            }
        }
    }

    /**
     * Renderiza la sección 6.7 de Espacios Vectoriales (Coseno, Distancia L2, Producto Punto),
     * Clustering Semántico k-Means, Recuperación RAG k-NN, Auto-Atención y Subpalabras FastText.
     */
    function renderVectorSpaceAndGuideFeatures(data) {
        const deep = data.deep_scan || {};
        const emb = data.embeddings || {};
        const vsm = deep.vector_space_metrics || emb.vector_space_metrics || {};
        const evo = deep.representation_evolution || emb.representation_evolution || {};
        const poly = deep.polysemy_analysis || emb.polysemy_analysis || {};
        const rag = deep.rag_knn_retrieval || emb.rag_knn_retrieval || {};
        const km = deep.kmeans_clustering || emb.kmeans_clustering || {};
        const prod = deep.production_guidelines || emb.production_guidelines || {};

        const setEl = (id, text) => {
            const el = document.getElementById(id);
            if (el) el.textContent = text;
        };

        const cosPct = vsm.cosine_percentage ?? emb.full_document_raw_cosine_percentage ?? 0;
        const euclidDist = vsm.euclidean_distance_l2 ?? 0;
        const euclidSimPct = vsm.euclidean_similarity_percentage ?? 0;
        const dotVal = vsm.dot_product ?? 0;
        const oneHotSparsity = evo.one_hot?.sparsity_percentage ?? 0;
        const overlapPct = prod.chunking_overlap_percentage ?? 15;

        setEl("vsmCosineValue", `${cosPct.toFixed(2)}%`);
        setEl(
            "vsmOneHotSparsity",
            `One-Hot dim=${evo.one_hot?.dimensionality ?? 0} (${oneHotSparsity.toFixed(1)}% dispersión)`
        );
        setEl("vsmEuclideanDist", euclidDist.toFixed(4));
        setEl("vsmEuclideanSim", `Similitud L2: ${euclidSimPct.toFixed(1)}%`);
        setEl("vsmDotProduct", dotVal.toFixed(4));
        setEl(
            "vsmL2Verified",
            vsm.dot_equals_cosine_verified
                ? "Identidad ||u||=1 verificada (u·v = cos)"
                : `Normas L2: ${(vsm.norm_l2_a ?? 0).toFixed(3)} / ${(vsm.norm_l2_b ?? 0).toFixed(3)}`
        );
        setEl("vsmProdOverlap", `${overlapPct}% overlap · ${prod.active_dimensions ?? emb.dimensions ?? 128}-D`);
        setEl(
            "vsmAnnIndexes",
            `${(prod.ann_index_architectures || ["HNSW", "IVF-PQ"]).slice(0, 2).join(" / ")} · L2 GPU`
        );

        // 1. Clustering k-Means
        const kmBody = document.getElementById("kmeansClustersBody");
        if (kmBody) {
            kmBody.innerHTML = "";
            const clusters = km.clusters || [];
            setEl(
                "kmeansMetaSubtitle",
                `${km.algorithm || "KMeans"} · ${km.num_clusters ?? clusters.length} clusters sobre ${km.total_sentences_clustered ?? 0} oraciones.`
            );
            if (clusters.length === 0) {
                const tr = document.createElement("tr");
                tr.innerHTML = `<td colspan="3" class="empty-state-cell">Se requieren al menos 2 oraciones para ejecutar KMeans.</td>`;
                kmBody.appendChild(tr);
            } else {
                clusters.forEach((c) => {
                    const tr = document.createElement("tr");

                    const tdCluster = document.createElement("td");
                    const badge = document.createElement("span");
                    badge.className = "sent-idx-badge";
                    badge.textContent = `${c.label} (${c.size})`;
                    const compSub = document.createElement("div");
                    compSub.style.fontSize = "0.72rem";
                    compSub.style.color = "var(--text-secondary)";
                    compSub.style.marginTop = "4px";
                    compSub.textContent = c.composition;
                    tdCluster.append(badge, compSub);

                    const tdSents = document.createElement("td");
                    (c.sentences || []).slice(0, 4).forEach((sObj) => {
                        const p = document.createElement("div");
                        p.className = "kwic-snippet";
                        p.style.marginBottom = "3px";
                        const shortTxt = sObj.text.length > 95 ? `${sObj.text.slice(0, 95)}…` : sObj.text;
                        p.textContent = `[${sObj.doc}#${sObj.sentence_index}] ${shortTxt}`;
                        tdSents.appendChild(p);
                    });

                    const tdKws = document.createElement("td");
                    tdKws.className = "mono";
                    tdKws.style.fontSize = "0.78rem";
                    tdKws.textContent = (c.top_keywords || []).join(", ") || "—";

                    tr.append(tdCluster, tdSents, tdKws);
                    kmBody.appendChild(tr);
                });
            }
        }

        // 2. Polisemia (Auto-Atención) y Subpalabras FastText
        const polyBody = document.getElementById("polysemyTermsBody");
        if (polyBody) {
            polyBody.innerHTML = "";
            const terms = poly.polysemous_terms || [];
            setEl(
                "polysemyMetaSubtitle",
                `${poly.mechanism || "Self-Attention"} · ${poly.terms_evaluated ?? 0} términos recurrentes evaluados.`
            );
            if (terms.length === 0) {
                const tr = document.createElement("tr");
                tr.innerHTML = `<td colspan="3" class="empty-state-cell">No se detectaron términos polisémicos con divergencia contextual significativa.</td>`;
                polyBody.appendChild(tr);
            } else {
                terms.forEach((t) => {
                    const tr = document.createElement("tr");

                    const tdW = document.createElement("td");
                    tdW.className = "word-cell";
                    tdW.textContent = `${t.word} (×${t.occurrences_count})`;

                    const tdSenses = document.createElement("td");
                    const div1 = document.createElement("div");
                    div1.className = "kwic-snippet";
                    div1.style.marginBottom = "3px";
                    div1.textContent = `1) ${t.instance_a?.inferred_sense || ""}: "${t.instance_a?.snippet || ""}"`;
                    const div2 = document.createElement("div");
                    div2.className = "kwic-snippet";
                    div2.textContent = `2) ${t.instance_b?.inferred_sense || ""}: "${t.instance_b?.snippet || ""}"`;
                    tdSenses.append(div1, div2);

                    const tdDiv = document.createElement("td");
                    tdDiv.className = "num-col";
                    const pill = document.createElement("span");
                    pill.className = `severity-pill ${t.polysemy_detected ? "severity-pill-mod" : ""}`;
                    pill.textContent = `${(t.contextual_divergence_percentage ?? 0).toFixed(1)}%`;
                    tdDiv.appendChild(pill);

                    tr.append(tdW, tdSenses, tdDiv);
                    polyBody.appendChild(tr);
                });
            }
        }

        const ftBox = document.getElementById("fasttextSubwordsContainer");
        if (ftBox) {
            ftBox.innerHTML = "";
            const sample = (emb.word_vectors_sample || []).slice(0, 4);
            if (sample.length === 0) {
                const p = document.createElement("p");
                p.className = "empty-chips-msg";
                p.textContent = "Sin subpalabras disponibles.";
                ftBox.appendChild(p);
            } else {
                sample.forEach((item) => {
                    const chip = document.createElement("span");
                    chip.className = "term-chip";
                    const subs = (item.fasttext_subwords || []).slice(0, 6).join(", ");
                    chip.textContent = `${item.word}: [${subs}]`;
                    ftBox.appendChild(chip);
                });
            }
        }

        // 3. Tabla RAG k-NN Retrieval
        const ragBody = document.getElementById("ragKnnBody");
        if (ragBody) {
            ragBody.innerHTML = "";
            const rows = rag.results || [];
            if (rows.length === 0) {
                const tr = document.createElement("tr");
                tr.innerHTML = `<td colspan="4" class="empty-state-cell">No hay fragmentos suficientes para recuperar vecinos más cercanos k-NN.</td>`;
                ragBody.appendChild(tr);
            } else {
                rows.slice(0, 8).forEach((r) => {
                    const tr = document.createElement("tr");

                    const tdQ = document.createElement("td");
                    tdQ.className = "sentence-cell";
                    const qBadge = document.createElement("span");
                    qBadge.className = "sent-idx-badge";
                    qBadge.textContent = `Q#${r.query_index}`;
                    const qTxt = document.createElement("span");
                    qTxt.textContent = r.query_text;
                    tdQ.append(qBadge, qTxt);

                    const tdR = document.createElement("td");
                    tdR.className = "sentence-cell";
                    const rBadge = document.createElement("span");
                    rBadge.className = "sent-idx-badge";
                    rBadge.textContent = `DocB#${r.retrieved_index}`;
                    const rTxt = document.createElement("span");
                    rTxt.textContent = r.retrieved_text;
                    tdR.append(rBadge, rTxt);

                    const tdCos = document.createElement("td");
                    tdCos.className = "num-col";
                    const cosBadge = document.createElement("span");
                    cosBadge.className = "diff-badge diff-badge-zero";
                    cosBadge.textContent = `${(r.cosine_percentage ?? 0).toFixed(1)}%`;
                    tdCos.appendChild(cosBadge);

                    const tdDist = document.createElement("td");
                    tdDist.className = "num-col mono";
                    tdDist.textContent = (r.vector_distance_l2 ?? 0).toFixed(4);

                    tr.append(tdQ, tdR, tdCos, tdDist);
                    ragBody.appendChild(tr);
                });
            }
        }
    }

    /**
     * Renderiza la sección desplegable de Métricas Avanzadas, Fragmentos Literales y Raíces Morfológicas.
     */
    function renderAdvancedMetricsAndFragments(data) {
        const { jaccard, cosine, embeddings, comparative_analysis: comp } = data;
        const embCal = embeddings.calibrated_similarity_percentage ?? embeddings.similarity_percentage ?? 0;
        const embRaw = embeddings.raw_similarity_percentage ?? embCal;
        const bs = embeddings.bertscore || { f1_percentage: 0, precision: 0, recall: 0 };

        const setEl = (id, text) => {
            const el = document.getElementById(id);
            if (el) el.textContent = text;
        };

        setEl("advEmbCalRaw", `${embCal.toFixed(2)}% (cal) / ${embRaw.toFixed(2)}% (crudo)`);
        setEl(
            "advBertScore",
            `F1: ${(bs.f1_percentage ?? 0).toFixed(2)}% (P: ${(bs.precision ?? 0).toFixed(3)}, R: ${(bs.recall ?? 0).toFixed(3)})`
        );
        setEl("advChunkSim", `${(embeddings.chunk_alignment_percentage ?? 0).toFixed(2)}%`);
        setEl("advCharNgramCos", `${(cosine.char_ngram_cosine_percentage ?? 0).toFixed(2)}%`);
        setEl("advNgram12Cos", `${(cosine.ngram_1_2_cosine_percentage ?? 0).toFixed(2)}%`);
        setEl(
            "advDiceContainment",
            `${(jaccard.dice_percentage ?? 0).toFixed(2)}% (Dice) / ${(jaccard.containment_percentage ?? 0).toFixed(2)}% (Containment)`
        );
        setEl("advWeightedJaccard", `${(jaccard.weighted_jaccard_percentage ?? 0).toFixed(2)}%`);
        setEl(
            "advShingleJaccard",
            `${(jaccard.bigram_jaccard_percentage ?? 0).toFixed(2)}% (2-gram) / ${(jaccard.trigram_jaccard_percentage ?? 0).toFixed(2)}% (3-gram)`
        );

        const exactInfo = data.exact_matching_fragments || comp?.exact_matching_fragments || {};
        const lcs = exactInfo.longest_common_substring || "";
        const lcsCount = exactInfo.longest_word_count || 0;
        setEl(
            "advLongestSubstring",
            lcs ? `"${lcs}" (${lcsCount} palabras)` : "(sin secuencia común destacable)"
        );

        const fragContainer = document.getElementById("exactFragmentsContainer");
        if (fragContainer) {
            fragContainer.innerHTML = "";
            const frags = exactInfo.fragments || [];
            if (frags.length === 0) {
                const p = document.createElement("p");
                p.className = "empty-chips-msg";
                p.textContent = "No se detectaron fragmentos exactos de 5 o más palabras consecutivas.";
                fragContainer.appendChild(p);
            } else {
                frags.forEach((f) => {
                    const chip = document.createElement("span");
                    chip.className = "term-chip";
                    chip.textContent = `"${f.fragment}" (${f.word_count} pal.)`;
                    fragContainer.appendChild(chip);
                });
            }
        }

        const morphContainer = document.getElementById("morphologicalMatchesContainer");
        if (morphContainer) {
            morphContainer.innerHTML = "";
            const morphs = comp?.morphological_matches || jaccard.morphological_matches || [];
            if (morphs.length === 0) {
                const p = document.createElement("p");
                p.className = "empty-chips-msg";
                p.textContent = "Todas las coincidencias léxicas ocurrieron con la misma forma escrita.";
                morphContainer.appendChild(p);
            } else {
                morphs.forEach((m) => {
                    const chip = document.createElement("span");
                    chip.className = "term-chip";
                    chip.textContent = `${m.word_a} ↔ ${m.word_b} (raíz: ${m.stem}-)`;
                    morphContainer.appendChild(chip);
                });
            }
        }
    }

    /**
     * Renderiza la tabla de Puentes Semánticos (palabras distintas con cercanía vectorial).
     */
    function renderSemanticBridgesTable(matches) {
        const tbody = document.getElementById("semanticBridgesBody");
        const badge = document.getElementById("semanticMatchesCountBadge");
        tbody.innerHTML = "";
        badge.textContent = matches.length;

        if (!matches || matches.length === 0) {
            const tr = document.createElement("tr");
            tr.innerHTML = `<td colspan="4" class="empty-state-cell">No se detectaron pares de palabras distintas con afinidad semántica/morfológica adicional a las coincidencias exactas.</td>`;
            tbody.appendChild(tr);
            return;
        }

        matches.forEach((m) => {
            const tr = document.createElement("tr");

            const tdA = document.createElement("td");
            tdA.className = "word-cell";
            tdA.textContent = m.word_a;

            const tdB = document.createElement("td");
            tdB.className = "word-cell";
            tdB.textContent = m.word_b;

            const tdRel = document.createElement("td");
            tdRel.textContent = m.relationship;

            const tdSim = document.createElement("td");
            tdSim.className = "num-col";
            const pill = document.createElement("span");
            pill.className = "diff-badge diff-badge-zero";
            pill.textContent = `${m.similarity_percentage.toFixed(1)}%`;
            tdSim.appendChild(pill);

            tr.append(tdA, tdB, tdRel, tdSim);
            tbody.appendChild(tr);
        });
    }

    /**
     * Renderiza la previsualización de los vectores densos de Embeddings por documento y por palabra.
     */
    function renderEmbeddingVectorsBox(embeddings) {
        document.getElementById("embeddingProviderSubtitle").textContent =
            `${embeddings.provider} · Vectores unitarios de ${embeddings.dimensions} dimensiones.`;

        const prevA = embeddings.vector_preview_a || [];
        const prevB = embeddings.vector_preview_b || [];
        document.getElementById("vecPreviewA").textContent =
            prevA.length > 0 ? `[${prevA.join(", ")}, ...]` : "[0.0]";
        document.getElementById("vecPreviewB").textContent =
            prevB.length > 0 ? `[${prevB.join(", ")}, ...]` : "[0.0]";

        const tbody = document.getElementById("wordVectorsSampleBody");
        tbody.innerHTML = "";
        const samples = embeddings.word_vectors_sample || [];

        if (samples.length === 0) {
            const tr = document.createElement("tr");
            tr.innerHTML = `<td colspan="3" class="empty-state-cell">Sin palabras vectorizadas.</td>`;
            tbody.appendChild(tr);
            return;
        }

        samples.forEach((item) => {
            const tr = document.createElement("tr");

            const tdWord = document.createElement("td");
            tdWord.className = "word-cell";
            tdWord.textContent = item.word;

            const tdStem = document.createElement("td");
            tdStem.className = "mono";
            tdStem.textContent = `${item.stem}-`;

            const tdVec = document.createElement("td");
            tdVec.className = "mono";
            tdVec.textContent = `[${item.vector_preview.join(", ")}, ...]`;

            tr.append(tdWord, tdStem, tdVec);
            tbody.appendChild(tr);
        });
    }

    /**
     * Renderiza la tabla de palabras más frecuentes de un documento individual.
     */
    function renderTopWordsTable(tbodyId, items) {
        const tbody = document.getElementById(tbodyId);
        tbody.innerHTML = "";

        if (!items || items.length === 0) {
            const tr = document.createElement("tr");
            tr.innerHTML = `<td colspan="4" class="empty-state-cell">Sin términos activos registrados.</td>`;
            tbody.appendChild(tr);
            return;
        }

        items.forEach((item) => {
            const tr = document.createElement("tr");

            const tdWord = document.createElement("td");
            tdWord.className = "word-cell";
            tdWord.textContent = item.word;

            const tdCount = document.createElement("td");
            tdCount.className = "num-col";
            tdCount.textContent = item.count;

            const tdRel = document.createElement("td");
            tdRel.className = "num-col";
            tdRel.textContent = `${item.relative_percentage.toFixed(2)}%`;

            const tdTfidf = document.createElement("td");
            tdTfidf.className = "num-col";
            tdTfidf.textContent = item.tfidf.toFixed(4);

            tr.append(tdWord, tdCount, tdRel, tdTfidf);
            tbody.appendChild(tr);
        });
    }

    /**
     * Renderiza la Tabla de Coincidencias (6.3):
     * Columnas: Palabra, Frecuencia en A, Frecuencia en B, Diferencia de frecuencia, Peso TF-IDF.
     */
    function renderSharedTermsTable(termsList) {
        const tbody = document.getElementById("sharedTermsBody");
        tbody.innerHTML = "";

        if (!termsList || termsList.length === 0) {
            const tr = document.createElement("tr");
            tr.innerHTML = `<td colspan="5" class="empty-state-cell">No se encontraron palabras idénticas coincidentes para el criterio actual.</td>`;
            tbody.appendChild(tr);
            return;
        }

        termsList.forEach((item) => {
            const tr = document.createElement("tr");

            const tdWord = document.createElement("td");
            tdWord.className = "word-cell";
            tdWord.textContent = item.word;

            const tdFreqA = document.createElement("td");
            tdFreqA.className = "num-col";
            tdFreqA.textContent = item.freq_a;

            const tdFreqB = document.createElement("td");
            tdFreqB.className = "num-col";
            tdFreqB.textContent = item.freq_b;

            const tdDiff = document.createElement("td");
            tdDiff.className = "num-col";
            const badge = document.createElement("span");
            badge.className = `diff-badge ${item.abs_diff === 0 ? "diff-badge-zero" : ""}`;
            const signed = item.diff > 0 ? `+${item.diff}` : `${item.diff}`;
            badge.textContent =
                item.abs_diff === 0 ? "0 (Igual)" : `${item.abs_diff} (${signed})`;
            tdDiff.appendChild(badge);

            const tdTfidf = document.createElement("td");
            tdTfidf.className = "num-col";
            tdTfidf.textContent = `${item.tfidf_a.toFixed(3)} / ${item.tfidf_b.toFixed(3)}`;

            tr.append(tdWord, tdFreqA, tdFreqB, tdDiff, tdTfidf);
            tbody.appendChild(tr);
        });
    }

    /**
     * Renderiza las etiquetas o chips visuales de palabras exclusivas (6.4).
     */
    function renderExclusiveChips(containerId, badgeId, items, emptyMessage) {
        const container = document.getElementById(containerId);
        const badge = document.getElementById(badgeId);
        container.innerHTML = "";
        badge.textContent = items.length;

        if (!items || items.length === 0) {
            const emptyP = document.createElement("p");
            emptyP.className = "empty-chips-msg";
            emptyP.textContent = emptyMessage;
            container.appendChild(emptyP);
            return;
        }

        items.forEach((item) => {
            const chip = document.createElement("span");
            chip.className = "term-chip";

            const wordSpan = document.createElement("span");
            wordSpan.textContent = item.word;

            const freqSpan = document.createElement("span");
            freqSpan.className = "chip-freq";
            freqSpan.title = `Frecuencia: ${item.count} | TF-IDF: ${item.tfidf.toFixed(3)}`;
            freqSpan.textContent = `×${item.count}`;

            chip.append(wordSpan, freqSpan);
            container.appendChild(chip);
        });
    }

    /**
     * GRÁFICO 1: Comparación de similitud (6 Técnicas de PLN o 3 algoritmos principales)
     */
    function renderSimilarityChart(embeddingPct, cosinePct, jaccardPct, nlpMatrix = []) {
        const ctx = document.getElementById("similarityChart");
        if (!ctx || !window.Chart) return;

        if (chartInstances.similarity) {
            chartInstances.similarity.destroy();
        }

        const hasMatrix = Array.isArray(nlpMatrix) && nlpMatrix.length === 6;
        const labels = hasMatrix
            ? nlpMatrix.map((row) => row.technique)
            : [
                  "Embeddings Semánticos (IA)",
                  "Similitud Coseno (TF-IDF)",
                  "Índice de Jaccard",
              ];
        const values = hasMatrix
            ? nlpMatrix.map((row) => row.percentage)
            : [embeddingPct, cosinePct, jaccardPct];
        const bgColors = hasMatrix
            ? [
                  "#B8E6DF",
                  PALETTE.turquoiseSecondary,
                  "#46A99B",
                  PALETTE.turquoisePrimary,
                  "#317D73",
                  PALETTE.textPrimary,
              ]
            : [
                  PALETTE.textPrimary,
                  PALETTE.turquoisePrimary,
                  PALETTE.turquoiseSecondary,
              ];

        chartInstances.similarity = new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: "Porcentaje de Similitud (%)",
                        data: values,
                        backgroundColor: bgColors,
                        borderColor: PALETTE.textPrimary,
                        borderWidth: 1.5,
                        borderRadius: 8,
                        maxBarThickness: 58,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false,
                    },
                    tooltip: {
                        backgroundColor: PALETTE.textPrimary,
                        padding: 12,
                        callbacks: {
                            label: (context) => ` Similitud: ${context.parsed.y.toFixed(2)}%`,
                        },
                    },
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: {
                            color: PALETTE.border,
                        },
                        ticks: {
                            callback: (val) => `${val}%`,
                        },
                    },
                    x: {
                        grid: {
                            display: false,
                        },
                        ticks: {
                            color: PALETTE.textPrimary,
                            font: {
                                weight: "600",
                            },
                        },
                    },
                },
            },
        });
    }

    /**
     * GRÁFICO 2: Distribución de vocabulario (Compartidas, Exclusivas A, Exclusivas B)
     */
    function renderVocabularyChart(sharedCount, exclusiveACount, exclusiveBCount) {
        const ctx = document.getElementById("vocabularyChart");
        if (!ctx || !window.Chart) return;

        if (chartInstances.vocabulary) {
            chartInstances.vocabulary.destroy();
        }

        const total = sharedCount + exclusiveACount + exclusiveBCount;
        const chartData =
            total > 0 ? [sharedCount, exclusiveACount, exclusiveBCount] : [1, 0, 0];
        const chartLabels =
            total > 0
                ? [
                      "Palabras compartidas (A ∩ B)",
                      "Exclusivas del Texto A (A \\ B)",
                      "Exclusivas del Texto B (B \\ A)",
                  ]
                : ["Sin vocabulario activo", "Exclusivas A", "Exclusivas B"];

        chartInstances.vocabulary = new Chart(ctx, {
            type: "doughnut",
            data: {
                labels: chartLabels,
                datasets: [
                    {
                        data: chartData,
                        backgroundColor: [
                            PALETTE.turquoisePrimary,
                            PALETTE.turquoiseSecondary,
                            PALETTE.textPrimary,
                        ],
                        borderColor: PALETTE.cardBg,
                        borderWidth: 3,
                        hoverOffset: 6,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: "62%",
                plugins: {
                    legend: {
                        position: "bottom",
                        labels: {
                            padding: 16,
                            usePointStyle: true,
                            pointStyle: "circle",
                            color: PALETTE.textPrimary,
                        },
                    },
                    tooltip: {
                        backgroundColor: PALETTE.textPrimary,
                        padding: 12,
                        callbacks: {
                            label: (context) => {
                                if (total === 0) return " 0 términos";
                                const value = context.parsed;
                                const pct = ((value / total) * 100).toFixed(1);
                                return ` ${value} términos (${pct}%)`;
                            },
                        },
                    },
                },
            },
        });
    }

    /**
     * GRÁFICO 3: Frecuencia comparativa de palabras (Texto A vs. Texto B)
     */
    function renderFrequencyChart(comparativeData) {
        const ctx = document.getElementById("frequencyChart");
        if (!ctx || !window.Chart) return;

        if (chartInstances.frequency) {
            chartInstances.frequency.destroy();
        }

        const labels = comparativeData?.labels || [];
        const freqA = comparativeData?.freq_a || [];
        const freqB = comparativeData?.freq_b || [];

        chartInstances.frequency = new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: "Frecuencia en Texto A",
                        data: freqA,
                        backgroundColor: PALETTE.turquoisePrimary,
                        borderColor: PALETTE.turquoisePrimary,
                        borderWidth: 1,
                        borderRadius: 6,
                    },
                    {
                        label: "Frecuencia en Texto B",
                        data: freqB,
                        backgroundColor: PALETTE.turquoiseSecondary,
                        borderColor: PALETTE.turquoisePrimary,
                        borderWidth: 1,
                        borderRadius: 6,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: "top",
                        align: "end",
                        labels: {
                            usePointStyle: true,
                            pointStyle: "rectRounded",
                            color: PALETTE.textPrimary,
                        },
                    },
                    tooltip: {
                        backgroundColor: PALETTE.textPrimary,
                        padding: 12,
                    },
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            precision: 0,
                        },
                        grid: {
                            color: PALETTE.border,
                        },
                        title: {
                            display: true,
                            text: "Frecuencia de aparición",
                            color: PALETTE.textSecondary,
                        },
                    },
                    x: {
                        grid: {
                            display: false,
                        },
                        ticks: {
                            color: PALETTE.textPrimary,
                            font: {
                                weight: "500",
                            },
                        },
                    },
                },
            },
        });
    }

    // =========================================================================
    // REGISTRO DE EVENTOS DE USUARIO
    // =========================================================================

    textA.addEventListener("input", () => {
        fileBadgeA.classList.add("hidden");
        updateCounters();
    });
    textB.addEventListener("input", () => {
        fileBadgeB.classList.add("hidden");
        updateCounters();
    });

    // Subida de archivos por botón
    fileInputA.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileUpload(e.target.files[0], "A");
            e.target.value = "";
        }
    });
    fileInputB.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileUpload(e.target.files[0], "B");
            e.target.value = "";
        }
    });

    // Drag & Drop en ambas tarjetas
    setupDragAndDrop(cardA, "A");
    setupDragAndDrop(cardB, "B");

    compareBtn.addEventListener("click", () => performComparison(true));

    // Botón Intercambiar textos
    swapBtn.addEventListener("click", () => {
        const temp = textA.value;
        textA.value = textB.value;
        textB.value = temp;

        const tempBadgeText = fileBadgeA.textContent;
        const tempBadgeHidden = fileBadgeA.classList.contains("hidden");
        fileBadgeA.textContent = fileBadgeB.textContent;
        fileBadgeA.classList.toggle("hidden", fileBadgeB.classList.contains("hidden"));
        fileBadgeB.textContent = tempBadgeText;
        fileBadgeB.classList.toggle("hidden", tempBadgeHidden);

        updateCounters();

        if (!resultsPanel.classList.contains("hidden") && textA.value.trim() && textB.value.trim()) {
            performComparison(false);
        }
    });

    // Botón Limpiar
    clearBtn.addEventListener("click", () => {
        textA.value = "";
        textB.value = "";
        sampleSelect.value = "";
        sharedSearchInput.value = "";
        fileBadgeA.classList.add("hidden");
        fileBadgeB.classList.add("hidden");
        hideValidationError();
        updateCounters();
        resultsPanel.classList.add("hidden");
        textA.focus();
    });

    // Cerrar alerta
    closeAlertBtn.addEventListener("click", hideValidationError);

    // Modal "¿Cómo funciona?"
    if (openGuideBtn && closeGuideBtn && guideModal) {
        openGuideBtn.addEventListener("click", () => {
            guideModal.classList.remove("hidden");
        });
        closeGuideBtn.addEventListener("click", () => {
            guideModal.classList.add("hidden");
        });
        guideModal.addEventListener("click", (e) => {
            if (e.target === guideModal) {
                guideModal.classList.add("hidden");
            }
        });
    }

    // Recalcular si se cambia la configuración de stopwords o el perfil principal y ya hay resultados visibles
    removeStopwordsToggle.addEventListener("change", () => {
        if (!resultsPanel.classList.contains("hidden") && textA.value.trim() && textB.value.trim()) {
            performComparison(false);
        }
    });

    if (profileSelect) {
        profileSelect.addEventListener("change", () => {
            if (!resultsPanel.classList.contains("hidden") && textA.value.trim() && textB.value.trim()) {
                performComparison(false);
            }
        });
    }

    // Selector de casos de estudio académicos
    sampleSelect.addEventListener("change", (e) => {
        const key = e.target.value;
        if (!key || !academicSamplesCache[key]) return;
        const sample = academicSamplesCache[key];
        textA.value = sample.text_a;
        textB.value = sample.text_b;
        fileBadgeA.classList.add("hidden");
        fileBadgeB.classList.add("hidden");
        updateCounters();
        performComparison(true);
    });

    // Buscador en vivo para la Tabla de Coincidencias
    sharedSearchInput.addEventListener("input", (e) => {
        const query = e.target.value.trim().toLowerCase();
        if (!query) {
            renderSharedTermsTable(currentSharedTerms);
            return;
        }
        const filtered = currentSharedTerms.filter((item) =>
            item.word.toLowerCase().includes(query)
        );
        renderSharedTermsTable(filtered);
    });

    // Inicializar contadores y verificar estado del sistema
    updateCounters();
    initializeSystem();
});
