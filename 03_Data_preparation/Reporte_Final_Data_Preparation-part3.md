# Reporte de Auditoría: Fase 3 - Preparación de los Datos (CRISP-DM)

**Fecha:** 25 de Enero de 2026
**Responsable:** Agente AI (Antigravity)
**Notebook Auditado:** `03_Data_preparation/CorrecciónDescriptivo_part2.ipynb`

---

## Resumen Ejecutivo

Esta fase transforma los datos crudos validados en la Fase 2 (Data Understanding) en un conjunto de datos analítico (ABT) listo para el modelado. El hito principal es el enriquecimiento del dataset con **69 vectores de conocimiento técnico** derivados de mallas curriculares, permitiendo analizar la influencia de las Hard Skills en la empleabilidad.

---

## 3.1 Selección de Datos (Select Data)

### Criterios de Inclusión
Se seleccionaron las siguientes características para el modelado final:
1.  **Habilidades Blandas (7 Variables):** `S1` a `S7` (Autopercepción).
2.  **Perfil Demográfico:** `Edad`, `Género`, `Facultad`.
3.  **Variable Objetivo ($T, E$):** Mapeo AFT interval-censored definido en Fase 2.
4.  **Vectores Técnicos (69 Variables):** Variables sintéticas derivadas de BERT que representan la exposición académica a tópicos específicos (ej. "Python", "Gestión de Proyectos").

> [!NOTE]
> **Decisión de Diseño:** Se excluyeron variables de texto libre (`TXT_Hard_Skills`) crudas para el modelo, utilizándolas solo para generar la matriz de vectores numéricos.

---

## 3.2 Limpieza de Datos (Clean Data)

### 3.2.1 Imputación de Valores Perdidos
Se aplicaron las estrategias definidas metodológicamente en la Fase 2:

1.  **Habilidades Blandas (S1-S6):**
    - **Acción (Código):** Se ejecutó la imputación por mediana para los casos residuales (<0.1%) detectados durante la carga.
    - **Evidencia Visual:** `03_Data_preparation/figures/11_diagnostico_nulos_soft_skills.png` (Confirma 0% de nulos en el dataset final).

2.  **Caso Crítico: S7 (Inglés):**
    - **Estado de Entrada:** La variable `S7_Ingles` se identificó como presente en el dataset de carga (`dataset_part1_analisis.csv`).
    - **Validación:** Se confirmó su integridad total (0% de nulos) mediante código de inspección.

3.  **Vectores Técnicos:**
    - **Acción (Código):** Se aplicó el relleno de `NaN` con ceros (`fillna(0.0)`) al generar la matriz `X_topics` para el modelado.
    - **Evidencia Visual mejorada:** `03_Data_preparation/figures/09_diagnostico_nulos_vectores.png` muestra explícitamente el conteo de nulos (0) para las 69 dimensiones, confirmando una matriz densa.

### 3.2.2 Normalización de Género
- Se estandarizaron categorías diversas a un binario `{M, F}`.

---

## 3.3 Construcción de Datos (Construct Data)

### 3.3.1 Ingeniería de Características: Vectores Académicos
Se integró una **matriz de 69 dimensiones técnicas** proporcionada originalmente por **Diego Rafael Arias Sarango (Cohorte 2025-B)**.
- **Función:** Reforzar las Hard Skills mencionadas en el cuestionario del Experimento B.
- **Método (BERT):** Se utilizó NLP para proyectar las habilidades declaradas hacia este espacio vectorial definido por Arias Sarango.

**Interpretación del Heatmap (Validación de Independencia):**
La Figura `03_Data_preparation/figures/06_heatmap_soft_vs_tech.png` presenta la correlación de Spearman ($\rho_s$) controlada por FDR:
1.  **Independencia Práctica:** Predominan coeficientes cercanos a cero, indicando que las Soft Skills autodeclaradas son casi independientes del perfil curricular.
2.  **Transversalidad:** Este patrón respalda la hipótesis de que las habilidades blandas son transversales y complementarias al perfil técnico, no meros proxys de la carrera.
3.  **Micro-patrones:** Se detectan asociaciones débiles pero significativas (p.ej., *Gestión de Proyectos* con *Protocolos de Comunicación*), sugiriendo leves variaciones por subdominio.

### 3.3.2 Validación de Poder Predictivo (Cox Univariante)
Se evaluó el impacto individual de los tópicos técnicos en la velocidad de inserción (FDR $q<0.05$).
- **Figura Generada:** `03_Data_preparation/figures/Univariate_Cox_Plot_Top15.png`
- **Interpretación del Hallazgo:**
    *   **Inserción más Rápida ($log(HR) > 0$):** Tópicos de perfil digital y herramientas (*ETL/LaTeX*, *Visualización de Datos*, *AWS*) muestran HR > 1, indicando una mayor tasa instantánea de inserción.
    *   **Inserción más Lenta ($log(HR) < 0$):** Tópicos como *Finanzas* o *Normativa Ambiental* muestran HR < 1.
    *   **Conclusión:** Existen marcadores curriculares específicos que aceleran o retrasan significativamente el tiempo de búsqueda en esta muestra.

### 3.3.3 Análisis Léxico de Hard Skills (WordCloud)
Se reforzó la validación de los vectores técnicos mediante una inspección visual de los términos más frecuentes en el campo de texto libre (`TXT_Hard_Skills`).
- **Figura Generada:** `03_Data_preparation/figures/B_wordcloud_hard_skills_clean.png`
- **Interpretación:** La nube de palabras confirma que los términos predominantes en la autodescripción técnica de los graduados (ej. *manejo, contabilidad, diseño, sistemas, gestión*) son coherentes con los tópicos vectoriales mapeados, validando la calidad del contenido semántico base antes de la vectorización con BERT.

---

## 3.5 Inventario Completo de Salidas Gráficas (Appendix)

Además de las evidencias principales, se generaron las siguientes visualizaciones exploratorias, disponibles en la carpeta `03_Data_preparation/figures/`, para uso en análisis suplementarios:

**A. Análisis de Censura:**
- `Tasa_de_Censura_por_Cohorte.png`: Diagnóstico de eventos vs. censura por periodo.
- `cap_para_infinito.png`: Validación del tratamiento de censura a la derecha (cota infinita).
- `Conteo_por_intervalos_T_Lower_T_Upper.png`: Verificación de la estructura del target de supervivencia.

**B. Curvas de Supervivencia (Kaplan-Meier no paramétrico):**
- `Curvas_KM_aproximadas_cotas_fontweightbold.png`: Visión global de la supervivencia.
- **Desglose por Facultad:** `fig_cap3_km_bounds_Facultad_*.png` (FCA, FC, FGP, FIC, FIE, FIM, FIQ, FIS).
- **Desglose por Género:** `fig_cap3_km_bounds_Genero_M.png`, `_F.png`.
- **Desglose por Cohorte:** `fig_cap3_km_bounds_Cohorte_*.png`.

**C. Validación de Tópicos (Terciles):**
- `km_terciles_GLOBAL__*.png`: Curvas KM segmentadas por nivel de exposición a tópicos clave (*aws, etl, visualización*), confirmando su relevancia predictiva preliminar.

---

## 3.4 Formato de Datos (Format Data)


### Estructura Final del Dataset (`df_model`)
El dataset resultante cumple con los requisitos del algoritmo AFT, aunque requiere pasos finales de transformación en la siguiente fase:
- **Sin Nulos:** Matriz densa.
- **Formato Mixto:** Se mantienen variables categóricas como `carrera` en formato texto/label (sin One-Hot Encoding aún), lo cual es lo esperado para este punto de control.
- **Target Definido:** Columna `duration` (tiempo) y `event` (censura) listas.

**Distribución del Target Final:**
La figura `03_Data_preparation/figures/Conteo_por_intervalos.png` confirma la estructura de censura por intervalos, con una mezcla saludable de observaciones exactas y censuradas (intervalos abiertos a infinito).
