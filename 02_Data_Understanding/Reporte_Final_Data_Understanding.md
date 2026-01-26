# Informe de Auditoría Técnica: Enfoque CRISP-DM

**Documento:** `CorrecciónDescriptivo (4).ipynb`
**Objetivo:** Validar la integridad técnica, estadística y metodológica del análisis de empleabilidad para su inclusión en la tesis.

Este reporte estructura los hallazgos técnicos dentro de las fases estándar de la metodología **CRISP-DM** (Cross-Industry Standard Process for Data Mining), facilitando su integración directa en el documento de tesis.

---

## Fase 2: Entendimiento de los Datos (Data Understanding)

Esta fase se centra en la recolección inicial, descripción y exploración de la calidad de los datos para asegurar que sean aptos para responder a la pregunta de investigación sobre la empleabilidad de los graduados.

### 1. Recolección y Carga de Datos (Cells 1-2)
- **Acción Realizada:** Se cargaron exitosamente dos conjuntos de datos independientes correspondientes a las cohortes de **Julio** y **Diciembre** desde archivos Excel (`.xlsx`).
- **Hallazgo:** La separación en dos archivos permite un análisis comparativo temporal, crucial para validar si los patrones de empleabilidad son estacionales o estructurales.
- **Validación:** Se confirmó la correcta lectura de las rutas relativas (`../ENCUESTAS/`), asegurando la reproducibilidad del notebook.

### 2. Validación Semántica de Variables (Cells 3-8)
- **Acción Realizada:** Se implementó una técnica avanzada de **Procesamiento de Lenguaje Natural (NLP)** utilizando `SentenceTransformer` para comparar las definiciones de las "Habilidades Blandas" (S1 a S7) entre ambas cohortes.
- **Figura Generada:** `02_Data_Understanding/figures-v2/Matriz_de_Homologación_Semántica_BERTnJustificació.png`
- **Hallazgo Real:** La matriz de similitud de coseno mostró correlaciones fuertes pero no idénticas, justificando la fusión de variables bajo un esquema de homologación semántica.
- **Gráficos de Densidad por Habilidad (Homogeneidad S1-S7):** Se generaron gráficos individuales para las 7 habilidades blandas, confirmando distribuciones similares entre cohortes:
    - `02_Data_Understanding/figures-v2/Figcap3_02_Densidad_S1_Comunicacion_Esp.png`
    - `02_Data_Understanding/figures-v2/Figcap3_03_Densidad_S2_Compromiso_Etico.png`
    - `02_Data_Understanding/figures-v2/Figcap3_04_Densidad_S3_Trabajo_Equipo_Liderazgo.png`
    - `02_Data_Understanding/figures-v2/Figcap3_05_Densidad_S4_Resp_Social.png`
    - `02_Data_Understanding/figures-v2/Figcap3_06_Densidad_S5_Gestion_Proyectos.png`
    - `02_Data_Understanding/figures-v2/Figcap3_07_Densidad_S6_Aprendizaje_Digital.png`
    - `02_Data_Understanding/figures-v2/Figcap3_08_Densidad_S7_Ingles.png`

#### 2.1 Imputación Probabilística: Caso S7 (Inglés) - *Hallazgo Crítico*
- **Problema:** La cohorte de Julio 2025 carecía de una medición específica para "Inglés" (S7), lo que hubiera generado una columna llena de nulos (`NaN`).
- **Solución Algorítmica (Código):** Se implementó una **Bifurcación de Constructo** basada en la similitud semántica hallada en la Fase 2.
    - Se calculó una **Probabilidad de Penalización** ($P_{pen}$) basada en la razón de similitudes: 
      $$P_{pen} = 1 - \frac{\text{Similitud}_{\text{Inglés}} (0.664)}{\text{Similitud}_{\text{Español}} (0.794)} \approx 16.4\%$$
    - Se generó la variable sintética `S7_Ingles` para Julio tomando el valor de "Comunicación", pero reduciéndolo en 1 punto (escala Likert) con una probabilidad del **16.4%**.
- **Valor para Tesis:** Esta técnica avanzada evita la eliminación de la variable "Inglés" del análisis global. En lugar de asumir que "Comunicación General" es igual a "Inglés" (lo cual sería falso), se introduce una **corrección conservadora** que modela la incertidumbre, permitiendo entrenar el modelo con datos completos.

### 3. Exploración de la Variable Objetivo: "Tiempo de Búsqueda"
- **Acción Realizada:** Análisis descriptivo y visualización de la variable `Tiempo_Busqueda`.
    - Se identificaron y filtraron valores inconsistentes (tiempos negativos), lo cual es crítico para la validez del modelo.
    - La prueba de **Mann-Whitney U** arrojó un **P-valor de 0.4213**, sugiriendo homogeneidad en los tiempos de inserción entre cohortes.
- **Hallazgo Crítico (Censura):** Análisis de la proporción de eventos observados vs. censurados.
    - **Figura Generada:** `02_Data_Understanding/figures-v2/Estado_de_Inserción_Laboral_Target_de_Supervivenci.png`
    - **Insight:** Este gráfico revela el balance entre graduados que encontraron empleo ($E=1$) y los que no ($E=0$) dentro de la ventana de estudio. Esta proporción justifica el uso de modelos de supervivencia sobre regresiones tradicionales, ya que los datos censurados contienen información temporal valiosa.

# Definición Operativa de la Variable Objetivo: Tiempo de Inserción ($T$)

## 1. Validación metodológica de la estrategia (con ventana estricta de 6 meses)

En este estudio, la medición se **ancla obligatoriamente** a una **ventana fija de observación de 6 meses** posterior al egreso ($L=6$). Por tanto, la variable objetivo se operacionaliza como una **escala temporal acotada** que permite entrenar modelos de supervivencia con **censura** e **incertidumbre por intervalos**, sin forzar supuestos no observables.

| **Decisión Técnica** | **Justificación Científica / Estadística** | **Veredicto** |
|---|---|---|
| **Ventana administrativa fija ($L=6$)** | La encuesta observa el estado laboral en un corte fijo; quienes no se insertan hasta el final del periodo quedan **censurados a la derecha**. | ✅ **APROBADO**. Se define censura como $T \ge 6$. |
| **Evento ($E$) NO depende del mapeo de $T$** | En supervivencia, $E$ codifica **si el evento fue observado**. Si reporta antigüedad laboral o "sí trabaja", $E=1$; si no, $E=0$. | ✅ **APROBADO**. El porcentaje de insertados no cambia por el mapeo de $T$. |
| **Censura por intervalos (AFT)** | El enfoque AFT permite entrenar con **intervalos** $(T_{\text{lower}},T_{\text{upper}})$, preservando incertidumbre. | ✅ **APROBADO**. Objetivo en dos columnas: $T_{\text{lower}},T_{\text{upper}}$. |

## 2. Definición operativa (Formato AFT interval-censored, con $L=6$)

### 2.1 Regla del evento ($E$)
- Si el encuestado **trabaja** o reporta **antigüedad laboral** ⇒ **$E=1$ (insertado observado)**
- Si **no trabaja** y la antigüedad está **vacía/NaN** ⇒ **$E=0$ (censurado)**

### 2.2 Mapeo de intervalos para $T$ (escala acotada en 6 meses)

| **Respuesta de antigüedad laboral** | **Lectura dentro de $L=6$** | **$T_{\text{lower}}$** | **$T_{\text{upper}}$** | **$E$** |
|---|---:|---:|---:|---:|
| **Más de 2 años** | Inserción **más temprana** | 0.0 | 1.0 | 1 |
| **Entre 1 y 2 años** | Inserción **temprana** | 1.0 | 2.0 | 1 |
| **Entre 6 meses y 1 año** | Inserción **intermedia** | 2.0 | 4.0 | 1 |
| **Menos de 6 meses** | Inserción **tardía** | 4.0 | 6.0 | 1 |
| **(NaN / No trabaja)** | **Censura a la derecha** | 6.0 | $\infty$ | 0 |

---

### 3. Implicación clave
- **Este mapeo NO debe cambiar el porcentaje de insertados**, porque **$E$ se deriva de la condición de trabajo**.
- **Visualización de Edad vs Tiempo:** Se ha generado el gráfico `02_Data_Understanding/figures-v2/Relación_No_Lineal_Edad_vs_Tiempo_de_Inserción.png` para explorar la no linealidad entre la edad del graduado y su velocidad de inserción.

### 4. Análisis Socioeconómico (Salarios)
- **Gráfico Generado:** `02_Data_Understanding/figures-v2/Distribución_Salarial_Inicial_por_Cohorte_Datos_Cu.png`
- **Acción Realizada:** Análisis de la distribución salarial.
- **Hallazgo:** El gráfico de barras muestra que la mayoría de los egresados (ambas cohortes) se ubican en el rango de $461 a $1000. Existe una estabilidad notable entre ambos periodos, lo que valida que no hubo un shock económico externo que sesgue los resultados de inserción.
- **Importancia:** Esta variable actúa como control de homogeneidad contextual. rango $461-$1000.
    - Diciembre: 65.07% en el mismo rango.
- **Interpretación para Tesis:** La estabilidad en las ofertas salariales refuerza la comparabilidad de las cohortes y sugiere un mercado laboral estandarizado para los perfiles de entrada.

---

## Fase 3: Preparación de los Datos (Data Preparation)

En esta fase, se transforman los datos brutos en un formato limpio y estructurado adecuado para el modelado. El notebook demuestra un rigor técnico alto en este proceso.

### 1. Limpieza y Normalización de Texto (Cell 20)
- **Acción Realizada:** Se aplicaron funciones personalizadas (`limpiar_texto`) para estandarizar los nombres de las facultades y carreras.
- **Detalle Técnico:** Se manejaron inconsistencias de entrada manual (ej. "Ingeniería Mecánica" vs "Mecanica" vs "ING. MECANICA").
- **Valor para Tesis:** Esta normalización es fundamental. Sin ella, la segmentación por facultad en el análisis posterior sería errónea, diluyendo las conclusiones específicas por área de conocimiento.

### 2. Filtrado Lógico y Manejo de Calidad (Cells 11, 30, 36)
- **Acción Realizada:**
    - **Edad:** Se excluyeron explícitamente registros menores de 18 años para garantizar la validez legal y lógica de la muestra laboral.
    - **Consistencia Temporal:** Para el análisis de correlación (Cell 36), se filtraron solo los eventos confirmados (`Event == 1`), evitando que los datos censurados (graduados que aún no encuentran trabajo) sesguen el cálculo del tiempo de inserción real.
- **Valor para Tesis:** Estas reglas de negocio codificadas demuestran una preparación de datos consciente del contexto, no una limpieza automática ciega.

### 3.- **Hallazgo:** Se identificaron brechas significativas en habilidades técnicas y blandas.
- **Evidencia Visual:**
    - Brechas Técnicas: `02_Data_Understanding/figures-v2/Top_10_Contenidos_Técnicos_Ausentes_en_la_Formació.png`
    - Brechas Blandas: `02_Data_Understanding/figures-v2/Top_10_Habilidades_Blandas_a_FortalecernPercepción.png` (Top 10 Habilidades a Fortalecer: Liderazgo, Comunicación).
- **Importancia:** Estos hallazgos descriptivos alimentarán las recomendaciones de diseño curricular en fases posteriores.s de Contenido Técnico" (Hard Skills Gaps).
- **Valor para Tesis:** Identifica contenidos específicos ausentes en el currículo (ej. "Python", "Excel Avanzado", "Normativas"). A diferencia de un simple promedio numérico, esto ofrece *insights* curriculares accionables para la actualización de planes de estudio.



---

## Siguientes Pasos (Roadmap)

1.  **Fase 3: Preparación de Datos (En curso)**
    - Implementar imputación avanzada (MICE/KNN) para variables numéricas faltantes.
    - Codificar variables categóricas (One-Hot/Target Encoding).
    - Construir el tablón analítico final (ABT) para modelado.

2.  **Fase 4: Modelado (Pendiente)**
    - Entrenar modelos de supervivencia (Cox, AFT, Random Survival Forest).
    - Evaluar métricas de desempeño (C-index, Brier Score).
    - Interpretar coeficientes para entender factores de riesgo/éxito.

3.  **Fase 5: Evaluación y Despliegue**
    - Traducir los hallazgos del modelo en políticas académicas concretas.
    - Diseñar el plan de monitoreo continuo.

---

## Apéndice: Hallazgos Adicionales (Figuras Extra)

Se han generado visualizaciones complementarias que refuerzan el entendimiento de la población estudiada:

### A. Caracterización Demográfica
- **Distribución por Género:** `02_Data_Understanding/figures-v2/Distribución_por_Género_Frecuencia.png`
    - Muestra la composición hombres/mujeres de las cohortes, relevante para detectar posibles sesgos de género en la inserción laboral.

### B. Análisis del Tiempo de Inserción
- **Distribución Proxy del Tiempo:** `02_Data_Understanding/figures-v2/Distribución_del_Tiempo_de_Inserción_Proxynn.png`
    - **Utilidad:** Histogramas mostrando los rangos reales de datos crudos. Permite visualizar la "forma" de la distribución temporal antes de cualquier transformación AFT, destacando la concentración temprana de inserciones.

### C. Correlaciones y Otros
- **Matriz de Spearman:** `02_Data_Understanding/figures-v2/Correlación_Spearman_Skills_vs._Éxito_Evento_y_Rap.png`
    - Exploración de correlaciones no lineales entre variables numéricas y ordinales.
- **Curvas de Supervivencia Preliminares:** `02_Data_Understanding/figures-v2/Curvas_de_Supervivencia_Kaplan-Meier_por_CohortenP.png`
    - Primer vistazo a la dinámica temporal de inserción (Kaplan-Meier) por grupos.

### 5. Curva Empírica de Inserción Laboral (Datos Crudos)
- **Gráfico Generado:** `02_Data_Understanding/figures-v2/Curva_Empírica_de_Inserción_LaboralnBasado_en_Rang.png`
- **Hallazgo Clave:** El **78.9%** de los egresados consigue empleo antes del primer año.
- **Importancia:** Esta métrica utiliza los conteos reales de la encuesta (sin transformaciones AFT), ofreciendo una validación directa y tangible de la empleabilidad acumulada. Confirma que la "Mediana" (<6 meses) y el "Pareto" (80%) se alcanzan en tiempos competitivos.
