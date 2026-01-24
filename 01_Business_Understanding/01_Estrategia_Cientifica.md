# 01 Estrategia Científica: Hibridación de Contextos y Supervivencia (CRISP-DM Fase 1)

**Autor:** Principal Academic Investigator (Antigravity)  
**Proyecto:** Predicción de Inserción Laboral - Enfoque Híbrido  
**Fecha:** Enero 2026

---

## 1. Metáfora Conceptual: "El Chasis y el Motor"

Para explicar la arquitectura de nuestros experimentos y la naturaleza de las variables, proponemos la siguiente metáfora de ingeniería, alineada con el enfoque de **hibridación de contextos** (`Saidani 2022`):

*   **El Chasis (Estructura Estática):** Representa el **Perfil Académico** (Matriz de Diego: 69 dimensiones técnicas + demografía). Es la estructura rígida que el graduado trae de la universidad. Es necesaria para "rodar", pero por sí sola no garantiza velocidad en la inserción.
*   **El Motor (Fuerza Dinámica):** Representa las **Habilidades Blandas e Interacciones Contextuales** (Pasantías, Soft Skills, Liderazgo). Según `Mwita (2024)`, estas variables actúan como "aceleradores" ($ \alpha > 1 $ en modelos AFT) que reducen drásticamente el tiempo de desempleo, independientemente del chasis.

**Objetivo de la Tesis:** Demostrar que un modelo que integra inteligentemente el motor sobre el chasis (Experimento B: Feature Weighting) supera significativamente a un modelo puramente estructural (Experimento A).

---

## 2. Definición de Experimentos

Diseñamos dos escenarios experimentales rigurosos para aislar el efecto de la hibridación:

### Experimento A: "Baseline Académico" (Solo Chasis)
*   **Input:** Vector de 69 dimensiones (Habilidades Técnicas) + Variables Demográficas (Edad, Género).
*   **Técnica:** Regresión AFT estándar sin ponderación externa.
*   **Hipótesis Nula ($H_0$):** La información curricular es suficiente para predecir el tiempo de inserción ($C_{index} \approx 0.6$).

### Experimento B: "Hibridación Contextual Ponderada" (Chasis + Motor)
*   **Input:** Experimento A + Variables de Contexto (Pasantías, Soft Skills).
*   **Técnica:** **Feature Weighting**. Se aplicarán pesos específicos ($w_j > 1$) a las dimensiones identificadas como críticas por `Mwita (2024)` y `Saidani (2022)`.
*   **Selección de Características:** Uso de **LASSO + RFE (Recursive Feature Elimination)**. Dado que tratamos con *Small Data* y alta dimensionalidad, `Jayachandran (2024)` y `Suresh (2022)` validan el uso de LASSO para penalizar coeficientes irrelevantes (sparsity) y evitar overfitting.
*   **Hipótesis Alternativa ($H_1$):** La incorporación de pesos contextuales mejora significativamente la predicción ($C_{index} > 0.7$).

---

## 3. Justificación del Modelo: ¿Por qué XGBoost-AFT y no Cox?

El "estándar de oro" médico (Cox Proportional Hazards) es inadecuado para nuestros datos educativos por dos razones fundamentales respaldadas por el corpus:

1.  **Violación de Riesgos Proporcionales (PH Assumption):**
    *   `Ayaneh et al. (2020)` y `Abdulazeez (2024)` demuestran que variables como **Género** y **Carrera STEM** no mantienen un riesgo constante en el tiempo (e.g., la brecha de género puede ser alta al inicio pero cerrarse a los 6 meses). Cox asume riesgo constante ($HR(t) = cte$), lo cual es falso aquí.
    *   **Solución:** **AFT (Accelerated Failure Time)** no asume proporcionalidad; modela directamente cómo las variables expanden o contraen el tiempo ($T = T_0 \cdot \exp(-\beta X)$), siendo robusto a estas variaciones (`Barnwal 2021`).

2.  **No-Linealidad e Interacciones:**
    *   La relación entre habilidades técnicas y empleo es compleja y no lineal. `Rossi (2025)` critica el uso de índices de concordancia simples cuando la asunción PH falla. **XGBoost** (Survival Embeddings) captura estas no-linealidades mediante árboles de decisión, superando a las regresiones lineales tradicionales.

---

## 4. Protocolo Longitudinal: Unificación Julio vs. Diciembre

Para maximizar el tamaño de la muestra sin introducir sesgo, aplicaremos un protocolo de **"Actualización de Censura a la Derecha"**:

*   **Escenario 1 (Aparece en Julio y Diciembre):**
    *   Si en Julio $E=0$ (Desempleado) y en Dic $E=1$ (Empleos): El evento ocurrió. Calculamos $T = \text{Fecha}_{Dic} - \text{Fecha}_{Grad}$. Estado final: $E=1$.
    *   Si en Julio $E=0$ y en Dic $E=0$: Aún no encuentra empleo. Actualizamos el tiempo de censura: $T = \text{Fecha}_{Dic} - \text{Fecha}_{Grad}$. Estado final: $E=0$. (Ganamos meses de información de "supervivencia").
*   **Escenario 2 (Solo aparece en uno):** Se mantiene el registro tal cual.

Este método evita duplicados y aprovecha la naturaleza longitudinal para reducir la incertidumbre de la censura, alineado con las prácticas de *Tracer Studies* (`Gecolea 2021`).

---

## 5. Hoja de Ruta (Next Steps)

1.  **Notebook 02:** Data Preparation implementando el Protocolo Longitudinal.
2.  **Notebook 03:** Feature Engineering (LASSO/RFE) para definir la matriz reducida.
3.  **Notebook 04:** Ejecución de Experimentos A y B comparativos.
