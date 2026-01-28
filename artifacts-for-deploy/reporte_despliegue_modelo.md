# Documentación de Despliegue: Modelo Predictivo de Empleabilidad STEM-EPN

Este documento detalla la identificación, selección y empaquetamiento del modelo final para su puesta en producción.

## 1. Comparativa de Modelos Finales (RSF)

Tras el proceso de optimización de hiperparámetros, se contrastaron los dos mejores candidatos de la arquitectura **Random Survival Forest (RSF)**:

| Modelo | C-index | IBS | Método de Búsqueda |
| :--- | :---: | :---: | :--- |
| **RSF RandomizedSearch** | **0.6983** | **0.1056** | 10 iteraciones de validación cruzada |
| RSF GridSearch manual | 0.6935 | 0.1062 | 50 iteraciones de validación cruzada |

### 🏆 Modelo Seleccionado: **RSF RandomizedSearchCV**

**Justificación Técnica:**
*   **Superioridad Predictiva:** Alcanzó el mayor C-index registrado (**0.6983**), superando al GridSearch manual por +0.0048.
*   **Calibración Probabilística:** El Integrated Brier Score (IBS) de **0.1056** indica una excelente capacidad para estimar probabilidades de supervivencia a través del tiempo.
*   **Eficiencia:** Logró un mejor punto óptimo en el espacio de búsqueda con solo 10 iteraciones, demostrando mayor robustez frente al sobreajuste.

---

## 2. Inventario de Artefactos para Producción

Los siguientes archivos han sido generados y validados para asegurar la reproducibilidad de las inferencias en el entorno de despliegue:

| Archivo | Función |
| :--- | :--- |
| `modelo_rsf_final.joblib` | Binario del modelo Random Survival Forest entrenado y optimizado. |
| `scaler_final.joblib` | Escalador (`StandardScaler`) ajustado con los parámetros de la muestra original. |
| `mapeo_carrera_encoded.json` | Diccionario de mapeo para la codificación consistente de las carreras de la EPN. |
| `modelo_metadata.json` | Metadatos que describen la versión del modelo, métricas de desempeño y parámetros de entrenamiento. |

---

## 3. Estado de Listo para Producción (Ready-to-Deploy)

> [!NOTE]
> Todos los artefactos han sido consolidados en el directorio `/home/desarrollo03/Documentos/UNIVERSIDAD/TIC/Escrito/new_format/artifacts-for-deploy`.

El sistema de inferencia debe cargar estos componentes secuencialmente:
1.  **Cargar Mapeo:** Para transformar las entradas de texto (carrera) en valores numéricos.
2.  **Aplicar Escalamiento:** Utilizando el `scaler_final.joblib` para normalizar las características de entrada.
3.  **Ejecutar Inferencia:** Invocando el método `predict_survival_function` o `predict_cumulative_hazard_function` del modelo RSF.

**Estado:** ✅ Validado y listo para integración con FastAPI.
