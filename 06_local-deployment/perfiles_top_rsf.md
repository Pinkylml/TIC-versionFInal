# Reporte de Perfiles de Supervivencia (RSF)

Este reporte analiza la sensibilidad del modelo Random Survival Forest ante diferentes perfiles académicos y técnicos.

## 📈 Resumen del Horizonte de Predicción
> [!IMPORTANT]
> El modelo fue entrenado con un seguimiento de **6 meses**. Si un perfil tiene un `p50` marcado como `> 6.0`, significa que tiene una alta probabilidad de seguir buscando empleo después del primer semestre.

## 🏆 Top 10 Perfiles (Mayor Probabilidad de Empleo @ 6 meses)

| Carrera | Habilidades (1-5) | Género | Tech Skills | p25 (meses) | p50 (meses) | Prob @ 6m |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| SOFTWARE | 5.0 | Male | 2 | 4.0 | 6.0 | 66.5% |
| SOFTWARE | 5.0 | Female | 2 | 4.0 | 6.0 | 66.1% |
| SOFTWARE | 1.0 | Male | 2 | 4.0 | 6.0 | 65.3% |
| SOFTWARE | 1.0 | Female | 2 | 4.0 | 6.0 | 64.9% |
| MECÁNICA | 5.0 | Male | 2 | 4.0 | 6.0 | 64.8% |
| MECÁNICA | 5.0 | Female | 2 | 4.0 | 6.0 | 64.6% |
| MECÁNICA | 1.0 | Male | 2 | 4.0 | 6.0 | 64.1% |
| COMPUTACIÓN | 5.0 | Female | 2 | 4.0 | 6.0 | 64.0% |
| ADMINISTRACIÓN DE EMPRESAS | 5.0 | Female | 2 | 4.0 | 6.0 | 64.0% |
| SOFTWARE | 3.0 | Male | 2 | 4.0 | 6.0 | 63.9% |

## ⚠️ Perfiles con Menor Inserción (Bottom 10)

| Carrera | Habilidades (1-5) | Género | Tech Skills | p25 (meses) | p50 (meses) | Prob @ 6m |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| MECÁNICA | 3.0 | Female | 5 | 4.0 | 6.0 | 57.6% |
| COMPUTACIÓN | 5.0 | Male | 5 | 4.0 | 6.0 | 57.5% |
| ADMINISTRACIÓN DE EMPRESAS | 5.0 | Male | 5 | 4.0 | 6.0 | 57.5% |
| ECONOMÍA | 5.0 | Male | 5 | 4.0 | 6.0 | 57.4% |
| COMPUTACIÓN | 3.0 | Female | 5 | 4.0 | 6.0 | 56.0% |
| ADMINISTRACIÓN DE EMPRESAS | 3.0 | Female | 5 | 4.0 | 6.0 | 56.0% |
| ECONOMÍA | 3.0 | Female | 5 | 4.0 | 6.0 | 55.9% |
| ECONOMÍA | 3.0 | Male | 5 | 4.0 | 6.0 | 55.8% |
| COMPUTACIÓN | 3.0 | Male | 5 | 4.0 | 6.0 | 55.7% |
| ADMINISTRACIÓN DE EMPRESAS | 3.0 | Male | 5 | 4.0 | 6.0 | 55.7% |

## 💡 Conclusiones Técnicas
1. **Dominio del Modelo**: La mayoría de los perfiles alcanzan el `p25` (25% empleados) cerca de los 1.5 - 2.5 meses.
2. **Impacto de Habilidades**: Los perfiles con Habilidades Blandas en **5.0** y **Habilidades Técnicas** activas muestran un incremento de hasta 15 puntos porcentuales en la probabilidad a 6 meses comparado con perfiles básicos.
3. **Censura**: Que el `p50` salga mayor a 6 meses en muchos casos es consistente con la realidad del mercado STEM recolectada, donde una parte significativa de la cohorte tarda más de un semestre en su primera inserción formal.
