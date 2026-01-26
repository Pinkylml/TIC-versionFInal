# %% [Code Cell]
# ==============================================================================
# 1. IMPORTACIÓN DE LIBRERÍAS Y CONFIGURACIÓN
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr # Para correlaciones estadísticas

# Configuración de Estilo Profesional para Gráficos
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.family'] = 'sans-serif'

# Configuración de Pandas para ver todas las columnas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# ==============================================================================
# 2. CARGA DEL DATASET PROCESADO (Checkpoint 1)
# ==============================================================================
# Cargamos el archivo que contiene las correcciones de Facultades y Skills (S1-S7)
df_base = pd.read_csv('dataset_part1_analisis.csv')

# Verificación de integridad
print("--- RESUMEN DE CARGA ---")
print(f"✅ Dimensiones: {df_base.shape[0]} filas x {df_base.shape[1]} columnas")

# Verificar que nuestras columnas críticas existen
cols_clave = ['S7_Ingles', 'Facultad', 'T_Lower', 'Event', 'Cohorte']
existencia = {c: c in df_base.columns for c in cols_clave}
print(f"✅ Columnas Críticas detectadas: {existencia}")

# Vista previa
df_base.head(3)
plt.savefig('figures/.png', bbox_inches='tight')
print('Saved figure to figures/.png')


# %% [Code Cell]
import os
import matplotlib.pyplot as plt
os.makedirs('figures', exist_ok=True)
# Set global figure params if needed
plt.rcParams.update({'figure.max_open_warning': 0})


# %% [Code Cell]
# ==============================================================================
# 3. SELECCIÓN DE FEATURES (ELIMINACIÓN DE RUIDO)
# ==============================================================================

# Definimos las columnas exactas que alimentarán el análisis y el modelo
cols_soft = [
    'S1_Comunicacion_Esp', 'S2_Compromiso_Etico', 'S3_Trabajo_Equipo_Liderazgo',
    'S4_Resp_Social', 'S5_Gestion_Proyectos', 'S6_Aprendizaje_Digital',
    'S7_Ingles' # <--- La variable crítica que rescatamos
]

cols_hard = ['TXT_Hard_Skills'] # <--- Tu solicitud explícita

cols_demograficas = ['Edad', 'Genero', 'Facultad', 'Cohorte', 'carrera']

cols_target_aft = ['T_Lower', 'T_Upper', 'Event'] # Para el modelo matemático
cols_target_desc = ['TARGET_Evento', 'TARGET_Tiempo'] # Para gráficos descriptivos

# Crear el DataFrame Limpio
df_model = df_base[cols_soft + cols_hard + cols_demograficas + cols_target_aft + cols_target_desc].copy()

# ==============================================================================
# 4. LIMPIEZA INICIAL
# ==============================================================================

# A. Tratamiento de Texto (Hard Skills)
# Rellenamos nulos con "Sin Información" para que el vectorizador no falle luego
df_model['TXT_Hard_Skills'] = df_model['TXT_Hard_Skills'].fillna('Sin información').astype(str)

# Normalización básica (minúsculas) para evitar duplicados como "Python" vs "python"
df_model['TXT_Hard_Skills'] = df_model['TXT_Hard_Skills'].str.lower().str.strip()

# B. Verificación de Nulos en Soft Skills
# Si queda algún nulo residual (raro tras la homologación), lo imputamos con la mediana
# Esto es seguridad para que XGBoost no reciba NaNs inesperados
for col in cols_soft:
    if df_model[col].isnull().sum() > 0:
        mediana = df_model[col].median()
        df_model[col] = df_model[col].fillna(mediana)
        print(f"⚠️ Imputados {df_model[col].isnull().sum()} nulos en {col} con mediana ({mediana})")

print("--- DATASET LIMPIO PARA MODELADO (df_model) ---")
print(f"Dimensiones: {df_model.shape}")
print("Muestra de Hard Skills:")
display(df_model['TXT_Hard_Skills'].head(5))

# Guardamos una copia de seguridad local por si acaso
# df_model.to_csv('dataset_part2_clean.csv', index=False)

# %% [Code Cell]
df_model

# %% [Code Cell]
df_base = df_model.copy()

# %% [Code Cell]
# ============================================================
# REASIGNAR "OTRO" -> {M,F} (estratificado + probabilístico)
# - Normaliza Genero a {M,F,OTRO,NaN}
# - Reasigna OTRO usando proporciones M/F dentro de (Carrera) si hay suficientes datos,
#   si no dentro de (Facultad), si no global.
# - (Opcional) también imputar NaN igual que OTRO.
# ============================================================
import numpy as np
import pandas as pd

def _norm_genero_4(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    s = s.replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u")
    s = s.replace(".", "").replace(",", "").replace("-", " ").replace("_", " ")
    s = " ".join(s.split())

    if s in {"m","masculino","hombre","varon","masc"}: return "M"
    if s in {"f","femenino","mujer","fem"}: return "F"
    if s in {"otro","otra","nonbinary","no binario","nb","x"}: return "OTRO"
    return np.nan

def _sample_mf_from_group(df_group, rng):
    # df_group ya es un subconjunto; usa distribución M/F (ignora OTRO/NaN)
    counts = df_group["Genero_norm"].value_counts()
    m = counts.get("M", 0)
    f = counts.get("F", 0)
    tot = m + f
    if tot == 0:
        return None
    p_m = m / tot
    return "M" if rng.random() < p_m else "F"

def reasignar_otro_a_mf(
    df,
    col_genero="Genero",
    col_carrera="carrera",
    col_facultad="Facultad",
    out_col="Genero_bin",
    min_ref_por_grupo=20,     # mínimo de casos M+F para usar distribución del grupo
    also_impute_nan=True,
    random_state=42
):
    df = df.copy()
    rng = np.random.default_rng(random_state)

    # 1) normalizar a {M,F,OTRO,NaN}
    df["Genero_norm"] = df[col_genero].apply(_norm_genero_4)

    # 2) preparar referencia global (M/F)
    ref_global = df[df["Genero_norm"].isin(["M","F"])].copy()

    # helper para conseguir subconjunto de referencia por llave
    def get_ref_by_key(key_col, key_val):
        if key_col is None or key_col not in df.columns:
            return None
        sub = df[(df[key_col] == key_val) & (df["Genero_norm"].isin(["M","F"]))]
        if sub.shape[0] >= min_ref_por_grupo:
            return sub
        return None

    # 3) decidir qué filas reasignar
    mask_target = df["Genero_norm"].eq("OTRO")
    if also_impute_nan:
        mask_target = mask_target | df["Genero_norm"].isna()

    # 4) reasignación fila por fila (estratificada)
    assigned = []
    for idx, row in df.loc[mask_target].iterrows():
        ref = None

        # prioridad: carrera -> facultad -> global
        if col_carrera in df.columns:
            ref = get_ref_by_key(col_carrera, row[col_carrera])
        if ref is None and col_facultad in df.columns:
            ref = get_ref_by_key(col_facultad, row[col_facultad])
        if ref is None:
            ref = ref_global  # global (puede ser pequeño, pero algo es algo)

        choice = _sample_mf_from_group(ref, rng)
        # si por algún motivo no hay M/F en todo el dataset, fallback determinista
        if choice is None:
            choice = "M"

        assigned.append((idx, choice))

    df[out_col] = df["Genero_norm"].copy()

    # 5) escribir asignaciones
    for idx, choice in assigned:
        df.at[idx, out_col] = choice

    # 6) asegurar solo {M,F} en salida
    df[out_col] = df[out_col].replace("OTRO", np.nan)
    if df[out_col].isna().any():
        # si quedaron NaN, fallback global por moda
        moda = df[df[out_col].isin(["M","F"])][out_col].value_counts().idxmax()
        df[out_col] = df[out_col].fillna(moda)

    # reporte
    print("✅ Reasignación completada.")
    print("Distribución original normalizada:\n", df["Genero_norm"].value_counts(dropna=False))
    print("\nDistribución final binaria:\n", df[out_col].value_counts(dropna=False))

    # limpiar columna auxiliar si no la quieres
    df.drop(columns=["Genero_norm"], inplace=True)

    return df


df_base = reasignar_otro_a_mf(df_base, col_genero="Genero", col_carrera="CARRERA_CLEAN", col_facultad="Facultad",
                             out_col="Genero_bin", min_ref_por_grupo=20, also_impute_nan=False, random_state=42)



# %% [Code Cell]
df_base['Genero_bin'].unique()

# %% [Code Cell]
# ==============================================================================
# FASE 3.2.2: INTEGRACIÓN DE VECTORES ACADÉMICOS (BERT 69D)
# Fuente: Diego Rafael Arias Sarango (Computación EPN)
# ==============================================================================

# 1. Definir el diccionario de mapeo (Normalizado a lo que tenemos en df_base['carrera'])
# Nota: En df_base la columna se llama 'carrera' y tiene valores como '(RRA20) INGENIERÍA CIVIL'
mapping_carreras = {
    '(RRA20) INGENIERÍA CIVIL': 'INGENIERÍA CIVIL',
    '(RRA20) ELECTRÓNICA Y AUTOMATIZACIÓN': 'ELECTRÓNICA Y AUTOMATIZACIÓN',
    '(RRA20) SOFTWARE': 'SOFTWARE',
    '(RRA20) DESARROLLO DE SOFTWARE': 'SOFTWARE',
    '(RRA20) INGENIERÍA DE LA PRODUCCIÓN': 'INGENIERÍA DE LA PRODUCCIÓN',
    '(RRA20) COMPUTACIÓN': 'COMPUTACIÓN',
    '(RRA20) MATEMÁTICA': 'MATEMÁTICA',
    '(RRA20) ELECTROMECÁNICA': 'MECATRÓNICA',
    '(RRA20) REDES Y TELECOMUNICACIONES': 'TELECOMUNICACIONES',
    '(RRA20) MECÁNICA': 'MECÁNICA',
    '(RRA20) GEOLOGÍA': 'GEOLOGÍA',
    '(RRA20) TECNOLOGÍAS DE LA INFORMACIÓN': 'TECNOLOGÍAS DE LA INFORMACIÓN',
    '(RRA20) ECONOMÍA': 'ECONOMÍA',
    '(RRA20) AGUA Y SANEAMIENTO AMBIENTAL': 'INGENIERÍA AMBIENTAL',
    '(RRA20) TELECOMUNICACIONES': 'TELECOMUNICACIONES',
    'INGENIERIA EN CIENCIAS ECONOMICAS Y FINANCIERAS': 'ECONOMÍA',
    '(RRA20) ELECTRICIDAD': 'ELECTRICIDAD',
    '(RRA20) AGROINDUSTRIA': 'AGROINDUSTRIA',
    '(RRA20) INGENIERÍA AMBIENTAL': 'INGENIERÍA AMBIENTAL',
    '(RRA20) FÍSICA': 'FÍSICA',
    '(RRA20) INGENIERÍA QUÍMICA': 'INGENIERÍA QUÍMICA',
    '(RRA20) ADMINISTRACIÓN DE EMPRESAS': 'ADMINISTRACIÓN DE EMPRESAS',
    '(RRA20) PETRÓLEOS': 'PETRÓLEOS',
    'FISICA': 'FÍSICA',
    'INGENIERIA EMPRESARIAL': 'ADMINISTRACIÓN DE EMPRESAS',
    'INGENIERIA MECANICA': 'MECÁNICA',
    'INGENIERIA GEOLOGICA': 'GEOLOGÍA',
    '(RRA20) MATEMÁTICA APLICADA': 'MATEMÁTICA APLICADA'
}

# 2. Aplicar el mapeo en df_base
# Usamos 'carrera' que es la columna real en nuestro dataset
print("--> Aplicando mapeo de carreras para compatibilidad con vectores...")
df_base['CARRERA_PARA_VECTORES'] = df_base['carrera'].map(mapping_carreras)

# Verificación de cobertura
nulos_map = df_base['CARRERA_PARA_VECTORES'].isnull().sum()
if nulos_map > 0:
    print(f"⚠️ ALERTA: {nulos_map} carreras no encontraron match en el diccionario de vectores.")
    print("Carreras sin vector:", df_base[df_base['CARRERA_PARA_VECTORES'].isnull()]['carrera'].unique())
else:
    print("✅ Todas las carreras fueron mapeadas correctamente.")

# 3. Cargar la matriz de Diego (Asegúrate de subir el archivo .csv)
try:
    df_vectores = pd.read_csv('Vectores_Academicos_69d.csv')
    print(f"✅ Matriz de Vectores Cargada: {df_vectores.shape}")

    # 4. Merge (Fusión)
    # Asumimos que el CSV de vectores tiene una columna llave (ej. 'CARRERA' o similar)
    # Ajusta 'on=' según el nombre real en df_vectores
    # df_merged = pd.merge(df_base, df_vectores, left_on='CARRERA_PARA_VECTORES', right_on='CARRERA_EN_VECTORES', how='left')

except FileNotFoundError:
    print("❌ ERROR: No se encuentra 'Vectores_Academicos_69d.csv'. Por favor, súbelo al entorno.")

# %% [Code Cell]
df_base

# %% [Code Cell]
df_vectores = pd.read_csv('Vectores_Academicos_69d.csv')


# %% [Code Cell]
df_vectores

# %% [Code Cell]
df_final_vectores = pd.merge(
    df_base,
    df_vectores,
    left_on='CARRERA_PARA_VECTORES',
    right_on='CARRERA', # Asumiendo que la primera columna de vectores se llama CARRERA
    how='left'
)

# Eliminamos columnas redundantes post-merge (CARRERA del csv de vectores)
if 'CARRERA' in df_final_vectores.columns:
    df_final_vectores.drop(columns=['CARRERA'], inplace=True)

# 5. Verificación Final
# ------------------------------------------------------------------
# Identificamos las columnas de vectores (las que no estaban antes)
cols_nuevas = set(df_final_vectores.columns) - set(df_base.columns)
print(f"✅ Se agregaron {len(cols_nuevas)} columnas de vectores técnicos.")
print(f"   Dimensiones finales: {df_final_vectores.shape}")

# Vista previa de un registro con sus vectores
cols_vista = ['carrera', 'CARRERA_PARA_VECTORES'] + list(cols_nuevas)[:3] # Primeras 3 columnas vectoriales
display(df_final_vectores[cols_vista].head(3))

# Guardamos el dataset enriquecido
df_final_vectores.to_csv('dataset_part3_vectores.csv', index=False)

# %% [Code Cell]
df_final_vectores

# %% [Code Cell]
df_base

# %% [Code Cell]
# ==============================================================================
# ANÁLISIS DE CORRELACIÓN CRUZADA: SOFT SKILLS vs. VECTORES TÉCNICOS (FIX)
# ==============================================================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Cargar Dataset Preparado
df_exp_A = df_final_vectores
print(f"Dimensiones: {df_exp_A.shape}")

# 2. Definir Grupos de Variables
cols_soft = [
    'S1_Comunicacion_Esp', 'S2_Compromiso_Etico', 'S3_Trabajo_Equipo_Liderazgo',
    'S4_Resp_Social', 'S5_Gestion_Proyectos', 'S6_Aprendizaje_Digital', 'S7_Ingles'
]

# Identificar columnas técnicas (excluyendo metadatos conocidos)
cols_conocidas = cols_soft + [
    'TXT_Hard_Skills', 'Edad', 'Genero', 'Facultad', 'Cohorte', 'carrera',
    'T_Lower', 'T_Upper', 'Event', 'TARGET_Evento', 'TARGET_Tiempo',
    'CARRERA_PARA_VECTORES', 'Unnamed: 0'
]

# 3) Candidatas a vectores (todo lo que NO es conocida)
cols_vectors_raw = [c for c in df_exp_A.columns if c not in cols_conocidas]

# 4) 🔥 FIX: quedarnos SOLO con columnas numéricas (evita 'M', 'F', textos, etc.)
cols_vectors = [c for c in cols_vectors_raw if pd.api.types.is_numeric_dtype(df_exp_A[c])]

# (Opcional) Diagnóstico: qué se estaba colando como no numérico
cols_vectors_bad = [c for c in cols_vectors_raw if c not in cols_vectors]
if cols_vectors_bad:
    print("⚠️ Columnas NO numéricas que se estaban colando como 'vectores':", cols_vectors_bad)
    # muestra ejemplo del contenido de la primera columna problemática
    print(df_exp_A[cols_vectors_bad[0]].head())

print(f"--> Calculando matriz: {len(cols_soft)} Soft Skills x {len(cols_vectors)} Vectores Técnicos (numéricos)")

# 5) Coerción defensiva: soft skills a numérico por si vienen como texto
for c in cols_soft:
    df_exp_A[c] = pd.to_numeric(df_exp_A[c], errors="coerce")

# 6) Cálculo de Correlación (Spearman) SOLO en variables numéricas
corr_matrix = df_exp_A[cols_soft + cols_vectors].corr(method='spearman', numeric_only=True)
soft_vs_tech = corr_matrix.loc[cols_soft, cols_vectors]

# 7) Visualización
plt.figure(figsize=(24, 8))
sns.heatmap(
    soft_vs_tech,
    annot=False,
    cmap='RdBu_r',
    center=0,
    linewidths=0.05,
    linecolor='white',
    cbar_kws={'label': 'Coeficiente Spearman', 'shrink': 0.8}
)

plt.title('Matriz de Independencia: Habilidades Blandas vs. Perfil Técnico Académico',
          fontsize=18, fontweight='bold', pad=20)
plt.ylabel('Soft Skills (Autopercepción)', fontsize=14, fontweight='bold')
plt.xlabel('Vectores de Conocimiento Técnico (Tópicos Curriculares)', fontsize=14, fontweight='bold')

plt.xticks(rotation=90, fontsize=9)
plt.yticks(rotation=0, fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('06_heatmap_soft_vs_tech.png', dpi=300)
plt.show()

# 8) Validación estadística rápida
max_corr = soft_vs_tech.abs().max().max()
print(f"\n📊 Máxima correlación absoluta encontrada: {max_corr:.3f}")

if max_corr < 0.3:
    print("✅ CONCLUSIÓN: Las variables son ORTOGONALES (Independientes).")
    print("   Esto es excelente para el modelo: significa que las Soft Skills agregan información NUEVA.")
else:
    print("⚠️ Se encontraron correlaciones moderadas. Revisar pares específicos.")


# %% [Code Cell]
# ==============================================================================
# ✅ P-VALUES para Spearman (Soft Skills vs Vectores Técnicos) + corrección FDR
#   - Calcula rho y pvalue por par (soft_i, tech_j)
#   - Aplica Benjamini–Hochberg (FDR) para múltiples comparaciones
#   - Devuelve: rho_matrix, p_matrix, q_matrix (FDR), y máscara significativa
# ==============================================================================

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

# --- Asegúrate de tener definidos: df_exp_A, cols_soft, cols_vectors ---

# 1) Subset con columnas relevantes
df_sub = df_exp_A[cols_soft + cols_vectors].copy()

# 2) Matrices vacías
rho = pd.DataFrame(index=cols_soft, columns=cols_vectors, dtype=float)
pval = pd.DataFrame(index=cols_soft, columns=cols_vectors, dtype=float)

# 3) Spearman por par (manejo de NaNs por eliminación por pares)
for s in cols_soft:
    xs = df_sub[s]
    for t in cols_vectors:
        yt = df_sub[t]
        mask = xs.notna() & yt.notna()
        if mask.sum() < 5:   # umbral mínimo de n para evitar inestabilidad
            rho.loc[s, t] = np.nan
            pval.loc[s, t] = np.nan
            continue
        r, p = spearmanr(xs[mask], yt[mask])
        rho.loc[s, t] = r
        pval.loc[s, t] = p

# 4) Corrección por múltiples comparaciones (FDR Benjamini–Hochberg)
p_flat = pval.values.flatten()
valid = ~np.isnan(p_flat)

q_flat = np.full_like(p_flat, np.nan, dtype=float)
rej_flat = np.full_like(p_flat, False, dtype=bool)

rej, q, _, _ = multipletests(p_flat[valid], alpha=0.05, method="fdr_bh")
q_flat[valid] = q
rej_flat[valid] = rej

qval = pd.DataFrame(q_flat.reshape(pval.shape), index=cols_soft, columns=cols_vectors)
sig_mask = pd.DataFrame(rej_flat.reshape(pval.shape), index=cols_soft, columns=cols_vectors)

# 5) (Opcional) Top correlaciones significativas por magnitud
top_sig = (
    rho.where(sig_mask)
       .stack()
       .rename("rho")
       .to_frame()
       .join(pval.stack().rename("p"))
       .join(qval.stack().rename("q"))
       .sort_values("rho", key=lambda x: x.abs(), ascending=False)
)

print("✅ Listo: rho, pval, qval (FDR) y sig_mask")
print("Significativas (FDR<0.05):", int(sig_mask.sum().sum()))
display(top_sig.head(20))


# %% [Code Cell]


# %% [Code Cell]
# ==============================================================================
# ✅ UNIVARIATE COX (69 tópicos) + FDR (Benjamini–Hochberg)
# Objetivo: evaluar asociación de cada tópico técnico con el TIEMPO A INSERCIÓN
#          tratando censura correctamente (Cox PH).
# Salida: tabla con HR, p, q(FDR), CI95 y ranking + export CSV + plot top.
# ==============================================================================

# --- (1) Dependencias ---
!pip -q install lifelines statsmodels

import numpy as np
import pandas as pd

from lifelines import CoxPHFitter
from statsmodels.stats.multitest import multipletests

import matplotlib.pyplot as plt

# --- (2) Cargar dataset ---
df = pd.read_csv("dataset_part3_vectores.csv")

# --- (3) Definir columnas (ajusta si tu archivo difiere) ---
cols_soft = [
    'S1_Comunicacion_Esp', 'S2_Compromiso_Etico', 'S3_Trabajo_Equipo_Liderazgo',
    'S4_Resp_Social', 'S5_Gestion_Proyectos', 'S6_Aprendizaje_Digital', 'S7_Ingles'
]
cols_conocidas = cols_soft + [
    'TXT_Hard_Skills', 'Edad', 'Genero', 'Facultad', 'Cohorte', 'carrera',
    'T_Lower', 'T_Upper', 'Event', 'TARGET_Evento', 'TARGET_Tiempo',
    'CARRERA_PARA_VECTORES', 'Unnamed: 0'
]
vector_cols = [c for c in df.columns if c not in cols_conocidas]

print("Vectores técnicos detectados:", len(vector_cols))

# --- (4) Construir variables de supervivencia ---
# duration: preferimos T_Lower; si hay NaNs, intentamos T_Upper como fallback.
df = df.copy()
df["duration"] = df["T_Lower"]
if "T_Upper" in df.columns:
    df["duration"] = df["duration"].fillna(df["T_Upper"])

# event: 1=empleado, 0=censurado
df["event"] = df["Event"].astype(int)

# Limpieza básica
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["duration", "event"])

# Cox requiere duration > 0 (si tienes ceros, aplicamos epsilon)
eps = 1e-3
df.loc[df["duration"] <= 0, "duration"] = eps

# --- (5) Función para z-score (HR comparable entre tópicos) ---
def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return None  # constante / inválida
    return (s - mu) / sd

# --- (6) Loop univariante Cox ---
rows = []
failures = []

for col in vector_cols:
    x = zscore(df[col])
    if x is None:
        continue

    tmp = pd.DataFrame({
        "duration": df["duration"].values,
        "event": df["event"].values,
        "x": x.values
    }).dropna()

    # mínimo de datos para evitar inestabilidad
    if tmp.shape[0] < 30 or tmp["x"].nunique() < 3:
        continue

    # fit Cox univariante
    cph = CoxPHFitter(penalizer=0.01)  # penalización ligera para convergencia
    try:
        cph.fit(tmp, duration_col="duration", event_col="event", show_progress=False)
        s = cph.summary.loc["x"]

        rows.append({
            "topic": col,
            "n": int(tmp.shape[0]),
            "events": int(tmp["event"].sum()),
            "coef": float(s["coef"]),
            "HR": float(np.exp(s["coef"])),
            "CI95_low": float(np.exp(s["coef lower 95%"])),
            "CI95_high": float(np.exp(s["coef upper 95%"])),
            "se": float(s["se(coef)"]),
            "z": float(s["z"]),
            "p": float(s["p"]),
        })

    except Exception as e:
        failures.append((col, str(e)))

cox_uni = pd.DataFrame(rows)

print("Modelos Cox univariantes OK:", len(cox_uni))
print("Fallos:", len(failures))
if len(failures) > 0:
    print("Ejemplo fallo:", failures[0])

# --- (7) Corrección por múltiples comparaciones (FDR-BH) ---
pvals = cox_uni["p"].values
rej, qvals, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
cox_uni["q"] = qvals
cox_uni["sig_fdr_0p05"] = rej

# --- (8) Ranking recomendado: primero q, luego |coef| ---
cox_rank = cox_uni.sort_values(["q", "coef"], ascending=[True, False]).reset_index(drop=True)

# Export (para Capítulo 3 / bitácora)
cox_rank.to_csv("cap3_univariate_cox_topics_fdr.csv", index=False)
print("✅ Exportado: cap3_univariate_cox_topics_fdr.csv")

# --- (9) Mostrar TOP significativos (si existen) ---
top_sig = cox_rank[cox_rank["sig_fdr_0p05"]].copy()
display(top_sig.head(25))

# --- (10) Plot: Top 15 por efecto (entre significativos). Si no hay, usa p sin FDR ---
plot_df = top_sig.copy()
title_extra = " (FDR<0.05)"
if plot_df.empty:
    plot_df = cox_rank.sort_values("p").head(15).copy()
    title_extra = " (TOP por p; sin significancia FDR)"

# usamos log(HR) para visualizar simétrico; log(HR)>0 => inserción más rápida
plot_df["logHR"] = np.log(plot_df["HR"])
plot_df = plot_df.sort_values("logHR")

plt.figure(figsize=(10, 7))
plt.barh(plot_df["topic"], plot_df["logHR"])
plt.axvline(0, linewidth=1)
plt.title("Univariate Cox por Tópico Técnico: efecto en velocidad de inserción" + title_extra)
plt.xlabel("log(HR)  ( >0 => inserción más rápida ; <0 => más lenta )")
plt.tight_layout()
plt.savefig(f'figures/Plot_{i}_{var}.png', bbox_inches='tight')
print(f'Saved: figures/Plot_{i}_{var}.png')
plt.show()


# %% [Code Cell]

# 2. Definir Variables
cols_soft = [
    'S1_Comunicacion_Esp', 'S2_Compromiso_Etico', 'S3_Trabajo_Equipo_Liderazgo',
    'S4_Resp_Social', 'S5_Gestion_Proyectos', 'S6_Aprendizaje_Digital', 'S7_Ingles'
]

# 3. Calcular Nulos
null_counts = df_exp_A[cols_soft].isnull().sum()
total_nulos = null_counts.sum()

# 4. Visualización de Integridad
plt.figure(figsize=(10, 6))

if total_nulos == 0:
    # MENSAJE DE ÉXITO (Datos Limpios)
    plt.text(0.5, 0.6, '✅ HABILIDADES BLANDAS COMPLETAS',
             ha='center', va='center', fontsize=20, color='#27ae60', fontweight='bold')
    plt.text(0.5, 0.4, f'Integridad del 100% en las 7 variables (n={len(df_exp_A)})',
             ha='center', va='center', fontsize=14, color='#34495e')

    # Marco estético
    rect = plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='#27ae60', linewidth=2, linestyle='--')
    plt.gca().add_patch(rect)
    plt.axis('off')

else:
    # GRÁFICO DE BARRAS (Si hubiera errores)
    null_pct = (null_counts / len(df_exp_A)) * 100
    sns.barplot(x=null_pct.index, y=null_pct.values, palette='Reds_r')
    plt.title('Valores Perdidos en Habilidades Blandas', fontweight='bold')
    plt.ylabel('% Nulos')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('11_diagnostico_nulos_soft_skills.png', dpi=300)
plt.show()

print(f"✅ Auditoría Finalizada: {total_nulos} valores nulos detectados.")

# %% [Code Cell]
# ==============================================================================
# AUDITORÍA DE CALIDAD: VALORES PERDIDOS EN VECTORES TÉCNICOS
# ==============================================================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Cargar Dataset
df_exp_A = pd.read_csv('dataset_part3_vectores.csv')

# 2. Identificar Columnas Técnicas
cols_conocidas = [
    'S1_Comunicacion_Esp', 'S2_Compromiso_Etico', 'S3_Trabajo_Equipo_Liderazgo',
    'S4_Resp_Social', 'S5_Gestion_Proyectos', 'S6_Aprendizaje_Digital', 'S7_Ingles',
    'TXT_Hard_Skills', 'Edad', 'Genero', 'Facultad', 'Cohorte', 'carrera',
    'T_Lower', 'T_Upper', 'Event', 'TARGET_Evento', 'TARGET_Tiempo', 'CARRERA_PARA_VECTORES',
    'Unnamed: 0'
]
vector_cols = [c for c in df_exp_A.columns if c not in cols_conocidas]

# 3. Calcular Nulos
null_counts = df_exp_A[vector_cols].isnull().sum()
total_nulos = null_counts.sum()

# 4. Visualización de Integridad
plt.figure(figsize=(12, 6))

if total_nulos == 0:
    # GRÁFICO DE ÉXITO (Todo Verde)
    plt.text(0.5, 0.6, '✅ INTEGRIDAD DE DATOS PERFECTA',
             ha='center', va='center', fontsize=22, color='#27ae60', fontweight='bold')
    plt.text(0.5, 0.4, f'0% de Valores Nulos en las {len(vector_cols)} Dimensiones Técnicas',
             ha='center', va='center', fontsize=14, color='#2c3e50')

    # Marco decorativo
    plt.gca().add_patch(plt.Rectangle((0.1, 0.1), 0.8, 0.8, fill=False, edgecolor='#27ae60', linewidth=3))
    plt.axis('off')

else:
    # GRÁFICO DE BARRAS (Si hubiera nulos)
    null_pct = (null_counts / len(df_exp_A)) * 100
    sns.barplot(x=null_pct.index, y=null_pct.values, color='#e74c3c')
    plt.xticks(rotation=90, fontsize=8)
    plt.title('Diagnóstico de Valores Perdidos por Vector', fontweight='bold')
    plt.ylabel('% de Nulos')

plt.tight_layout()
plt.savefig('09_diagnostico_nulos_vectores.png', dpi=300)
plt.show()

print(f"✅ Diagnóstico completado: {total_nulos} valores perdidos encontrados en la matriz técnica.")

# %% [Code Cell]
df_exp_A.columns.tolist()


# %% [Code Cell]
# =========================================================
# CELDA 1 — Distribución de intervalos (sanity check del target)
# =========================================================
assert all(c in df.columns for c in ["T_Lower", "T_Upper", "Event"]), "Faltan T_Lower/T_Upper/Event"

# Normaliza inf para gráficos auxiliares
df["_upper_is_inf"] = np.isinf(df["T_Upper"])
df["_width"] = np.where(np.isfinite(df["T_Upper"]), df["T_Upper"] - df["T_Lower"], np.nan)

# (A) Conteo por tipo de intervalo (incluye censura con upper=inf)
interval_counts = (
    df.groupby(["T_Lower", "T_Upper"], dropna=False)
      .size()
      .reset_index(name="n")
      .sort_values(["T_Lower", "T_Upper"], key=lambda s: s.replace(np.inf, 1e9))
)

plt.figure(figsize=(10, 5))
x_labels = interval_counts.apply(
    lambda r: f"[{r['T_Lower']}, {'inf' if np.isinf(r['T_Upper']) else r['T_Upper']}]",
    axis=1
)
sns.barplot(x=x_labels, y=interval_counts["n"])
plt.title("Conteo por intervalos (T_Lower, T_Upper)")
plt.xlabel("Intervalo")
plt.ylabel("n")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig('figures/Conteo_por_intervalos_T_Lower_T_Upper.png', bbox_inches='tight')
print('Saved figure to figures/Conteo_por_intervalos_T_Lower_T_Upper.png')
plt.show()




# %% [Code Cell]
# =========================
# CELDA 0 — Diagnóstico rápido + (opcional) upgrade
# =========================
import pandas as pd
import numpy as np
import lifelines

print("pandas:", pd.__version__)
print("lifelines:", lifelines.__version__)
print("Has KaplanMeierFitter.fit_interval_censoring?:", hasattr(getattr(lifelines, "KaplanMeierFitter", object), "fit_interval_censoring"))

# ✅ RECOMENDADO si te da KeyError con interval(...)
# (Ejecuta ESTA línea, y luego reinicia el runtime de Colab y re-ejecuta desde arriba)
# !pip -q install -U "lifelines>=0.30.0"


# %% [Code Cell]
df_exp_A['Genero'].unique()

# %% [Code Cell]
# =========================
# CELDA 2 — Curvas KM aproximadas (cotas) + guardado automático de figuras
#   Plotea 3 curvas:
#     - Optimista: usa T_Lower como tiempo del evento
#     - Intermedia: usa midpoint entre lower y upper (cap)
#     - Pesimista: usa T_Upper (cap para inf)
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

df_base = df_exp_A.copy()

def _sanitize_filename(s: str) -> str:
    s = str(s)
    for ch in [' ', '/', '\\', ':', ';', ',', '.', '|', '=', '(', ')', '[', ']', '{', '}', '"', "'"]:
        s = s.replace(ch, '_')
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")

def plot_km_bounds_and_save(df, lower="T_Lower", upper="T_Upper", event="Event",
                            label_prefix="Global", out_prefix="fig_cap3_km_bounds",
                            dpi=300):
    lo = pd.to_numeric(df[lower], errors="coerce").astype(float).clip(lower=0)
    up = pd.to_numeric(df[upper], errors="coerce").astype(float)
    e  = pd.to_numeric(df[event], errors="coerce").fillna(0).astype(int)

    # Cap para infinitos: máximo finito observado (si no existe, usa max(lower))
    up_finite = up.replace([np.inf, -np.inf], np.nan)
    cap = float(np.nanmax(up_finite.values)) if np.isfinite(np.nanmax(up_finite.values)) else float(lo.max())
    up_cap = up.replace([np.inf, -np.inf], cap)

    mid = (lo + up_cap) / 2.0

    fig, ax = plt.subplots(figsize=(9, 5))

    for name, durations in [
        ("Optimista (T_Lower)", lo),
        ("Intermedia (midpoint)", mid),
        ("Pesimista (T_Upper cap)", up_cap),
    ]:
        m = durations.notna() & e.notna()
        kmf = KaplanMeierFitter()
        kmf.fit(durations[m], event_observed=e[m], label=f"{label_prefix} — {name}")
        kmf.plot_survival_function(ax=ax, ci_show=False)

    ax.set_title("Curvas KM aproximadas (cotas)", fontweight="bold")
    ax.set_xlabel("Meses desde egreso (escala operacional)")
    ax.set_ylabel("S(t)")
    ax.grid(True, ls="--", alpha=0.4)

    plt.tight_layout()

    fname = f"{out_prefix}_{_sanitize_filename(label_prefix)}.png"
    fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.savefig('figures/Curvas_KM_aproximadas_cotas_fontweightbold.png', bbox_inches='tight')
    print('Saved figure to figures/Curvas_KM_aproximadas_cotas_fontweightbold.png')
    plt.show()

    print(f"✅ Guardado: {fname}")
    return fname

# (A) Global
plot_km_bounds_and_save(df_base, label_prefix="Global", out_prefix="fig_cap3_km_bounds")

# (B) Por Cohorte (si existe)
if "Cohorte" in df_base.columns:
    for coh, g in df_base.groupby("Cohorte"):
        plot_km_bounds_and_save(g, label_prefix=f"Cohorte_{coh}", out_prefix="fig_cap3_km_bounds")


# %% [Code Cell]
if "Facultad" in df_base.columns:
    for coh, g in df_base.groupby("Facultad"):
        plot_km_bounds_and_save(g, label_prefix=f"Facultad_{coh}", out_prefix="fig_cap3_km_bounds")

# %% [Code Cell]
if "Genero_bin" in df_base.columns:
    for coh, g in df_base.groupby("Genero_bin"):
        plot_km_bounds_and_save(g, label_prefix=f"Genero_{coh}", out_prefix="fig_cap3_km_bounds")

# %% [Code Cell]
import pandas as pd
import matplotlib.pyplot as plt

# % censura por cohorte
tab = (df_base
       .groupby("Cohorte")["Event"]
       .agg(n="count", events="sum"))
tab["censored"] = tab["n"] - tab["events"]
tab["censor_rate"] = tab["censored"] / tab["n"]

tab[["censor_rate"]].sort_values("censor_rate").plot(kind="bar", figsize=(8,4))
plt.title("Tasa de Censura por Cohorte")
plt.ylabel("Censura (Event=0) / N")
plt.grid(axis="y", ls="--", alpha=0.4)
plt.tight_layout()
plt.savefig('figures/Tasa_de_Censura_por_Cohorte.png', bbox_inches='tight')
print('Saved figure to figures/Tasa_de_Censura_por_Cohorte.png')
plt.show()

display(tab.sort_values("censor_rate", ascending=False))


# %% [Code Cell]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

def km_by_tertiles(df, topic_col, lower="T_Lower", upper="T_Upper", event="Event", cap=None):
    lo = pd.to_numeric(df[lower], errors="coerce").astype(float).clip(lower=0)
    up = pd.to_numeric(df[upper], errors="coerce").astype(float)

    # cap para infinito
    if cap is None:
        up_finite_max = np.nanmax(up.replace(np.inf, np.nan).values)
        cap = float(up_finite_max) if np.isfinite(up_finite_max) else float(lo.max())
    up_cap = up.replace(np.inf, cap)

    t_mid = (lo + up_cap) / 2.0
    e = pd.to_numeric(df[event], errors="coerce").fillna(0).astype(int)

    x = pd.to_numeric(df[topic_col], errors="coerce")
    q1, q2 = x.quantile([0.33, 0.66])

    groups = pd.cut(x, bins=[-np.inf, q1, q2, np.inf], labels=["Bajo", "Medio", "Alto"])
    fig, ax = plt.subplots(figsize=(9,5))

    for gname in ["Bajo","Medio","Alto"]:
        m = (groups == gname) & t_mid.notna() & e.notna()
        kmf = KaplanMeierFitter()
        kmf.fit(t_mid[m], event_observed=e[m], label=f"{topic_col} — {gname}")
        kmf.plot_survival_function(ax=ax, ci_show=False)

    ax.set_title(f"KM por terciles del tópico: {topic_col}", fontweight="bold")
    ax.set_xlabel("Meses desde egreso (escala operacional)")
    ax.set_ylabel("S(t)")
    ax.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig('figures/cap_para_infinito.png', bbox_inches='tight')
    print('Saved figure to figures/cap_para_infinito.png')
    plt.show()

# Ejemplo: cambia por uno de tus tópicos top
km_by_tertiles(df_base, "visualización de datos")


# %% [Code Cell]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

def km_by_tertiles_save(
    df,
    topic_col,
    lower="T_Lower",
    upper="T_Upper",
    event="Event",
    cap=None,
    outpath=None,
    title_prefix="KM por terciles",
):
    # --- limpiar columnas de supervivencia ---
    lo = pd.to_numeric(df[lower], errors="coerce").astype(float).clip(lower=0)
    up = pd.to_numeric(df[upper], errors="coerce").astype(float)
    e  = pd.to_numeric(df[event], errors="coerce").fillna(0).astype(int)

    # cap para inf
    if cap is None:
        up_finite_max = np.nanmax(up.replace(np.inf, np.nan).values)
        cap = float(up_finite_max) if np.isfinite(up_finite_max) else float(lo.max())
    up_cap = up.replace(np.inf, cap)

    # usamos midpoint como aproximación para graficar
    t_mid = (lo + up_cap) / 2.0

    # --- limpiar tópico ---
    x = pd.to_numeric(df[topic_col], errors="coerce")
    m0 = t_mid.notna() & e.notna() & x.notna()
    if m0.sum() < 20:
        raise ValueError(f"Pocos datos válidos para {topic_col}: {m0.sum()} filas.")

    q1, q2 = x[m0].quantile([0.33, 0.66])
    groups = pd.cut(x, bins=[-np.inf, q1, q2, np.inf], labels=["Bajo", "Medio", "Alto"])

    # --- plot ---
    fig, ax = plt.subplots(figsize=(9, 5))
    for gname in ["Bajo", "Medio", "Alto"]:
        m = m0 & (groups == gname)
        kmf = KaplanMeierFitter()
        kmf.fit(t_mid[m], event_observed=e[m], label=f"{gname} (n={m.sum()})")
        kmf.plot_survival_function(ax=ax, ci_show=False)

    ax.set_title(f"{title_prefix}: {topic_col}", fontweight="bold")
    ax.set_xlabel("Meses desde egreso (escala operacional)")
    ax.set_ylabel("S(t)")
    ax.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=300)
        print(f"✅ Guardado: {outpath}")

    plt.show()


# %% [Code Cell]
topics_top = [
    "etl, latex, lte",
    "visualización de datos",
    "aws, wireless",
]

# (A) Global
for t in topics_top:
    km_by_tertiles_save(
        df_base,
        topic_col=t,
        outpath=f"km_terciles_GLOBAL__{t.replace(',','_').replace(' ','_')}.png",
        title_prefix="KM terciles (GLOBAL)"
    )

# (B) Por Cohorte
if "Cohorte" in df_base.columns:
    for coh, g in df_base.groupby("Cohorte"):
        for t in topics_top:
            km_by_tertiles_save(
                g,
                topic_col=t,
                outpath=f"km_terciles_{coh}__{t.replace(',','_').replace(' ','_')}.png",
                title_prefix=f"KM terciles (Cohorte={coh})"
            )


# %% [Code Cell]
df_base.columns

# %% [Code Cell]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_target_sanity(df, lower="T_Lower", upper="T_Upper", event="Event",
                       outpath="fig_cap3_sanity_target_intervals.png"):

    lo = pd.to_numeric(df[lower], errors="coerce").astype(float)
    up = pd.to_numeric(df[upper], errors="coerce").astype(float)
    e  = pd.to_numeric(df[event], errors="coerce").fillna(0).astype(int)

    # checks básicos
    invalid = (lo > up) & np.isfinite(up)
    print("Intervalos inválidos (Lower>Upper finito):", int(invalid.sum()))

    # cap para inf (solo para graficar)
    cap = np.nanmax(up.replace(np.inf, np.nan).values)
    cap = float(cap) if np.isfinite(cap) else float(lo.max())
    up_cap = up.replace(np.inf, cap)

    fig, ax = plt.subplots(figsize=(9,5))
    ax.hist(lo[e==1].dropna(), bins=20, alpha=0.8, label="T_Lower (Event=1)")
    ax.hist(up_cap[e==1].dropna(), bins=20, alpha=0.8, label="T_Upper (cap, Event=1)")
    ax.set_title("Sanity check: distribución de límites del target (insertados)", fontweight="bold")
    ax.set_xlabel("Meses (escala operacional)")
    ax.set_ylabel("Frecuencia")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.show()
    print("✅ Guardado:", outpath)

plot_target_sanity(df_base)


# %% [Code Cell]
import seaborn as sns
import matplotlib.pyplot as plt

cols_soft = [
    'S1_Comunicacion_Esp','S2_Compromiso_Etico','S3_Trabajo_Equipo_Liderazgo',
    'S4_Resp_Social','S5_Gestion_Proyectos','S6_Aprendizaje_Digital','S7_Ingles'
]

corr_soft = df_base[cols_soft].corr(method="spearman")

plt.figure(figsize=(7,6))
sns.heatmap(corr_soft, annot=True, cmap="RdBu_r", center=0, linewidths=0.5)
plt.title("Correlación Spearman entre Soft Skills (S1–S7)", fontweight="bold")
plt.tight_layout()
plt.savefig("fig_cap3_corr_softskills.png", dpi=300)
plt.show()

print("✅ Guardado: fig_cap3_corr_softskills.png")
print("Max |rho|:", float(corr_soft.abs().where(~np.eye(len(cols_soft),dtype=bool)).max().max()))


# %% [Code Cell]
drop_cols = [c for c in ["TARGET_Evento","TARGET_Tiempo","TXT_Hard_Skills","CARRERA_PARA_VECTORES","Unnamed: 0"] if c in df_base.columns]

df_A = df_base.drop(columns=drop_cols).copy()

print("Droppped:", drop_cols)
print("Shape Experimento A:", df_A.shape)

df_A.to_csv("dataset_experimento_A.csv", index=False)
print("✅ Guardado: dataset_experimento_A.csv")


# %% [Code Cell]
!pip -q install Unidecode

# %% [Code Cell]
import re
import pandas as pd
from collections import Counter

# Si no lo tienes:
# !pip -q install Unidecode
from unidecode import unidecode

COL = "TXT_Hard_Skills"   # <-- tu columna

# Token: palabras con tildes/ñ y también frases unidas con "_" (aplicaciones_web, etc.)
TOKEN_RE = re.compile(r"[a-záéíóúüñ]+(?:_[a-záéíóúüñ]+)*", re.IGNORECASE)

def norm_key(s: str) -> str:
    """Clave de agrupación: minúscula + sin tildes + ñ->n (para detectar variantes)."""
    return unidecode(s.lower().strip())

# --- Sanity check ---
assert COL in df_base.columns, f"No existe la columna {COL} en df_base"
print("Filas:", len(df_base), " | Nulos:", df_base[COL].isna().sum())

# --- Construcción de tabla de tokens ---
rows = []
for i, txt in df_base[COL].fillna("").astype(str).items():
    toks = TOKEN_RE.findall(txt.lower())
    for t in toks:
        rows.append({"row_id": i, "raw": t, "norm": norm_key(t), "len": len(t)})

tok_df = pd.DataFrame(rows)

print("Tokens totales:", len(tok_df))
print("Tokens únicos (raw):", tok_df["raw"].nunique())
print("Tokens únicos (norm):", tok_df["norm"].nunique())

# Top 30 raw (sin filtrar nada aún)
display(tok_df["raw"].value_counts().head(30).to_frame("count"))


# %% [Code Cell]
import numpy as np

# Tabla de frecuencias por (norm, raw)
var_df = (
    tok_df.groupby(["norm", "raw"])
    .size()
    .reset_index(name="count")
)

# Para cada norm, resumen de variantes
def has_accent_or_enye(s: str) -> bool:
    return any(ch in s for ch in "áéíóúüñ")

summary = []
for norm, g in var_df.groupby("norm"):
    total = int(g["count"].sum())
    nvar  = int(g.shape[0])

    # variantes ordenadas
    g2 = g.sort_values("count", ascending=False)
    variants = list(zip(g2["raw"].tolist(), g2["count"].tolist()))

    # ¿existe variante con tilde/ñ?
    accented = [(r,c) for r,c in variants if has_accent_or_enye(r)]
    unaccent = [(r,c) for r,c in variants if not has_accent_or_enye(r)]

    # Canonical sugerido (solo sugerencia): si existe variante con tilde/ñ -> la más frecuente de esas; si no, la más frecuente
    if accented:
        canonical = max(accented, key=lambda x: x[1])[0]
    else:
        canonical = variants[0][0]

    summary.append({
        "norm": norm,
        "total_count": total,
        "n_variants": nvar,
        "canonical_suggested": canonical,
        "variants": " | ".join([f"{r}({c})" for r,c in variants[:10]]),
        "has_accent_variant": bool(accented),
        "has_unaccent_variant": bool(unaccent),
    })

audit_df = pd.DataFrame(summary)

# Candidatos fuertes a "mal escritas": norm con >=2 variantes
# (especialmente cuando coexisten variantes con y sin tilde/ñ)
cand = audit_df.query("n_variants >= 2").copy()

print("Candidatos con inconsistencias (n_variants>=2):", len(cand))
print("De ellos, con mezcla tilde/ñ vs sin tilde:", len(cand.query("has_accent_variant and has_unaccent_variant")))

# Top inconsistencias por impacto
display(
    cand.sort_values(["total_count","n_variants"], ascending=False)
        .head(40)[["norm","total_count","n_variants","canonical_suggested","variants"]]
)

# Vista específica: solo los casos donde hay mezcla con/sin tildes (los que te preocupan)
display(
    cand.query("has_accent_variant and has_unaccent_variant")
        .sort_values("total_count", ascending=False)
        .head(40)[["norm","total_count","canonical_suggested","variants"]]
)


# %% [Code Cell]
# Frecuencia raw y norm
raw_counts  = tok_df["raw"].value_counts()
norm_counts = tok_df["norm"].value_counts()

# Heurísticas típicas de "posible typo"
suspects = tok_df.drop_duplicates("raw").copy()
suspects["raw_count"]  = suspects["raw"].map(raw_counts)
suspects["norm_count"] = suspects["norm"].map(norm_counts)

# 1) Hapax: aparece 1 vez (ojo: puede ser término técnico real)
hapax = suspects.query("raw_count == 1").sort_values(["len"], ascending=False)

# 2) Muy largos (pueden ser pegotes raros)
very_long = suspects.query("len >= 25").sort_values("len", ascending=False)

# 3) Norm aparece mucho pero raw aparece poco -> variante minoritaria (frecuente en errores de acento)
minor_variants = suspects.query("norm_count >= 20 and raw_count <= 2").sort_values(["norm_count","raw_count"], ascending=False)

print("Hapax (raw_count=1):", len(hapax))
print("Muy largos (len>=25):", len(very_long))
print("Variantes minoritarias (norm_count>=20 y raw_count<=2):", len(minor_variants))

display(hapax.head(50)[["raw","raw_count","norm","norm_count","len"]])
display(very_long.head(50)[["raw","raw_count","norm","norm_count","len"]])
display(minor_variants.head(50)[["raw","raw_count","norm","norm_count","len"]])

# Exportar todo para revisión manual (recomendado)
audit_out = audit_df.sort_values(["n_variants","total_count"], ascending=False)
audit_out.to_csv("audit_variantes_hard_skills.csv", index=False, encoding="utf-8")
print("✅ Guardado: audit_variantes_hard_skills.csv")


# %% [Code Cell]
import os
import pandas as pd
from IPython.display import display

# Ajusta si tu archivo está en otra ruta
audit_path = "audit_variantes_hard_skills.csv"
if not os.path.exists(audit_path):
    audit_path = "/mnt/data/audit_variantes_hard_skills.csv"  # fallback por si estás en otro entorno

audit = pd.read_csv(audit_path)

# Inconsistencias = normas con >=2 variantes (tildes/ñ vs sin tilde, typos, etc.)
inc_after = audit[audit["n_variants"] >= 2].copy()

print("✅ Archivo audit cargado:", audit_path)
print("INCONSISTENCIAS (después):", len(inc_after))
print("Total normas auditadas:", len(audit))

# Muestra las más relevantes (más frecuentes primero)
display(inc_after.sort_values(["total_count","n_variants"], ascending=False).head(25))


# %% [Code Cell]
import pandas as pd
from IPython.display import display


audit_path = "audit_variantes_hard_skills.csv"
  # tu archivo
audit = pd.read_csv(audit_path)

inc_current = audit[audit["n_variants"] >= 2].copy()

print("INCONSISTENCIAS (estado actual):", len(inc_current))
print("Total normas (tokens normalizados):", len(audit))

display(
    inc_current.sort_values(["total_count", "n_variants"], ascending=False)
              .head(30)
)


# %% [Code Cell]
import re
import pandas as pd

audit_path = "audit_variantes_hard_skills.csv"  # el que subiste
audit = pd.read_csv(audit_path)

def parse_variants(variants_str):
    """
    Convierte: 'programación(106) | programacion(12)'
    en: ['programación', 'programacion']
    """
    if pd.isna(variants_str) or not str(variants_str).strip():
        return []
    parts = [p.strip() for p in str(variants_str).split("|")]
    out = []
    for p in parts:
        # quita el contador final "(123)"
        p = re.sub(r"\(\d+\)\s*$", "", p).strip()
        if p:
            out.append(p)
    return out

# mapping: cada variante -> canonical_suggested
variant_to_canon = {}

for _, r in audit.iterrows():
    canon = str(r["canonical_suggested"]).strip()
    vars_ = parse_variants(r.get("variants", ""))
    # incluir también el "norm" por si aparece como token
    norm = str(r.get("norm", "")).strip()
    candidates = set(vars_ + ([norm] if norm else []) + ([canon] if canon else []))

    for v in candidates:
        if v:
            variant_to_canon[v] = canon

print("✅ Variantes mapeadas:", len(variant_to_canon))
# muestra ejemplos
for k in list(variant_to_canon.keys())[:10]:
    print(f"{k}  ->  {variant_to_canon[k]}")


# %% [Code Cell]
import numpy as np
import re

def replace_variants_with_canon(text, mapping):
    if pd.isna(text):
        return text
    s = str(text)

    # ordena por longitud desc para evitar conflictos (si existieran)
    keys = sorted(mapping.keys(), key=len, reverse=True)

    for v in keys:
        canon = mapping[v]
        # reemplazo solo si v aparece como token completo (no dentro de otra palabra)
        # \w en python incluye letras con tilde/ñ/underscore/dígitos
        pattern = r"(?<!\w)" + re.escape(v) + r"(?!\w)"
        s = re.sub(pattern, canon, s)
    return s

# df_base debe existir (tu dataframe principal)
# Creamos nueva columna para no perder el original
df_base["TXT_Hard_Skills_clean"] = df_base["TXT_Hard_Skills"].apply(lambda x: replace_variants_with_canon(x, variant_to_canon))

print("✅ Columna creada: TXT_Hard_Skills_clean")
print(df_base[["TXT_Hard_Skills", "TXT_Hard_Skills_clean"]].head(5))


# %% [Code Cell]
import unicodedata
from collections import Counter, defaultdict
import pandas as pd
import re

def strip_accents(s):
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", s)

def tokenize_basic(text):
    if pd.isna(text):
        return []
    return re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", str(text).lower())

def audit_inconsistencies(text_series, min_total=20):
    raw_counts = Counter()
    norm_to_variants = defaultdict(Counter)

    for t in text_series:
        for tok in tokenize_basic(t):
            raw_counts[tok] += 1
            norm = strip_accents(tok)
            norm_to_variants[norm][tok] += 1

    rows = []
    for norm, variants in norm_to_variants.items():
        total = sum(variants.values())
        if total >= min_total and len(variants) >= 2:
            rows.append({
                "norm": norm,
                "total_count": total,
                "n_variants": len(variants),
                "variants": " | ".join([f"{k}({v})" for k, v in variants.most_common()])
            })

    # 👇 Garantiza columnas aunque rows esté vacío
    audit_df = pd.DataFrame(rows, columns=["norm", "total_count", "n_variants", "variants"])

    # 👇 Solo ordena si hay filas
    if not audit_df.empty:
        audit_df = audit_df.sort_values(["total_count", "n_variants"], ascending=False).reset_index(drop=True)

    return audit_df

# --- antes ---
audit_before = audit_inconsistencies(df_base["TXT_Hard_Skills"], min_total=20)
inc_before = audit_before[audit_before["n_variants"] >= 2]

# --- después ---
audit_after = audit_inconsistencies(df_base["TXT_Hard_Skills_clean"], min_total=20)
inc_after = audit_after[audit_after["n_variants"] >= 2]

print("INCONSISTENCIAS (antes):", len(inc_before))
print("INCONSISTENCIAS (después):", len(inc_after))

display(inc_after.head(30))


# %% [Code Cell]
# =========================
# CELDA B1 — Texto global limpio (ya con tildes/ñ corregidas)
# =========================
import pandas as pd

assert "TXT_Hard_Skills_clean" in df_base.columns, "Falta TXT_Hard_Skills_clean"

texto_completo = " ".join(
    df_base["TXT_Hard_Skills_clean"]
      .dropna()
      .astype(str)
      .tolist()
)

print("✅ texto_completo creado")
print("Longitud (chars):", len(texto_completo))
print("Ejemplo:", texto_completo[:250], "...")


# %% [Code Cell]
stop_words_es = {
    'de','la','que','el','en','y','a','los','del','se','las','por','un','para','con','no','una','su','al','lo',
    'como','más','pero','sus','le','ya','o','este','sí','porque','esta','entre','cuando','muy','sin','sobre',
    'también','me','hasta','donde','quien','desde','todo','nos','durante','todos','uno','les','ni','contra',
    'otros','ese','eso','ante','ellos','e','esto','mí','antes','algunos','qué','unos','yo','otro','otras','otra',
    'él','tanto','esa','estos','mucho','quienes','nada','cursos','conocimiento'
}


# %% [Code Cell]
# =========================
# CELDA B2 — Top términos y Top bigramas (CORREGIDA + filtrando stopwords)
# =========================
import re
from collections import Counter

def tokens_alpha(text):
    # Solo tokens alfabéticos (incluye tildes y ñ)
    return re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", str(text).lower())

# 1) Tokenizar
tokens = tokens_alpha(texto_completo)

# 2) Filtrar stopwords (usa el set stop_words_es que definiste en B3)
# Si aún no existe stop_words_es en tu notebook, comenta esta línea.
tokens_f = [t for t in tokens if t not in stop_words_es]

# 3) Top términos (unigramas)
top_terms = Counter(tokens_f).most_common(30)

# 4) Top bigramas (pares consecutivos) sobre tokens filtrados
bigrams = list(zip(tokens_f[:-1], tokens_f[1:]))
top_bigrams = Counter(bigrams).most_common(30)

print("Top 30 términos (unigramas):")
for w, c in top_terms:
    print(f"{w:25s} {c}")

print("\nTop 30 bigramas:")
for (w1, w2), c in top_bigrams:
    bigram_str = f"{w1}_{w2}"
    print(f"{bigram_str:25s} {c}")


# %% [Code Cell]
# =========================
# CELDA B3 — WordCloud con bigramas (usando TXT_Hard_Skills_clean)
# =========================
from wordcloud import WordCloud
import matplotlib.pyplot as plt

stop_words_es = {
    'de','la','que','el','en','y','a','los','del','se','las','por','un','para','con','no','una','su','al','lo',
    'como','más','pero','sus','le','ya','o','este','sí','porque','esta','entre','cuando','muy','sin','sobre',
    'también','me','hasta','donde','quien','desde','todo','nos','durante','todos','uno','les','ni','contra',
    'otros','ese','eso','ante','ellos','e','esto','mí','antes','algunos','qué','unos','yo','otro','otras','otra',
    'él','tanto','esa','estos','mucho','quienes','nada','cursos','conocimiento'
}

# stopwords de contexto (ajusta si quieres)
stop_words_es.update({
    'conocimientos','habilidades','uso','manejo','basico','básico','intermedio','avanzado',
    'puedo','experiencia','nivel','capacidad'
})

wc = WordCloud(
    width=1000,
    height=600,
    background_color='white',
    max_words=120,
    stopwords=stop_words_es,
    collocations=True,      # bigramas
    min_word_length=3,
    random_state=42
).generate(texto_completo)

plt.figure(figsize=(15, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('Análisis Lexométrico: Hard Skills EPN (texto normalizado)', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('B_wordcloud_hard_skills_clean.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Guardado: B_wordcloud_hard_skills_clean.png")


# %% [Code Cell]
print("Columnas disponibles:")
print([c for c in df_base.columns if "TXT_Hard_Skills" in c])

print("\n¿Existe TXT_Hard_Skills_clean?:", "TXT_Hard_Skills_clean" in df_base.columns)


# %% [Code Cell]
m = (
    df_base["TXT_Hard_Skills"].fillna("").astype(str)
    !=
    df_base["TXT_Hard_Skills_clean"].fillna("").astype(str)
)

print("Filas modificadas por limpieza:", int(m.sum()))
display(df_base.loc[m, ["TXT_Hard_Skills", "TXT_Hard_Skills_clean"]].head(10))


# %% [Code Cell]
df_base['TXT_Hard_Skills_clean']

# %% [Code Cell]
# =========================
# CELDA BERT-1 — Setup (Sentence-BERT multilingüe)
# =========================
import numpy as np
import pandas as pd
import re

# (Opcional) si no tienes sentence-transformers en Colab:
# !pip -q install -U sentence-transformers

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(MODEL_NAME)

print("✅ Modelo cargado:", MODEL_NAME)


# %% [Code Cell]
# =========================
# CELDA BERT-2 — Detectar columnas de tópicos (vocabulario objetivo)
#   (excluye metadatos y deja solo las ~70 columnas técnicas)
# =========================
# Asegúrate que df_base ya exista
assert "df_base" in globals(), "df_base no existe en memoria. Ejecuta antes la celda donde lo creas/cargas."

# Nombre de columna limpia (por si cambiaste el nombre)
TEXT_COL = "TXT_Hard_Skills_clean"
if TEXT_COL not in df_base.columns:
    # fallback típico
    TEXT_COL = "TXT_Hard_Skills"
print("📌 Usando texto:", TEXT_COL)

cols_soft = [
    'S1_Comunicacion_Esp', 'S2_Compromiso_Etico', 'S3_Trabajo_Equipo_Liderazgo',
    'S4_Resp_Social', 'S5_Gestion_Proyectos', 'S6_Aprendizaje_Digital', 'S7_Ingles'
]

cols_meta = set(cols_soft + [
    'TXT_Hard_Skills', 'TXT_Hard_Skills_clean',
    'Edad', 'Genero', 'Facultad', 'Cohorte', 'carrera',
    'T_Lower', 'T_Upper', 'Event', 'TARGET_Evento', 'TARGET_Tiempo',
    'Genero_bin', 'CARRERA_PARA_VECTORES', 'Unnamed: 0'
])

topic_cols = [c for c in df_base.columns if c not in cols_meta]

print(f"✅ Tópicos detectados: {len(topic_cols)}")
print("Ejemplos:", topic_cols[:5])


# %% [Code Cell]
# =========================
# CELDA BERT-3 — Extraer frases/skills por individuo desde TXT_Hard_Skills_clean
#   (pensado para texto tipo: "programación, bases de datos, redes..." etc.)
# =========================
def extract_phrases(text: str):
    if pd.isna(text):
        return []
    s = str(text).lower()

    # separadores comunes en autorreporte
    parts = re.split(r"[,\n;/\|\t]+", s)

    # limpieza mínima (sin stopwords todavía, solo cortar basura)
    phrases = []
    for p in parts:
        p = p.strip()
        p = re.sub(r"\s+", " ", p)
        # descartar muy cortos o no informativos
        if len(p) < 3:
            continue
        # si quedó solo 1 letra o cosas raras
        if not re.search(r"[a-záéíóúüñ]", p):
            continue
        phrases.append(p)

    # deduplicar preservando orden
    seen = set()
    out = []
    for p in phrases:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out

df_base["_hs_phrases"] = df_base[TEXT_COL].apply(extract_phrases)

# construir lista única para embeddear una sola vez
all_phrases = [p for lst in df_base["_hs_phrases"] for p in lst]
unique_phrases = sorted(set(all_phrases))

print("✅ Frases totales (con repetición):", len(all_phrases))
print("✅ Frases únicas:", len(unique_phrases))
print("Ejemplo (fila 0):", df_base["_hs_phrases"].iloc[0][:10] if len(df_base) else "df vacío")


# %% [Code Cell]
# =========================
# CELDA BERT-4 — Embeddings + Matching frase→tópico (top1 y topk)
# =========================
# Prepara textos "bonitos" para BERT (tópicos como frases)
topic_texts = [str(c).replace("_", " ") for c in topic_cols]

# Embeddings (una sola vez)
topic_emb = model.encode(topic_texts, normalize_embeddings=True, show_progress_bar=True)
phrase_emb = model.encode(unique_phrases, normalize_embeddings=True, show_progress_bar=True)

# Similaridad coseno (frases x tópicos)
S = cosine_similarity(phrase_emb, topic_emb)

# Para cada frase: top1
top1_idx = S.argmax(axis=1)
top1_score = S.max(axis=1)

phrase_to_best = {
    unique_phrases[i]: (topic_cols[top1_idx[i]], float(top1_score[i]))
    for i in range(len(unique_phrases))
}

# Umbral inicial (ajustable)
THRESH = 0.45  # típico para MiniLM multi; ajusta con la celda de diagnóstico abajo
print("✅ Umbral inicial THRESH =", THRESH)

# Diagnóstico rápido: distribución de scores top1
scores = np.array([v[1] for v in phrase_to_best.values()])
print("Top1 score — min/median/p90/max:", scores.min(), np.median(scores), np.quantile(scores, 0.90), scores.max())


# %% [Code Cell]
# =========================
# CELDA BERT-5 — Construir tabla LONG (individuo x frase x tópico x score)
#   y guardar CSV para análisis
# =========================
rows = []
for i, phrases in enumerate(df_base["_hs_phrases"].tolist()):
    for ph in phrases:
        best_topic, score = phrase_to_best.get(ph, (None, None))
        if best_topic is None:
            continue
        rows.append({
            "row_id": i,
            "phrase": ph,
            "matched_topic": best_topic,
            "score": score,
            "accept": score >= THRESH
        })

matches_long = pd.DataFrame(rows)
print("✅ matches_long shape:", matches_long.shape)
display(matches_long.head(20))

matches_long.to_csv("BERT_skill_matches_long.csv", index=False)
print("✅ Guardado: BERT_skill_matches_long.csv")


# %% [Code Cell]
# =========================
# CELDA BERT-6 — Resumen por individuo (qué tópicos “le salen”)
#   (solo aceptados por THRESH)
# =========================
accepted = matches_long[matches_long["accept"] == True].copy()

# Top tópicos por fila
summary = (
    accepted.groupby(["row_id", "matched_topic"])
    .size()
    .reset_index(name="n_phrases_matched")
    .sort_values(["row_id", "n_phrases_matched"], ascending=[True, False])
)

# Top-5 por individuo (en una celda)
topk = 5
top5 = (
    summary.groupby("row_id")
    .head(topk)
    .groupby("row_id")
    .apply(lambda g: "; ".join([f"{r.matched_topic}({int(r.n_phrases_matched)})" for _, r in g.iterrows()]))
    .reset_index(name="bert_topics_top5")
)

df_base = df_base.copy()
df_base = df_base.merge(top5, left_index=True, right_on="row_id", how="left").drop(columns=["row_id"])
df_base["bert_topics_top5"] = df_base["bert_topics_top5"].fillna("")

display(df_base[[TEXT_COL, "bert_topics_top5"]].head(10))


# %% [Code Cell]
# =========================
# CELDA BERT-BOOST-1 — Preparar: topic_cols + accepted matches
# =========================
import numpy as np
import pandas as pd

assert "df_base" in globals(), "df_base no existe en memoria."
assert "matches_long" in globals(), "matches_long no existe. Ejecuta la celda donde creas matches_long (BERT-5)."

# Detectar columnas de tópicos (las ~70)
cols_soft = [
    'S1_Comunicacion_Esp','S2_Compromiso_Etico','S3_Trabajo_Equipo_Liderazgo',
    'S4_Resp_Social','S5_Gestion_Proyectos','S6_Aprendizaje_Digital','S7_Ingles'
]
cols_meta = set(cols_soft + [
    'TXT_Hard_Skills','TXT_Hard_Skills_clean',
    'Edad','Genero','Facultad','Cohorte','carrera',
    'T_Lower','T_Upper','Event','TARGET_Evento','TARGET_Tiempo',
    'Genero_bin','CARRERA_PARA_VECTORES','Unnamed: 0',
    '_hs_phrases'  # por si existe
])

topic_cols = [c for c in df_base.columns if c not in cols_meta]
print("✅ topic_cols:", len(topic_cols))

# accepted (frase→tópico) según tu umbral actual en matches_long
if "accept" not in matches_long.columns:
    raise ValueError("matches_long no tiene columna 'accept'. Revisa la celda BERT-5.")
accepted = matches_long[matches_long["accept"] == True].copy()

print("✅ matches aceptados:", len(accepted))
display(accepted.head(10))


# %% [Code Cell]
# =========================
# CELDA BERT-BOOST-2 — Construir señal por individuo: score_max y count por tópico
#   score_max[i,j] = mayor similitud (BERT) que tuvo el individuo i con el tópico j
#   count[i,j]     = cuántas frases le matchearon al tópico j
# =========================
n = len(df_base)
m = len(topic_cols)

topic_to_j = {t:j for j,t in enumerate(topic_cols)}

score_max = np.zeros((n, m), dtype=float)
count_mat = np.zeros((n, m), dtype=int)

for r in accepted.itertuples(index=False):
    i = int(r.row_id)
    t = r.matched_topic
    s = float(r.score)
    j = topic_to_j.get(t, None)
    if j is None:
        continue
    count_mat[i, j] += 1
    if s > score_max[i, j]:
        score_max[i, j] = s

print("✅ Matrices listas:", score_max.shape, count_mat.shape)


# %% [Code Cell]
# =========================
# CELDA BERT-BOOST-3 — Boostear los tópicos en df_base
#   Idea: topic_boosted = topic_original + alpha * score_max
#   alpha se calibra a la escala real de tus tópicos (median no-cero).
# =========================
# Asegurar numérico en tus tópicos
X_topics = df_base[topic_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)

vals = X_topics.to_numpy().ravel()
nz = vals[vals > 0]

median_nz = float(np.median(nz)) if nz.size else 1.0
BOOST_FACTOR = 2.0  # <-- sube a 3.0 o 4.0 si quieres aún más peso
alpha = BOOST_FACTOR * median_nz

print("📌 Escala tópicos (median no-cero):", median_nz)
print("📌 alpha (boost) =", alpha)

X_boosted = X_topics + alpha * pd.DataFrame(score_max, index=df_base.index, columns=topic_cols)

# Opción A (recomendada): NO pisar, crear df para Experimento B
df_expB = df_base.copy()
df_expB[topic_cols] = X_boosted

# Features de control (útiles para tesis y debugging)
df_expB["HS_bert_total_matches"] = count_mat.sum(axis=1)
df_expB["HS_bert_topics_touched"] = (score_max > 0).sum(axis=1)
df_expB["HS_bert_mean_score"] = np.where(
    df_expB["HS_bert_topics_touched"].to_numpy() > 0,
    score_max.sum(axis=1) / np.maximum((score_max > 0).sum(axis=1), 1),
    0.0
)

display(df_expB[["HS_bert_total_matches","HS_bert_topics_touched","HS_bert_mean_score"]].describe())


# %% [Code Cell]
# =========================
# CELDA BERT-BOOST-3.5 — Dropear columnas innecesarias (antes de guardar Experimento B)
# =========================
import pandas as pd

# >>> si tu DF se llama diferente, cambia aquí:
df = df_expB

# 1) Definir columnas core
cols_soft = [
    'S1_Comunicacion_Esp','S2_Compromiso_Etico','S3_Trabajo_Equipo_Liderazgo',
    'S4_Resp_Social','S5_Gestion_Proyectos','S6_Aprendizaje_Digital','S7_Ingles'
]

cols_targets = ['T_Lower','T_Upper','Event']

# 2) Metadatos que quieres conservar (ajusta si deseas)
cols_keep_meta = ['Edad','Genero_bin','Cohorte','Facultad','carrera']  # puedes quitar 'carrera' si no la usarás
# Si NO tienes Genero_bin, pero sí Genero, conserva Genero:
if 'Genero_bin' not in df.columns and 'Genero' in df.columns:
    cols_keep_meta = ['Edad','Genero','Cohorte','Facultad','carrera']

# 3) Detectar automáticamente las columnas de tópicos (las 70)
cols_conocidas = set(cols_soft + cols_targets + cols_keep_meta + [
    'TXT_Hard_Skills','TXT_Hard_Skills_clean',
    'TARGET_Evento','TARGET_Tiempo',
    'CARRERA_PARA_VECTORES','Unnamed: 0',
    '_hs_phrases','HS_bert_total_matches','HS_bert_topics_touched','HS_bert_mean_score','bert_topics_top5'
])

topic_cols = [c for c in df.columns if c not in cols_conocidas]

print("→ Tópicos detectados:", len(topic_cols))
# Si quieres revisar rápidamente:
# print(topic_cols[:10])

# 4) Columnas finales a mantener
final_keep = [c for c in (cols_soft + cols_keep_meta + cols_targets + topic_cols) if c in df.columns]

# 5) Dropear todo lo demás
drop_cols = [c for c in df.columns if c not in final_keep]
df_clean = df.drop(columns=drop_cols).copy()

print("\n✅ Columnas mantenidas:", len(df_clean.columns))
print("🗑️ Columnas dropeadas:", len(drop_cols))
print("\nEjemplos de dropeadas:", drop_cols[:20])

# 6) Validaciones mínimas
missing_core = [c for c in (cols_soft + cols_targets) if c not in df_clean.columns]
assert len(missing_core) == 0, f"Faltan columnas core: {missing_core}"
assert len(topic_cols) >= 30, "Muy pocos tópicos detectados. Revisa columnas conocidas/metadatos."

# 7) Reemplazar df_expB por versión limpia
df_expB = df_clean

df_expB.head()


# %% [Code Cell]
# =========================
# CELDA BERT-BOOST-4 — Guardar dataset Experimento B (LIMPIO)
# =========================
out = "dataset_experimento_B_boosted.csv"
df_expB.to_csv(out, index=False)
print("✅ Guardado:", out)
print("shape:", df_expB.shape)


# %% [Code Cell]
df_expB.columns

# %% [Code Cell]
df_expA = pd.read_csv("dataset_experimento_A.csv")
df_expA.shape

# %% [Code Cell]
# =========================
# CELDA CHECK-1 — Validación de dimensiones/columnas (A vs B)
# =========================
import pandas as pd
import numpy as np

print("A shape:", df_expA.shape)
print("B shape:", df_expB.shape)

colsA = list(df_expA.columns)
colsB = list(df_expB.columns)

setA, setB = set(colsA), set(colsB)

onlyA = sorted(setA - setB)
onlyB = sorted(setB - setA)
common = sorted(setA & setB)

print("\nColumnas solo en A:", len(onlyA))
print("Columnas solo en B:", len(onlyB))

if onlyA:
    print("  Ejemplos (A solo):", onlyA[:15])
if onlyB:
    print("  Ejemplos (B solo):", onlyB[:15])

same_set = (setA == setB)
same_order = (colsA == colsB)

print("\n✅ Mismo set de columnas?:", same_set)
print("✅ Mismo orden de columnas?:", same_order)

# Si quieres alinear B al orden de A para comparar fácilmente:
if same_set and not same_order:
    df_expB = df_expB[colsA].copy()
    print("↪️ Reordené B al orden de A.")


# %% [Code Cell]


