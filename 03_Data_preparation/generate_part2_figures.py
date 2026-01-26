
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from scipy.stats import spearmanr
from lifelines import CoxPHFitter
import os

# Ensure figures dir
os.makedirs('figures', exist_ok=True)
# sns.set_theme(style="whitegrid") # Removed
plt.rcParams['figure.figsize'] = (12, 6)

print("⏳ Cargando dataset pre-procesado (Experimento A)...")
try:
    # Load dataset that already has vectors integrated
    df = pd.read_csv('dataset_experimento_A.csv')
    print(f"✅ Dataset cargado. Dimensiones: {df.shape}")
except FileNotFoundError:
    print("❌ Error: No se encuentra dataset_experimento_A.csv")
    exit()

# Define columns
cols_soft = [
    'S1_Comunicacion_Esp', 'S2_Compromiso_Etico', 'S3_Trabajo_Equipo_Liderazgo',
    'S4_Resp_Social', 'S5_Gestion_Proyectos', 'S6_Aprendizaje_Digital', 'S7_Ingles'
]
cols_known = cols_soft + [
    'TXT_Hard_Skills', 'Edad', 'Genero', 'Facultad', 'Cohorte', 'carrera', 
    'T_Lower', 'T_Upper', 'Event', 'TARGET_Evento', 'TARGET_Tiempo', 
    'CARRERA_PARA_VECTORES', 'Unnamed: 0', 'Genero_bin'
]

# Identify Vector Columns (Technical Topics)
vector_cols = [c for c in df.columns if c not in cols_known and pd.api.types.is_numeric_dtype(df[c])]
print(f"ℹ️ Detectados {len(vector_cols)} vectores técnicos numéricos.")

# ================= 1. NULL DIAGNOSTICS (Soft Skills) =================
print("📊 Generando Diagnóstico Nulos Soft Skills...")
null_counts = df[cols_soft].isnull().sum()
total_nulos = null_counts.sum()

plt.figure(figsize=(10, 6))
if total_nulos == 0:
    plt.text(0.5, 0.6, '✅ HABILIDADES BLANDAS COMPLETAS', 
             ha='center', va='center', fontsize=20, color='#27ae60', fontweight='bold')
    plt.text(0.5, 0.4, f'Integridad del 100% en las 7 variables (n={len(df)})', 
             ha='center', va='center', fontsize=14, color='#34495e')
    rect = plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='#27ae60', linewidth=2, linestyle='--')
    plt.gca().add_patch(rect)
    plt.axis('off')
else:
    # Matplotlib Barplot alternative to Seaborn
    plt.bar(null_pct.index, null_pct.values, color='#e74c3c')
    plt.title('Valores Perdidos en Habilidades Blandas')
    plt.ylabel('% Nulos')
    plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('figures/11_diagnostico_nulos_soft_skills.png', dpi=300)
plt.close()

# ================= 2. NULL DIAGNOSTICS (Vectors) =================
print("📊 Generando Diagnóstico Nulos Vectores...")
null_counts_vec = df[vector_cols].isnull().sum()
total_nulos_vec = null_counts_vec.sum()

plt.figure(figsize=(12, 6))

# Always show the bar plot even if 0, to PROVE it
plt.bar(range(len(null_counts_vec)), null_counts_vec.values, color='#e74c3c', width=1.0, edgecolor='none')
plt.title(f'Diagnóstico de Integridad: {len(vector_cols)} Vectores Técnicos (Total Nulos: {total_nulos_vec})', fontweight='bold')
plt.xlabel('Índice del Vector (0-69)')
plt.ylabel('Cantidad de Nulos Detectados')
plt.ylim(0, 10)  # Zoom in to show even 1 null would be visible
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add text annotation
if total_nulos_vec == 0:
    plt.text(len(null_counts_vec)/2, 5, '✅ MATRIZ DENSE COMPLETA (0 Nulos)', 
             ha='center', va='center', fontsize=16, color='green', fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='green'))

plt.tight_layout()
plt.savefig('figures/09_diagnostico_nulos_vectores.png', dpi=300)
plt.close()

# ================= 3. HEATMAP SOFT vs TECH =================
print("📊 Generando Heatmap Soft vs Tech...")
# Ensure numeric
for c in cols_soft: df[c] = pd.to_numeric(df[c], errors='coerce')

# Slice relevant data
df_corr = df[cols_soft + vector_cols].copy()
corr_matrix = df_corr.corr(method='spearman', numeric_only=True)
soft_vs_tech = corr_matrix.loc[cols_soft, vector_cols]

plt.figure(figsize=(24, 8))
# Matplotlib Imshow (Heatmap)
im = plt.imshow(soft_vs_tech, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, label='Coeficiente Spearman')

plt.title('Matriz de Independencia: Habilidades Blandas vs. Perfil Técnico Académico', fontsize=18, fontweight='bold', pad=20)
plt.ylabel('Soft Skills', fontweight='bold')
plt.xlabel('Vectores Técnicos', fontweight='bold')

# Ticks
plt.xticks(np.arange(len(vector_cols)), vector_cols, rotation=90, fontsize=8)
plt.yticks(np.arange(len(cols_soft)), cols_soft, rotation=0, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/06_heatmap_soft_vs_tech.png', dpi=300)
plt.close()

# ================= 4. UNIVARIATE COX PLOT (Top 15) =================
print("📊 Generando Cox Univariante (Top 15)...")
# Prepare survival data
df_surv = df.copy()
df_surv["duration"] = df_surv["T_Lower"].fillna(df_surv["T_Upper"])
df_surv["event"] = df_surv["Event"].fillna(0).astype(int)
df_surv = df_surv.dropna(subset=["duration", "event"])
df_surv.loc[df_surv["duration"] <= 0, "duration"] = 0.001

# Univariate Cox Loop
def zscore(s):
    if s.std() == 0: return None
    return (s - s.mean()) / s.std()

rows = []
for col in vector_cols:
    x = zscore(df_surv[col])
    if x is None: continue
    
    tmp = pd.DataFrame({"duration": df_surv["duration"], "event": df_surv["event"], "x": x}).dropna()
    if len(tmp) < 30: continue
    
    try:
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(tmp, duration_col="duration", event_col="event")
        rows.append({
            "topic": col, 
            "HR": cph.hazard_ratios_["x"], 
            "p": cph.summary.loc["x", "p"]
        })
    except:
        continue

if rows:
    cox_df = pd.DataFrame(rows).sort_values("p")
    # Top 15 by significance (lowest p)
    plot_df = cox_df.head(15).copy()
    plot_df["logHR"] = np.log(plot_df["HR"])
    plot_df = plot_df.sort_values("logHR")
    
    plt.figure(figsize=(10, 7))
    plt.barh(plot_df["topic"], plot_df["logHR"], color='#3498db')
    plt.axvline(0, color='black', linewidth=1)
    plt.title("Univariate Cox: Top 15 Tópicos con Mayor Impacto (P-Value)")
    plt.xlabel("log(HR) (>0: Acelera inserción)")
    plt.tight_layout()
    plt.savefig('figures/Univariate_Cox_Plot_Top15.png', dpi=300)
    plt.close()
    print("✅ Cox Plot guardado.")
else:
    print("⚠️ No se pudieron ajustar modelos Cox.")

# ================= 5. INTERVAL COUNTS =================
print("📊 Generando Conteo Intervalos...")
interval_counts = df.groupby(["T_Lower", "T_Upper"], dropna=False).size().reset_index(name="n")
x_labels = interval_counts.apply(lambda r: f"[{r['T_Lower']}, {r['T_Upper']}]", axis=1)

plt.figure(figsize=(10, 5))
# Matplotlib Barplot
plt.bar(x_labels, interval_counts["n"], color="#2ecc71")
plt.title("Distribución de Intervalos de Tiempo (Censura por Intervalos)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig('figures/Conteo_por_intervalos.png', dpi=300)
plt.close()

# ================= 6. WORDCLOUD (Hard Skills) =================
print("📊 Generando WordCloud Hard Skills...")
try:
    from wordcloud import WordCloud
    
    # Custom stopwords from notebook
    stop_words_es = {
        'de','la','que','el','en','y','a','los','del','se','las','por','un','para','con','no','una','su','al','lo',
        'como','más','pero','sus','le','ya','o','este','sí','porque','esta','entre','cuando','muy','sin','sobre',
        'también','me','hasta','donde','quien','desde','todo','nos','durante','todos','uno','les','ni','contra',
        'otros','ese','eso','ante','ellos','e','esto','mí','antes','algunos','qué','unos','yo','otro','otras','otra',
        'él','tanto','esa','estos','mucho','quienes','nada','cursos','conocimiento',
        'conocimientos','habilidades','uso','manejo','basico','básico','intermedio','avanzado',
        'puedo','experiencia','nivel','capacidad'
    }

    # Prepare text
    # The 'clean' text might be in Part 1 CSV if not in Experiment A
    if 'TXT_Hard_Skills_clean' in df.columns:
        text_col = 'TXT_Hard_Skills_clean'
        df_wc = df
    elif 'TXT_Hard_Skills' in df.columns:
        text_col = 'TXT_Hard_Skills'
        df_wc = df
    else:
        print("ℹ️ Columna de texto no encontrada en Dataset A. Cargando Dataset Parte 1...")
        df_part1 = pd.read_csv('../02_Data_Understanding/dataset_part1_analisis.csv')
        # Check standard names in Part 1
        if 'TXT_Hard_Skills_clean' in df_part1.columns:
            text_col = 'TXT_Hard_Skills_clean'
            df_wc = df_part1
        else:
            text_col = 'TXT_Hard_Skills'
            df_wc = df_part1

    print(f"ℹ️ Usando columna '{text_col}' para WordCloud.")
    
    texto_completo = " ".join(df_wc[text_col].dropna().astype(str).tolist()) # Only non-nulls

    wc = WordCloud(
        width=1000, height=600, background_color='white',
        max_words=120, stopwords=stop_words_es, collocations=True,
        min_word_length=3, random_state=42,
        font_path='/usr/share/fonts/truetype/freefont/FreeMono.ttf'
    ).generate(texto_completo)

    plt.figure(figsize=(15, 10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Análisis Lexométrico: Hard Skills EPN', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/B_wordcloud_hard_skills_clean.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ WordCloud guardado.")

except ImportError:
    print("⚠️ Librería 'wordcloud' no instalada. Saltando generación de WordCloud.")
except Exception as e:
    print(f"⚠️ Error generando WordCloud: {e}")

print("✅ Generación de figuras finalizada.")
