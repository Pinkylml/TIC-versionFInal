import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sentence_transformers import SentenceTransformer, util
import os

# Ensure output directory exists
if not os.path.exists('figures-v2'):
    os.makedirs('figures-v2')

print("✅ Loading Data...")
try:
    df_julio = pd.read_excel('../ENCUESTAS/Julio.xlsx')
    df_dic = pd.read_excel('../ENCUESTAS/Diciembre.xlsx')
except Exception as e:
    print(f"❌ Error loading files: {e}")
    exit()

def extraer_definicion_habilidad(col_name):
    # Captura todo el contenido dentro de los corchetes [...]
    match = re.search(r'\[(.*?)\]', col_name)
    if match:
        full_tag = match.group(1).strip()
        # Expecting format like "S1: Comunicación" or just "S1"
        # We want to use the "S#" as the sorting key if present
        return full_tag
    return None

patron_pregunta = "A continuación, por favor evalúa"
skills_julio = {}
skills_dic = {}

print("✅ Extracting Skills...")
for col in df_julio.columns:
    if patron_pregunta in col:
        def_texto = extraer_definicion_habilidad(col)
        if def_texto:
            skills_julio[col] = def_texto

for col in df_dic.columns:
    if patron_pregunta in col:
        def_texto = extraer_definicion_habilidad(col)
        if def_texto:
            skills_dic[col] = def_texto

# SORTING LOGIC: Extract S1, S2 etc to sort
def get_sort_key(full_col_name):
    tag = extraer_definicion_habilidad(full_col_name)
    if tag:
        # Extract number from S1, S2...
        match_num = re.search(r'S(\d+)', tag)
        if match_num:
            return int(match_num.group(1))
    return 999 

# Sort keys based on S# number
sorted_keys_julio = sorted(skills_julio.keys(), key=get_sort_key)
sorted_keys_dic = sorted(skills_dic.keys(), key=get_sort_key)

# Prepare lists in sorted order
textos_julio = [skills_julio[k] for k in sorted_keys_julio]
textos_dic = [skills_dic[k] for k in sorted_keys_dic]

# Short Labels for Plotting
labels_julio = [extraer_definicion_habilidad(k).split(':')[0].strip() for k in sorted_keys_julio]
labels_dic = [extraer_definicion_habilidad(k).split(':')[0].strip() for k in sorted_keys_dic]

print(f"✅ Sorted Labels Julio: {labels_julio}")
print(f"✅ Sorted Labels Dic: {labels_dic}")

print("✅ Loading Model & Encoding...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

embeddings_julio = model.encode(textos_julio, convert_to_tensor=True)
embeddings_dic = model.encode(textos_dic, convert_to_tensor=True)

print("✅ Calculating Cosine Similarity...")
cosine_scores = util.cos_sim(embeddings_julio, embeddings_dic)

# Visualización
plt.figure(figsize=(12, 8))
sns.heatmap(cosine_scores.cpu(), annot=True, cmap='RdBu_r', fmt=".2f",
            xticklabels=labels_dic,
            yticklabels=labels_julio, vmin=0, vmax=1)

plt.title('Matriz de Homologación Semántica (BERT) - ORDENADA')
plt.xlabel('Experiment B (Diciembre)')
plt.ylabel('Experiment A (Julio)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

out_path = 'figures-v2/Matriz_de_Homologación_Semántica_BERTnJustificació_FIXED.png'
plt.savefig(out_path, bbox_inches='tight')
print(f"✅ Saved sorted heatmap to {out_path}")
