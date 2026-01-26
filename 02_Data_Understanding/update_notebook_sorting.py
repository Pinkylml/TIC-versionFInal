import json

nb_path = 'CorreccionDescriptivo-part1-v2.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The target cell contains:
target_snippet = "model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# We want to replace the list creation lines:
# textos_julio = list(skills_julio.values())
# with a block that sorts them first.

replacement_code = """
# 1. Cargar Modelo SOTA (State of the Art)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# --- FIX: SORT KEYS TO ENSURE DIAGONAL ALIGNMENT ---
def get_sort_key(full_col_name):
    # Extract S1, S2... from the key string
    match = re.search(r'S(\d+)', full_col_name)
    if match:
        return int(match.group(1))
    return 999

# Sort the keys
sorted_keys_julio = sorted(skills_julio.keys(), key=get_sort_key)
sorted_keys_dic = sorted(skills_dic.keys(), key=get_sort_key)

# 2. Preparar listas de textos y etiquetas cortas para el gráfico (ORDENADO)
textos_julio = [skills_julio[k] for k in sorted_keys_julio]
textos_dic = [skills_dic[k] for k in sorted_keys_dic]

# Etiquetas cortas para visualización (eje X e Y)
labels_julio = [k.split('[')[1].split(':')[0].strip() for k in sorted_keys_julio]
labels_dic = [k.split('[')[1].split(':')[0].strip() for k in sorted_keys_dic]
# ---------------------------------------------------
"""

count = 0
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if target_snippet in source_str:
            # Found the cell. Now we replace the lines.
            # We will replace the top part of the cell up to the cosine calculation line if possible,
            # or just replace the whole cell content if we are sure.
            # The user provided the cell content in the prompt.
            # Let's match line by line or Replace the top section.
            
            new_source = []
            
            # We'll rewrite the source completely based on the user's provided snippet + fix
            new_source_lines = [
                "# 1. Cargar Modelo SOTA (State of the Art)\n",
                "model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')\n",
                "\n",
                "# --- FIX: ORDERNAR CLAVES PARA ALINEAR DIAGONAL ---\n",
                "def get_sort_key(full_col_name):\n",
                "    match = re.search(r'S(\d+)', full_col_name)\n",
                "    if match:\n",
                "        return int(match.group(1))\n",
                "    return 999\n",
                "\n",
                "sorted_keys_julio = sorted(skills_julio.keys(), key=get_sort_key)\n",
                "sorted_keys_dic = sorted(skills_dic.keys(), key=get_sort_key)\n",
                "# --------------------------------------------------\n",
                "\n",
                "# 2. Preparar listas de textos y etiquetas cortas para el gráfico\n",
                "textos_julio = [skills_julio[k] for k in sorted_keys_julio]\n",
                "textos_dic = [skills_dic[k] for k in sorted_keys_dic]\n",
                "\n",
                "# Etiquetas cortas para visualización (eje X e Y)\n",
                "labels_julio = [k.split('[')[1].split(':')[0].strip() for k in sorted_keys_julio]\n",
                "labels_dic = [k.split('[')[1].split(':')[0].strip() for k in sorted_keys_dic]\n",
                "\n",
                "# 3. Generar Embeddings y Calcular Similitud\n",
                "embeddings_julio = model.encode(textos_julio, convert_to_tensor=True)\n",
                "embeddings_dic = model.encode(textos_dic, convert_to_tensor=True)\n",
                "\n",
                "cosine_scores = util.cos_sim(embeddings_julio, embeddings_dic)\n",
                "\n",
                "# 4. Visualización (Heatmap)\n",
                "plt.figure(figsize=(12, 8))\n",
                "sns.heatmap(cosine_scores.cpu(), annot=True, cmap='RdBu_r', fmt=\".2f\",\n",
                "            xticklabels=labels_dic,\n",
                "            yticklabels=labels_julio, vmin=0, vmax=1)\n",
                "\n",
                "plt.title('Matriz de Homologación Semántica (BERT)\\nJustificación para la Fusión de Variables')\n",
                "plt.xticks(rotation=45, ha='right')\n",
                "plt.tight_layout()\n",
                "plt.savefig('figures-v2/Matriz_de_Homologación_Semántica_BERTnJustificació.png', bbox_inches='tight')\n",
                "print(f'Saved figure to figures-v2/Matriz_de_Homologación_Semántica_BERTnJustificació.png')\n",
                "plt.show()\n"
            ]
            cell['source'] = new_source_lines
            count += 1

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"✅ Injected sorting logic into {count} cell(s) in {nb_path}")
