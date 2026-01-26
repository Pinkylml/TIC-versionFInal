#!/usr/bin/env python3
"""
Script para ejecutar el notebook desde Python directamente usando el venv correcto.
"""
import json
import sys
from pathlib import Path

# Agregar path del venv al sys.path
venv_site_packages = Path(__file__).parent / 'venv' / 'lib' / 'python3.11' / 'site-packages'
sys.path.insert(0, str(venv_site_packages))

# Ahora importar las librerías del notebook
try:
    import pandas as pd
    import numpy as np
    from sentence_transformers import SentenceTransformer
    print("✅ Todas las librerías cargadas correctamente desde venv")
    print(f"  - pandas: {pd.__version__}")
    print(f"  - numpy: {np.__version__}")
    print(f"  - Python: {sys.version}")
except ImportError as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# Cargar el notebook
nb_path = Path(__file__).parent / '02_Data_Understanding' / 'CorrecciónDescriptivo-part1.ipynb'
with open(nb_path) as f:
    nb = json.load(f)

print(f"\n📓 Notebook cargado: {nb_path.name}")
print(f"   Total celdas: {len(nb['cells'])}")

# Ejecutar solo las primeras 3 celdas como prueba
print("\n🔬 Ejecutando primeras 3 celdas (TEST)...\n")

for i in range(min(3, len(nb['cells']))):
    cell = nb['cells'][i]
    if cell['cell_type'] == 'code':
        code = ''.join(cell['source'])
        print(f"\n{'='*60}")
        print(f"CELDA {i}: {'='*50}")
        print(f"{'='*60}")
        try:
            exec(code, globals())
        except Exception as e:
            print(f"❌ Error en celda {i}: {e}")
            break
    elif cell['cell_type'] == 'markdown':
        print(f"\n[Celda {i}: MARKDOWN - saltada]")

print("\n✅ Prueba completada")
