"""
Script para analizar rangos de tiempo de inserción y enriquecer el reporte.
Categorías: 0-1 mes, < 3 meses, < 4 meses, Más Lentos.
"""

import pandas as pd

# Cargar resultados v2
df = pd.read_csv('mass_predictions_v2_results.csv')

print("="*80)
print("📊 ANÁLISIS DETALLADO POR RANGOS DE TIEMPO")
print("="*80)

# 1. Rango 0-1 Mes
range_0_1 = df[df['p50'] <= 1.0]
print(f"\n🟢 Rango 0-1 Mes: {len(range_0_1)} perfiles")
if not range_0_1.empty:
    print(range_0_1[['Carrera', 'soft_config', 'p50']].head())

# 2. Rango 0-3 Meses (Ya analizado, pero para confirmar)
range_0_3 = df[df['p50'] <= 3.0]
print(f"\n🟡 Rango 0-3 Meses: {len(range_0_3)} perfiles")
print(range_0_3['Carrera'].value_counts())

# 3. Rango 0-4 Meses
range_0_4 = df[df['p50'] <= 4.0]
print(f"\n🟠 Rango 0-4 Meses: {len(range_0_4)} perfiles")
print("Top Carreras en < 4 meses:")
print(range_0_4['Carrera'].value_counts())
print("\nTop Configuraciones Soft en < 4 meses:")
print(range_0_4['soft_config'].value_counts())

# 4. Los Más Rápidos (Top 5 absoluto)
print(f"\n🏆 TOP 5 MÁS RÁPIDOS ABSOLUTOS:")
print(df.nsmallest(5, 'p50')[['Carrera', 'soft_config', 'p50', 'Edad', 'Genero']])

# 5. Los Más Lentos (Top 10 absoluto)
print(f"\n🐢 TOP 10 MÁS LENTOS ABSOLUTOS:")
slowest = df.nlargest(10, 'p50')[['Carrera', 'soft_config', 'p50', 'Edad', 'Genero']]
print(slowest)

# 6. Análisis por Carrera (Min/Max/Promedio)
print(f"\n📈 RESUMEN POR CARRERA (Ordenado por p50 promedio):")
print(df.groupby('Carrera')['p50'].agg(['min', 'mean', 'max']).sort_values('mean'))

# 7. Análisis por Perfil Soft (Min/Max/Promedio)
print(f"\n🧠 RESUMEN POR PERFIL SOFT (Ordenado por p50 promedio):")
print(df.groupby('soft_config')['p50'].agg(['min', 'mean', 'max']).sort_values('mean'))
