"""
Script mejorado de predicciones masivas (SIN DESARROLLO DE SOFTWARE)
- Varía soft skills en grid completo
- Identifica configuraciones óptimas (p50 < 3)
- Genera reporte detallado para replicar configuraciones ganadoras
"""

import requests
import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm

API_URL = "http://127.0.0.1:8000/predict"

# CARRERAS EXCLUYEN DESARROLLO DE SOFTWARE
CARRERAS = [
    "COMPUTACIÓN",
    "REDES Y TELECOMUNICACIONES",
    "ECONOMÍA",
    "INGENIERÍA CIVIL",
    "INGENIERÍA QUÍMICA",
    "ADMINISTRACIÓN DE EMPRESAS",
    "SOFTWARE",  # Mantener esta, solo excluir DESARROLLO DE SOFTWARE
]

# Top technical skills de SHAP
TOP_TECH_SKILLS = [
    "react, revit",
    "estructura de datos, estructuras",
    "simulación, simulación de procesos",
    "telefonía ip, voz sobre ip",
    "finanzas, mercados financieros",
]

def generate_varied_profiles():
    """
    Genera perfiles variando soft skills de forma más exhaustiva.
    """
    profiles = []
    
    # Configuraciones de soft skills variadas
    soft_skill_configs = [
        # Perfiles completos
        {"name": "Excelente_All", "S1": 5, "S2": 5, "S3": 5, "S4": 5, "S5": 5, "S6": 5, "S7": 5},
        {"name": "MuyBueno_All", "S1": 4, "S2": 4, "S3": 4, "S4": 4, "S5": 4, "S6": 4, "S7": 4},
        {"name": "Bueno_All", "S1": 3, "S2": 3, "S3": 3, "S4": 3, "S5": 3, "S6": 3, "S7": 3},
        
        # Perfiles focalizados en TOP SHAP (S2, S3, S6)
        {"name": "TopSHAP_Excelente", "S1": 3, "S2": 5, "S3": 5, "S4": 3, "S5": 3, "S6": 5, "S7": 3},
        {"name": "TopSHAP_MuyBueno", "S1": 3, "S2": 4, "S3": 4, "S4": 3, "S5": 3, "S6": 4, "S7": 3},
        
        # Perfiles desequilibrados (fuerte técnico, débil soft)
        {"name": "TecnicoFuerte_SoftDebil", "S1": 2, "S2": 2, "S3": 2, "S4": 2, "S5": 4, "S6": 5, "S7": 2},
        
        # Perfiles desequilibrados (débil técnico, fuerte soft)
        {"name": "SoftFuerte_TecnicoDebil", "S1": 5, "S2": 5, "S3": 5, "S4": 5, "S5": 2, "S6": 2, "S7": 5},
        
        # Mix intermedio
        {"name": "Mix_Equilibrado", "S1": 4, "S2": 4, "S3": 4, "S4": 3, "S5": 3, "S6": 4, "S7": 3},
    ]
    
    # Edades variadas
    edades = [22, 25, 28, 30, 35]
    
    # Géneros
    generos = [0, 1]
    
    for skill_config in soft_skill_configs:
        for edad in edades:
            for genero in generos:
                for carrera in CARRERAS:
                    profile = {
                        "profile_name": f"{skill_config['name']}_E{edad}_G{genero}_{carrera[:20]}",
                        "S1": skill_config["S1"],
                        "S2": skill_config["S2"],
                        "S3": skill_config["S3"],
                        "S4": skill_config["S4"],
                        "S5": skill_config["S5"],
                        "S6": skill_config["S6"],
                        "S7": skill_config["S7"],
                        "Edad": edad,
                        "Genero_bin": genero,
                        "Carrera": carrera,
                        "soft_config": skill_config['name']
                    }
                    profiles.append(profile)
    
    return profiles

def add_technical_variations(base_profiles, max_per_profile=5):
    """
    Añade variaciones de technical skills.
    """
    all_profiles = []
    
    for base in tqdm(base_profiles, desc="Añadiendo variaciones técnicas"):
        # Baseline sin skills técnicas
        profile_baseline = base.copy()
        for skill in TOP_TECH_SKILLS:
            profile_baseline[skill] = 0
        profile_baseline["tech_config"] = "None"
        all_profiles.append(profile_baseline)
        
        # Con 1 skill a la vez (solo top 3)
        for skill in TOP_TECH_SKILLS[:3]:
            profile_single = base.copy()
            for s in TOP_TECH_SKILLS:
                profile_single[s] = 1 if s == skill else 0
            profile_single["tech_config"] = skill.split(",")[0].strip()
            all_profiles.append(profile_single)
        
        # Combinación top 2
        profile_top2 = base.copy()
        for i, skill in enumerate(TOP_TECH_SKILLS):
            profile_top2[skill] = 1 if i < 2 else 0
        profile_top2["tech_config"] = "Top2_Combined"
        all_profiles.append(profile_top2)
    
    return all_profiles

def predict_profile(profile):
    """Hace predicción para un perfil."""
    payload = {k: v for k, v in profile.items() 
               if k not in ["profile_name", "tech_config", "soft_config"]}
    
    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            return {
                "profile_name": profile.get("profile_name", "Unknown"),
                "soft_config": profile.get("soft_config", "Unknown"),
                "tech_config": profile.get("tech_config", "Unknown"),
                "mu": result["mu"],
                "p50": result["percentiles"]["p50"],
                "p75": result["percentiles"]["p75"],
                "p90": result["percentiles"]["p90"],
                "S1": profile["S1"],
                "S2": profile["S2"],
                "S3": profile["S3"],
                "S4": profile["S4"],
                "S5": profile["S5"],
                "S6": profile["S6"],
                "S7": profile["S7"],
                "Edad": profile["Edad"],
                "Genero": "F" if profile["Genero_bin"] == 0 else "M",
                "Carrera": profile["Carrera"]
            }
        else:
            print(f"Error {response.status_code}: {response.text[:100]}")
            return None
    except Exception as e:
        print(f"Error: {str(e)[:100]}")
        return None

def main():
    print("🚀 Generando perfiles base (SIN DESARROLLO DE SOFTWARE)...")
    base_profiles = generate_varied_profiles()
    print(f"✅ {len(base_profiles)} perfiles base generados")
    
    print("\n🔧 Añadiendo variaciones técnicas...")
    # Limitar a primeros 100 perfiles base para no saturar
    all_profiles = add_technical_variations(base_profiles[:100])
    print(f"✅ {len(all_profiles)} perfiles totales a probar")
    
    print("\n🔮 Haciendo predicciones masivas...")
    results = []
    for profile in tqdm(all_profiles, desc="Prediciendo"):
        result = predict_profile(profile)
        if result:
            results.append(result)
    
    print(f"\n✅ {len(results)} predicciones exitosas")
    
    # Guardar resultados
    df = pd.DataFrame(results)
    output_file = "mass_predictions_v2_results.csv"
    df.to_csv(output_file, index=False)
    print(f"💾 Resultados guardados en: {output_file}")
    
    # ANÁLISIS CRÍTICO: Configuraciones con p50 < 3 meses
    print("\n" + "="*80)
    print("🎯 CONFIGURACIONES ÓPTIMAS (p50 < 3 meses)")
    print("="*80)
    optimal = df[df['p50'] < 3.0].sort_values('p50')
    
    if len(optimal) > 0:
        print(f"\n✅ Encontradas {len(optimal)} configuraciones óptimas\n")
        
        # Agrupar por configuración de soft skills
        print("📊 RESUMEN POR CONFIGURACIÓN SOFT SKILLS:")
        print("-"*80)
        soft_summary = optimal.groupby('soft_config').agg({
            'p50': ['mean', 'min', 'count'],
            'Carrera': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'N/A'
        }).round(2)
        print(soft_summary)
        
        print("\n📊 RESUMEN POR CARRERA:")
        print("-"*80)
        career_summary = optimal.groupby('Carrera')['p50'].agg(['mean', 'min', 'count']).sort_values('mean')
        print(career_summary)
        
        print("\n📊 TOP 20 CONFIGURACIONES GANADORAS:")
        print("-"*80)
        top20 = optimal.head(20)
        for i, row in top20.iterrows():
            print(f"\n🏆 #{len(optimal[optimal['p50'] <= row['p50']])} - p50 = {row['p50']:.2f} meses")
            print(f"   Perfil: {row['soft_config']}")
            print(f"   Carrera: {row['Carrera']}")
            print(f"   Edad: {row['Edad']}, Género: {row['Genero']}")
            print(f"   Soft Skills: S1={row['S1']}, S2={row['S2']}, S3={row['S3']}, S4={row['S4']}, S5={row['S5']}, S6={row['S6']}, S7={row['S7']}")
            print(f"   Tech Config: {row['tech_config']}")
        
        # Guardar reporte detallado
        with open("configuraciones_optimas.txt", "w") as f:
            f.write("="*80 + "\n")
            f.write("CONFIGURACIONES ÓPTIMAS PARA INSERCIÓN RÁPIDA (p50 < 3 meses)\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total configuraciones óptimas: {len(optimal)}\n")
            f.write(f"p50 mínimo alcanzado: {optimal['p50'].min():.2f} meses\n")
            f.write(f"p50 máximo en óptimas: {optimal['p50'].max():.2f} meses\n\n")
            
            f.write("GUÍA DE REPLICACIÓN:\n")
            f.write("-"*80 + "\n\n")
            
            for config in optimal['soft_config'].unique():
                subset = optimal[optimal['soft_config'] == config]
                f.write(f"\n### CONFIGURACIÓN: {config}\n")
                f.write(f"Cantidad de perfiles óptimos: {len(subset)}\n")
                f.write(f"p50 promedio: {subset['p50'].mean():.2f} meses\n")
                f.write(f"Carreras compatibles: {', '.join(subset['Carrera'].unique())}\n")
                
                # Tomar ejemplo representativo
                ejemplo = subset.iloc[0]
                f.write(f"\nEJEMPLO DE CONFIGURACIÓN:\n")
                f.write(f"  S1 (Comunicación Esp): {ejemplo['S1']}\n")
                f.write(f"  S2 (Compromiso Ético): {ejemplo['S2']}\n")
                f.write(f"  S3 (Trabajo Equipo): {ejemplo['S3']}\n")
                f.write(f"  S4 (Resp. Social): {ejemplo['S4']}\n")
                f.write(f"  S5 (Gestión Proyectos): {ejemplo['S5']}\n")
                f.write(f"  S6 (Aprendizaje Digital): {ejemplo['S6']}\n")
                f.write(f"  S7 (Inglés): {ejemplo['S7']}\n")
                f.write(f"  Edad recomendada: {subset['Edad'].mode()[0] if len(subset) > 0 else 'N/A'}\n")
                f.write(f"\n")
        
        print(f"\n✅ Reporte detallado guardado en: configuraciones_optimas.txt")
    else:
        print("\n⚠️ NO se encontraron configuraciones con p50 < 3 meses")
        print("Mostrando las 10 mejores:")
        print(df.nsmallest(10, 'p50')[['soft_config', 'Carrera', 'p50', 'S2', 'S3', 'S6', 'Edad']])
    
    # Stats globales
    print("\n" + "="*80)
    print("📊 ESTADÍSTICAS GLOBALES")
    print("="*80)
    print(f"p50 promedio: {df['p50'].mean():.2f} meses")
    print(f"p50 mínimo: {df['p50'].min():.2f} meses")
    print(f"p50 máximo: {df['p50'].max():.2f} meses")
    print(f"Desviación estándar: {df['p50'].std():.2f}")

if __name__ == "__main__":
    main()
