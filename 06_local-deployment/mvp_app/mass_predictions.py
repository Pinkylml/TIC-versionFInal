"""
Script para hacer predicciones masivas y encontrar perfiles con inserción rápida.

Genera múltiples combinaciones de perfiles variando:
- Soft skills (S1-S7)
- Edad
- Género  
- Carrera
- Top technical skills

Identifica perfiles con menor tiempo de inserción (μ bajo, p50 bajo).
"""

import requests
import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm
import json

# Configuración
API_URL = "http://127.0.0.1:8000/predict"

# Carreras a probar
CARRERAS = [
    "SOFTWARE",
    "COMPUTACIÓN",
    "DESARROLLO DE SOFTWARE",
    "REDES Y TELECOMUNICACIONES",
    "ECONOMÍA",
    "INGENIERÍA CIVIL",
    "INGENIERÍA QUÍMICA",
    "ADMINISTRACIÓN DE EMPRESAS"
]

# Top technical skills de SHAP
TOP_TECH_SKILLS = [
    "react, revit",
    "estructura de datos, estructuras",
    "simulación, simulación de procesos",
    "telefonía ip, voz sobre ip",
    "finanzas, mercados financieros",
    "arquitectura de computadoras",
    "optimización, optimización de procesos",
    "análisis de datos, análisis de materiales, análisis de sistemas ...",
    "etl, latex, lte",
    "logística, supply chain"
]

def generate_base_profiles():
    """
    Genera perfiles base variando soft skills, edad, género y carrera.
    """
    profiles = []
    
    # Perfiles típicos de soft skills
    soft_skill_configs = [
        {"name": "Excelente", "S1": 5, "S2": 5, "S3": 5, "S4": 5, "S5": 5, "S6": 5, "S7": 5},
        {"name": "Muy Bueno", "S1": 4, "S2": 4, "S3": 4, "S4": 4, "S5": 4, "S6": 4, "S7": 4},
        {"name": "Bueno", "S1": 3, "S2": 3, "S3": 3, "S4": 3, "S5": 3, "S6": 3, "S7": 3},
        {"name": "Regular", "S1": 2, "S2": 2, "S3": 2, "S4": 2, "S5": 2, "S6": 2, "S7": 2},
        # Perfil mixto (fuerte técnico, débil soft)
        {"name": "Técnico Fuerte", "S1": 3, "S2": 5, "S3": 4, "S4": 3, "S5": 4, "S6": 5, "S7": 3},
    ]
    
    # Edades
    edades = [22, 25, 28, 30]
    
    # Géneros
    generos = [0, 1]  # 0=Femenino, 1=Masculino
    
    for skill_config in soft_skill_configs:
        for edad in edades:
            for genero in generos:
                for carrera in CARRERAS:
                    profile = {
                        "profile_name": f"{skill_config['name']}_E{edad}_G{genero}_{carrera[:15]}",
                        "S1": skill_config["S1"],
                        "S2": skill_config["S2"],
                        "S3": skill_config["S3"],
                        "S4": skill_config["S4"],
                        "S5": skill_config["S5"],
                        "S6": skill_config["S6"],
                        "S7": skill_config["S7"],
                        "Edad": edad,
                        "Genero_bin": genero,
                        "Carrera": carrera
                    }
                    profiles.append(profile)
    
    return profiles

def add_technical_skills_variations(base_profile, max_skills=3):
    """
    Genera variaciones del perfil base añadiendo combinaciones de technical skills.
    
    Args:
        base_profile: Perfil base
        max_skills: Número máximo de skills técnicas a activar simultáneamente
    """
    variations = []
    
    # Sin skills técnicas (baseline)
    profile_baseline = base_profile.copy()
    for skill in TOP_TECH_SKILLS:
        profile_baseline[skill] = 0
    profile_baseline["tech_config"] = "None"
    variations.append(profile_baseline)
    
    # Con 1 skill a la vez
    for skill in TOP_TECH_SKILLS[:5]:  # Solo top 5 para no explotar combinatoria
        profile_single = base_profile.copy()
        for s in TOP_TECH_SKILLS:
            profile_single[s] = 1 if s == skill else 0
        profile_single["tech_config"] = skill.split(",")[0]
        variations.append(profile_single)
    
    # Combinación de top 3 skills
    if max_skills >= 3:
        profile_top3 = base_profile.copy()
        for i, skill in enumerate(TOP_TECH_SKILLS):
            profile_top3[skill] = 1 if i < 3 else 0
        profile_top3["tech_config"] = "Top3_Combined"
        variations.append(profile_top3)
    
    return variations

def predict_profile(profile):
    """
    Hace predicción para un perfil usando la API.
    """
    # Preparar payload (excluir metadatos)
    payload = {k: v for k, v in profile.items() if k not in ["profile_name", "tech_config"]}
    
    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        if response.status_code == 200:
            result = response.json()
            return {
                "profile_name": profile.get("profile_name", "Unknown"),
                "tech_config": profile.get("tech_config", "Unknown"),
                "mu": result["mu"],
                "p50": result["percentiles"]["p50"],
                "p75": result["percentiles"]["p75"],
                "p90": result["percentiles"]["p90"],
                "S1": profile["S1"],
                "S2": profile["S2"],
                "S3": profile["S3"],
                "Edad": profile["Edad"],
                "Genero": "F" if profile["Genero_bin"] == 0 else "M",
                "Carrera": profile["Carrera"]
            }
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"Error en predicción: {e}")
        return None

def main():
    print("🚀 Generando perfiles base...")
    base_profiles = generate_base_profiles()
    print(f"✅ {len(base_profiles)} perfiles base generados")
    
    print("\n🔧 Añadiendo variaciones de technical skills...")
    all_profiles = []
    for base in tqdm(base_profiles[:50], desc="Expandiendo perfiles"):  # Limitar para prueba
        variations = add_technical_skills_variations(base, max_skills=3)
        all_profiles.extend(variations)
    
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
    output_file = "mass_predictions_results.csv"
    df.to_csv(output_file, index=False)
    print(f"💾 Resultados guardados en: {output_file}")
    
    # Análisis: Top perfiles con inserción más rápida
    print("\n" + "="*80)
    print("🏆 TOP 20 PERFILES CON INSERCIÓN MÁS RÁPIDA (menor p50)")
    print("="*80)
    top_fast = df.nsmallest(20, 'p50')
    print(top_fast[['profile_name', 'tech_config', 'mu', 'p50', 'p75', 'Carrera', 'S2', 'S3', 'S6']].to_string())
    
    # Estadísticas por carrera
    print("\n" + "="*80)
    print("📊 ESTADÍSTICAS POR CARRERA (p50 promedio)")
    print("="*80)
    career_stats = df.groupby('Carrera')['p50'].agg(['mean', 'min', 'max', 'count']).sort_values('mean')
    print(career_stats)
    
    # Impacto de technical skills
    print("\n" + "="*80)
    print("🔧 IMPACTO DE TECHNICAL SKILLS (p50 promedio)")
    print("="*80)
    tech_stats = df.groupby('tech_config')['p50'].agg(['mean', 'count']).sort_values('mean')
    print(tech_stats.head(15))
    
    # Guardar análisis adicional
    with open("mass_predictions_analysis.txt", "w") as f:
        f.write("="*80 + "\n")
        f.write("ANÁLISIS DE PREDICCIONES MASIVAS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total perfiles evaluados: {len(results)}\n")
        f.write(f"μ promedio: {df['mu'].mean():.4f}\n")
        f.write(f"p50 promedio: {df['p50'].mean():.2f} meses\n")
        f.write(f"p50 mínimo: {df['p50'].min():.2f} meses\n")
        f.write(f"p50 máximo: {df['p50'].max():.2f} meses\n\n")
        
        f.write("\nTOP 10 PERFILES MÁS RÁPIDOS:\n")
        f.write("-"*80 + "\n")
        for i, row in top_fast.head(10).iterrows():
            f.write(f"{row['profile_name']} | {row['tech_config']}\n")
            f.write(f"  → μ={row['mu']:.4f}, p50={row['p50']:.2f}, Carrera={row['Carrera']}\n")
            f.write(f"  → Skills: S2={row['S2']}, S3={row['S3']}, S6={row['S6']}\n\n")
    
    print("\n✅ Análisis completo guardado en: mass_predictions_analysis.txt")

if __name__ == "__main__":
    main()
