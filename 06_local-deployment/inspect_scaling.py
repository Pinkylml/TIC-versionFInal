import sys
import os
import pandas as pd
import numpy as np
import joblib

# Add app to path
sys.path.append(os.path.join(os.getcwd(), 'mvp_app'))

from core.engine import SurvivalEngine

def inspect_scaling():
    engine = SurvivalEngine()
    
    input_data = {
        "S1": 3.0, "S2": 3.0, "S3": 3.0, "S4": 3.0, "S5": 3.0, "S6": 3.0, "S7": 3.0,
        "Edad": 24, "Genero_bin": 1, "Carrera": "(RRA20) SOFTWARE",
        "technical_skills": {}
    }
    
    mapping = engine._get_career_mapping()
    print(f"Carrera: {input_data['Carrera']} -> ID: {mapping.get(input_data['Carrera'])}")
    
    # Get raw vector
    X_raw = engine.preprocess(input_data) # This actually returns scaled unfortunately in my current implementation
    
    # Let's modify engine.py temporarily or re-implement logic here to see RAW vs SCALED
    scaler = engine._get_scaler()
    
    # 1. Career Encoding
    data = input_data.copy()
    career_id = mapping.get(data["Carrera"], 0)
    data["carrera_encoded"] = float(career_id)
    data["genero_x_carrera"] = float(data["Genero_bin"]) * data["carrera_encoded"]
    
    # 2. Raw Vector
    ser = pd.Series(0.0, index=engine.feature_names)
    key_map = {
        "S1": "S1_Comunicacion_Esp", "S2": "S2_Compromiso_Etico", "S3": "S3_Trabajo_Equipo_Liderazgo",
        "S4": "S4_Resp_Social", "S5": "S5_Gestion_Proyectos", "S6": "S6_Aprendizaje_Digital", "S7": "S7_Ingles",
        "Edad": "Edad", "Genero_bin": "Genero_bin", "carrera_encoded": "carrera_encoded", "genero_x_carrera": "genero_x_carrera"
    }
    for k, v in key_map.items():
        ser[v] = float(data[k])
        
    print("\n--- Top 11 Raw Features ---")
    print(ser.head(11))
    
    # 3. Scaled
    X_scaled = scaler.transform(ser.values.reshape(1, -1))[0]
    scaled_ser = pd.Series(X_scaled, index=engine.feature_names)
    
    print("\n--- Top 11 Scaled Features ---")
    print(scaled_ser.head(11))
    
    print(f"\nTotal scaled sum: {np.sum(X_scaled):.4f}")

if __name__ == "__main__":
    inspect_scaling()
