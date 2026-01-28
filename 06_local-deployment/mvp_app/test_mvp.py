import pandas as pd
import requests
import json
import os
from pathlib import Path

# Config
BASE_URL = "http://localhost:8000"
DATASET_PATH = Path("/home/desarrollo03/Documentos/UNIVERSIDAD/TIC/TIC-workspacev4-definitive/06_Deployment/df_expA_cleaned.csv")
OUTPUT_DIR = Path("mvp_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def run_tests():
    print(f"Loading dataset from {DATASET_PATH}...")
    if not DATASET_PATH.exists():
        print(f"Error: Dataset not found at {DATASET_PATH.resolve()}")
        # Try finding it in current dir or up
        return

    df = pd.read_csv(DATASET_PATH)
    
    # Pick 5 random rows
    sample = df.sample(5, random_state=42)
    
    results = []
    
    print("Running tests against API...")
    
    # Needs to match PredictionInput schema
    # df columns mapped to schema fields
    # S1_Comunicacion_Esp -> S1
    # ...
    # gender -> Genero_bin ?
    # career -> ?
    
    # We need to map df columns to schema inputs
    # Let's inspect columns briefly (simulation) or just try mapping based on known names
    # S1_Comunicacion_Esp -> S1
    
    for idx, row in sample.iterrows():
        # Construct payload
        payload = {
            "S1": row.get("S1_Comunicacion_Esp", 5.0),
            "S2": row.get("S2_Compromiso_Etico", 5.0),
            "S3": row.get("S3_Trabajo_Equipo_Liderazgo", 5.0),
            "S4": row.get("S4_Resp_Social", 5.0),
            "S5": row.get("S5_Gestion_Proyectos", 5.0),
            "S6": row.get("S6_Aprendizaje_Digital", 5.0),
            "S7": row.get("S7_Ingles", 5.0),
            "Edad": int(row.get("Edad", 25)),
            "Genero_bin": int(row.get("Genero_bin", 1)),
            # Use 'Career' or find one-hot?
            # The DF usually has one-hot columns for career.
            # We need to reverse one-hot or just send a dummy/default if not easily extractable
            # For this test, we might fallback to "SOFTWARE" if extracting from one-hot is hard
            # But let's try to find the col starting with carrera_clean_ that is 1
            "Carrera": "SOFTWARE" # Default for MVP test simplicity unless we parse row keys
        }
        
        # Try to find actual career
        for col in df.columns:
            if col.startswith("carrera_clean_") and row[col] == 1:
                career_name = col.replace("carrera_clean_", "")
                payload["Carrera"] = career_name
                break
        
        # Requests
        # Note: We rely on the server being UP. 
        # Alternatively, import app and run directly for unit testing without network
        # But 'run_tests' usually implies integration.
        # Since I cannot easily background the server and run this in one go without 'wait',
        # I will use TestClient from starlette/fastapi which is better for this environment!
        
        from fastapi.testclient import TestClient
        from main import app
        
        client = TestClient(app)
        
        try:
            response = client.post("/predict", json=payload)
            if response.status_code == 200:
                print(f"Row {idx}: Success")
                res_json = response.json()
                results.append({
                    "row_id": idx,
                    "input": payload,
                    "output": res_json
                })
            else:
                print(f"Row {idx}: Failed {response.text}")
        except Exception as e:
            print(f"Row {idx}: Error {e}")

    # Save results
    out_file = OUTPUT_DIR / "test_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    run_tests()
