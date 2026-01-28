import sys
import os
import pandas as pd
import numpy as np
import joblib

# Add app to path
sys.path.append(os.path.join(os.getcwd(), 'mvp_app'))

from core.engine import SurvivalEngine

def debug_rsf():
    engine = SurvivalEngine()
    
    # Test cases
    cases = [
        {
            "name": "High Skills (All 5), Male, Software, With Tech Skills",
            "input": {
                "S1": 5.0, "S2": 5.0, "S3": 5.0, "S4": 5.0, "S5": 5.0, "S6": 5.0, "S7": 5.0,
                "Edad": 24, "Genero_bin": 1, "Carrera": "(RRA20) SOFTWARE",
                "technical_skills": {"react, revit": 1.0, "sql, python": 1.0, "base de datos": 1.0}
            }
        },
        {
            "name": "Low Skills (All 1), Female, Software, No Tech Skills",
            "input": {
                "S1": 1.0, "S2": 1.0, "S3": 1.0, "S4": 1.0, "S5": 1.0, "S6": 1.0, "S7": 1.0,
                "Edad": 30, "Genero_bin": 0, "Carrera": "(RRA20) SOFTWARE",
                "technical_skills": {}
            }
        }
    ]
    
    for case in cases:
        print(f"\n--- {case['name']} ---")
        # Preprocess manually to see feature vector
        X_scaled = engine.preprocess(case['input'])
        print(f"X_scaled sum: {np.sum(X_scaled):.4f}")
        print(f"X_scaled mean: {np.mean(X_scaled):.4f}")
        
        # Prediction
        results = engine.predict(case['input'])
        p50 = results['percentiles']['p50']
        print(f"p50: {p50:.2f} months")
        
        # Check first 5 points of survival curve
        curve = results['survival_curve']
        print("S(t) points [0, 1, 2, 3, 4, 5]:")
        for i in range(0, 60, 10):
            t = curve[i]['t']
            s = curve[i]['S_t']
            print(f"  t={t:.1f}: S(t)={s:.4f}")

if __name__ == "__main__":
    debug_rsf()
