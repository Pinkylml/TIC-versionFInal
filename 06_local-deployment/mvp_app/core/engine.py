import numpy as np
import pandas as pd
import joblib
from .loader import ModelLoader

class SurvivalEngine:
    def __init__(self):
        self.loader = ModelLoader
        # Metadata
        self.meta = self.loader.get_metadata()
        self.feature_names = self.meta.get("datos", {}).get("feature_names", [])
        
    def _get_model(self):
        return self.loader.get_model()
        
    def _get_scaler(self):
        return self.loader.get_scaler()
        
    def _get_career_mapping(self):
        return self.loader.get_career_mapping()

    def preprocess(self, input_data: dict) -> np.ndarray:
        """
        Preprocess input dictionary for RSF model:
        1. Encode career
        2. Interaction term
        3. Align features
        4. Apply Scaler
        """
        # Create a deep copy to avoid modifying original
        data = input_data.copy()
        
        # 1. Career Encoding
        mapping = self._get_career_mapping()
        career_name = data.get("Carrera", "")
        # The mapping is Name -> ID. Let's find matches.
        # Normalize career name if needed, but let's assume it matches the keys in the JSON
        career_id = mapping.get(career_name, 0) # Default to 0 if not found
        data["carrera_encoded"] = float(career_id)
        
        # 2. Interaction Term
        gender = float(data.get("Genero_bin", 0))
        data["genero_x_carrera"] = gender * data["carrera_encoded"]
        
        # 3. Align Features (80 features expected by RSF)
        ser = pd.Series(0.0, index=self.feature_names)
        
        # Base mapping
        key_map = {
            "S1": "S1_Comunicacion_Esp",
            "S2": "S2_Compromiso_Etico",
            "S3": "S3_Trabajo_Equipo_Liderazgo",
            "S4": "S4_Resp_Social",
            "S5": "S5_Gestion_Proyectos",
            "S6": "S6_Aprendizaje_Digital",
            "S7": "S7_Ingles",
            "Edad": "Edad",
            "Genero_bin": "Genero_bin",
            "carrera_encoded": "carrera_encoded",
            "genero_x_carrera": "genero_x_carrera"
        }
        
        for input_key, feature_name in key_map.items():
            if input_key in data:
                ser[feature_name] = float(data[input_key])
        
        # 4. Smart Technical Skill Mapping (Keyword search in groups)
        # The frontend sends a dict of { checkbox_id: 0/1 }
        user_skills = data.get("technical_skills", {})
        selected_skills = [k.lower() for k, v in user_skills.items() if v > 0]
        
        if selected_skills:
            for feature_name in self.feature_names:
                # If feature_name is one of the 70 NLP groups (contains commas or keywords)
                for skill in selected_skills:
                    # Check if the user's selected skill (e.g. 'react') is in the group name ('react, revit')
                    if skill in feature_name.lower():
                        ser[feature_name] = 1.0
        
        # 5. Scale
        X_scaled = self._get_scaler().transform(ser.values.reshape(1, -1))
        return X_scaled

    def predict(self, input_data: dict) -> dict:
        """
        Predict survival curve using RSF.
        """
        X_scaled = self.preprocess(input_data)
        model = self._get_model()
        surv_funcs = model.predict_survival_function(X_scaled)
        surv_func = surv_funcs[0]
        # 3. Evaluate at time points (Higher resolution for smoother curves)
        times = np.linspace(0.0, 6.0, 100)
        S_t = surv_func(times)
        
        # Percentiles (Precise search)
        percentiles = {}
        for p_name, p_value in [("p25", 0.25), ("p50", 0.50), ("p75", 0.75)]:
            target_survival = 1.0 - p_value
            # Find the first time point where survival drops below threshold
            idx = np.where(S_t <= target_survival)[0]
            if len(idx) > 0:
                percentiles[p_name] = float(times[idx[0]])
            else:
                percentiles[p_name] = -1.0 # Not reached within 6 months
        
        # Employment Probability @ 6 months (Complementary probability)
        # 1 - S(6) = Prob(T <= 6)
        prob_6m = 1.0 - float(S_t[-1])
        
        return {
            "mu": 0.0,
            "percentiles": percentiles,
            "survival_curve": [{"t": float(t), "S_t": float(s)} for t, s in zip(times, S_t)],
            "prob_6m": prob_6m,
            "algo": "Random Survival Forest (RSF)",
            "citations": "Cando Santos (2026), Ishwaran et al. (2008)"
        }
