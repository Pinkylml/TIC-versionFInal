import joblib
import json
import os
from pathlib import Path

class ModelLoader:
    _model = None
    _scaler = None
    _metadata = None
    _career_mapping = None
    
    @classmethod
    def get_base_path(cls):
        return Path(__file__).resolve().parent.parent

    @classmethod
    def get_model(cls):
        if cls._model is None:
            path = cls.get_base_path() / "models" / "model.joblib"
            if not path.exists():
                raise FileNotFoundError(f"Model not found at: {path}")
            cls._model = joblib.load(path)
            print(f"[INFO] RSF Model loaded from {path}")
        return cls._model

    @classmethod
    def get_scaler(cls):
        if cls._scaler is None:
            path = cls.get_base_path() / "models" / "scaler.joblib"
            if not path.exists():
                raise FileNotFoundError(f"Scaler not found at: {path}")
            cls._scaler = joblib.load(path)
            print(f"[INFO] Scaler loaded from {path}")
        return cls._scaler

    @classmethod
    def get_metadata(cls):
        if cls._metadata is None:
            path = cls.get_base_path() / "models" / "metadata.json"
            if not path.exists():
                # Fallback to some defaults if metadata missing
                return {"feature_names": []}
            with open(path, 'r', encoding='utf-8') as f:
                cls._metadata = json.load(f)
        return cls._metadata

    @classmethod
    def get_career_mapping(cls):
        if cls._career_mapping is None:
            path = cls.get_base_path() / "data" / "mapeo_carrera.json"
            if not path.exists():
                raise FileNotFoundError(f"Career mapping not found at: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
                # Reverse mapping: Name -> ID
                cls._career_mapping = {v: int(k) for k, v in mapping.items()}
            print(f"[INFO] Career mapping loaded with {len(cls._career_mapping)} entries")
        return cls._career_mapping
