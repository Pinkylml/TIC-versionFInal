import pandas as pd
from pathlib import Path

class CareerVectorLoader:
    """Loads career-to-technical-skills vectors from CSV."""
    _instance = None
    _vectors_df = None
    
    @classmethod
    def get_vectors(cls):
        if cls._vectors_df is None:
            # Relative path from this file
            base_path = Path(__file__).resolve().parent.parent  # mvp_app/
            csv_path = base_path / "data" / "Vectores_Academicos_69d.csv"
            
            if not csv_path.exists():
                raise FileNotFoundError(f"Career vectors CSV not found at: {csv_path}")
            
            cls._vectors_df = pd.read_csv(csv_path, index_col=0)
            print(f"[INFO] Loaded career vectors: {cls._vectors_df.shape}")
        
        return cls._vectors_df
    
    @classmethod
    def get_vector_for_career(cls, career_name: str):
        """
        Returns the technical skills vector for a given career.
        career_name should match the index in the CSV.
        """
        df = cls.get_vectors()
        
        # Normalize career name
        career_clean = career_name.strip().upper()
        
        if career_clean not in df.index:
            print(f"[WARNING] Career '{career_clean}' not found in vectors. Using zeros.")
            # Return a dict with all columns = 0
            return {col: 0.0 for col in df.columns}
        
        # Return as dict
        return df.loc[career_clean].to_dict()
