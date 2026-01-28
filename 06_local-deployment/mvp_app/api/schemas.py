from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class PredictionInput(BaseModel):
    # Soft Skills (1-5 Likert Scale)
    s1_comunicacion: float = Field(..., alias="S1", ge=1, le=5)
    s2_compromiso: float = Field(..., alias="S2", ge=1, le=5)
    s3_trabajo_equipo: float = Field(..., alias="S3", ge=1, le=5)
    s4_resp_social: float = Field(..., alias="S4", ge=1, le=5)
    s5_gestion_proyectos: float = Field(..., alias="S5", ge=1, le=5)
    s6_aprendizaje_digital: float = Field(..., alias="S6", ge=1, le=5)
    s7_ingles: float = Field(..., alias="S7", ge=1, le=5)
    
    age: int = Field(..., alias="Edad", ge=15, le=100)
    gender: int = Field(..., alias="Genero_bin", description="0: Female, 1: Male")
    
    career: str = Field(..., alias="Carrera", description="Name of the career/faculty")
    
    # Technical skills from frontend checkboxes
    technical_skills: Dict[str, float] = Field(default_factory=dict)

    def to_engine_dict(self) -> dict:
        """
        Converts to a simple dictionary for the SurvivalEngine.
        """
        return {
            "S1": self.s1_comunicacion,
            "S2": self.s2_compromiso,
            "S3": self.s3_trabajo_equipo,
            "S4": self.s4_resp_social,
            "S5": self.s5_gestion_proyectos,
            "S6": self.s6_aprendizaje_digital,
            "S7": self.s7_ingles,
            "Edad": float(self.age),
            "Genero_bin": float(self.gender),
            "Carrera": self.career,
            "technical_skills": self.technical_skills
        }

class SurvivalOutput(BaseModel):
    mu: float
    percentiles: Dict[str, float]
    survival_curve: List[Dict[str, float]]
    prob_6m: float
    algo: str = "Random Survival Forest"
    citations: Optional[str] = None
