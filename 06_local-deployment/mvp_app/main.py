from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from core.engine import SurvivalEngine
from api.schemas import PredictionInput, SurvivalOutput
import uvicorn
import os

app = FastAPI(
    title="RSF Survival Prediction MVP",
    description="API for predicting graduate employability survival time using Random Survival Forest.",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Engine
engine = SurvivalEngine()

# Templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=SurvivalOutput)
async def predict_survival(input_data: PredictionInput):
    try:
        # Convert Pydantic model to dictionary for the engine
        model_input = input_data.to_engine_dict()
        
        # Run prediction
        results = engine.predict(model_input)
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
