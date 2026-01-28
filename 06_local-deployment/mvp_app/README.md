# XGB-AFT Survival MVP

This is a Minimum Viable Product (MVP) for predicting graduate labor insertion using an XGBoost Accelerated Failure Time (AFT) model.

## Model Architecture
- **Model**: XGBoost AFT (LogNormal distribution).
- **Inputs**: 7 Soft Skills (S1-S7), Age, Gender, Career, Technical Skills.
- **Output**: Survival Function $S(t)$ and Percentiles ($p_{50}, p_{75}, p_{90}$).

## Directory Structure
- `main.py`: FastAPI application.
- `core/`: Business logic and model loading.
- `api/`: Pydantic schemas.
- `templates/`: Simple UI.

## Environment
- **Python**: 3.11.11
- **Venv**: `/home/desarrollo03/Documentos/UNIVERSIDAD/TIC/TIC-workspacev4-definitive/venv`

## Local Development
1. Activate venv:
   ```bash
   source ../../venv/bin/activate
   ```
2. Run server:
   ```bash
   uvicorn main:app --reload
   ```
3. Open browser: http://localhost:8000

## Deployment (Vercel)
This app is configured for Vercel with `@vercel/python`.
1. `npm install -g vercel` (if not installed)
2. `vercel`

## API Usage
**POST** `/predict`
```json
{
  "S1": 5.5,
  "S2": 6.0,
  "S3": 5.0,
  "S4": 4.5,
  "S5": 6.0,
  "S6": 5.5,
  "S7": 4.0,
  "Edad": 24,
  "Genero_bin": 1,
  "Carrera": "SOFTWARE"
}
```
