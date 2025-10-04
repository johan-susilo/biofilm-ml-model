## Team NCKU-Tainan 2025 Software Tool

The Biofilm Prediction Software Tool is a modular, containerized service that predicts and optimizes biofilm degradation using enzyme mixtures. 

It combines a FastAPI server (serving both the REST API and the web UI) with an XGBoost model trained on experimental data. 

For more details, feel free to visit our team wiki. The tool provides:

- Biofilm degradation predictions.
- Optimization of enzyme ratios.
- Suggestions for informative experiments.



## Installation

### Requirements
- Docker (for containerized deployment)
- Git (to clone the repository)

### Quick start
1) Clone the repository

```bash
git clone https://gitlab.igem.org/2025/software-tools/ncku-tainan
cd ncku-tainan
```

2) Build and run the software using Docker Compose

```bash
docker-compose up --build -d
```

3) Open the UI in a browser

- Web UI: http://localhost:8000/ui
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health



## Usage

### Web client
- Open http://localhost:8000/ui
- Add enzyme ratio rows and reaction time
- Click "Predict" to get predicted removal and uncertainty
- Use "Find Optimal Mix" to run an optimization over mixtures
- Use "Suggest New Experiment" to get candidate experiments to run


### Training (optional)
- To retrain the models using your own dataset, place a CSV file at `data/polished.csv` with the required columns and run the training script:

```bash
# Place your CSV as ml-model/polished.csv
# Required columns: dspb, dnase, prok, reaction_time, degrade
# Make sure you're located inside ncku-tainan main folder

#Give permission to run the script
chmod +x ./train.sh 
#Run the script
./train.sh
```

This will run the training pipeline and save models in `ml-model/`.

## API

The API is served by the FastAPI backend package (`api/`), with the ASGI entrypoint at `api/main.py` (import path `api.main:app`). Below is a concise list of available endpoints, their purpose, and short examples showing how to call them.

Endpoints

- GET /
    - Description: Basic API status and model availability.
    - Returns: JSON with keys `status`, `model`, `xgb_loaded`, `rf_loaded`.

- GET /health
    - Description: Simple health check for container orchestration.
    - Returns: `{ "status": "healthy" }`

- GET /training-schema-stats
    - Description: Parameter ranges used for frontend validation.
    - Returns: Ranges for `DspB_ratio`, `DNase_I_ratio`, `ProK_ratio`, `Total_Volume`, `pH`, `Temperature`, `Reaction_Time`, and `biofilm_age_hours`.

- POST /predict
    - Description: Predict biofilm removal percentage for a single experimental configuration.
    - Request JSON (example fields):
        - `DspB_ratio`, `DNase_I_ratio`, `ProK_ratio` (floats, non-negative; will be normalized to sum=1)
        - `Total_Volume` (float)
        - `pH` (float)
        - `Temperature` (float, currently fixed 43°C by the UI but accepted in schema)
        - `Reaction_Time` (float, hours)
        - `biofilm_age_hours` (float)
    - Response: `PredictionResponse` with `mean_prediction`, `prediction_interval_low`, `prediction_interval_high`, `epistemic_uncertainty`, etc.

### Example (curl):

```bash
curl -s -X POST http://localhost:8000/predict \
    -H 'Content-Type: application/json' \
    -d '{
        "DspB_ratio": 0.34,
        "DNase_I_ratio": 0.33,
        "ProK_ratio": 0.33,
        "Total_Volume": 100,
        "pH": 7.2,
        "Reaction_Time": 24,
        "biofilm_age_hours": 24
    }'
```


- GET /feature-importance
    - Description: Returns feature names and values (basic 4-feature importances: `Dispersin B`, `DNase I`, `Proteinase K`, `Reaction Time`).
    - Returns: `{ "features": [...], "values": [...] }`

- POST /optimal-mix
    - Description: Run Bayesian optimization (or fallback random search) to find an enzyme ratio and reaction time that maximizes predicted removal plus an uncertainty bonus.
    - Request JSON: `fixed_conditions` (dict), optional `prior_experiments` (list), `n_trials` (int)
    - Returns: optimal `ratios` (array), `integer_counts`, `predicted`, `uncertainty`, and `recommended_reaction_time`.

- GET /optimal-mix
    - Description: Lightweight default optimal mix used by the UI on startup (equal ratios, 24h).

- POST /suggest-experiments
    - Description: Active-learning suggestions for new experiments (prioritizes high-uncertainty regions). Returns up to 3 suggestions with `dspb`, `dnase`, `prok`, `reaction_time`, `predicted`, `uncertainty`, and `score`.

Notes
- The server exposes a static web UI at `/static/index.html` (and `/ui` redirects there).
- API docs (OpenAPI/Swagger) are available at `/docs` when the service is running.


## Project layout

```
ncku-tainan/
├── Dockerfile                 # Production build (uvicorn api.main:app)
├── docker-compose.yml         # Service orchestration
├── api/                       # FastAPI backend package (serves UI + API)
│   ├── main.py                # App factory and ASGI entrypoint (api.main:app)
│   ├── core/                  # Config and shared pydantic models
│   │   ├── config.py          # Model paths, feature names
│   │   └── models_io.py       # Request/response schemas
│   ├── ml/                    # Model loading and inference helpers
│   │   └── engine.py          # XGBoost/RF loading, prediction, importance
│   ├── optimization/          # Optimization and active-learning logic
│   │   └── optimizer.py       # Optimal mix and suggestions
│   └── routers/               # API route groupings
│       ├── health.py          # Health and status endpoints
│       ├── misc.py            # Schema stats and misc endpoints
│       ├── predict.py         # Prediction and feature importance
│       └── optimize.py        # Optimal mix and experiment suggestions
├── requirements.txt           # Python dependencies
├── start.sh                   # Helper to run the service
├── train.sh                   # Helper to run training
├── data/                      # Experiment data folder
│   └── polished.csv           # Training dataset 
├── ml-model/                  # ML training + models
│   ├── train.py               # Training pipeline
│   ├── xgb_biofilm_model.json # XGBoost prediction model
│   └── rf_uncertainty_model.joblib # Random Forest uncertainty model
└── static/                    # Web UI
    ├── index.html             # Main interface
    └── assets/                # JS modules and styles
```

## Contributing

We welcome contributions. To get started:
1) Open an issue or discussion describing your proposed change
2) Fork the repository and create a feature branch
3) Run tests / linting if available and open a pull request

Development notes
- Use `./start.sh` to run the app in development mode (or run `uvicorn api.main:app --host 0.0.0.0 --port 8000` after installing dependencies)
- Linting and tests are not required for small fixes, but please keep changes focused and documented

## Authors and acknowledgment

This software tool was developed by:

- Johan Susilo
