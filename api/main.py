"""App factory and ASGI entrypoint for the Biofilm Prediction API.

- Configures CORS and mounts the static UI under `/static`
- Provides a `/ui` redirect to `static/index.html`
- Registers routers for health, prediction, and optimization endpoints
- Loads ML models on startup so endpoints are ready to serve
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from starlette.responses import RedirectResponse

from .routers import health, predict, optimize, misc
from .ml.engine import load_models


def create_app() -> FastAPI:
    app = FastAPI(
        title="Biofilm Prediction API",
        version="1.0.0",
        description="Predict biofilm removal effectiveness using enzyme combinations",
    )

    # CORS settings (allow all origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static UI
    app.mount("/static", StaticFiles(directory="static"), name="static")

    # Redirect /ui -> /static/index.html
    @app.get("/ui", include_in_schema=False)
    def ui_redirect():
        return RedirectResponse(url="/static/index.html", status_code=307)

    # Register routers
    app.include_router(health.router)
    app.include_router(misc.router)
    app.include_router(predict.router)
    app.include_router(optimize.router)

    # Load models at startup
    @app.on_event("startup")
    def _startup_load_models():
        load_models()

    return app


# ASGI entrypoint (uvicorn: `uvicorn api.main:app`)
app = create_app()