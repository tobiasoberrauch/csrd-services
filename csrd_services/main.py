import uvicorn
from csrd_services.api.v1.esrs_classifier import router as esrs_router
from csrd_services.config import settings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # Add this import


def create_app() -> FastAPI:
    app = FastAPI(
        title="CSRD Services",
        description="APIs for Document Matching, Classification, etc.",
        version="1.0.0",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8080", "http://localhost:3000"],  # Remove trailing slash
        allow_credentials=True,
        allow_methods=["*"],  # Allow all methods
        allow_headers=["*"],  # Allow all headers
    )
    
    app.include_router(esrs_router, prefix="/api/v1/esrs-classification", tags=["ESRS Classification"])
    return app


# Create the app instance at module level
app = create_app()


def run():
    uvicorn.run(
        app,
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        log_level=settings.LOG_LEVEL.lower(),
    )


def dev():
    uvicorn.run(
        "csrd_services.main:app",
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=True,
    )


if __name__ == "__main__":
    run()
