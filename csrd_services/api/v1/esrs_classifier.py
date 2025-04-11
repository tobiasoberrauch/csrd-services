from enum import Enum
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from csrd_services.config import settings
from csrd_services.core.embedder import classify_file, classify_text, classify_text_file, ESRS_CATEGORIES

router = APIRouter()


class ModelName(str, Enum):
    multiLingual_Mpnet_v2 = "paraphrase-multilingual-mpnet-base-v2"
    multiLingual_Distiluse_v1 = "distiluse-base-multilingual-cased-v1"
    multiLingual_Distiluse_v2 = "distiluse-base-multilingual-cased-v2"
    multiLingual_MiniLM_L12_v2 = "paraphrase-multilingual-MiniLM-L12-v2"
    german_Paraphrase_Cosine = "deutsche-telekom/gbert-large-paraphrase-cosine"

@router.get("/models", summary="Get available models")
async def get_available_models():
    return {"available_models": list(ModelName)}


@router.get("/categories", summary="Get available esrs categories")
async def get_available_categories():
    return ESRS_CATEGORIES


@router.post("/upload", summary="Upload and classify a file")
async def upload_file(File: UploadFile = File(...), Model: ModelName = Form(...)):
    max_size_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    contents = await File.read()

    if len(contents) > max_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {settings.MAX_FILE_SIZE_MB} MB",
        )

    file_extension = File.filename.split(".")[-1].lower()
    if file_extension not in settings.SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_extension}. Supported formats: {", ".join(settings.SUPPORTED_FORMATS)}",
        )

    try:
        if file_extension == "txt":
            result = await classify_text_file(file_extension, contents, Model)
        else:
            result = await classify_file(file_extension, contents, Model)
        return {"filename": File.filename, "classification_result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


"""classifying input text endpoint"""


@router.post("/text", summary="Classify plain text")
async def classify_text_endpoint(
    Text: List[str] = Query(description="Input text to be classified"),
    Model: ModelName = Form(...),
):
    if not Text or not any(Text):
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    try:
        results = classify_text(Text, Model)
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error classifying text: {str(e)}")
