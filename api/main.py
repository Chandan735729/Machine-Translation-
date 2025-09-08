"""
FastAPI Backend for English-Assamese Translation Platform
Provides REST API endpoints for translation services
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import logging
import os
import sys
import uvicorn

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from translate import EnglishToAssameseTranslator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Modal Translation Platform",
    description="English-Assamese Translation API for NGOs and Organizations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Global translator instance
translator = None

# Pydantic models for request/response
class TranslationRequest(BaseModel):
    text: str
    source_language: str = "en"
    target_language: str = "as"
    max_length: Optional[int] = 512

class BatchTranslationRequest(BaseModel):
    texts: List[str]
    source_language: str = "en"
    target_language: str = "as"
    max_length: Optional[int] = 512

class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: Optional[float] = None

class BatchTranslationResponse(BaseModel):
    translations: List[TranslationResponse]
    total_count: int

class HealthResponse(BaseModel):
    status: str
    message: str
    model_loaded: bool

@app.on_event("startup")
async def startup_event():
    """Initialize the translation model on startup"""
    global translator
    try:
        logger.info("Loading translation model...")
        translator = EnglishToAssameseTranslator()
        logger.info("Translation model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load translation model: {e}")
        translator = None

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main frontend page"""
    try:
        with open("frontend/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
            <head><title>Translation Platform</title></head>
            <body>
                <h1>Multi-Modal Translation Platform</h1>
                <p>API is running! Visit <a href="/docs">/docs</a> for API documentation.</p>
                <p>Frontend not found. Please ensure frontend files are in the 'frontend' directory.</p>
            </body>
        </html>
        """)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_loaded = translator is not None
    status = "healthy" if model_loaded else "unhealthy"
    message = "Translation service is ready" if model_loaded else "Translation model not loaded"
    
    return HealthResponse(
        status=status,
        message=message,
        model_loaded=model_loaded
    )

@app.post("/api/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """
    Translate single text from English to Assamese
    """
    if translator is None:
        raise HTTPException(
            status_code=503, 
            detail="Translation service unavailable. Model not loaded."
        )
    
    try:
        logger.info(f"Translating text: {request.text[:50]}...")
        
        # Currently only supports English to Assamese
        if request.source_language != "en" or request.target_language != "as":
            raise HTTPException(
                status_code=400,
                detail="Currently only English to Assamese translation is supported"
            )
        
        translated_text = translator.translate(request.text, request.max_length)
        
        return TranslationResponse(
            original_text=request.text,
            translated_text=translated_text,
            source_language=request.source_language,
            target_language=request.target_language
        )
        
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.post("/api/translate/batch", response_model=BatchTranslationResponse)
async def translate_batch(request: BatchTranslationRequest):
    """
    Translate multiple texts from English to Assamese
    """
    if translator is None:
        raise HTTPException(
            status_code=503, 
            detail="Translation service unavailable. Model not loaded."
        )
    
    try:
        logger.info(f"Batch translating {len(request.texts)} texts...")
        
        # Currently only supports English to Assamese
        if request.source_language != "en" or request.target_language != "as":
            raise HTTPException(
                status_code=400,
                detail="Currently only English to Assamese translation is supported"
            )
        
        translated_texts = translator.batch_translate(request.texts, request.max_length)
        
        translations = [
            TranslationResponse(
                original_text=original,
                translated_text=translated,
                source_language=request.source_language,
                target_language=request.target_language
            )
            for original, translated in zip(request.texts, translated_texts)
        ]
        
        return BatchTranslationResponse(
            translations=translations,
            total_count=len(translations)
        )
        
    except Exception as e:
        logger.error(f"Batch translation error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch translation failed: {str(e)}")

@app.get("/api/languages")
async def get_supported_languages():
    """
    Get list of supported languages
    """
    return {
        "supported_pairs": [
            {
                "source": "en",
                "target": "as",
                "source_name": "English",
                "target_name": "Assamese",
                "description": "English to Assamese translation using fine-tuned NLLB model"
            }
        ],
        "future_languages": [
            "bodo (brx_Deva)",
            "dogri (dgo_Deva)",
            "hindi (hin_Deva)",
            "bengali (ben_Beng)"
        ]
    }

@app.get("/api/stats")
async def get_translation_stats():
    """
    Get translation service statistics
    """
    return {
        "model_info": {
            "base_model": "facebook/nllb-200-distilled-600M",
            "fine_tuned": "English-Assamese",
            "model_size": "600M parameters"
        },
        "service_info": {
            "version": "1.0.0",
            "status": "active" if translator else "inactive",
            "supported_languages": 1
        }
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "detail": "The requested endpoint does not exist"}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "detail": "An unexpected error occurred"}

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
