"""
FastAPI Backend for Multi-Language Translation Platform
Provides REST API endpoints for bidirectional translation services
Supports English, Assamese, Bengali, Manipuri, and Santali
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
from typing import List, Optional, Dict
import logging
import os
import sys
import uvicorn

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from translate import MultiLanguageTranslator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Language Translation Platform",
    description="Bidirectional Translation API for English, Assamese, Bengali, Manipuri, and Santali",
    version="2.3.0",
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
frontend_path = os.path.join(os.path.dirname(__file__), '..', 'frontend')
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

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
    source_language_name: str
    target_language_name: str
    confidence: Optional[float] = None

class BatchTranslationResponse(BaseModel):
    translations: List[TranslationResponse]
    total_count: int

class LanguageDetectionRequest(BaseModel):
    text: str

class LanguageDetectionResponse(BaseModel):
    detected_language: str
    detected_language_name: str
    confidence: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    message: str
    model_loaded: bool
    supported_languages: int

@app.on_event("startup")
async def startup_event():
    """Initialize the translation model on startup"""
    global translator
    try:
        logger.info("Loading multi-language translation model...")
        translator = MultiLanguageTranslator()
        logger.info("Multi-language translation model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load translation model: {e}")
        translator = None

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main frontend page"""
    frontend_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'index.html')
    try:
        with open(frontend_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
            <head><title>Multi-Language Translation Platform</title></head>
            <body>
                <h1>Multi-Language Translation Platform</h1>
                <p>API is running! Visit <a href="/docs">/docs</a> for API documentation.</p>
                <p>Supported Languages: English, Assamese (অসমীয়া), Bengali (বাংলা), Manipuri (ꯃꯅꯤꯄꯨꯔꯤ), Santali (ᱥᱟᱱᱛᱟᱲᱤ)</p>
                <p>Frontend not found. Please ensure frontend files are in the 'frontend' directory.</p>
            </body>
        </html>
        """)

@app.get("/script.js")
async def get_script():
    """Serve the JavaScript file"""
    script_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'script.js')
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            return Response(content=f.read(), media_type="application/javascript")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Script file not found")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_loaded = translator is not None
    status = "healthy" if model_loaded else "unhealthy"
    message = "Multi-language translation service is ready" if model_loaded else "Translation model not loaded"
    supported_languages = len(translator.get_supported_languages()) if translator else 0
    
    return HealthResponse(
        status=status,
        message=message,
        model_loaded=model_loaded,
        supported_languages=supported_languages
    )

@app.post("/api/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """
    Translate text between any supported language pair
    """
    if translator is None:
        raise HTTPException(
            status_code=503, 
            detail="Translation service unavailable. Model not loaded."
        )
    
    try:
        logger.info(f"Translating text ({request.source_language}->{request.target_language}): {request.text[:50]}...")
        
        # Validate language pair
        if not translator.validate_language_pair(request.source_language, request.target_language):
            supported_langs = list(translator.get_supported_languages().keys())
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language pair: {request.source_language} -> {request.target_language}. Supported languages: {supported_langs}"
            )
        
        translated_text = translator.translate(
            request.text, 
            request.source_language, 
            request.target_language, 
            request.max_length
        )
        
        # Get language names
        language_names = translator.get_supported_languages()
        
        return TranslationResponse(
            original_text=request.text,
            translated_text=translated_text,
            source_language=request.source_language,
            target_language=request.target_language,
            source_language_name=language_names[request.source_language],
            target_language_name=language_names[request.target_language]
        )
        
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.post("/api/translate/batch", response_model=BatchTranslationResponse)
async def translate_batch(request: BatchTranslationRequest):
    """
    Translate multiple texts between any supported language pair
    """
    if translator is None:
        raise HTTPException(
            status_code=503, 
            detail="Translation service unavailable. Model not loaded."
        )
    
    try:
        logger.info(f"Batch translating {len(request.texts)} texts ({request.source_language}->{request.target_language})...")
        
        # Validate language pair
        if not translator.validate_language_pair(request.source_language, request.target_language):
            supported_langs = list(translator.get_supported_languages().keys())
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language pair: {request.source_language} -> {request.target_language}. Supported languages: {supported_langs}"
            )
        
        translated_texts = translator.batch_translate(
            request.texts, 
            request.source_language, 
            request.target_language, 
            request.max_length
        )
        
        # Get language names
        language_names = translator.get_supported_languages()
        
        translations = [
            TranslationResponse(
                original_text=original,
                translated_text=translated,
                source_language=request.source_language,
                target_language=request.target_language,
                source_language_name=language_names[request.source_language],
                target_language_name=language_names[request.target_language]
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

@app.post("/api/detect-language", response_model=LanguageDetectionResponse)
async def detect_language(request: LanguageDetectionRequest):
    """
    Detect the language of input text
    """
    if translator is None:
        raise HTTPException(
            status_code=503, 
            detail="Translation service unavailable. Model not loaded."
        )
    
    try:
        detected_lang = translator.detect_language(request.text)
        language_names = translator.get_supported_languages()
        
        return LanguageDetectionResponse(
            detected_language=detected_lang,
            detected_language_name=language_names.get(detected_lang, "Unknown")
        )
        
    except Exception as e:
        logger.error(f"Language detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Language detection failed: {str(e)}")

@app.get("/api/languages")
async def get_supported_languages():
    """
    Get list of supported languages and language pairs
    """
    if translator is None:
        return {"error": "Translation service not available"}
    
    return {
        "supported_languages": translator.get_supported_languages(),
        "supported_pairs": translator.get_language_pairs(),
        "total_languages": len(translator.get_supported_languages()),
        "total_pairs": len(translator.get_language_pairs())
    }

@app.get("/api/stats")
async def get_translation_stats():
    """
    Get translation service statistics
    """
    if translator is None:
        return {"error": "Translation service not available"}
    
    return {
        "model_info": {
            "base_model": "facebook/nllb-200-distilled-600M",
            "model_type": "Pre-trained (No fine-tuning required)",
            "model_size": "600M parameters"
        },
        "service_info": {
            "version": "2.3.0",
            "status": "active" if translator else "inactive",
            "supported_languages": len(translator.get_supported_languages()),
            "bidirectional": True,
            "features": ["Text Translation", "Batch Translation", "Language Detection"]
        },
        "supported_languages": translator.get_supported_languages()
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "detail": "The requested endpoint does not exist"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
