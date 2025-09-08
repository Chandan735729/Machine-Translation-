# 🚀 Deployment Guide - Multi-Modal Translation Platform

This guide covers different deployment options for your English-Assamese translation platform.

## 🏠 Local Development Setup

### Quick Start
```bash
# 1. Clone and setup
git clone https://github.com/your-username/Machine-Translation-.git
cd Machine-Translation-
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
python run_server.py
```

### Access Points
- **Web Interface:** http://localhost:8000
- **API Documentation:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

## ☁️ Google Colab Training

### Setup Process
1. **Upload Notebook:** Upload `notebooks/train_colab.ipynb` to Google Colab
2. **Set GPU Runtime:** Runtime → Change runtime type → GPU (T4 recommended)
3. **Run Training:** Execute all cells in sequence
4. **Download Model:** Save the trained model to your local machine

### Training Time
- **T4 GPU:** ~30-45 minutes
- **V100 GPU:** ~15-25 minutes
- **CPU Only:** Not recommended (6+ hours)

## 🐳 Docker Deployment

### Create Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "run_server.py"]
```

### Build and Run
```bash
# Build image
docker build -t translation-platform .

# Run container
docker run -p 8000:8000 translation-platform
```

## 🌐 Cloud Deployment Options

### 1. Hugging Face Spaces
```python
# Create app.py for Hugging Face Spaces
import gradio as gr
import requests

def translate_text(text):
    response = requests.post("http://localhost:8000/api/translate", 
                           json={"text": text})
    return response.json()["translated_text"]

iface = gr.Interface(
    fn=translate_text,
    inputs="text",
    outputs="text",
    title="English-Assamese Translator"
)

iface.launch()
```

### 2. Railway Deployment
```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "python run_server.py",
    "healthcheckPath": "/health"
  }
}
```

### 3. Render Deployment
```yaml
# render.yaml
services:
  - type: web
    name: translation-platform
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "python run_server.py"
    envVars:
      - key: PORT
        value: 8000
```

## 📱 Mobile App Architecture

### Flutter Integration
```dart
// lib/services/translation_service.dart
import 'dart:convert';
import 'package:http/http.dart' as http;

class TranslationService {
  static const String baseUrl = 'https://your-api-url.com';
  
  static Future<String> translate(String text) async {
    final response = await http.post(
      Uri.parse('$baseUrl/api/translate'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'text': text}),
    );
    
    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return data['translated_text'];
    } else {
      throw Exception('Translation failed');
    }
  }
}
```

### Flutter UI Example
```dart
// lib/screens/translation_screen.dart
class TranslationScreen extends StatefulWidget {
  @override
  _TranslationScreenState createState() => _TranslationScreenState();
}

class _TranslationScreenState extends State<TranslationScreen> {
  final TextEditingController _controller = TextEditingController();
  String _translation = '';
  bool _isLoading = false;

  Future<void> _translateText() async {
    setState(() => _isLoading = true);
    try {
      final translation = await TranslationService.translate(_controller.text);
      setState(() => _translation = translation);
    } catch (e) {
      // Handle error
    } finally {
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('English-Assamese Translator')),
      body: Padding(
        padding: EdgeInsets.all(16.0),
        child: Column(
          children: [
            TextField(
              controller: _controller,
              decoration: InputDecoration(labelText: 'Enter English text'),
              maxLines: 3,
            ),
            SizedBox(height: 16),
            ElevatedButton(
              onPressed: _isLoading ? null : _translateText,
              child: _isLoading 
                ? CircularProgressIndicator() 
                : Text('Translate'),
            ),
            SizedBox(height: 16),
            Container(
              width: double.infinity,
              padding: EdgeInsets.all(16),
              decoration: BoxDecoration(
                border: Border.all(color: Colors.grey),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(_translation.isEmpty ? 'Translation will appear here' : _translation),
            ),
          ],
        ),
      ),
    );
  }
}
```

## 🔧 Production Considerations

### Environment Variables
```bash
# .env file
MODEL_PATH=models/nllb-finetuned-en-to-asm-final
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
MAX_TEXT_LENGTH=1000
```

### Performance Optimization
```python
# api/config.py
import os

class Settings:
    model_path = os.getenv("MODEL_PATH", "models/nllb-finetuned-en-to-asm-final")
    api_host = os.getenv("API_HOST", "0.0.0.0")
    api_port = int(os.getenv("API_PORT", 8000))
    max_text_length = int(os.getenv("MAX_TEXT_LENGTH", 1000))
    enable_gpu = os.getenv("ENABLE_GPU", "true").lower() == "true"
```

### Load Balancing
```nginx
# nginx.conf
upstream translation_backend {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://translation_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🔒 Security Best Practices

### API Rate Limiting
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/translate")
@limiter.limit("10/minute")
async def translate_text(request: Request, translation_request: TranslationRequest):
    # Translation logic
    pass
```

### Input Validation
```python
from pydantic import BaseModel, validator

class TranslationRequest(BaseModel):
    text: str
    max_length: int = 512
    
    @validator('text')
    def validate_text(cls, v):
        if len(v.strip()) == 0:
            raise ValueError('Text cannot be empty')
        if len(v) > 1000:
            raise ValueError('Text too long (max 1000 characters)')
        return v.strip()
```

## 📊 Monitoring and Logging

### Health Checks
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": translator is not None,
        "version": "1.0.0"
    }
```

### Metrics Collection
```python
import time
from prometheus_client import Counter, Histogram

translation_requests = Counter('translation_requests_total', 'Total translation requests')
translation_duration = Histogram('translation_duration_seconds', 'Translation duration')

@app.post("/api/translate")
async def translate_text(request: TranslationRequest):
    start_time = time.time()
    translation_requests.inc()
    
    try:
        result = translator.translate(request.text)
        return {"translated_text": result}
    finally:
        translation_duration.observe(time.time() - start_time)
```

## 🚀 Scaling Strategies

### Horizontal Scaling
- Deploy multiple API instances behind a load balancer
- Use Redis for session management and caching
- Implement model serving with TensorFlow Serving or TorchServe

### Vertical Scaling
- Use GPU instances for faster inference
- Implement model quantization for reduced memory usage
- Add caching layer for frequently translated texts

---

**Ready to deploy your translation platform! 🌐**
