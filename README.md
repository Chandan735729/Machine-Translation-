# 🌐 Multi-Language Translation Platform: Breaking Language Barriers

An open-source, scalable translation platform designed to break down language barriers for NGOs and organizations in India. This project creates a comprehensive ecosystem including pre-trained model integration, web APIs, and user interfaces for **5 languages**: English, Assamese, Bengali, Manipuri, and Santali using Meta's powerful NLLB model.

## 🎯 Project Vision

Our mission extends beyond simple translation:
- **Bridge Communication Gaps**: Empower NGOs to share health awareness, educational materials, and public announcements in regional languages
- **Preserve Low-Resource Languages**: Support languages like Assamese, Bengali, Manipuri, and Santali to promote linguistic diversity
- **Create Multi-Modal Access**: Enable text, voice, and mobile access to translation services
- **Build Open-Source Impact**: Provide a foundation for researchers and developers to expand language support

-----

## ✨ Key Features

* **🤖 AI-Powered Translation**: Uses Meta's NLLB-200 pre-trained model for high-quality translations
* **🌐 5-Language Support**: English, Assamese (অসমীয়া), Bengali (বাংলা), Manipuri (ꯃꯅꯤꯄꯨꯔꯤ), Santali (ᱥᱟᱱᱛᱟᱲᱤ)
* **🔄 Bidirectional Translation**: Translate between any supported language pair
* **🌐 Web API**: RESTful FastAPI backend with comprehensive endpoints for integration
* **💻 Modern Web Interface**: Beautiful, responsive frontend with real-time translation
* **📱 Mobile-Ready**: Architecture designed for future Flutter mobile app integration
* **🚀 No Training Required**: Uses pre-trained models - ready to use out of the box
* **🔧 Production-Ready**: Complete deployment pipeline with health checks and error handling
* **📊 Extensible**: Modular design for easy addition of new language pairs

-----

## 🗣️ Supported Languages

| Language | Code | Script | NLLB Code | Support Level |
|----------|------|--------|-----------|---------------|
| English | `en` | Latin | `eng_Latn` | ⭐⭐⭐ Excellent |
| Assamese | `as` | Bengali | `asm_Beng` | ⭐⭐⭐ Strong |
| Bengali | `bn` | Bengali | `ben_Beng` | ⭐⭐⭐ Excellent |
| Manipuri | `mni` | Bengali/Meetei | `mni_Beng` | ⭐⭐ Good (via Assamese) |
| Santali | `sat` | Ol Chiki | `sat_Olck` | ⭐⭐ Good (via Assamese) |

**Translation Strategy:**
- Direct translation for English ↔ Assamese, English ↔ Bengali
- Two-step translation for Manipuri/Santali (via Assamese for better quality)
- Cross-language translation between all 5 languages supported

-----

## 🛠️ Technology Stack

**Core ML & NLP:**
* Python 3.9+
* PyTorch 2.0+
* Hugging Face Transformers
* Meta's NLLB-200-distilled-600M Model

**Backend & API:**
* FastAPI
* Uvicorn ASGI Server
* Pydantic for data validation

**Frontend:**
* Modern HTML5/CSS3/JavaScript
* Responsive design with CSS Grid/Flexbox
* Font Awesome icons

**No Training Required:**
* Uses pre-trained NLLB model
* No GPU/CUDA requirements
* Ready to use immediately

-----

## 📂 Project Structure

```
Machine-Translation-/
├── 📁 src/                     # Core Python modules
│   ├── data_preparation.py     # Dataset loading and preprocessing (legacy)
│   ├── train.py               # Model training script (legacy)
│   └── translate.py           # Multi-language translation engine
├── 📁 api/                     # FastAPI backend
│   └── main.py                # REST API server with endpoints
├── 📁 frontend/                # Web interface
│   ├── index.html             # Main web application
│   └── script.js              # Frontend JavaScript logic
├── 📁 notebooks/               # Training notebooks (legacy)
│   └── train_colab.ipynb      # Google Colab training notebook
├── requirements.txt            # Python dependencies
├── run_server.py              # Simple server runner
└── README.md                   # This file
```

-----

## 🚀 Quick Start Guide

### Prerequisites

* **Python 3.9+**
* **Git**
* **Internet connection** (for downloading pre-trained models)

### Installation & Setup

1. **Clone and Setup:**
   ```bash
   git clone https://github.com/Chandan735729/Machine-Translation-.git
   cd Machine-Translation-
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   
   pip install -r requirements.txt
   ```

2. **Run the Application:**
   ```bash
   python run_server.py
   ```
   
3. **Access the Interface:**
   Open your browser and go to `http://localhost:8000`

**That's it!** The first run will automatically download the pre-trained NLLB model (~2.4GB). No training or GPU required.

-----

## 📖 Usage Guide

### 🌐 Web Interface

The easiest way to use the translation platform:

1. **Start the server:** `python run_server.py`
2. **Open your browser:** Navigate to `http://localhost:8000`
3. **Select languages** from the dropdown menus
4. **Enter text** in the source language panel
5. **Click "Translate"** to get translation in target language
6. **Use language detection** for automatic source language identification

### 🔧 API Integration

For developers integrating the translation service:

```python
import requests

# Single translation
response = requests.post("http://localhost:8000/api/translate", 
    json={
        "text": "Hello, how are you?",
        "source_language": "en",
        "target_language": "as"
    })
result = response.json()
print(result["translated_text"])

# Batch translation
response = requests.post("http://localhost:8000/api/translate/batch",
    json={
        "texts": ["Hello", "Thank you", "Good morning"],
        "source_language": "en",
        "target_language": "bn"
    })
results = response.json()

# Language detection
response = requests.post("http://localhost:8000/api/detect",
    json={"text": "আপনি কেমন আছেন?"})
detected = response.json()
print(f"Detected: {detected['detected_language']}")
```

### 📱 API Endpoints

- `GET /` - Web interface
- `POST /api/translate` - Single text translation
- `POST /api/translate/batch` - Batch translation
- `POST /api/detect` - Language detection
- `GET /api/languages` - Supported language pairs
- `GET /health` - Service health check
- `GET /docs` - Interactive API documentation

**Example Translations:**
* **English → Assamese:** `"Hello"` → `"নমস্কাৰ"`
* **English → Bengali:** `"Thank you"` → `"ধন্যবাদ"`
* **English → Manipuri:** `"Good morning"` → `"নুংঙাইরবা"`
* **English → Santali:** `"Welcome"` → `"ᱡᱚᱦᱟᱨ"`

-----

## 🧠 Technical Details

### Model Architecture
* **Base Model:** Meta's `facebook/nllb-200-distilled-600M`
* **Parameters:** 600M parameters, optimized for multilingual translation
* **Languages Supported:** 200+ languages with focus on low-resource languages
* **No Fine-tuning Required:** Uses pre-trained weights directly

### Language Specifications
* **Supported Scripts:** Latin, Bengali, Ol Chiki, Meetei Mayek
* **Language Detection:** Automatic script-based detection with fallbacks
* **Quality Optimization:** Two-step translation for challenging language pairs

### Performance Metrics
* **Setup Time:** ~5-10 minutes (model download)
* **Model Size:** ~2.4GB (downloaded once)
* **Inference Speed:** ~1-3 seconds per sentence
* **Supported Text Length:** Up to 512 tokens per translation
* **Memory Usage:** ~4-6GB RAM recommended

-----



## 🤝 Contributing

We welcome contributions from developers, linguists, and domain experts!

**Ways to Contribute:**
- 🐛 **Bug Reports:** Found an issue? Open a GitHub issue
- 💡 **Feature Requests:** Have ideas? We'd love to hear them
- 🌐 **Language Support:** Help us add new language pairs
- 📝 **Documentation:** Improve our guides and examples
- 🧪 **Testing:** Help us test with real-world scenarios

**Development Setup:**
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and test thoroughly
4. Submit a pull request with detailed description

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

* **Meta AI** - For the groundbreaking NLLB models
* **Hugging Face** - For the incredible transformers library and model hub
* **FastAPI** - For the modern, fast web framework
* **Regional Language Communities** - For inspiring this project's mission
* **Open Source Community** - For tools and libraries that made this possible

---

<div align="center">

**🌐 Breaking Language Barriers, One Translation at a Time 🌐**

**Supported Languages:** English • অসমীয়া • বাংলা • ꯃꯅꯤꯄꯨꯔꯤ • ᱥᱟᱱᱛᱟᱲᱤ

[🚀 Get Started](#-quick-start-guide) • [📖 Documentation](#-usage-guide) • [🤝 Contribute](#-contributing) • [📱 API Docs](http://localhost:8000/docs)

</div>
