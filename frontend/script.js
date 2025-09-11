/**
 * Frontend JavaScript for Multi-Language Translation Platform
 * Handles user interactions and API communication for bidirectional translation
 */

class MultiLanguageTranslationApp {
    constructor() {
        this.apiBaseUrl = window.location.origin;
        this.supportedLanguages = {};
        this.initializeElements();
        this.bindEvents();
        this.loadSupportedLanguages();
        this.checkServiceHealth();
    }

    initializeElements() {
        // Get DOM elements
        this.sourceText = document.getElementById('sourceText');
        this.targetText = document.getElementById('targetText');
        this.sourceLanguage = document.getElementById('sourceLanguage');
        this.targetLanguage = document.getElementById('targetLanguage');
        this.sourceLabel = document.getElementById('sourceLabel');
        this.targetLabel = document.getElementById('targetLabel');
        this.translateBtn = document.getElementById('translateBtn');
        this.detectBtn = document.getElementById('detectBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.copyBtn = document.getElementById('copyBtn');
        this.swapBtn = document.getElementById('swapBtn');
        this.loading = document.getElementById('loading');
        this.error = document.getElementById('error');
        this.success = document.getElementById('success');
        this.errorMessage = document.getElementById('errorMessage');
        this.successMessage = document.getElementById('successMessage');
        this.charCount = document.getElementById('charCount');
    }

    bindEvents() {
        // Translation button
        this.translateBtn.addEventListener('click', () => this.translateText());
        
        // Language detection button
        this.detectBtn.addEventListener('click', () => this.detectLanguage());
        
        // Clear button
        this.clearBtn.addEventListener('click', () => this.clearText());
        
        // Copy button
        this.copyBtn.addEventListener('click', () => this.copyTranslation());
        
        // Swap languages button
        this.swapBtn.addEventListener('click', () => this.swapLanguages());
        
        // Language selection changes
        this.sourceLanguage.addEventListener('change', () => this.updateLabels());
        this.targetLanguage.addEventListener('change', () => this.updateLabels());
        
        // Character counter
        this.sourceText.addEventListener('input', () => this.updateCharCount());
        
        // Enter key for translation (Ctrl+Enter)
        this.sourceText.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                this.translateText();
            }
        });
    }

    async loadSupportedLanguages() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/languages`);
            const data = await response.json();
            this.supportedLanguages = data.supported_languages || {};
        } catch (error) {
            console.warn('Could not load supported languages:', error);
        }
    }

    updateLabels() {
        const sourceLang = this.sourceLanguage.value;
        const targetLang = this.targetLanguage.value;
        
        const sourceName = this.supportedLanguages[sourceLang] || this.getLanguageName(sourceLang);
        const targetName = this.supportedLanguages[targetLang] || this.getLanguageName(targetLang);
        
        this.sourceLabel.textContent = `Enter ${sourceName} Text`;
        this.targetLabel.textContent = `${targetName} Translation`;
        
        // Update placeholder
        this.sourceText.placeholder = `Type your ${sourceName.toLowerCase()} text here...`;
    }

    getLanguageName(code) {
        const names = {
            'en': 'English',
            'as': 'Assamese',
            'brx': 'Bodo',
            'dgo': 'Dogri',
            'sat': 'Santali',
            'mni': 'Manipuri'
        };
        return names[code] || code.toUpperCase();
    }

    swapLanguages() {
        const sourceValue = this.sourceLanguage.value;
        const targetValue = this.targetLanguage.value;
        const sourceTextValue = this.sourceText.value;
        const targetTextValue = this.targetText.value;
        
        // Swap language selections
        this.sourceLanguage.value = targetValue;
        this.targetLanguage.value = sourceValue;
        
        // Swap text content
        this.sourceText.value = targetTextValue;
        this.targetText.value = sourceTextValue;
        
        // Update labels
        this.updateLabels();
        this.updateCharCount();
        
        // Enable/disable copy button based on target text
        this.copyBtn.disabled = !this.targetText.value;
        
        this.showSuccess('Languages swapped successfully!');
    }

    updateCharCount() {
        const count = this.sourceText.value.length;
        this.charCount.textContent = count;
        
        // Change color based on character count
        if (count > 800) {
            this.charCount.style.color = '#dc3545';
        } else if (count > 600) {
            this.charCount.style.color = '#ffc107';
        } else {
            this.charCount.style.color = '#6c757d';
        }
    }

    async checkServiceHealth() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            const data = await response.json();
            
            if (!data.model_loaded) {
                this.showError('Translation service is starting up. Please wait a moment and try again.');
                this.translateBtn.disabled = true;
                this.detectBtn.disabled = true;
            } else {
                this.showSuccess(`Service ready! Supporting ${data.supported_languages} languages.`);
            }
        } catch (error) {
            console.warn('Could not check service health:', error);
        }
    }

    async detectLanguage() {
        const text = this.sourceText.value.trim();
        
        if (!text) {
            this.showError('Please enter some text for language detection.');
            return;
        }

        this.showLoading(true);
        this.hideMessages();
        this.detectBtn.disabled = true;

        try {
            const response = await fetch(`${this.apiBaseUrl}/api/detect-language`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Language detection failed');
            }

            const data = await response.json();
            
            // Update source language if different
            if (data.detected_language !== this.sourceLanguage.value) {
                this.sourceLanguage.value = data.detected_language;
                this.updateLabels();
                this.showSuccess(`Detected language: ${data.detected_language_name}. Source language updated.`);
            } else {
                this.showSuccess(`Confirmed language: ${data.detected_language_name}`);
            }

        } catch (error) {
            console.error('Language detection error:', error);
            this.showError(error.message || 'Failed to detect language. Please try again.');
        } finally {
            this.showLoading(false);
            this.detectBtn.disabled = false;
        }
    }

    async translateText() {
        const text = this.sourceText.value.trim();
        const sourceLang = this.sourceLanguage.value;
        const targetLang = this.targetLanguage.value;
        
        if (!text) {
            this.showError('Please enter some text to translate.');
            return;
        }

        if (text.length > 1000) {
            this.showError('Text is too long. Please limit to 1000 characters.');
            return;
        }

        if (sourceLang === targetLang) {
            this.showError('Source and target languages cannot be the same.');
            return;
        }

        this.showLoading(true);
        this.hideMessages();
        this.translateBtn.disabled = true;

        try {
            const response = await fetch(`${this.apiBaseUrl}/api/translate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    source_language: sourceLang,
                    target_language: targetLang,
                    max_length: 512
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Translation failed');
            }

            const data = await response.json();
            this.targetText.value = data.translated_text;
            this.copyBtn.disabled = false;
            
            // Show success message
            this.showSuccess(`Translation completed: ${data.source_language_name} → ${data.target_language_name}`);
            
            // Add success animation
            this.targetText.style.backgroundColor = '#d4edda';
            setTimeout(() => {
                this.targetText.style.backgroundColor = '';
            }, 1000);

        } catch (error) {
            console.error('Translation error:', error);
            this.showError(error.message || 'Failed to translate text. Please try again.');
        } finally {
            this.showLoading(false);
            this.translateBtn.disabled = false;
        }
    }

    clearText() {
        this.sourceText.value = '';
        this.targetText.value = '';
        this.copyBtn.disabled = true;
        this.hideMessages();
        this.updateCharCount();
        this.sourceText.focus();
    }

    async copyTranslation() {
        const text = this.targetText.value;
        
        if (!text) {
            this.showError('No translation to copy.');
            return;
        }

        try {
            await navigator.clipboard.writeText(text);
            
            // Visual feedback
            const originalText = this.copyBtn.innerHTML;
            this.copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
            this.copyBtn.style.backgroundColor = '#28a745';
            
            setTimeout(() => {
                this.copyBtn.innerHTML = originalText;
                this.copyBtn.style.backgroundColor = '';
            }, 2000);
            
            this.showSuccess('Translation copied to clipboard!');
            
        } catch (error) {
            console.error('Copy failed:', error);
            this.showError('Failed to copy text to clipboard.');
        }
    }

    showLoading(show) {
        if (show) {
            this.loading.classList.add('show');
        } else {
            this.loading.classList.remove('show');
        }
    }

    showError(message) {
        this.errorMessage.textContent = message;
        this.error.classList.add('show');
        this.success.classList.remove('show');
        
        // Auto-hide error after 5 seconds
        setTimeout(() => {
            this.hideMessages();
        }, 5000);
    }

    showSuccess(message) {
        this.successMessage.textContent = message;
        this.success.classList.add('show');
        this.error.classList.remove('show');
        
        // Auto-hide success after 3 seconds
        setTimeout(() => {
            this.hideMessages();
        }, 3000);
    }

    hideMessages() {
        this.error.classList.remove('show');
        this.success.classList.remove('show');
    }
}

// Sample texts for different languages
const multiLanguageSamples = {
    'en': [
        "Hello, how are you today?",
        "Community health workers are important.",
        "Education is the key to development.",
        "Clean water is essential for health.",
        "Thank you for your help and support."
    ],
    'as': [
        "আপোনাৰ স্বাস্থ্য কেনে আছে?",
        "শিক্ষা উন্নতিৰ চাবিকাঠি।"
    ],
    'brx': [
        "नमस्कार, आप कैसे हैं?",
        "शिक्षा विकास की कुंजी है।"
    ]
};

// Add sample text functionality
function addSampleTextButtons() {
    const controlsDiv = document.querySelector('.controls');
    
    // Create sample text dropdown
    const sampleBtn = document.createElement('button');
    sampleBtn.className = 'btn btn-secondary';
    sampleBtn.innerHTML = '<i class="fas fa-lightbulb"></i> Sample Texts';
    sampleBtn.onclick = toggleSampleTexts;
    
    controlsDiv.appendChild(sampleBtn);
    
    // Create sample texts container
    const samplesContainer = document.createElement('div');
    samplesContainer.id = 'samplesContainer';
    samplesContainer.style.display = 'none';
    samplesContainer.style.marginTop = '15px';
    samplesContainer.style.padding = '15px';
    samplesContainer.style.backgroundColor = '#f8f9fa';
    samplesContainer.style.borderRadius = '8px';
    
    const samplesTitle = document.createElement('h4');
    samplesTitle.textContent = 'Click a sample text to try:';
    samplesTitle.style.marginBottom = '10px';
    samplesTitle.style.fontSize = '14px';
    samplesContainer.appendChild(samplesTitle);
    
    // Add samples for current source language
    updateSampleTexts(samplesContainer);
    
    controlsDiv.parentNode.insertBefore(samplesContainer, controlsDiv.nextSibling);
}

function updateSampleTexts(container) {
    const sourceLang = document.getElementById('sourceLanguage').value;
    const samples = multiLanguageSamples[sourceLang] || multiLanguageSamples['en'];
    
    // Clear existing samples
    const existingSamples = container.querySelectorAll('.sample-text');
    existingSamples.forEach(sample => sample.remove());
    
    samples.forEach((text, index) => {
        const sampleDiv = document.createElement('div');
        sampleDiv.className = 'sample-text';
        sampleDiv.style.padding = '8px';
        sampleDiv.style.margin = '5px 0';
        sampleDiv.style.backgroundColor = 'white';
        sampleDiv.style.borderRadius = '5px';
        sampleDiv.style.cursor = 'pointer';
        sampleDiv.style.border = '1px solid #dee2e6';
        sampleDiv.textContent = text;
        
        sampleDiv.onclick = () => {
            document.getElementById('sourceText').value = text;
            app.updateCharCount();
            toggleSampleTexts();
        };
        
        container.appendChild(sampleDiv);
    });
}

function toggleSampleTexts() {
    const container = document.getElementById('samplesContainer');
    if (container.style.display === 'none') {
        updateSampleTexts(container);
        container.style.display = 'block';
    } else {
        container.style.display = 'none';
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new MultiLanguageTranslationApp();
    addSampleTextButtons();
    
    // Update labels initially
    setTimeout(() => {
        app.updateLabels();
    }, 500);
    
    // Add keyboard shortcuts info
    const footer = document.querySelector('.footer p');
    footer.innerHTML += ' | <span style="font-size: 0.9em;">💡 Tip: Press Ctrl+Enter to translate</span>';
});
