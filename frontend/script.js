/**
 * Frontend JavaScript for Multi-Modal Translation Platform
 * Handles user interactions and API communication
 */

class TranslationApp {
    constructor() {
        this.apiBaseUrl = window.location.origin;
        this.initializeElements();
        this.bindEvents();
        this.checkServiceHealth();
    }

    initializeElements() {
        // Get DOM elements
        this.sourceText = document.getElementById('sourceText');
        this.targetText = document.getElementById('targetText');
        this.translateBtn = document.getElementById('translateBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.copyBtn = document.getElementById('copyBtn');
        this.loading = document.getElementById('loading');
        this.error = document.getElementById('error');
        this.errorMessage = document.getElementById('errorMessage');
        this.charCount = document.getElementById('charCount');
    }

    bindEvents() {
        // Translate button
        this.translateBtn.addEventListener('click', () => this.translateText());
        
        // Clear button
        this.clearBtn.addEventListener('click', () => this.clearText());
        
        // Copy button
        this.copyBtn.addEventListener('click', () => this.copyTranslation());
        
        // Character counter
        this.sourceText.addEventListener('input', () => this.updateCharCount());
        
        // Enter key for translation (Ctrl+Enter)
        this.sourceText.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                this.translateText();
            }
        });
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
            }
        } catch (error) {
            console.warn('Could not check service health:', error);
        }
    }

    async translateText() {
        const text = this.sourceText.value.trim();
        
        if (!text) {
            this.showError('Please enter some text to translate.');
            return;
        }

        if (text.length > 1000) {
            this.showError('Text is too long. Please limit to 1000 characters.');
            return;
        }

        this.showLoading(true);
        this.hideError();
        this.translateBtn.disabled = true;

        try {
            const response = await fetch(`${this.apiBaseUrl}/api/translate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    source_language: 'en',
                    target_language: 'as',
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
        this.hideError();
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
        
        // Auto-hide error after 5 seconds
        setTimeout(() => {
            this.hideError();
        }, 5000);
    }

    hideError() {
        this.error.classList.remove('show');
    }
}

// Sample texts for quick testing
const sampleTexts = [
    "Community health workers are the backbone of our medical system.",
    "Education is the key to development.",
    "Clean water is essential for good health.",
    "Vaccination protects children from diseases.",
    "Women's empowerment leads to stronger communities.",
    "Hello, how are you today?",
    "Thank you for your help and support.",
    "The weather is very nice today."
];

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
    
    sampleTexts.forEach((text, index) => {
        const sampleDiv = document.createElement('div');
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
        
        samplesContainer.appendChild(sampleDiv);
    });
    
    controlsDiv.parentNode.insertBefore(samplesContainer, controlsDiv.nextSibling);
}

function toggleSampleTexts() {
    const container = document.getElementById('samplesContainer');
    container.style.display = container.style.display === 'none' ? 'block' : 'none';
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new TranslationApp();
    addSampleTextButtons();
    
    // Add keyboard shortcuts info
    const footer = document.querySelector('.footer p');
    footer.innerHTML += ' | <span style="font-size: 0.9em;">💡 Tip: Press Ctrl+Enter to translate</span>';
});
