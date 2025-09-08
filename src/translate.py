"""
Translation Inference Script
Uses trained NLLB model to translate English text to Assamese
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnglishToAssameseTranslator:
    """
    Handles translation from English to Assamese using fine-tuned NLLB model
    """
    
    def __init__(self, model_path: str = "models/nllb-finetuned-en-to-asm-final"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.source_lang = "eng_Latn"
        self.target_lang = "asm_Beng"
        
        # Load model and tokenizer
        self.load_model()
    
    def load_model(self):
        """
        Load the fine-tuned model and tokenizer
        """
        try:
            if os.path.exists(self.model_path):
                logger.info(f"Loading fine-tuned model from {self.model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            else:
                logger.info("Fine-tuned model not found. Using base NLLB model.")
                model_name = "facebook/nllb-200-distilled-600M"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def translate(self, text: str, max_length: int = 512) -> str:
        """
        Translate English text to Assamese
        
        Args:
            text (str): English text to translate
            max_length (int): Maximum length of generated translation
            
        Returns:
            str: Translated Assamese text
        """
        try:
            # Prepare input
            input_text = f"{self.source_lang}: {text}"
            inputs = self.tokenizer(
                input_text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate translation
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.lang_code_to_id[self.target_lang],
                    max_length=max_length,
                    num_beams=5,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode translation
            translation = self.tokenizer.decode(
                generated_tokens[0], 
                skip_special_tokens=True
            )
            
            # Remove language prefix if present
            if translation.startswith(f"{self.target_lang}:"):
                translation = translation[len(f"{self.target_lang}:"):].strip()
            
            return translation
            
        except Exception as e:
            logger.error(f"Error during translation: {e}")
            return f"Translation error: {str(e)}"
    
    def batch_translate(self, texts: list, max_length: int = 512) -> list:
        """
        Translate multiple texts at once
        
        Args:
            texts (list): List of English texts to translate
            max_length (int): Maximum length of generated translations
            
        Returns:
            list: List of translated Assamese texts
        """
        translations = []
        for text in texts:
            translation = self.translate(text, max_length)
            translations.append(translation)
        
        return translations

def main():
    """
    Main function for testing translation
    """
    # Initialize translator
    translator = EnglishToAssameseTranslator()
    
    # Test sentences
    test_sentences = [
        "Community health workers are the backbone of our medical system.",
        "Education is the key to development.",
        "Clean water is essential for good health.",
        "Vaccination protects children from diseases.",
        "Women's empowerment leads to stronger communities.",
        "Hello, how are you?",
        "Thank you for your help.",
        "The weather is nice today."
    ]
    
    print("\n" + "="*60)
    print("ENGLISH TO ASSAMESE TRANSLATION DEMO")
    print("="*60)
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n{i}. English: {sentence}")
        translation = translator.translate(sentence)
        print(f"   Assamese: {translation}")
    
    print("\n" + "="*60)
    print("Translation demo completed!")
    print("="*60)

if __name__ == "__main__":
    main()
