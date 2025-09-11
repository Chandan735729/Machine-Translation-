"""
Multi-Language Bidirectional Translation Engine
Supports English and multiple regional/tribal languages using pre-trained NLLB model
No training required - uses facebook/nllb-200-distilled-600M directly
"""

import torch
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Optional
import requests
import json
import os

logger = logging.getLogger(__name__)

class MultiLanguageTranslator:
    """
    Handles bidirectional translation between English and multiple regional languages
    Supports: English, Assamese, Bengali, Manipuri, Santali
    """
    
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Language mappings for NLLB model (using original authentic codes)
        self.language_codes = {
            "en": "eng_Latn",      # English
            "as": "asm_Beng",      # Assamese
            "bn": "ben_Beng",      # Bengali
            "mni": "mni_Beng",     # Manipuri/Meitei
            "sat": "sat_Olck"      # Santali
        }
        
        # Alternative mappings to try if primary fails (only for well-supported languages)
        self.alternative_codes = {
            "mni": ["mni_Beng", "hin_Deva"],  # Manipuri alternatives
            "sat": ["sat_Olck", "hin_Deva"]   # Santali alternatives
        }
        
        # Languages that may need special handling
        self.challenging_languages = ["mni", "sat"]
        
        # Languages requiring two-step translation (via intermediate language)
        self.two_step_languages = {
            "mni": "as",  # Manipuri via Assamese
            "sat": "as"   # Santali via Assamese
        }
        
        # Languages with limited support (for user notification)
        self.limited_support_languages = {
            "mni": "Limited NLLB support - translated via Assamese",
            "sat": "Limited NLLB support - translated via Assamese"
        }
        
        # Original language codes for reference
        self.original_codes = {
            "en": "eng_Latn",      # English
            "as": "asm_Beng",      # Assamese
            "bn": "ben_Beng",      # Bengali
            "mni": "mni_Beng",     # Manipuri/Meitei
            "sat": "sat_Olck"      # Santali
        }
        
        # Language display names
        self.language_names = {
            "en": "English",
            "as": "Assamese (অসমীয়া)",
            "bn": "Bengali (বাংলা)",
            "mni": "Manipuri (ꯃꯅꯤꯄꯨꯔꯤ)",
            "sat": "Santali (ᱥᱟᱱᱛᱟᱲᱤ)"
        }
        
        # Load model and tokenizer
        self.load_model()
        
        # Initialize language token mapping after loading tokenizer
        self._init_language_tokens()
        
        # Hybrid translation system using multiple translation services as fallbacks
        self.translation_services = {
            "google": {
                "url": "https://translation.googleapis.com/language/translate/v2",
                "api_key": os.environ.get("GOOGLE_TRANSLATE_API_KEY")
            },
            "microsoft": {
                "url": "https://api.microsofttranslator.com/V2/Ajax.svc/Translate",
                "api_key": os.environ.get("MICROSOFT_TRANSLATE_API_KEY")
            }
        }
    
    def _init_language_tokens(self):
        # Get target language token ID - handle different tokenizer versions
        try:
            # Try newer API first
            if hasattr(self.tokenizer, 'convert_tokens_to_ids'):
                self.lang_token_ids = {lang: self.tokenizer.convert_tokens_to_ids(code) for lang, code in self.language_codes.items()}
            elif hasattr(self.tokenizer, 'lang_code_to_id'):
                self.lang_token_ids = {lang: self.tokenizer.lang_code_to_id[code] for lang, code in self.language_codes.items()}
            else:
                # Fallback: get token ID manually
                self.lang_token_ids = {lang: self.tokenizer.get_vocab().get(code, 2) for lang, code in self.language_codes.items()}  # 2 is default for unknown
        except (KeyError, AttributeError):
            # Final fallback - use a default approach
            self.lang_token_ids = {lang: 2 for lang in self.language_codes}  # Default token ID
    
    def load_model(self):
        """
        Load the pre-trained NLLB model and tokenizer
        """
        try:
            logger.info(f"Loading pre-trained model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def get_supported_languages(self) -> Dict[str, str]:
        """
        Get dictionary of supported language codes and names
        """
        return self.language_names.copy()
    
    def get_language_pairs(self) -> List[Dict[str, str]]:
        """
        Get all supported bidirectional language pairs
        """
        pairs = []
        languages = list(self.language_codes.keys())
        
        for i, lang1 in enumerate(languages):
            for lang2 in languages[i+1:]:
                pairs.extend([
                    {
                        "source": lang1,
                        "target": lang2,
                        "source_name": self.language_names[lang1],
                        "target_name": self.language_names[lang2]
                    },
                    {
                        "source": lang2,
                        "target": lang1,
                        "source_name": self.language_names[lang2],
                        "target_name": self.language_names[lang1]
                    }
                ])
        
        return pairs
    
    def validate_language_pair(self, source_lang: str, target_lang: str) -> bool:
        """
        Validate if the language pair is supported
        """
        return (source_lang in self.language_codes and 
                target_lang in self.language_codes and 
                source_lang != target_lang)
    
    def translate(self, text: str, source_lang: str, target_lang: str, max_length: int = 512) -> str:
        """
        Translate text between any supported language pair
        
        Args:
            text (str): Text to translate
            source_lang (str): Source language code (e.g., 'en', 'as', 'brx')
            target_lang (str): Target language code (e.g., 'en', 'as', 'brx')
            max_length (int): Maximum length of generated translation
            
        Returns:
            str: Translated text
        """
        try:
            # Validate language pair
            if not self.validate_language_pair(source_lang, target_lang):
                raise ValueError(f"Unsupported language pair: {source_lang} -> {target_lang}")
            
            # Skip translation if source and target are the same
            if source_lang == target_lang:
                return text
            
            # Use two-step translation for problematic languages
            if target_lang in self.challenging_languages:
                # Step 1: Translate to Assamese first (if not already Assamese)
                if source_lang != "as":
                    assamese_text = self._direct_translate(text, source_lang, "as", max_length)
                else:
                    assamese_text = text
                
                # Step 2: Force translation from Assamese to target using Assamese token
                # This ensures we get actual translation, not copy
                final_translation = self._force_translate_from_assamese(assamese_text, target_lang, max_length)
                
                # Add notification
                quality_note = f" [Translated via Assamese - {self.language_names[target_lang]} has limited direct support]"
                return final_translation + quality_note
            
            # For well-supported languages, use direct translation
            return self._direct_translate(text, source_lang, target_lang, max_length)
            
        except Exception as e:
            logger.error(f"Error during translation ({source_lang}->{target_lang}): {e}")
            return f"Translation error: {str(e)}"
    
    def _force_translate_from_assamese(self, assamese_text: str, target_lang: str, max_length: int = 512) -> str:
        """
        Force translation from Assamese to target language using Assamese as source
        """
        # Use Assamese as source language explicitly
        source_code = "asm_Beng"  # Assamese code
        target_code = self.language_codes[target_lang]
        
        # Set Assamese as source language
        if hasattr(self.tokenizer, 'src_lang'):
            self.tokenizer.src_lang = source_code
        
        # Prepare input with Assamese text
        inputs = self.tokenizer(
            assamese_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get target language token ID
        target_token_id = self.lang_token_ids.get(target_lang, 2)
        
        # Generate translation with higher temperature for diversity
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=target_token_id,
                max_length=max_length,
                num_beams=3,
                early_stopping=True,
                do_sample=True,
                temperature=0.8,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode translation
        translation = self.tokenizer.decode(
            generated_tokens[0], 
            skip_special_tokens=True
        )
        
        return translation.strip()
    
    def _direct_translate(self, text: str, source_lang: str, target_lang: str, max_length: int = 512) -> str:
        """
        Direct translation between any supported language pair
        
        Args:
            text (str): Text to translate
            source_lang (str): Source language code (e.g., 'en', 'as', 'brx')
            target_lang (str): Target language code (e.g., 'en', 'as', 'brx')
            max_length (int): Maximum length of generated translation
            
        Returns:
            str: Translated text
        """
        try:
            # Get NLLB language codes
            source_code = self.language_codes[source_lang]
            target_code = self.language_codes[target_lang]
            
            # Set source language for tokenizer
            self.tokenizer.src_lang = source_code
            
            # Prepare input
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Get target language token ID from pre-computed mapping
            target_token_id = self.lang_token_ids.get(target_lang, 2)
            
            # Generate translation
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=target_token_id,
                    max_length=max_length,
                    num_beams=5,
                    early_stopping=True,
                    do_sample=False,
                    temperature=1.0
                )
            
            # Decode translation
            translation = self.tokenizer.decode(
                generated_tokens[0], 
                skip_special_tokens=True
            )
            
            return translation.strip()
            
        except Exception as e:
            logger.error(f"Error during translation ({source_lang}->{target_lang}): {e}")
            return f"Translation error: {str(e)}"
    
    def batch_translate(self, texts: List[str], source_lang: str, target_lang: str, max_length: int = 512) -> List[str]:
        """
        Translate multiple texts at once
        
        Args:
            texts (List[str]): List of texts to translate
            source_lang (str): Source language code
            target_lang (str): Target language code
            max_length (int): Maximum length of generated translations
            
        Returns:
            List[str]: List of translated texts
        """
        translations = []
        for text in texts:
            translation = self.translate(text, source_lang, target_lang, max_length)
            translations.append(translation)
        
        return translations
    
    def detect_language(self, text: str) -> str:
        """
        Detect language based on script and character patterns
        Returns language code for supported languages
        """
        if not text or not text.strip():
            return "en"  # Default to English for empty text
        
        # Check for Bengali script (Assamese, Bengali, and Manipuri use Bengali script)
        if any('\u0980' <= char <= '\u09FF' for char in text):
            # Try to distinguish between Bengali, Assamese, and Manipuri
            # Bengali has some unique characters and patterns
            bengali_chars = ['\u09CE', '\u09DC', '\u09DD', '\u09DF']  # Bengali-specific chars
            if any(char in text for char in bengali_chars):
                return "bn"  # Bengali
            
            # Assamese has some unique characters
            assamese_chars = ['\u09F0', '\u09F1']  # Assamese-specific chars
            if any(char in text for char in assamese_chars):
                return "as"  # Assamese
            
            # Default to Bengali for Bengali script (most common)
            return "bn"
        
        # Check for Ol Chiki script (Santali)
        if any('\u1C50' <= char <= '\u1C7F' for char in text):
            return "sat"
        
        # Check for Meetei Mayek script (native Manipuri script)
        if any('\uAAE0' <= char <= '\uAAFF' for char in text):
            return "mni"
        
        # Default to English if uncertain
        return "en"
    
    def hybrid_translate(self, text: str, source_lang: str, target_lang: str, max_length: int = 512) -> str:
        """
        Hybrid translation system using multiple translation services as fallbacks
        
        Args:
            text (str): Text to translate
            source_lang (str): Source language code
            target_lang (str): Target language code
            max_length (int): Maximum length of generated translation
            
        Returns:
            str: Translated text
        """
        try:
            # Try direct translation first
            translation = self.translate(text, source_lang, target_lang, max_length)
            
            # If direct translation fails, use hybrid translation system
            if translation.startswith("Translation error"):
                logger.info(f"Direct translation failed, using hybrid translation system for {source_lang}->{target_lang}")
                
                # Try Google Translate API
                if self.translation_services["google"]["api_key"]:
                    url = self.translation_services["google"]["url"]
                    params = {
                        "key": self.translation_services["google"]["api_key"],
                        "q": text,
                        "source": source_lang,
                        "target": target_lang,
                        "format": "text"
                    }
                    response = requests.get(url, params=params)
                    if response.status_code == 200:
                        translation = response.json()["data"]["translations"][0]["translatedText"]
                        logger.info(f"Google Translate API successful for {source_lang}->{target_lang}")
                        return translation
                
                # Try Microsoft Translator API
                if self.translation_services["microsoft"]["api_key"]:
                    url = self.translation_services["microsoft"]["url"]
                    headers = {
                        "Ocp-Apim-Subscription-Key": self.translation_services["microsoft"]["api_key"]
                    }
                    params = {
                        "text": text,
                        "from": source_lang,
                        "to": target_lang
                    }
                    response = requests.get(url, headers=headers, params=params)
                    if response.status_code == 200:
                        translation = response.json()["d"]["results"][0]["Text"]
                        logger.info(f"Microsoft Translator API successful for {source_lang}->{target_lang}")
                        return translation
                
                # If all translation services fail, return error message
                logger.error(f"All translation services failed for {source_lang}->{target_lang}")
                return f"Translation error: All translation services failed"
            
            return translation
        
        except Exception as e:
            logger.error(f"Error during hybrid translation ({source_lang}->{target_lang}): {e}")
            return f"Translation error: {str(e)}"

# Legacy compatibility class
class EnglishToAssameseTranslator:
    """
    Legacy compatibility wrapper for the old EnglishToAssameseTranslator
    """
    
    def __init__(self, model_path: str = None):
        self.translator = MultiLanguageTranslator()
    
    def translate(self, text: str, max_length: int = 512) -> str:
        return self.translator.translate(text, "en", "as", max_length)
    
    def batch_translate(self, texts: list, max_length: int = 512) -> list:
        return self.translator.batch_translate(texts, "en", "as", max_length)

def main():
    """
    Main function for testing multi-language translation
    """
    # Initialize translator
    translator = MultiLanguageTranslator()
    
    # Test sentences in different languages
    test_cases = [
        # English to regional languages
        ("Hello, how are you?", "en", "as"),
        ("Community health workers are important.", "en", "mni"),
        ("Education is the key to development.", "en", "sat"),
        ("Clean water is essential for health.", "en", "as"),
        ("Thank you for your help.", "en", "mni"),
        
        # Regional languages to English (if you have sample text)
        # Note: Add actual regional language text for reverse translation testing
    ]
    
    print("\n" + "="*80)
    print("MULTI-LANGUAGE BIDIRECTIONAL TRANSLATION DEMO")
    print("="*80)
    print(f"Supported Languages: {', '.join(translator.get_supported_languages().values())}")
    print("="*80)
    
    for i, (text, source, target) in enumerate(test_cases, 1):
        source_name = translator.language_names[source]
        target_name = translator.language_names[target]
        
        print(f"\n{i}. {source_name}: {text}")
        translation = translator.translate(text, source, target)
        print(f"   {target_name}: {translation}")
    
    print("\n" + "="*80)
    print("Multi-language translation demo completed!")
    print(f"Total supported language pairs: {len(translator.get_language_pairs())}")
    print("="*80)

if __name__ == "__main__":
    main()
