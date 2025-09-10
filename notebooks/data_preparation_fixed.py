"""
Fixed Data Preparation for Colab - Incorporates all recent improvements
"""

import os
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer
import json
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreparatorFixed:
    """
    FIXED Data Preparation class with all recent improvements
    """
    
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.source_lang = "eng_Latn"
        self.target_lang = "asm_Beng"
        
        # Set source and target languages for NLLB tokenizer
        self.tokenizer.src_lang = self.source_lang
        self.tokenizer.tgt_lang = self.target_lang
        
    def load_dataset_with_fallbacks(self):
        """
        Load dataset with multiple fallback options - FIXED VERSION
        """
        logger.info("🔄 Loading dataset with fallback options...")
        
        # Try multiple dataset sources for English-Assamese
        dataset_configs = [
            ("ai4bharat/sangraha", "eng-asm"),
            ("ai4bharat/sangraha", "en-as"), 
            ("ai4bharat/sangraha", "english-assamese"),
            ("Helsinki-NLP/opus-100", "en-as"),
            ("facebook/flores", "eng_Latn-asm_Beng"),
        ]
        
        for dataset_name, config in dataset_configs:
            try:
                logger.info(f"🔍 Trying {dataset_name} with config {config}")
                dataset = load_dataset(dataset_name, config)
                logger.info(f"✅ Dataset loaded successfully! Train size: {len(dataset['train'])}")
                return dataset
            except Exception as e:
                logger.warning(f"❌ Failed to load {dataset_name} with {config}: {e}")
                continue
        
        # If all fail, create expanded sample dataset
        logger.info("📝 All dataset sources failed. Creating expanded sample dataset")
        return self._create_expanded_sample_dataset()
    
    def _create_expanded_sample_dataset(self) -> DatasetDict:
        """
        Create expanded sample dataset with 25 examples - FIXED VERSION
        """
        logger.info("📚 Creating expanded sample dataset with more examples")
        
        sample_data = [
            # Health & Medical (5 examples)
            {"en": "Community health workers are the backbone of our medical system.", "as": "সামূহিক স্বাস্থ্য কৰ্মীসকল আমাৰ চিকিৎসা ব্যৱস্থাৰ মেৰুদণ্ড।"},
            {"en": "Clean water is essential for good health.", "as": "ভাল স্বাস্থ্যৰ বাবে পৰিষ্কাৰ পানী অপৰিহাৰ্য।"},
            {"en": "Vaccination protects children from diseases.", "as": "টিকাকৰণে শিশুসকলক ৰোগৰ পৰা সুৰক্ষা দিয়ে।"},
            {"en": "Regular exercise keeps the body healthy.", "as": "নিয়মীয়া ব্যায়ামে শৰীৰক সুস্থ ৰাখে।"},
            {"en": "Proper nutrition is important for growth.", "as": "বৃদ্ধিৰ বাবে সঠিক পুষ্টি গুৰুত্বপূৰ্ণ।"},
            
            # Education & Development (5 examples)
            {"en": "Education is the key to development.", "as": "শিক্ষা উন্নয়নৰ চাবিকাঠি।"},
            {"en": "Women's empowerment leads to stronger communities.", "as": "মহিলা সৱলীকৰণে শক্তিশালী সমাজৰ সৃষ্টি কৰে।"},
            {"en": "Knowledge is power.", "as": "জ্ঞানেই শক্তি।"},
            {"en": "Every child has the right to education.", "as": "প্ৰতিটো শিশুৰ শিক্ষাৰ অধিকাৰ আছে।"},
            {"en": "Teachers shape the future of society.", "as": "শিক্ষকসকলে সমাজৰ ভৱিষ্যত গঢ় দিয়ে।"},
            
            # Daily Conversations (5 examples)
            {"en": "Hello, how are you?", "as": "নমস্কাৰ, আপুনি কেনে আছে?"},
            {"en": "Thank you for your help.", "as": "আপোনাৰ সহায়ৰ বাবে ধন্যবাদ।"},
            {"en": "The weather is nice today.", "as": "আজি বতৰটো ভাল।"},
            {"en": "What is your name?", "as": "আপোনাৰ নাম কি?"},
            {"en": "I am fine, thank you.", "as": "মই ভাল আছো, ধন্যবাদ।"},
            
            # Technology & Modern Life (5 examples)
            {"en": "Technology has changed our lives.", "as": "প্ৰযুক্তিয়ে আমাৰ জীৱন সলনি কৰিছে।"},
            {"en": "Mobile phones are very useful.", "as": "মোবাইল ফোন অতি উপযোগী।"},
            {"en": "The internet connects the world.", "as": "ইণ্টাৰনেটে বিশ্বক সংযুক্ত কৰে।"},
            {"en": "Digital literacy is important today.", "as": "আজি ডিজিটেল সাক্ষৰতা গুৰুত্বপূৰ্ণ।"},
            {"en": "Computers help us work faster.", "as": "কম্পিউটাৰে আমাক দ্ৰুত কাম কৰাত সহায় কৰে।"},
            
            # Culture & Environment (5 examples)
            {"en": "Assamese culture is very rich.", "as": "অসমীয়া সংস্কৃতি অতি চহকী।"},
            {"en": "We must protect our environment.", "as": "আমি আমাৰ পৰিৱেশ ৰক্ষা কৰিব লাগিব।"},
            {"en": "Trees are important for clean air.", "as": "বিশুদ্ধ বায়ুৰ বাবে গছ গুৰুত্বপূৰ্ণ।"},
            {"en": "Festivals bring people together.", "as": "উৎসৱে মানুহক একগোট কৰে।"},
            {"en": "Climate change affects everyone.", "as": "জলবায়ু পৰিৱৰ্তনে সকলোকে প্ৰভাৱিত কৰে।"}
        ]
        
        # Create train/validation split (80/20 = 20 train, 5 validation)
        split_idx = int(0.8 * len(sample_data))
        train_data = sample_data[:split_idx]
        val_data = sample_data[split_idx:]
        
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        logger.info(f"✅ Created expanded dataset: {len(train_data)} training, {len(val_data)} validation samples")
        
        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
    
    def preprocess_function_fixed(self, examples):
        """
        FIXED preprocessing function - resolves tokenizer deprecation warning
        """
        # Handle different possible column names
        if "src" in examples and "tgt" in examples:
            source_texts = examples["src"]
            target_texts = examples["tgt"]
        elif "en" in examples and "as" in examples:
            source_texts = examples["en"]
            target_texts = examples["as"]
        elif "english" in examples and "assamese" in examples:
            source_texts = examples["english"]
            target_texts = examples["assamese"]
        else:
            keys = list(examples.keys())
            source_texts = examples[keys[0]]
            target_texts = examples[keys[1]]
        
        inputs = [f"{self.source_lang}: {text}" for text in source_texts]
        targets = [f"{self.target_lang}: {text}" for text in target_texts]
        
        # Tokenize inputs
        model_inputs = self.tokenizer(
            inputs, 
            max_length=512, 
            truncation=True, 
            padding=True
        )
        
        # FIXED: Use text_target parameter instead of deprecated as_target_tokenizer
        labels = self.tokenizer(
            text_target=targets,
            max_length=512, 
            truncation=True, 
            padding=True
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def prepare_datasets_fixed(self, dataset: DatasetDict) -> DatasetDict:
        """
        Apply FIXED preprocessing to the entire dataset
        """
        logger.info("⚙️ Preprocessing datasets with FIXED tokenizer...")
        
        tokenized_datasets = dataset.map(
            self.preprocess_function_fixed,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        logger.info("✅ Dataset preprocessing completed")
        return tokenized_datasets
    
    def save_processed_data(self, dataset: DatasetDict, output_dir: str = "data/processed"):
        """
        Save processed datasets to disk
        """
        os.makedirs(output_dir, exist_ok=True)
        dataset.save_to_disk(output_dir)
        logger.info(f"💾 Processed data saved to {output_dir}")
    
    def get_data_stats(self, dataset: DatasetDict) -> Dict:
        """
        Get statistics about the dataset
        """
        stats = {
            "train_size": len(dataset["train"]),
            "validation_size": len(dataset.get("validation", [])),
            "source_language": self.source_lang,
            "target_language": self.target_lang,
            "model_name": self.model_name,
            "tokenizer_fixed": True,
            "expanded_dataset": True
        }
        return stats
