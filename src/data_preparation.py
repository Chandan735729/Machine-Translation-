
"""
Data Preparation Pipeline for English-Assamese Translation
Handles dataset loading, cleaning, and preprocessing for NLLB fine-tuning
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

class DataPreparator:
    """
    Handles data preparation for English-Assamese translation model training
    """
    
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.source_lang = "eng_Latn"
        self.target_lang = "asm_Beng"
        
        # Set source and target languages for NLLB tokenizer
        self.tokenizer.src_lang = self.source_lang
        self.tokenizer.tgt_lang = self.target_lang
        
    def load_dataset(self, dataset_name: str = "ai4bharat/sangraha") -> DatasetDict:
        """
        Load the Sangraha dataset for English-Assamese translation
        """
        logger.info(f"Loading dataset: {dataset_name}")
        
        # Try multiple dataset sources for English-Assamese
        dataset_configs = [
            ("ai4bharat/sangraha", "eng-asm"),
            ("ai4bharat/sangraha", "en-as"),
            ("ai4bharat/sangraha", "english-assamese"),
            ("Helsinki-NLP/opus-100", "en-as"),
            ("facebook/flores", "eng_Latn-asm_Beng"),
        ]
        
        for dataset_name_try, config in dataset_configs:
            try:
                logger.info(f"Trying {dataset_name_try} with config {config}")
                dataset = load_dataset(dataset_name_try, config)
                logger.info(f"Dataset loaded successfully! Train size: {len(dataset['train'])}")
                return dataset
            except Exception as e:
                logger.warning(f"Failed to load {dataset_name_try} with {config}: {e}")
                continue
        
        # If all fail, create an expanded sample dataset
        logger.info("All dataset sources failed. Creating expanded sample dataset")
        return self._create_expanded_sample_dataset()
    
    def _create_expanded_sample_dataset(self) -> DatasetDict:
        """
        Create an expanded sample dataset with more training examples
        """
        logger.info("Creating expanded sample dataset with more examples")
        
        sample_data = [
            # Health & Medical
            {"en": "Community health workers are the backbone of our medical system.", "as": "সামূহিক স্বাস্থ্য কৰ্মীসকল আমাৰ চিকিৎসা ব্যৱস্থাৰ মেৰুদণ্ড।"},
            {"en": "Clean water is essential for good health.", "as": "ভাল স্বাস্থ্যৰ বাবে পৰিষ্কাৰ পানী অপৰিহাৰ্য।"},
            {"en": "Vaccination protects children from diseases.", "as": "টিকাকৰণে শিশুসকলক ৰোগৰ পৰা সুৰক্ষা দিয়ে।"},
            {"en": "Regular exercise keeps the body healthy.", "as": "নিয়মীয়া ব্যায়ামে শৰীৰক সুস্থ ৰাখে।"},
            {"en": "Proper nutrition is important for growth.", "as": "বৃদ্ধিৰ বাবে সঠিক পুষ্টি গুৰুত্বপূৰ্ণ।"},
            
            # Education & Development
            {"en": "Education is the key to development.", "as": "শিক্ষা উন্নয়নৰ চাবিকাঠি।"},
            {"en": "Women's empowerment leads to stronger communities.", "as": "মহিলা সৱলীকৰণে শক্তিশালী সমাজৰ সৃষ্টি কৰে।"},
            {"en": "Knowledge is power.", "as": "জ্ঞানেই শক্তি।"},
            {"en": "Every child has the right to education.", "as": "প্ৰতিটো শিশুৰ শিক্ষাৰ অধিকাৰ আছে।"},
            {"en": "Teachers shape the future of society.", "as": "শিক্ষকসকলে সমাজৰ ভৱিষ্যত গঢ় দিয়ে।"},
            
            # Daily Conversations
            {"en": "Hello, how are you?", "as": "নমস্কাৰ, আপুনি কেনে আছে?"},
            {"en": "Thank you for your help.", "as": "আপোনাৰ সহায়ৰ বাবে ধন্যবাদ।"},
            {"en": "The weather is nice today.", "as": "আজি বতৰটো ভাল।"},
            {"en": "What is your name?", "as": "আপোনাৰ নাম কি?"},
            {"en": "I am fine, thank you.", "as": "মই ভাল আছো, ধন্যবাদ।"},
            
            # Technology & Modern Life
            {"en": "Technology has changed our lives.", "as": "প্ৰযুক্তিয়ে আমাৰ জীৱন সলনি কৰিছে।"},
            {"en": "Mobile phones are very useful.", "as": "মোবাইল ফোন অতি উপযোগী।"},
            {"en": "The internet connects the world.", "as": "ইণ্টাৰনেটে বিশ্বক সংযুক্ত কৰে।"},
            {"en": "Digital literacy is important today.", "as": "আজি ডিজিটেল সাক্ষৰতা গুৰুত্বপূৰ্ণ।"},
            
            # Culture & Society
            {"en": "Assamese culture is very rich.", "as": "অসমীয়া সংস্কৃতি অতি চহকী।"},
            {"en": "Unity in diversity is our strength.", "as": "বৈচিত্ৰ্যৰ মাজত একতাই আমাৰ শক্তি।"},
            {"en": "Respect for elders is important.", "as": "বয়োজ্যেষ্ঠসকলৰ প্ৰতি সন্মান গুৰুত্বপূৰ্ণ।"},
            {"en": "Festivals bring people together.", "as": "উৎসৱে মানুহক একগোট কৰে।"},
            
            # Environment & Nature
            {"en": "We must protect our environment.", "as": "আমি আমাৰ পৰিৱেশ ৰক্ষা কৰিব লাগিব।"},
            {"en": "Trees are important for clean air.", "as": "বিশুদ্ধ বায়ুৰ বাবে গছ গুৰুত্বপূৰ্ণ।"},
            {"en": "Water pollution is a serious problem.", "as": "পানী প্ৰদূষণ এটা গুৰুতৰ সমস্যা।"},
            {"en": "Climate change affects everyone.", "as": "জলবায়ু পৰিৱৰ্তনে সকলোকে প্ৰভাৱিত কৰে।"}
        ]
        
        # Create train/validation split (80/20)
        split_idx = int(0.8 * len(sample_data))
        train_data = sample_data[:split_idx]
        val_data = sample_data[split_idx:]
        
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        logger.info(f"Created expanded dataset: {len(train_data)} training, {len(val_data)} validation samples")
        
        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
    
    def _create_sample_dataset(self) -> DatasetDict:
        """
        Create a sample dataset for demonstration purposes (kept for backward compatibility)
        """
        return self._create_expanded_sample_dataset()
    
    def preprocess_function(self, examples):
        """
        Preprocess the data for NLLB model training
        """
        # Handle different possible column names from sangraha dataset
        if "src" in examples and "tgt" in examples:
            # Sangraha dataset format
            source_texts = examples["src"]
            target_texts = examples["tgt"]
        elif "en" in examples and "as" in examples:
            # Alternative format
            source_texts = examples["en"]
            target_texts = examples["as"]
        elif "english" in examples and "assamese" in examples:
            # Another possible format
            source_texts = examples["english"]
            target_texts = examples["assamese"]
        else:
            # Fallback - use first two columns
            keys = list(examples.keys())
            source_texts = examples[keys[0]]
            target_texts = examples[keys[1]]
        
        inputs = [f"{self.source_lang}: {text}" for text in source_texts]
        targets = [f"{self.target_lang}: {text}" for text in target_texts]
        
        model_inputs = self.tokenizer(
            inputs, 
            max_length=512, 
            truncation=True, 
            padding=True
        )
        
        # For NLLB tokenizer, use the updated approach to avoid deprecation warning
        # Use text_target parameter instead of as_target_tokenizer
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                max_length=512, 
                truncation=True, 
                padding=True
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def prepare_datasets(self, dataset: DatasetDict) -> DatasetDict:
        """
        Apply preprocessing to the entire dataset
        """
        logger.info("Preprocessing datasets...")
        
        tokenized_datasets = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        logger.info("Dataset preprocessing completed")
        return tokenized_datasets
    
    def save_processed_data(self, dataset: DatasetDict, output_dir: str = "data/processed"):
        """
        Save processed datasets to disk
        """
        os.makedirs(output_dir, exist_ok=True)
        dataset.save_to_disk(output_dir)
        logger.info(f"Processed data saved to {output_dir}")
    
    def get_data_stats(self, dataset: DatasetDict) -> Dict:
        """
        Get statistics about the dataset
        """
        stats = {
            "train_size": len(dataset["train"]),
            "validation_size": len(dataset.get("validation", [])),
            "source_language": self.source_lang,
            "target_language": self.target_lang,
            "model_name": self.model_name
        }
        return stats

def main():
    """
    Main function to run data preparation pipeline
    """
    logger.info("Starting data preparation pipeline...")
    
    # Initialize data preparator
    preparator = DataPreparator()
    
    # Load dataset
    raw_dataset = preparator.load_dataset()
    
    # Preprocess data
    processed_dataset = preparator.prepare_datasets(raw_dataset)
    
    # Save processed data
    preparator.save_processed_data(processed_dataset)
    
    # Print statistics
    stats = preparator.get_data_stats(processed_dataset)
    logger.info(f"Data preparation completed. Stats: {stats}")
    
    # Save stats to file
    os.makedirs("data", exist_ok=True)
    with open("data/data_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()
