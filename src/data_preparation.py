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
        
    def load_dataset(self, dataset_name: str = "ai4bharat/sangraha") -> DatasetDict:
        """
        Load the Sangraha dataset for English-Assamese translation
        """
        logger.info(f"Loading dataset: {dataset_name}")
        try:
            # Try loading the sangraha dataset with English-Assamese pair
            dataset = load_dataset(dataset_name, "eng-asm")
            logger.info(f"Dataset loaded successfully. Train size: {len(dataset['train'])}")
            return dataset
        except Exception as e:
            logger.error(f"Error loading dataset with eng-asm: {e}")
            try:
                # Try alternative configuration
                dataset = load_dataset(dataset_name, "en-as")
                logger.info(f"Dataset loaded with en-as config. Train size: {len(dataset['train'])}")
                return dataset
            except Exception as e2:
                logger.error(f"Error loading dataset with en-as: {e2}")
                # Fallback to creating a sample dataset
                logger.info("Falling back to sample dataset")
                return self._create_sample_dataset()
    
    def _create_sample_dataset(self) -> DatasetDict:
        """
        Create a sample dataset for demonstration purposes
        """
        logger.info("Creating sample dataset for demonstration")
        
        sample_data = [
            {
                "en": "Community health workers are the backbone of our medical system.",
                "as": "সামূহিক স্বাস্থ্য কৰ্মীসকল আমাৰ চিকিৎসা ব্যৱস্থাৰ মেৰুদণ্ড।"
            },
            {
                "en": "Education is the key to development.",
                "as": "শিক্ষা উন্নয়নৰ চাবিকাঠি।"
            },
            {
                "en": "Clean water is essential for good health.",
                "as": "ভাল স্বাস্থ্যৰ বাবে পৰিষ্কাৰ পানী অপৰিহাৰ্য।"
            },
            {
                "en": "Vaccination protects children from diseases.",
                "as": "টিকাকৰণে শিশুসকলক ৰোগৰ পৰা সুৰক্ষা দিয়ে।"
            },
            {
                "en": "Women's empowerment leads to stronger communities.",
                "as": "মহিলা সৱলীকৰণে শক্তিশালী সমাজৰ সৃষ্টি কৰে।"
            }
        ]
        
        # Create train/validation split
        train_data = sample_data[:4]
        val_data = sample_data[4:]
        
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
    
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
        
        # Use text_target parameter instead of deprecated as_target_tokenizer
        labels = self.tokenizer(
            text_target=targets,
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
