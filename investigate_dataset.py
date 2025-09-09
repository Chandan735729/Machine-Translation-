#!/usr/bin/env python3
"""
Script to investigate available configurations in ai4bharat/sangraha dataset
"""

from datasets import get_dataset_config_names, load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def investigate_sangraha_dataset():
    """
    Investigate the ai4bharat/sangraha dataset configurations
    """
    try:
        # Get available configurations
        logger.info("Fetching available configurations for ai4bharat/sangraha...")
        configs = get_dataset_config_names("ai4bharat/sangraha")
        logger.info(f"Available configurations: {configs}")
        
        # Look for English-Assamese configurations
        en_as_configs = [config for config in configs if 'en' in config.lower() and 'as' in config.lower()]
        logger.info(f"English-Assamese related configs: {en_as_configs}")
        
        # Try to load a few configurations to see their structure
        for config in configs[:5]:  # Check first 5 configs
            try:
                logger.info(f"\n--- Checking configuration: {config} ---")
                dataset = load_dataset("ai4bharat/sangraha", config, split="train[:10]")  # Load only 10 samples
                logger.info(f"Columns: {dataset.column_names}")
                logger.info(f"Sample data: {dataset[0]}")
                
                # Check if this might be English-Assamese
                sample = dataset[0]
                if any(key in sample for key in ['en', 'eng', 'english']) and any(key in sample for key in ['as', 'asm', 'assamese']):
                    logger.info(f"*** FOUND POTENTIAL ENGLISH-ASSAMESE CONFIG: {config} ***")
                    
            except Exception as e:
                logger.warning(f"Failed to load config {config}: {e}")
                
    except Exception as e:
        logger.error(f"Error investigating dataset: {e}")
        return None
        
    return configs

def try_alternative_datasets():
    """
    Try alternative datasets for English-Assamese translation
    """
    alternative_datasets = [
        "cfilt/iitb-english-hindi",  # Large Indian language dataset
        "Helsinki-NLP/opus-100",     # Multilingual dataset
        "facebook/flores",           # Facebook's multilingual dataset
    ]
    
    for dataset_name in alternative_datasets:
        try:
            logger.info(f"\n--- Checking alternative dataset: {dataset_name} ---")
            configs = get_dataset_config_names(dataset_name)
            logger.info(f"Available configs: {configs}")
            
            # Look for Assamese-related configs
            as_configs = [config for config in configs if 'as' in config.lower() or 'asm' in config.lower()]
            if as_configs:
                logger.info(f"*** FOUND ASSAMESE CONFIGS in {dataset_name}: {as_configs} ***")
                
        except Exception as e:
            logger.warning(f"Failed to check {dataset_name}: {e}")

if __name__ == "__main__":
    logger.info("Starting dataset investigation...")
    configs = investigate_sangraha_dataset()
    try_alternative_datasets()
    logger.info("Investigation completed!")
