"""
Model Training Script for English-Assamese Translation
Fine-tunes NLLB model on English-Assamese dataset
"""

import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import load_from_disk
import json
import logging
from data_preparation import DataPreparator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationTrainer:
    """
    Handles the fine-tuning of NLLB model for English-Assamese translation
    """
    
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Move model to device
        self.model.to(self.device)
        
    def load_processed_data(self, data_path: str = "data/processed"):
        """
        Load preprocessed datasets
        """
        if os.path.exists(data_path):
            logger.info(f"Loading processed data from {data_path}")
            return load_from_disk(data_path)
        else:
            logger.info("Processed data not found. Running data preparation...")
            preparator = DataPreparator()
            raw_dataset = preparator.load_dataset()
            processed_dataset = preparator.prepare_datasets(raw_dataset)
            preparator.save_processed_data(processed_dataset)
            return processed_dataset
    
    def setup_training_arguments(self, output_dir: str = "models/nllb-finetuned-en-to-asm"):
        """
        Configure training arguments
        """
        return TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=5e-5,
            num_train_epochs=3,
            warmup_steps=500,
            logging_steps=100,
            eval_steps=500,
            save_steps=1000,
            eval_strategy="steps",  # Changed from evaluation_strategy
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # Disable wandb logging
            dataloader_pin_memory=False,
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        )
    
    def train_model(self, dataset, output_dir: str = "models/nllb-finetuned-en-to-asm"):
        """
        Fine-tune the model on the dataset
        """
        logger.info("Starting model training...")
        
        # Setup training arguments
        training_args = self.setup_training_arguments(output_dir)
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"),
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train the model
        trainer.train()
        
        # Save the final model
        final_output_dir = f"{output_dir}-final"
        trainer.save_model(final_output_dir)
        self.tokenizer.save_pretrained(final_output_dir)
        
        logger.info(f"Training completed. Model saved to {final_output_dir}")
        
        return trainer, final_output_dir
    
    def evaluate_model(self, trainer, dataset):
        """
        Evaluate the trained model
        """
        logger.info("Evaluating model...")
        eval_results = trainer.evaluate()
        
        # Save evaluation results
        os.makedirs("results", exist_ok=True)
        with open("results/evaluation_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)
        
        logger.info(f"Evaluation completed. Results: {eval_results}")
        return eval_results

def main():
    """
    Main training pipeline
    """
    logger.info("Starting training pipeline...")
    
    # Initialize trainer
    trainer_obj = TranslationTrainer()
    
    # Load processed data
    dataset = trainer_obj.load_processed_data()
    
    # Train model
    trainer, model_path = trainer_obj.train_model(dataset)
    
    # Evaluate model
    eval_results = trainer_obj.evaluate_model(trainer, dataset)
    
    logger.info("Training pipeline completed successfully!")
    logger.info(f"Final model saved at: {model_path}")
    
    return model_path

if __name__ == "__main__":
    main()
