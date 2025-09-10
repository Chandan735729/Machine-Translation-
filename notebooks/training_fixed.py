"""
Fixed Training Module for Colab - Optimized parameters for small datasets
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
import gc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationTrainerFixed:
    """
    FIXED Translation Trainer with optimized parameters for small datasets
    """
    
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🎮 Using device: {self.device}")
        
        # Clear GPU memory before loading model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Load tokenizer and model
        logger.info(f"📥 Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Move model to device
        self.model.to(self.device)
        logger.info(f"✅ Model loaded and moved to {self.device}")
        
        # Print GPU memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(0) / 1e9
            logger.info(f"💾 GPU Memory used: {memory_used:.2f} GB")
    
    def setup_training_arguments_optimized(self, output_dir: str = "models/nllb-finetuned-en-to-asm"):
        """
        OPTIMIZED training arguments for small datasets based on memory fixes
        """
        logger.info("⚙️ Setting up OPTIMIZED training arguments...")
        
        return TrainingArguments(
            output_dir=output_dir,
            # OPTIMIZED: Reduced batch size for stability with small dataset
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            # OPTIMIZED: Increased gradient accumulation to maintain effective batch size
            gradient_accumulation_steps=4,
            # OPTIMIZED: Adjusted learning rate for fine-tuning
            learning_rate=3e-5,
            # OPTIMIZED: More epochs for small dataset
            num_train_epochs=5,
            # OPTIMIZED: Reduced warmup for small dataset
            warmup_steps=50,
            # OPTIMIZED: More frequent logging and evaluation
            logging_steps=10,
            eval_steps=50,
            save_steps=100,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # Disable wandb logging for Colab
            dataloader_pin_memory=False,
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
            # OPTIMIZED: Additional regularization for small datasets
            weight_decay=0.01,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            seed=42,  # For reproducibility
            remove_unused_columns=False,
            # OPTIMIZED: Colab-specific settings
            dataloader_num_workers=0,  # Avoid multiprocessing issues in Colab
            save_total_limit=2,  # Limit saved checkpoints to save space
            prediction_loss_only=True,  # Simplify evaluation
        )
    
    def train_model_optimized(self, dataset, output_dir: str = "models/nllb-finetuned-en-to-asm"):
        """
        OPTIMIZED model training with better memory management
        """
        logger.info("🚀 Starting OPTIMIZED model training...")
        
        # Setup training arguments
        training_args = self.setup_training_arguments_optimized(output_dir)
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Print training info
        logger.info(f"📊 Training Configuration:")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Train samples: {len(dataset['train'])}")
        logger.info(f"  Validation samples: {len(dataset.get('validation', []))}")
        logger.info(f"  Batch size: {training_args.per_device_train_batch_size}")
        logger.info(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
        logger.info(f"  Learning rate: {training_args.learning_rate}")
        logger.info(f"  Epochs: {training_args.num_train_epochs}")
        logger.info(f"  Device: {self.device}")
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation"),
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Clear memory before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Train the model
        logger.info("🔥 Starting training process...")
        trainer.train()
        
        # Save the final model
        final_output_dir = f"{output_dir}-final"
        trainer.save_model(final_output_dir)
        self.tokenizer.save_pretrained(final_output_dir)
        
        logger.info(f"✅ Training completed! Model saved to {final_output_dir}")
        
        # Clear memory after training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        return trainer, final_output_dir
    
    def evaluate_model_optimized(self, trainer, dataset):
        """
        OPTIMIZED model evaluation
        """
        logger.info("📊 Evaluating model performance...")
        
        try:
            eval_results = trainer.evaluate()
            
            # Save evaluation results
            os.makedirs("results", exist_ok=True)
            with open("results/evaluation_results.json", "w") as f:
                json.dump(eval_results, f, indent=2)
            
            logger.info(f"✅ Evaluation completed")
            return eval_results
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            return {"eval_loss": "N/A", "error": str(e)}

def run_complete_training_pipeline():
    """
    Complete training pipeline for Colab
    """
    from data_preparation_fixed import DataPreparatorFixed
    
    logger.info("🌟 Starting COMPLETE OPTIMIZED training pipeline...")
    
    # Step 1: Data Preparation
    logger.info("📚 Step 1: Data Preparation")
    preparator = DataPreparatorFixed()
    raw_dataset = preparator.load_dataset_with_fallbacks()
    processed_dataset = preparator.prepare_datasets_fixed(raw_dataset)
    preparator.save_processed_data(processed_dataset)
    
    # Print dataset stats
    stats = preparator.get_data_stats(processed_dataset)
    logger.info(f"📊 Dataset Statistics: {stats}")
    
    # Step 2: Model Training
    logger.info("🤖 Step 2: Model Training")
    trainer_obj = TranslationTrainerFixed()
    trainer, model_path = trainer_obj.train_model_optimized(processed_dataset)
    
    # Step 3: Model Evaluation
    logger.info("📈 Step 3: Model Evaluation")
    eval_results = trainer_obj.evaluate_model_optimized(trainer, processed_dataset)
    
    logger.info("🎉 COMPLETE training pipeline finished!")
    logger.info(f"📁 Final model saved at: {model_path}")
    
    return model_path, eval_results, stats
