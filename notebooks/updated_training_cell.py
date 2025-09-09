# Updated Training Cell - Replace the content in cell "5. Model Training"

# Import training modules with improved parameters
from train import TranslationTrainer
import os
import torch
from transformers import TrainingArguments

class ImprovedTranslationTrainer(TranslationTrainer):
    """Enhanced trainer with optimized parameters for small datasets"""
    
    def setup_training_arguments(self, output_dir: str = "models/nllb-finetuned-en-to-asm"):
        """Configure optimized training arguments"""
        return TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2,  # Reduced for stability
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,  # Increased to maintain effective batch size
            learning_rate=3e-5,  # Optimized learning rate
            num_train_epochs=5,  # More epochs for small dataset
            warmup_steps=50,  # Reduced warmup for small dataset
            logging_steps=10,  # More frequent logging
            eval_steps=50,  # More frequent evaluation
            save_steps=100,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # Disable wandb logging
            dataloader_pin_memory=False,
            fp16=torch.cuda.is_available(),
            # Additional optimizations
            weight_decay=0.01,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            seed=42,  # For reproducibility
            remove_unused_columns=False,
        )

# Initialize improved trainer
print("🤖 Initializing enhanced translation trainer...")
trainer_obj = ImprovedTranslationTrainer()

# Load processed data
print("📂 Loading processed dataset...")
dataset = trainer_obj.load_processed_data()

print(f"\n📋 Enhanced Training Configuration:")
print(f"  Model: facebook/nllb-200-distilled-600M")
print(f"  Dataset: Enhanced sample dataset (English-Assamese)")
print(f"  Train samples: {len(dataset['train'])}")
print(f"  Validation samples: {len(dataset.get('validation', []))}")
print(f"  Device: {trainer_obj.device}")
print(f"  Batch size: 2 (effective: 8 with gradient accumulation)")
print(f"  Learning rate: 3e-5")
print(f"  Epochs: 5")
print(f"  Mixed precision: {torch.cuda.is_available()}")

# Start training
print("\n🚀 Starting enhanced model training...")
print("This will take 45-90 minutes depending on your GPU.")
print("Training is optimized for small datasets with:")
print("  ✅ Expanded dataset (25 examples)")
print("  ✅ Optimized batch size and learning rate")
print("  ✅ More frequent evaluation and logging")
print("  ✅ Regularization to prevent overfitting")

trainer, model_path = trainer_obj.train_model(dataset)

print(f"\n✅ Enhanced training completed!")
print(f"📁 Model saved to: {model_path}")
print(f"🎯 Training used {len(dataset['train'])} samples across 5 epochs")
