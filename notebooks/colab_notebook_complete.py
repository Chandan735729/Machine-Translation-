"""
Complete Colab Notebook Content - Ready to copy into Jupyter cells
This file contains all the notebook cells with fixes applied
"""

# =============================================================================
# CELL 1: Data Preparation with Fixed Tokenizer
# =============================================================================

cell_data_prep = '''
# Data Preparation with FIXED tokenizer and multiple fallbacks
import sys
import os

# Copy the fixed data preparation code
exec(open('data_preparation_fixed.py').read())

# Initialize and run data preparation
print("🔄 Starting data preparation with FIXED tokenizer...")
preparator = DataPreparatorFixed()

# Load dataset with fallbacks
print("📥 Loading dataset with multiple fallback options...")
raw_dataset = preparator.load_dataset_with_fallbacks()

# Check dataset structure
print("\\n📊 Dataset structure:")
print(f"Available splits: {list(raw_dataset.keys())}")
if 'train' in raw_dataset:
    print(f"Train columns: {raw_dataset['train'].column_names}")
    print(f"Train size: {len(raw_dataset['train'])}")
    
    # Show sample data
    print("\\n📝 Sample data:")
    for i in range(min(3, len(raw_dataset['train']))):
        sample = raw_dataset['train'][i]
        print(f"Sample {i+1}: {sample}")

# Process dataset with FIXED tokenizer
print("\\n⚙️ Processing dataset with FIXED tokenizer (no deprecation warnings)...")
processed_dataset = preparator.prepare_datasets_fixed(raw_dataset)

# Save processed data
print("💾 Saving processed data...")
preparator.save_processed_data(processed_dataset)

# Print statistics
stats = preparator.get_data_stats(processed_dataset)
print(f"\\n📊 Dataset Statistics:")
for key, value in stats.items():
    print(f"  {key}: {value}")

print("✅ Data preparation completed successfully!")
'''

# =============================================================================
# CELL 2: Model Training with Optimized Parameters
# =============================================================================

cell_training = '''
# Model Training with OPTIMIZED parameters for small datasets
exec(open('training_fixed.py').read())

# Initialize trainer with optimized settings
print("🤖 Initializing OPTIMIZED translation trainer...")
trainer_obj = TranslationTrainerFixed()

# Load processed data
print("📂 Loading processed dataset...")
try:
    from datasets import load_from_disk
    dataset = load_from_disk("data/processed")
    print(f"✅ Loaded processed dataset from disk")
except:
    print("⚠️ Processed data not found, using current dataset")
    dataset = processed_dataset

print(f"\\n📋 OPTIMIZED Training Configuration:")
print(f"  Model: facebook/nllb-200-distilled-600M")
print(f"  Train samples: {len(dataset['train'])}")
print(f"  Validation samples: {len(dataset.get('validation', []))}")
print(f"  Batch size: 2 (optimized for small dataset)")
print(f"  Gradient accumulation: 4 (maintains effective batch size)")
print(f"  Learning rate: 3e-5 (optimized for fine-tuning)")
print(f"  Epochs: 5 (more epochs for small dataset)")
print(f"  Device: {trainer_obj.device}")

# Start training with optimized parameters
print("\\n🚀 Starting OPTIMIZED model training...")
print("⏱️ This may take 30-60 minutes depending on your GPU")
print("📊 Training uses optimized parameters based on recent fixes")

trainer, model_path = trainer_obj.train_model_optimized(dataset)

print(f"\\n🎉 Training completed successfully!")
print(f"📁 Model saved to: {model_path}")
'''

# =============================================================================
# CELL 3: Model Evaluation and Testing
# =============================================================================

cell_evaluation = '''
# Model Evaluation and Testing
print("📊 Evaluating OPTIMIZED model performance...")
eval_results = trainer_obj.evaluate_model_optimized(trainer, dataset)

print(f"\\n📈 Evaluation Results:")
for key, value in eval_results.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

# Test the trained model
print("\\n🧪 Testing translations with OPTIMIZED trained model:")
print("=" * 80)

# Test sentences covering different domains
test_sentences = [
    "Community health workers are the backbone of our medical system.",
    "Education is the key to development.", 
    "Clean water is essential for good health.",
    "Technology has changed our lives.",
    "Hello, how are you?",
    "Thank you for your help.",
    "The weather is nice today.",
    "We must protect our environment."
]

# Load the trained model for testing
from transformers import pipeline
try:
    # Create translation pipeline with trained model
    translator_pipeline = pipeline(
        "translation", 
        model=model_path,
        tokenizer=model_path,
        device=0 if torch.cuda.is_available() else -1
    )
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\\n{i}. English: {sentence}")
        try:
            # Use the pipeline for translation
            result = translator_pipeline(sentence, 
                                       src_lang="eng_Latn", 
                                       tgt_lang="asm_Beng")
            translation = result[0]['translation_text']
            print(f"   Assamese: {translation}")
        except Exception as e:
            print(f"   Translation error: {e}")
        print("-" * 60)
        
except Exception as e:
    print(f"❌ Error loading trained model for testing: {e}")
    print("💡 Model was saved successfully, but testing pipeline failed")

print("\\n✅ Model evaluation and testing completed!")
'''

# =============================================================================
# CELL 4: Download and Save Model
# =============================================================================

cell_download = '''
# Create downloadable model archive
import shutil
import json
from datetime import datetime

model_dir = model_path
archive_name = "trained_model_optimized_fixed"

if os.path.exists(model_dir):
    print("📦 Creating OPTIMIZED model archive...")
    
    # Create zip file
    shutil.make_archive(archive_name, 'zip', model_dir)
    
    print(f"✅ Model archived as '{archive_name}.zip'")
    print("📥 Download it from the Files panel on the left")
    
    # Show file size
    size_mb = os.path.getsize(f"{archive_name}.zip") / (1024 * 1024)
    print(f"📊 Archive size: {size_mb:.1f} MB")
    
    # Save comprehensive training info
    training_info = {
        "dataset_source": "Multiple fallbacks (ai4bharat/sangraha, expanded sample)",
        "base_model": "facebook/nllb-200-distilled-600M",
        "language_pair": "English-Assamese (eng_Latn-asm_Beng)",
        "training_date": datetime.now().isoformat(),
        "model_path": model_dir,
        "fixes_applied": {
            "tokenizer_deprecation_warning": "Fixed - using text_target parameter",
            "training_parameters": "Optimized for small datasets",
            "batch_size": 2,
            "gradient_accumulation": 4,
            "learning_rate": "3e-5",
            "epochs": 5,
            "dataset_size": len(dataset['train'])
        },
        "dataset_stats": stats,
        "evaluation_results": eval_results
    }
    
    with open("training_info_complete.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    print("📄 Complete training info saved to 'training_info_complete.json'")
    
else:
    print("❌ Model directory not found. Training may have failed.")

# Optional: Upload to Google Drive
print("\\n🔗 Optional: Upload to Google Drive")
try:
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Copy to Drive
    drive_path = "/content/drive/MyDrive/translation_models_fixed/"
    os.makedirs(drive_path, exist_ok=True)
    
    if os.path.exists(f"{archive_name}.zip"):
        shutil.copy(f"{archive_name}.zip", f"{drive_path}{archive_name}.zip")
        shutil.copy("training_info_complete.json", f"{drive_path}training_info_complete.json")
        print(f"✅ OPTIMIZED model uploaded to Google Drive: {drive_path}")
    else:
        print("❌ Model archive not found")
        
except Exception as e:
    print(f"⚠️ Google Drive upload failed: {e}")
    print("💡 You can still download the files manually")
'''

print("📝 Complete Colab notebook content created!")
print("🔧 All fixes applied:")
print("  ✅ Fixed tokenizer deprecation warning")
print("  ✅ Optimized training parameters for small datasets") 
print("  ✅ Multiple dataset fallback options")
print("  ✅ Better GPU memory management")
print("  ✅ Enhanced error handling")
print("\\n📋 To use in Colab:")
print("1. Copy data_preparation_fixed.py and training_fixed.py to Colab")
print("2. Create notebook cells with the content from each cell_ variable")
print("3. Run cells in order")
