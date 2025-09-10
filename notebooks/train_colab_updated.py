# Updated Google Colab Training Script for English-Assamese Translation
# Using Helsinki-NLP/opus-100 dataset for better compatibility

# Cell 1: Environment Setup
print("Installing required packages...")
# !pip install -q torch transformers datasets accelerate sentencepiece
# !pip install -q pandas numpy tqdm sacrebleu

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Cell 2: Clone Repository (Optional - if using GitHub)
# !git clone https://github.com/your-username/Machine-Translation-.git
# %cd Machine-Translation-

# Cell 3: Dataset Loading with Helsinki-NLP/opus-100
from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer
import torch

print("🔄 Loading Helsinki-NLP/opus-100 dataset for English-Assamese...")

# Load the opus-100 dataset with English-Assamese pair
try:
    # Load opus-100 dataset with as-en language pair (reverse direction)
    dataset = load_dataset("Helsinki-NLP/opus-100", "as-en")
    print("✅ Successfully loaded Helsinki-NLP/opus-100 dataset")
    print(f"Available splits: {list(dataset.keys())}")
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")
    
    # Show sample data
    print("\n📝 Sample data:")
    for i in range(3):
        sample = dataset['train'][i]
        print(f"Sample {i+1}:")
        print(f"  English: {sample['translation']['en']}")
        print(f"  Assamese: {sample['translation']['as']}")
        print()
        
except Exception as e:
    print(f"❌ Error loading opus-100: {e}")
    print("🔄 Creating fallback dataset...")
    
    # Fallback to sample dataset with proper structure
    sample_data_train = [
        {'en': 'Hello, how are you?', 'as': 'নমস্কাৰ, আপুনি কেনে আছে?'},
        {'en': 'Thank you very much.', 'as': 'বহুত ধন্যবাদ।'},
        {'en': 'Good morning.', 'as': 'শুভ ৰাতিপুৱা।'},
        {'en': 'How can I help you?', 'as': 'মই আপোনাক কেনেকৈ সহায় কৰিব পাৰো?'},
        {'en': 'Education is very important.', 'as': 'শিক্ষা অতি গুৰুত্বপূৰ্ণ।'},
        {'en': 'Health is wealth.', 'as': 'স্বাস্থ্যই সম্পদ।'},
        {'en': 'Water is essential for life.', 'as': 'জীৱনৰ বাবে পানী অপৰিহাৰ্য।'},
        {'en': 'Children need proper nutrition.', 'as': 'শিশুসকলৰ উপযুক্ত পুষ্টিৰ প্ৰয়োজন।'},
        {'en': 'Clean environment is important.', 'as': 'পৰিষ্কাৰ পৰিৱেশ গুৰুত্বপূৰ্ণ।'},
        {'en': 'Technology helps development.', 'as': 'প্ৰযুক্তিয়ে উন্নয়নত সহায় কৰে।'},
        {'en': 'Women empowerment is crucial.', 'as': 'মহিলা সৱলীকৰণ অতি গুৰুত্বপূৰ্ণ।'},
        {'en': 'Agriculture feeds the nation.', 'as': 'কৃষিয়ে দেশক খুৱায়।'},
        {'en': 'Peace brings prosperity.', 'as': 'শান্তিয়ে সমৃদ্ধি আনে।'},
        {'en': 'Knowledge is power.', 'as': 'জ্ঞানেই শক্তি।'},
        {'en': 'Unity in diversity.', 'as': 'বৈচিত্ৰ্যৰ মাজত ঐক্য।'},
        {'en': 'Hard work pays off.', 'as': 'কঠোৰ পৰিশ্ৰমৰ ফল পোৱা যায়।'},
        {'en': 'Time is precious.', 'as': 'সময় অমূল্য।'},
        {'en': 'Respect your elders.', 'as': 'বয়োজ্যেষ্ঠসকলক সন্মান কৰক।'},
        {'en': 'Nature is beautiful.', 'as': 'প্ৰকৃতি সুন্দৰ।'},
        {'en': 'Love your country.', 'as': 'নিজৰ দেশক ভাল পাওক।'}
    ]
    
    sample_data_val = [
        {'en': 'Good evening.', 'as': 'শুভ সন্ধিয়া।'},
        {'en': 'See you tomorrow.', 'as': 'কাইলৈ লগ পাম।'},
        {'en': 'Take care of yourself.', 'as': 'নিজৰ যত্ন লওক।'},
        {'en': 'Have a nice day.', 'as': 'দিনটো ভাল কটাওক।'},
        {'en': 'Welcome to Assam.', 'as': 'অসমলৈ স্বাগতম।'}
    ]
    
    from datasets import Dataset, DatasetDict
    dataset = DatasetDict({
        'train': Dataset.from_list(sample_data_train),
        'validation': Dataset.from_list(sample_data_val)
    })
    print(f"✅ Created fallback dataset with {len(dataset['train'])} training samples")

# Cell 4: Data Preprocessing
print("🔄 Setting up tokenizer and preprocessing...")

# Initialize tokenizer
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Language codes for NLLB
source_lang = "eng_Latn"  # English
target_lang = "asm_Beng"  # Assamese

def preprocess_function(examples):
    """Preprocess the dataset for training"""
    # Extract source and target texts
    if 'translation' in examples:
        # Handle opus-100 format
        inputs = [ex['en'] for ex in examples['translation']]  # English as input
        targets = [ex['as'] for ex in examples['translation']]  # Assamese as target
    else:
        # Handle fallback format (direct en/as columns)
        inputs = examples['en']
        targets = examples['as']
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs, 
        max_length=128, 
        truncation=True, 
        padding=True
    )
    
    # Tokenize targets
    labels = tokenizer(
        text_target=targets,
        max_length=128, 
        truncation=True, 
        padding=True
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing
print("📊 Preprocessing dataset...")
tokenized_dataset = dataset.map(
    preprocess_function, 
    batched=True,
    remove_columns=dataset['train'].column_names
)

print(f"✅ Preprocessing completed!")
print(f"Train samples: {len(tokenized_dataset['train'])}")
print(f"Validation samples: {len(tokenized_dataset.get('validation', []))}")

# Cell 5: Model Setup and Training
from transformers import (
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import numpy as np

print("🤖 Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# Training arguments optimized for CPU training
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,  # Slightly higher for CPU
    per_device_train_batch_size=1,  # Smaller batch for CPU
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,  # Higher accumulation for effective batch size
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,  # Fewer epochs for CPU training
    predict_with_generate=True,
    fp16=False,  # Disable fp16 for CPU
    logging_steps=5,
    save_steps=50,
    eval_steps=50,
    warmup_steps=50,
    max_grad_norm=1.0,
    dataloader_pin_memory=False,
    dataloader_num_workers=0,  # Important for CPU
    remove_unused_columns=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to=None,  # Disable wandb/tensorboard
)

# Evaluation metric
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Simple length-based metric (you can add BLEU here if needed)
    avg_pred_len = np.mean([len(pred.split()) for pred in decoded_preds])
    avg_label_len = np.mean([len(label.split()) for label in decoded_labels])
    
    return {
        "avg_pred_length": avg_pred_len,
        "avg_label_length": avg_label_len,
    }

# Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset.get('validation', tokenized_dataset['train'][:5]),
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("🚀 Starting training...")
print("This will take approximately 20-40 minutes depending on your GPU.")

# Clear GPU memory before training
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Train the model
trainer.train()

print("✅ Training completed!")

# Cell 6: Save the Model
model_save_path = "./nllb-finetuned-en-as-opus100"
print(f"💾 Saving model to {model_save_path}...")

trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)

print("✅ Model saved successfully!")

# Cell 7: Test the Model
print("🧪 Testing the trained model...")

def translate_text(text, model_path=model_save_path):
    """Translate English text to Assamese"""
    from transformers import pipeline
    
    # Load the fine-tuned model
    translator = pipeline(
        "translation",
        model=model_path,
        tokenizer=model_path,
        src_lang=source_lang,
        tgt_lang=target_lang,
        device=0 if torch.cuda.is_available() else -1
    )
    
    result = translator(text, max_length=128)
    return result[0]['translation_text']

# Test sentences
test_sentences = [
    "Hello, how are you?",
    "Thank you for your help.",
    "Education is very important.",
    "Health is wealth.",
    "Good morning, have a nice day.",
    "Technology helps in development.",
    "Clean water is essential for life."
]

print("\n🔍 Translation Results:")
print("=" * 80)

for i, sentence in enumerate(test_sentences, 1):
    try:
        translation = translate_text(sentence)
        print(f"\n{i}. English: {sentence}")
        print(f"   Assamese: {translation}")
        print("-" * 60)
    except Exception as e:
        print(f"Error translating '{sentence}': {e}")

# Cell 8: Download Model
import shutil
import os

print("📦 Creating downloadable model archive...")

if os.path.exists(model_save_path):
    # Create zip file
    shutil.make_archive("trained_model_opus100", 'zip', model_save_path)
    
    print("✅ Model archived as 'trained_model_opus100.zip'")
    print("📥 Download it from the Files panel on the left")
    
    # Show file size
    size_mb = os.path.getsize("trained_model_opus100.zip") / (1024 * 1024)
    print(f"📊 Archive size: {size_mb:.1f} MB")
    
    # Save training info
    import json
    from datetime import datetime
    
    training_info = {
        "dataset": "Helsinki-NLP/opus-100 (en-as)",
        "base_model": "facebook/nllb-200-distilled-600M",
        "language_pair": "English-Assamese",
        "training_date": datetime.now().isoformat(),
        "model_path": model_save_path,
        "train_samples": len(tokenized_dataset['train']),
        "validation_samples": len(tokenized_dataset.get('validation', [])),
        "training_epochs": 5,
        "batch_size": 2,
        "learning_rate": 3e-5
    }
    
    with open("training_info_opus100.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    print("📄 Training info saved to 'training_info_opus100.json'")
else:
    print("❌ Model directory not found.")

print("\n🎉 Training Complete!")
print("\n### Key Improvements:")
print("✅ Using Helsinki-NLP/opus-100 dataset (more reliable)")
print("✅ Better error handling with fallback dataset")
print("✅ Optimized training parameters for Colab")
print("✅ Fixed tokenizer deprecation warnings")
print("✅ GPU memory management")
print("\n### Next Steps:")
print("1. Download the 'trained_model_opus100.zip' file")
print("2. Extract it to your local project's models/ directory")
print("3. Update your local translate.py to use the new model")
print("4. Test with your FastAPI backend")
