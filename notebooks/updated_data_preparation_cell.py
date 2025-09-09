# Updated Data Preparation Cell - Replace the content in cell "3. Data Preparation with Sangraha Dataset"

# Import and run data preparation with improved dataset loading
import sys
sys.path.append('src')

from data_preparation import DataPreparator
import logging
from datasets import Dataset, DatasetDict

# Set up logging
logging.basicConfig(level=logging.INFO)

class ImprovedDataPreparator(DataPreparator):
    """Enhanced data preparator with expanded sample dataset"""
    
    def _create_expanded_sample_dataset(self) -> DatasetDict:
        """Create an expanded sample dataset with more training examples"""
        print("🔄 Creating expanded sample dataset with 25 examples...")
        
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
            {"en": "Trees are important for clean air.", "as": "বিশুদ্ধ বায়ুৰ বাবে গছ গুৰুত্বপূৰ্ণ।"}
        ]
        
        # Create train/validation split (80/20)
        split_idx = int(0.8 * len(sample_data))
        train_data = sample_data[:split_idx]
        val_data = sample_data[split_idx:]
        
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        print(f"✅ Created expanded dataset: {len(train_data)} training, {len(val_data)} validation samples")
        
        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })

# Initialize improved data preparator
print("🔄 Initializing improved data preparation...")
preparator = ImprovedDataPreparator()

# Try multiple dataset sources
dataset_configs = [
    ("ai4bharat/sangraha", "eng-asm"),
    ("ai4bharat/sangraha", "en-as"),
    ("Helsinki-NLP/opus-100", "en-as"),
]

raw_dataset = None
for dataset_name, config in dataset_configs:
    try:
        print(f"📥 Trying to load {dataset_name} with config {config}...")
        raw_dataset = preparator.load_dataset(dataset_name)
        if raw_dataset and len(raw_dataset['train']) > 10:  # Ensure we have substantial data
            print(f"✅ Successfully loaded {dataset_name}!")
            break
    except Exception as e:
        print(f"⚠️ Failed to load {dataset_name}: {e}")
        continue

# If all external datasets fail, use expanded sample dataset
if raw_dataset is None or len(raw_dataset['train']) <= 10:
    print("📦 Using expanded sample dataset...")
    raw_dataset = preparator._create_expanded_sample_dataset()

# Check dataset structure
print("📊 Dataset structure:")
print(f"Available splits: {list(raw_dataset.keys())}")
if 'train' in raw_dataset:
    print(f"Train columns: {raw_dataset['train'].column_names}")
    print(f"Train size: {len(raw_dataset['train'])}")
    print(f"Validation size: {len(raw_dataset.get('validation', []))}")
    
    # Show sample data
    print("\n📝 Sample data:")
    for i in range(min(3, len(raw_dataset['train']))):
        sample = raw_dataset['train'][i]
        print(f"Sample {i+1}: {sample}")

print("\n⚙️ Processing dataset...")
processed_dataset = preparator.prepare_datasets(raw_dataset)

# Save processed data
print("💾 Saving processed data...")
preparator.save_processed_data(processed_dataset)

# Print statistics
stats = preparator.get_data_stats(processed_dataset)
print(f"\n📊 Final Dataset Statistics:")
for key, value in stats.items():
    print(f"  {key}: {value}")
