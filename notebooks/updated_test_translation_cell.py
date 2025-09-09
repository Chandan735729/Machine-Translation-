# Updated Test Translation Cell - Replace the content in cell "7. Test Translation"

# Enhanced testing with more comprehensive examples
from translate import EnglishToAssameseTranslator

# Initialize translator with trained model
print("🔄 Loading enhanced trained model for testing...")
translator = EnglishToAssameseTranslator(model_path)

# Expanded test sentences covering different domains
test_sentences = [
    # Health & Medical (from training data)
    "Community health workers are the backbone of our medical system.",
    "Clean water is essential for good health.",
    "Vaccination protects children from diseases.",
    
    # Education & Development (from training data)
    "Education is the key to development.",
    "Knowledge is power.",
    "Every child has the right to education.",
    
    # Daily Conversations (from training data)
    "Hello, how are you?",
    "Thank you for your help.",
    "What is your name?",
    
    # New test sentences (not in training data)
    "The sun is shining brightly today.",
    "I love reading books in the evening.",
    "Technology makes life easier.",
    "Family is very important to me.",
    "Learning new languages is fun.",
    "Good morning, have a nice day.",
    "The food tastes delicious.",
    "Children are playing in the garden."
]

print("\n🧪 Testing translations with enhanced trained model:")
print("=" * 80)
print("📝 Note: First 9 sentences were in training data, rest are new")
print("=" * 80)

for i, sentence in enumerate(test_sentences, 1):
    print(f"\n{i}. English: {sentence}")
    try:
        translation = translator.translate(sentence)
        print(f"   Assamese: {translation}")
        
        # Indicate if this was in training data
        if i <= 9:
            print("   📚 (Training data - should be well translated)")
        else:
            print("   🆕 (New sentence - tests generalization)")
            
    except Exception as e:
        print(f"   ❌ Translation failed: {e}")
    
    print("-" * 60)

print("\n✅ Enhanced translation testing completed!")
print("\n📊 Translation Quality Assessment:")
print("  🎯 Training data translations should be very accurate")
print("  🔍 New sentence translations test model generalization")
print("  📈 Compare quality between trained vs new sentences")
print("\n💡 Tips for better results:")
print("  • Use more training data for production models")
print("  • Fine-tune on domain-specific data")
print("  • Consider data augmentation techniques")
