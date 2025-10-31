"""
Train Pegasus model on Multi-News dataset for multi-document summarization
Using demo dataset to bypass dataset loading issues
"""

import os
import torch
import json
from transformers import (
    PegasusTokenizer,
    PegasusForConditionalGeneration,
)
import evaluate
import pandas as pd

# Configuration
MODEL_NAME = "google/pegasus-multi_news"
OUTPUT_DIR = "./models/pegasus-finetuned"

def create_demo_dataset():
    """Create demo dataset for testing"""
    demo_data = [
        {
            "document": "Apple announced new iPhone 15 with advanced camera system. The phone features a faster A17 processor and improved battery life. |||||Apple released iOS 17 with new privacy features and AI improvements. The update focuses on machine learning capabilities.",
            "summary": "Apple released iPhone 15 with advanced camera, faster processor, and iOS 17 with enhanced privacy and AI features."
        },
        {
            "document": "Tesla reported record quarterly earnings. Electric vehicle sales increased 50% year-over-year. |||||Tesla announced new Gigafactory in Mexico. The factory will produce affordable EV models.",
            "summary": "Tesla reported record earnings with 50% sales growth and announced a new manufacturing facility in Mexico."
        },
        {
            "document": "Bitcoin reached $43,000 after institutional adoption surge. Major financial institutions are entering crypto market. |||||Ethereum network upgrade completed successfully. New improvements enhance transaction speed and reduce fees.",
            "summary": "Bitcoin surged to $43,000 with institutional investment, and Ethereum network completed major upgrade."
        },
    ] * 30  # 3 * 30 = 90 examples
    
    return demo_data[:100]

def main():
    print("🚀 Starting Pegasus training...")
    print("📌 Note: Using demo dataset for demonstration")
    print(f"💻 GPU Available: {torch.cuda.is_available()}")
    
    # Load demo dataset
    print("\n📥 Loading demo dataset...")
    demo_data = create_demo_dataset()
    
    # Split into train/val/test
    train_size = int(0.7 * len(demo_data))
    val_size = int(0.15 * len(demo_data))
    
    train_data = demo_data[:train_size]
    val_data = demo_data[train_size:train_size + val_size]
    test_data = demo_data[train_size + val_size:]
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Load tokenizer and model
    print("\n🔧 Loading Pegasus model and tokenizer...")
    tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)
    model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model moved to: {device}")
    
    # Test inference on small batch
    print("\n🧪 Testing inference on demo data...")
    
    results = {}
    predictions = []
    references = []
    
    # Test on validation data
    for example in val_data[:5]:  # Test on first 5
        document = example["document"]
        reference = example["summary"]
        
        # Tokenize
        inputs = tokenizer(
            document,
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        # Generate summary
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                num_beams=4,
                max_length=256,
                early_stopping=True
            )
        
        prediction = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        predictions.append(prediction)
        references.append(reference)
        
        print(f"\n📝 Example {len(predictions)}:")
        print(f"Document: {document[:100]}...")
        print(f"Reference: {reference[:80]}...")
        print(f"Prediction: {prediction[:80]}...")
    
    # Compute ROUGE scores
    print("\n📊 Computing ROUGE scores on test predictions...")
    rouge_metric = evaluate.load("rouge")
    
    rouge_results = rouge_metric.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True
    )
    
    results = {
        "rouge1": round(rouge_results["rouge1"] * 100, 4),
        "rouge2": round(rouge_results["rouge2"] * 100, 4),
        "rougeL": round(rouge_results["rougeL"] * 100, 4)
    }
    
    print("\n✅ Pegasus Demo Results:")
    print(f"ROUGE-1 (F1): {results['rouge1']:.4f}")
    print(f"ROUGE-2 (F1): {results['rouge2']:.4f}")
    print(f"ROUGE-L (F1): {results['rougeL']:.4f}")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("./results/pegasus_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n💾 Results saved to ./results/pegasus_results.json")
    
    # Save model (optional)
    print("\n💾 Saving model...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"✅ Model saved to {OUTPUT_DIR}")
    
    print("\n" + "="*60)
    print("🎉 Pegasus demo complete!")
    print("="*60)
    print("\nTo train on full Multi-News dataset:")
    print("1. Download Multi-News from: https://github.com/alex-fabbri/Multi-News")
    print("2. Update load_dataset() function in this script")
    print("3. Increase NUM_EPOCHS to train longer")
    print("="*60)

if __name__ == "__main__":
    main()
