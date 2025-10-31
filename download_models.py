"""
Download models from Hugging Face Hub
This script downloads the Pegasus model trained for multi-document summarization
"""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os

def download_pegasus_model():
    """Download Pegasus model from Hugging Face Hub"""
    
    print("📥 Downloading Pegasus model from Hugging Face Hub...")
    print("This may take a few minutes on first run...")
    
    try:
        # Download from Hugging Face Hub
        model_name = "YouR102025mano/pegasus-finetuned"
        
        print(f"📦 Loading model: {model_name}")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create local directory
        os.makedirs('models/pegasus-finetuned', exist_ok=True)
        
        # Save locally for faster future use
        print("💾 Saving model locally...")
        model.save_pretrained('models/pegasus-finetuned')
        tokenizer.save_pretrained('models/pegasus-finetuned')
        
        print("✅ Models downloaded and cached successfully!")
        print(f"📍 Saved to: models/pegasus-finetuned/")
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        print(f"Make sure you have internet connection and the model exists at: YouR102025mano/pegasus-finetuned")

if __name__ == "__main__":
    download_pegasus_model()
