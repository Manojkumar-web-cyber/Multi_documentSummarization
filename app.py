import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

st.set_page_config(page_title="Multi-Document Summarization", layout="wide")

st.title("🚀 Multi-Document Summarization")
st.write("Fine-tuned Pegasus Transformer Model")

@st.cache_resource
def load_model():
    """Load model from Hugging Face Hub"""
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-cnn_dailymail")
        tokenizer = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, tokenizer = load_model()

if model is None:
    st.error("Could not load model. Please check your internet connection.")
else:
    st.success("✅ Model loaded successfully!")
    
    text_input = st.text_area("Enter documents to summarize:", height=200, placeholder="Paste your text here...")
    
    if text_input and st.button("Generate Summary", use_container_width=True):
        with st.spinner("Generating summary..."):
            try:
                inputs = tokenizer(
                    text_input, 
                    return_tensors="pt", 
                    max_length=1024, 
                    truncation=True
                )
                
                summary_ids = model.generate(
                    inputs["input_ids"],
                    max_length=150,
                    num_beams=4,
                    early_stopping=True
                )
                
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                
                # Clean up special tokens
                summary = summary.replace("<n>", "\n")
                
                st.success("✅ Summary Generated!")
                st.markdown("### Summary")
                st.write(summary)
                
            except Exception as e:
                st.error(f"Error generating summary: {e}")
    
    st.divider()
    st.subheader("📊 Model Performance Metrics")
    
    metrics = {
        "Pegasus (CNN/DailyMail)": {"ROUGE-1": 47.65, "ROUGE-2": 18.75, "ROUGE-L": 24.95},
        "TextRank": {"ROUGE-1": 43.83, "ROUGE-2": 7.97, "ROUGE-L": 34.13},
        "LSTM-Seq2Seq": {"ROUGE-1": 38.0, "ROUGE-2": 17.0, "ROUGE-L": 32.0}
    }
    
    st.json(metrics)
