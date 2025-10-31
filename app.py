import streamlit as st
import sys
sys.path.insert(0, 'src')

from train_textrank import TextRankSummarizer
from train_pegasus import PegasusSummarizer

st.set_page_config(page_title="Multi-Document Summarization", layout="wide")

st.title("🚀 Multi-Document Summarization")
st.write("Compare different summarization models on your text")

# Input text
text_input = st.text_area("Enter documents to summarize:", height=200)

if text_input:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("TextRank")
        textrank = TextRankSummarizer()
        summary_tr = textrank.summarize(text_input)
        st.write(summary_tr)
    
    with col2:
        st.subheader("Pegasus")
        pegasus = PegasusSummarizer()
        summary_peg = pegasus.summarize(text_input)
        st.write(summary_peg)
    
    with col3:
        st.subheader("LSTM-Seq2Seq")
        st.write("Comparison metrics from evaluation")

# Show metrics
st.divider()
st.subheader("📊 Model Comparison")
st.json({
    "TextRank": {"ROUGE-1": 43.83, "ROUGE-2": 7.97, "ROUGE-L": 34.13},
    "Transformer": {"ROUGE-1": 47.65, "ROUGE-2": 18.75, "ROUGE-L": 24.95},
    "LSTM-Seq2Seq": {"ROUGE-1": 38.0, "ROUGE-2": 17.0, "ROUGE-L": 32.0}
})
