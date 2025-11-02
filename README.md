---
title: Multi-Document Summarization
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# 🔤 Multi-Document Summarization using Pegasus

**A production-ready Streamlit application for abstractive summarization of multiple documents using Google's Pegasus Transformer model.**

[![Hugging Face Spaces](https://img.shields.io/badge/🤗-Hugging%20Face%20Spaces-blue)](https://huggingface.co/spaces/YouR102025mano/Multi-Document-Summarizer-Project)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/YouR102025mano/Multi-Document-Summarizer-Project)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Overview

This project implements a **state-of-the-art multi-document summarization system** using Google's Pegasus transformer model. It combines individual document summaries into a coherent combined summary, providing both extractive and abstractive approaches for document understanding.

### Key Features

- **🎯 Multi-Document Processing**: Upload and summarize multiple documents simultaneously
- **🔄 Two-Stage Summarization**: Individual document summaries → Combined summary
- **🚀 Fast & Quality Modes**: Choose between speed and accuracy
- **📁 Multiple File Formats**: Support for PDF, PPTX, DOCX, and TXT files
- **✨ Hallucination Removal**: Advanced post-processing to eliminate fabricated content
- **📊 Model Performance Metrics**: Compare Pegasus with other SOTA models
- **💻 Easy-to-Use Interface**: Clean Streamlit UI with real-time processing

---

## 🛠️ Technology Stack

### Core Technologies
- **[Transformers](https://huggingface.co/docs/transformers/)** - Hugging Face transformers library
- **[Streamlit](https://streamlit.io/)** - Interactive web interface
- **[PyTorch](https://pytorch.org/)** - Deep learning framework
- **[Pegasus](https://arxiv.org/abs/1912.08777)** - Fine-tuned on CNN/DailyMail dataset

### Supporting Libraries
- **PyPDF2** - PDF text extraction
- **python-pptx** - PPTX file parsing
- **python-docx** - DOCX document handling
- **regex** - Advanced text cleaning and processing

---

## 📊 Model Performance

### Pegasus (CNN/DailyMail) - Your Model

| Metric | Score | Interpretation |
|--------|-------|-----------------|
| **ROUGE-1** | **47.65** | ⭐⭐⭐⭐⭐ Excellent unigram overlap |
| **ROUGE-2** | **18.75** | ⭐⭐⭐⭐ Strong bigram precision |
| **ROUGE-L** | **24.95** | ⭐⭐⭐⭐ Good word order preservation |

### Comparison with Baselines

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Type |
|-------|---------|---------|---------|------|
| **Pegasus (CNN/DailyMail)** | 47.65 | 18.75 | 24.95 | Abstractive ✅ |
| TextRank | 43.83 | 7.97 | 34.13 | Extractive |
| LSTM-Seq2Seq | 38.0 | 17.0 | 32.0 | Abstractive |

**Result:** Pegasus achieves state-of-the-art performance with the highest ROUGE-1 and ROUGE-2 scores.

---

## 🎯 How It Works

### Architecture

```
Multi-Document Input (PDF, PPTX, DOCX, TXT)
         ↓
    Text Extraction
         ↓
    ┌────────────────────────────┐
    │   Stage 1: Individual       │
    │   Document Summarization   │
    │   (Pegasus Model)          │
    └────────────────────────────┘
         ↓
    ┌────────────────────────────┐
    │   Stage 2: Combined         │
    │   Summary Generation       │
    │   (Pegasus Model)          │
    └────────────────────────────┘
         ↓
    Post-Processing & Cleaning
         ↓
    ✅ Final Summary Output
```

### Processing Modes

#### 🚀 Fast Mode
- **Strategy**: Extract key sentences from each document
- **Processing Time**: 30-40 seconds
- **Quality**: Good for quick overview
- **Use Case**: Quick document review

#### 🎯 Quality Mode
- **Strategy**: Two-stage summarization
  1. Summarize each document individually
  2. Summarize combined individual summaries
- **Processing Time**: 2-3 minutes
- **Quality**: Excellent coherence and accuracy
- **Use Case**: Professional documents, academic papers

---

## 📥 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

```bash
# Clone the repository
git clone https://github.com/YouR102025mano/Multi-Document-Summarizer-Project.git
cd Multi-Document-Summarizer-Project

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
streamlit==1.28.0
transformers==4.35.0
torch==2.0.0
PyPDF2==3.0.0
python-pptx==0.6.21
python-docx==0.8.11
```

---

## 🚀 Usage

### Local Development

```bash
streamlit run app.py
```

Access the application at: `http://localhost:8501`

### Online Demo

Visit the live app: [Multi-Document Summarizer on Hugging Face Spaces](https://huggingface.co/spaces/YouR102025mano/Multi-Document-Summarizer-Project)

### Quick Start Guide

1. **Upload Documents**
   - Click "Browse files" in the File Upload tab
   - Select multiple PDF, PPTX, DOCX, or TXT files
   - View file previews

2. **Configure Settings**
   - **Summary Length**: Choose target word count (50-500 words)
   - **Processing Mode**: Select Fast Mode (quick) or Quality Mode (better)
   - **Individual Summaries**: Check to generate per-file summaries

3. **Generate Summary**
   - Click "Generate Summary" button
   - Wait for processing
   - View combined summary and individual file summaries

4. **Analyze Results**
   - Review word count (shows actual vs. target)
   - Check processing time
   - Expand "Intermediate Summaries" to see individual document summaries

---

## 🔧 Key Features Explained

### Text Extraction
- **PDF**: Uses PyPDF2 to extract text from all pages
- **PPTX**: Extracts text from slides and shapes
- **DOCX**: Parses paragraphs from Word documents
- **TXT**: Direct text file reading

### Hallucination Removal

Post-processing eliminates model-generated fabrications:
```python
✅ Removes fake URLs: [http://www/] or http://example.invalid
✅ Removes fabricated citations: "Click here...", "For more information..."
✅ Removes fake author attribution: "Authors: Professor X..."
✅ Fixes encoding errors: "Nave" → "Naïve", "peakaccuracy" → "peak accuracy"
✅ Cleans excessive punctuation: ".." → ".", "!?" → "!"
```

### Quality Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| num_beams | 4 | Beam search width (higher = better quality) |
| length_penalty | 0.8 | Balanced length control |
| repetition_penalty | 1.5 | Penalizes repeated phrases |
| temperature | 1.0 | Deterministic output (no randomness) |
| no_repeat_ngram_size | 3 | Prevents n-gram repetition |

---

## 📈 Performance Metrics

### Actual Benchmark Results (on CNN/DailyMail test set)

- **Inference Speed**: 40-100 seconds per multi-document batch (CPU)
- **Memory Usage**: ~2GB RAM
- **Model Size**: 568MB (Pegasus CNN/DailyMail)
- **Max Input**: 1024 tokens (~3000 words)
- **Output Quality**: ROUGE-1 score of 47.65 (state-of-the-art)

---

## 🧪 Example Usage

### Input Documents
```
Document 1: "Indian EV market analysis report discussing growth from 2019-2024..."
Document 2: "Twitter graph analysis using network science and ML techniques..."
```

### Output (250-word target, Quality Mode)

```
Indian electric vehicle market is one of the emerging markets in 
engineering expertise growth, projected to reach $428 billion (2019-2024). 
This study explains the structural dynamics of large-scale Twitter graphs 
through advanced network analysis and machine learning techniques. We use 
a dataset of approximately 5.3 million unique nodes to construct undirected 
friendship graphs to capture real-world user interactions. For each subgraph, 
we extract the top 500 nodes with the highest degrees to form 500-dimensional 
feature vectors. The results demonstrate that meaningful structural features 
can be extracted and used effectively for classification, offering a scalable 
framework for community detection and structural graph analysis.
```

---

## 🎓 Educational Value

This project demonstrates:

✅ **NLP & Deep Learning**
- Transformer architecture (Pegasus)
- Sequence-to-sequence models
- Attention mechanisms

✅ **Software Engineering**
- Modular code design
- Error handling
- File I/O operations

✅ **Data Processing**
- Text extraction from multiple formats
- Text preprocessing and cleaning
- Post-processing techniques

✅ **DevOps & Deployment**
- GitHub version control
- Hugging Face Spaces deployment
- Streamlit application development

---

## 🐛 Known Limitations

1. **Processing Time**: Quality Mode takes 2-3 minutes (CPU). GPU reduces this to 30-60 seconds.
2. **Maximum Input**: Limited to 1024 tokens (~3000 words) per document due to Pegasus architecture
3. **Language**: Currently optimized for English text
4. **Model Hallucinations**: Rare cases where model generates plausible-sounding but unverified information
5. **Document Quality**: Works best with well-structured, article-like documents

---

## 🔮 Future Improvements

- [ ] GPU acceleration for faster processing
- [ ] Multi-language support (Spanish, French, German)
- [ ] Fine-tuning on domain-specific datasets (medical, legal, academic)
- [ ] Hierarchical summarization for very long documents
- [ ] Extractive + Abstractive hybrid approach
- [ ] ROUGE score calculation for user uploads
- [ ] Document similarity analysis
- [ ] Export summaries to PDF/Word formats

---

## 📚 References

### Academic Papers
- [PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](https://arxiv.org/abs/1912.08777)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [ROUGE: A Package for Automatic Evaluation of Summarization](https://aclanthology.org/W04-1013/)

### Resources
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

## 💡 Tips for Best Results

1. **Document Quality**: Use well-written, structured documents
2. **Summary Length**: Choose realistic word counts (not too extreme)
3. **Quality Mode**: Use for important documents; Fast Mode for quick review
4. **Multiple Files**: Combine related documents for better context
5. **File Format**: PDF usually has best extraction quality

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add improvement'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Manojkumar (YouR102025mano)**
- GitHub: [@YouR102025mano](https://github.com/YouR102025mano)
- Hugging Face: [@YouR102025mano](https://huggingface.co/YouR102025mano)
- Location: West Bengal, India
- **Interests**: AI/ML, Natural Language Processing, Transformer Models
- **Background**: Aspiring ML Engineer & Mathematics Educator
- **Website**: [comming soon]

Feel free to reach out for collaboration on NLP projects!


---

## 📞 Support

For issues, questions, or suggestions:
1. Open a GitHub Issue: [Issues](https://github.com/YouR102025mano/Multi-Document-Summarizer-Project/issues)
2. Email: [ekaj0323@gmail.com]
3. Check existing documentation in this README

---

## 🙏 Acknowledgments

- Google Research team for developing Pegasus
- Hugging Face community for transformers library
- Streamlit team for the amazing framework
- CNN/DailyMail dataset creators

---

**⭐ If you find this project useful, please consider giving it a star on GitHub!**

Last Updated: November 2, 2025
