import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import PyPDF2  # For reading PDFs
from pptx import Presentation  # For reading PowerPoint files
from docx import Document  # For reading Word docs
import io

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

def summarize_text(text, model, tokenizer):
    """Generate summary for a single text"""
    # Pre-process text to handle potential empty inputs or very short text
    if not text or len(text.strip()) < 50: # Added a minimum length check
        return "Not enough content to summarize."
        
    try:
        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary.replace("<n>", "\n")
    except Exception as e:
        return f"Error during summarization: {e}"

# --- Text Extraction Functions ---

def extract_text_from_pdf(file_like_object):
    """Extracts text from a PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(file_like_object)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def extract_text_from_pptx(file_like_object):
    """Extracts text from a PowerPoint (pptx) file."""
    try:
        prs = Presentation(file_like_object)
        text = ""
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PPTX: {e}")
        return ""

def extract_text_from_docx(file_like_object):
    """Extracts text from a Word (docx) file."""
    try:
        doc = Document(file_like_object)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""

# --- Main App Logic ---

model, tokenizer = load_model()

if model is None:
    st.error("Could not load model. Please check your internet connection or model name.")
else:
    st.success("✅ Model loaded successfully!")
    
    # Tabs for different input methods
    tab1, tab2 = st.tabs(["📝 Multiple Text Inputs", "📄 File Upload"])
    
    with tab1:
        st.write("**Enter multiple documents below (one per box):**")
        
        # Initialize session state for documents
        if 'num_docs' not in st.session_state:
            st.session_state.num_docs = 2
        
        # Add/Remove document buttons
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("➕ Add Document"):
                st.session_state.num_docs += 1
        with col2:
            if st.button("➖ Remove Document") and st.session_state.num_docs > 1:
                st.session_state.num_docs -= 1
        
        # Multiple text inputs
        documents = []
        for i in range(st.session_state.num_docs):
            doc = st.text_area(
                f"Document {i+1}:", 
                height=150, 
                key=f"doc_{i}",
                placeholder=f"Paste document {i+1} here..."
            )
            if doc.strip():
                documents.append(doc)
        
        # Summary type selection
        summary_type = st.radio(
            "Summary Type:",
            ["Combined Summary", "Individual Summaries"],
            horizontal=True,
            key="text_summary_type"
        )
        
        if documents and st.button("🚀 Generate Summary", key="text_summary"):
            with st.spinner("Generating summary..."):
                try:
                    if summary_type == "Combined Summary":
                        # Combine all documents
                        combined_text = "\n\n--- End of Document ---\n\n".join(documents)
                        summary = summarize_text(combined_text, model, tokenizer)
                        
                        st.success("✅ Summary Generated!")
                        st.markdown("### 📋 Combined Summary")
                        st.info(summary)
                        
                    else:  # Individual Summaries
                        st.success("✅ Summaries Generated!")
                        for idx, doc in enumerate(documents, 1):
                            summary = summarize_text(doc, model, tokenizer)
                            st.markdown(f"### 📄 Document {idx} Summary")
                            st.info(summary)
                            st.divider()
                            
                except Exception as e:
                    st.error(f"Error generating summary: {e}")
    
    with tab2:
        st.write("**Upload files (.txt, .pdf, .pptx, .docx) for summarization:**")
        
        uploaded_files = st.file_uploader(
            "Choose files", 
            # *** ADDED 'docx' HERE ***
            type=['txt', 'pdf', 'pptx', 'docx'], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.write(f"**{len(uploaded_files)} file(s) uploaded**")
            
            file_summary_type = st.radio(
                "Summary Type:",
                ["Combined Summary", "Individual Summaries"],
                horizontal=True,
                key="file_summary_type"
            )
            
            if st.button("🚀 Generate Summary", key="file_summary"):
                with st.spinner("Processing files and generating summary..."):
                    try:
                        file_contents = []
                        file_names = []
                        
                        for file in uploaded_files:
                            content = ""
                            # Read file based on its type
                            if file.type == "text/plain":
                                content = file.read().decode("utf-8")
                            elif file.type == "application/pdf":
                                content = extract_text_from_pdf(file)
                            elif file.type == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
                                content = extract_text_from_pptx(file)
                            # *** ADDED THIS BLOCK FOR DOCX ***
                            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                                content = extract_text_from_docx(file)
                            
                            if content.strip():
                                file_contents.append(content)
                                file_names.append(file.name)
                            else:
                                st.warning(f"Could not extract text from {file.name} or file is empty.")
                        
                        if not file_contents:
                            st.error("No text could be extracted from the uploaded files.")
                        else:
                            if file_summary_type == "Combined Summary":
                                combined_text = "\n\n--- End of Document ---\n\n".join(file_contents)
                                summary = summarize_text(combined_text, model, tokenizer)
                                
                                st.success("✅ Summary Generated!")
                                st.markdown("### 📋 Combined Summary")
                                st.info(summary)
                            else:
                                st.success("✅ Summaries Generated!")
                                for idx, (name, content) in enumerate(zip(file_names, file_contents), 1):
                                    summary = summarize_text(content, model, tokenizer)
                                    st.markdown(f"### 📄 {name}")
                                    st.info(summary)
                                    st.divider()
                                    
                    except Exception as e:
                        st.error(f"Error processing files: {e}")
    
    # Model metrics
    st.divider()
    st.subheader("📊 Model Performance Metrics")
    
    metrics = {
        "Pegasus (CNN/DailyMail)": {"ROUGE-1": 47.65, "ROUGE-2": 18.75, "ROUGE-L": 24.95},
        "TextRank": {"ROUGE-1": 43.83, "ROUGE-2": 7.97, "ROUGE-L": 34.13},
        "LSTM-Seq2Seq": {"ROUGE-1": 38.0, "ROUGE-2": 17.0, "ROUGE-L": 32.0}
    }
    
    st.json(metrics)
