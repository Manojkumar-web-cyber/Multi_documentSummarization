import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import io
from PyPDF2 import PdfReader
from pptx import Presentation
from docx import Document

st.set_page_config(page_title="Multi-Document Summarization", layout="wide")

st.title("🚀 Multi-Document Summarization")
st.write("Fine-tuned Pegasus Transformer Model")

# --- 1. Text Extraction Functions (Remains the same) ---

def extract_text_from_pdf(file):
    """Extracts text content from a PDF file."""
    text = ""
    try:
        reader = PdfReader(io.BytesIO(file.read()))
        for page in reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""
    return text

def extract_text_from_pptx(file):
    """Extracts text content from a PPTX file."""
    text = ""
    try:
        prs = Presentation(io.BytesIO(file.read()))
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
    except Exception as e:
        st.error(f"Error reading PPTX: {e}")
        return ""
    return text

def extract_text_from_docx(file):
    """Extracts text content from a DOCX file."""
    text = ""
    try:
        doc = Document(io.BytesIO(file.read()))
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""
    return text

# --- 2. Model Loading and Summarization (Remains the same) ---

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
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary.replace("<n>", "\n")

model, tokenizer = load_model()

if model is None:
    st.error("Could not load model. Please check your internet connection.")
    st.stop()
else:
    # Use a check to prevent success message from flashing on every rerun
    if 'model_loaded_success' not in st.session_state:
        st.success("✅ Model loaded successfully!")
        st.session_state.model_loaded_success = True

# --- 3. New Unified Extraction Function ---

def extract_all_file_contents(uploaded_files):
    """Extracts text from a list of uploaded Streamlit files."""
    file_contents = []
    if uploaded_files:
        for file in uploaded_files:
            try:
                # Need to reset pointer for multi-file processing
                file.seek(0)
                
                if file.name.endswith('.pdf'):
                    content = extract_text_from_pdf(file)
                elif file.name.endswith('.pptx'):
                    content = extract_text_from_pptx(file)
                elif file.name.endswith('.docx'):
                    content = extract_text_from_docx(file)
                else: # Assume .txt or other raw text file
                    content = file.read().decode("utf-8")
                
                if content.strip():
                    file_contents.append(content)
            except Exception as e:
                st.warning(f"Could not read content from file '{file.name}'. Error: {e}")
    return file_contents


# --- 4. Streamlit UI with Session State for Inputs ---

# Initialize session state for text input and file input
if 'text_documents' not in st.session_state:
    st.session_state.text_documents = []
if 'uploaded_files_state' not in st.session_state:
    st.session_state.uploaded_files_state = None


# Tabs for different input methods
tab1, tab2 = st.tabs(["📝 Multiple Text Inputs", "📄 File Upload"])

with tab1:
    st.write("**Enter multiple documents below (one per box):**")

    # Initialize session state for number of text boxes
    if 'num_docs' not in st.session_state:
        st.session_state.num_docs = 2
    
    # Add/Remove document buttons
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("➕ Add Document", key="add_doc_text"):
            st.session_state.num_docs += 1
    with col2:
        if st.button("➖ Remove Document", key="remove_doc_text") and st.session_state.num_docs > 1:
            st.session_state.num_docs -= 1
    
    # Capture multiple text inputs
    current_text_documents = []
    for i in range(st.session_state.num_docs):
        doc = st.text_area(
            f"Document {i+1}:", 
            height=150, 
            key=f"doc_input_{i}",
            placeholder=f"Paste document {i+1} here..."
        )
        if doc.strip():
            current_text_documents.append(doc)
    
    # Save text documents to session state for cross-tab access
    st.session_state.text_documents = current_text_documents

    # Summary type selection for this tab
    summary_type_text = st.radio(
        "Summary Scope:",
        ["Current Tab (Text Inputs Only)", "Combined Tab Data (Text Inputs + Uploaded Files)"],
        horizontal=True,
        key="summary_type_text"
    )
    
    if st.session_state.text_documents and st.button("🚀 Generate Summary", key="text_summary_btn"):
        with st.spinner("Generating summary..."):
            try:
                
                # Determine which documents to summarize
                documents_to_summarize = st.session_state.text_documents
                
                if summary_type_text == "Combined Tab Data (Text Inputs + Uploaded Files)" and st.session_state.uploaded_files_state:
                    # Get file contents from session state
                    file_contents = extract_all_file_contents(st.session_state.uploaded_files_state)
                    documents_to_summarize.extend(file_contents)

                if not documents_to_summarize:
                    st.warning("Please provide some text or files to summarize.")
                else:
                    # Combine all documents
                    combined_text = "\n\n".join(documents_to_summarize)
                    summary = summarize_text(combined_text, model, tokenizer)
                    
                    st.success("✅ Combined Summary Generated!")
                    st.markdown("### 📋 Combined Summary")
                    st.info(summary)
                        
            except Exception as e:
                st.error(f"Error generating summary: {e}")

with tab2:
    st.write("**Upload files (.txt, .pdf, .pptx, .docx) for summarization:**")
    
    # File Uploader
    uploaded_files = st.file_uploader(
        "Choose files", 
        type=['txt', 'pdf', 'pptx', 'docx'], 
        accept_multiple_files=True
    )

    # Store uploaded files in session state immediately (uploader returns list/None on every run)
    st.session_state.uploaded_files_state = uploaded_files
    
    if uploaded_files:
        st.write(f"**{len(uploaded_files)} file(s) uploaded**")
        
        file_summary_type = st.radio(
            "Summary Scope:",
            ["Current Tab (Uploaded Files Only)", "Combined Tab Data (Uploaded Files + Text Inputs)"],
            horizontal=True,
            key="file_summary_type"
        )
        
        individual_summary_check = st.checkbox("Also generate individual summaries for uploaded files?", key="individual_check")

        if st.button("🚀 Generate Summary", key="file_summary_btn"):
            with st.spinner("Processing files and generating summary..."):
                try:
                    
                    # Start with file contents
                    documents_to_summarize = extract_all_file_contents(uploaded_files)
                    
                    # Check if we should add text inputs
                    if file_summary_type == "Combined Tab Data (Uploaded Files + Text Inputs)":
                        documents_to_summarize.extend(st.session_state.text_documents)

                    if not documents_to_summarize:
                        st.warning("No readable content was found in the selected files and text inputs.")
                    else:
                        # Generate Combined Summary
                        combined_text = "\n\n".join(documents_to_summarize)
                        summary = summarize_text(combined_text, model, tokenizer)
                        
                        st.success("✅ Combined Summary Generated!")
                        st.markdown("### 📋 Combined Summary")
                        st.info(summary)
                        
                        # Generate Individual Summaries if requested
                        if individual_summary_check:
                            st.divider()
                            st.markdown("### 📄 Individual File Summaries")
                            for file, content in zip(uploaded_files, extract_all_file_contents(uploaded_files)):
                                individual_summary = summarize_text(content, model, tokenizer)
                                st.markdown(f"**{file.name}**")
                                st.info(individual_summary)
                                st.divider()
                                
                except Exception as e:
                    st.error(f"Error generating summary: {e}")

# --- 5. Model Metrics (Remains the same) ---
st.divider()
st.subheader("📊 Model Performance Metrics")

metrics = {
    "Pegasus (CNN/DailyMail)": {"ROUGE-1": 47.65, "ROUGE-2": 18.75, "ROUGE-L": 24.95},
    "TextRank": {"ROUGE-1": 43.83, "ROUGE-2": 7.97, "ROUGE-L": 34.13},
    "LSTM-Seq2Seq": {"ROUGE-1": 38.0, "ROUGE-2": 17.0, "ROUGE-L": 32.0}
}

st.json(metrics)
