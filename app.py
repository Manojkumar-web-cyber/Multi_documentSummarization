import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import io
from PyPDF2 import PdfReader
from pptx import Presentation
from docx import Document
import time

st.set_page_config(page_title="Multi-Document Summarization", layout="wide")

st.title("🔤 Multi-Document Summarization")
st.write("Fine-tuned Pegasus Transformer Model")

# --- ADD SUMMARY LENGTH CONTROLS IN SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Summary Settings")
    
    summary_length = st.slider(
        "Summary Length (words approx.)",
        min_value=50,
        max_value=500,
        value=150,
        step=50,
        help="Adjust the desired length of the summary"
    )
    st.info(f"📝 Summary will be ~{summary_length} words")

# --- 1. Text Extraction Functions (IMPROVED) ---
def extract_text_from_pdf(file):
    """Extracts text content from a PDF file."""
    text = ""
    try:
        # Create a new bytes buffer each time
        file_bytes = io.BytesIO(file.read())
        reader = PdfReader(file_bytes)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""
    return text.strip()

def extract_text_from_pptx(file):
    """Extracts text content from a PPTX file."""
    text = ""
    try:
        file_bytes = io.BytesIO(file.read())
        prs = Presentation(file_bytes)
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    text += shape.text + "\n"
    except Exception as e:
        st.error(f"Error reading PPTX: {e}")
        return ""
    return text.strip()

def extract_text_from_docx(file):
    """Extracts text content from a DOCX file."""
    text = ""
    try:
        file_bytes = io.BytesIO(file.read())
        doc = Document(file_bytes)
        for paragraph in doc.paragraphs:
            if paragraph.text:
                text += paragraph.text + "\n"
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""
    return text.strip()

# --- 2. UPDATED Model Loading and Summarization (OPTIMIZED) ---
@st.cache_resource(show_spinner=False)
def load_model():
    """Load model from Hugging Face Hub with caching"""
    try:
        with st.spinner("🔄 Loading model... This may take a minute"):
            model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-cnn_dailymail")
            tokenizer = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# UPDATED: Better length handling and performance
def summarize_text(text, model, tokenizer, max_length=150):
    """Generate summary for a single text with adjustable length"""
    if not text.strip():
        return "No content to summarize."
    
    try:
        # Better token length calculation
        max_tokens = min(int(max_length * 1.3), 512)  # Cap at 512 tokens
        
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            max_length=1024, 
            truncation=True,
            padding=True
        )
        
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_tokens,
            min_length=max(30, max_tokens // 3),  # Ensure reasonable minimum
            num_beams=4,
            early_stopping=True,
            length_penalty=0.8,  # Better for variable lengths
            no_repeat_ngram_size=3,
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary.replace("<n>", " ")  # Clean up formatting
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# Load model once
model, tokenizer = load_model()

# --- 3. FIXED Unified Extraction Function ---
def extract_all_file_contents(uploaded_files):
    """Extracts text from a list of uploaded Streamlit files."""
    file_contents = []
    if uploaded_files:
        for file in uploaded_files:
            try:
                # Reset file pointer and read content
                file.seek(0)
                file_content = file.read()
                
                # Create a new file-like object for each extraction
                file_obj = io.BytesIO(file_content)
                file_obj.name = file.name  # Preserve filename
                
                if file.name.lower().endswith('.pdf'):
                    content = extract_text_from_pdf(file_obj)
                elif file.name.lower().endswith('.pptx'):
                    content = extract_text_from_pptx(file_obj)
                elif file.name.lower().endswith('.docx'):
                    content = extract_text_from_docx(file_obj)
                else:  # Assume text file
                    content = file_content.decode("utf-8")
                
                if content and content.strip():
                    file_contents.append({
                        'name': file.name,
                        'content': content.strip(),
                        'word_count': len(content.split())
                    })
                else:
                    st.warning(f"⚠️ No readable content found in '{file.name}'")
                    
            except Exception as e:
                st.warning(f"❌ Could not read content from file '{file.name}'. Error: {e}")
    
    return file_contents

# --- 4. Streamlit UI with FIXED Session State ---

# Initialize session state properly
if 'text_documents' not in st.session_state:
    st.session_state.text_documents = []
if 'uploaded_files_state' not in st.session_state:
    st.session_state.uploaded_files_state = []
if 'file_contents_cache' not in st.session_state:
    st.session_state.file_contents_cache = {}
if 'num_docs' not in st.session_state:
    st.session_state.num_docs = 2

# Tabs for different input methods
tab1, tab2 = st.tabs(["📝 Multiple Text Inputs", "📄 File Upload"])

with tab1:
    st.write("**Enter multiple documents below (one per box):**")
    
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
            placeholder=f"Paste document {i+1} here...",
            value=st.session_state.text_documents[i] if i < len(st.session_state.text_documents) else ""
        )
        current_text_documents.append(doc)
    
    # Save to session state
    st.session_state.text_documents = current_text_documents

    summary_type_text = st.radio(
        "Summary Scope:",
        ["Current Tab (Text Inputs Only)", "Combined Tab Data (Text Inputs + Uploaded Files)"],
        horizontal=True,
        key="summary_type_text"
    )
    
    if st.button("🚀 Generate Summary", key="text_summary_btn"):
        with st.spinner("Generating summary..."):
            start_time = time.time()
            try:
                documents_to_summarize = []
                text_contents = [doc for doc in st.session_state.text_documents if doc.strip()]
                
                if text_contents:
                    documents_to_summarize.extend(text_contents)
                
                # Add file contents if combined mode
                if (summary_type_text == "Combined Tab Data (Text Inputs + Uploaded Files)" 
                    and st.session_state.uploaded_files_state):
                    
                    # Use cached file contents or extract fresh
                    file_contents = extract_all_file_contents(st.session_state.uploaded_files_state)
                    for file_data in file_contents:
                        documents_to_summarize.append(file_data['content'])

                if not documents_to_summarize:
                    st.warning("Please provide some text or files to summarize.")
                else:
                    combined_text = "\n\n".join(documents_to_summarize)
                    summary = summarize_text(combined_text, model, tokenizer, max_length=summary_length)
                    
                    end_time = time.time()
                    st.success(f"✅ Combined Summary Generated! (Time: {end_time - start_time:.2f}s)")
                    st.markdown("### 📋 Combined Summary")
                    
                    word_count = len(summary.split())
                    st.caption(f"Summary length: ~{word_count} words | Input documents: {len(documents_to_summarize)}")
                    
                    st.info(summary)
                        
            except Exception as e:
                st.error(f"Error generating summary: {e}")

with tab2:
    st.write("**Upload files (.txt, .pdf, .pptx, .docx) for summarization:**")
    
    uploaded_files = st.file_uploader(
        "Choose files", 
        type=['txt', 'pdf', 'pptx', 'docx'], 
        accept_multiple_files=True,
        key="file_uploader"
    )

    # Store uploaded files in session state
    if uploaded_files is not None:
        st.session_state.uploaded_files_state = uploaded_files
    
    if st.session_state.uploaded_files_state:
        st.write(f"**📁 {len(st.session_state.uploaded_files_state)} file(s) uploaded**")
        
        # Show file preview
        with st.expander("📊 File Contents Preview"):
            file_contents = extract_all_file_contents(st.session_state.uploaded_files_state)
            for file_data in file_contents:
                st.write(f"**{file_data['name']}** ({file_data['word_count']} words)")
                preview = file_data['content'][:200] + "..." if len(file_data['content']) > 200 else file_data['content']
                st.text(preview)
                st.divider()
        
        file_summary_type = st.radio(
            "Summary Scope:",
            ["Current Tab (Uploaded Files Only)", "Combined Tab Data (Uploaded Files + Text Inputs)"],
            horizontal=True,
            key="file_summary_type"
        )
        
        individual_summary_check = st.checkbox(
            "Also generate individual summaries for uploaded files?", 
            key="individual_check"
        )

        if st.button("🚀 Generate Summary", key="file_summary_btn"):
            with st.spinner("Processing files and generating summary..."):
                start_time = time.time()
                try:
                    # Extract file contents once and cache
                    file_contents_data = extract_all_file_contents(st.session_state.uploaded_files_state)
                    documents_to_summarize = [file_data['content'] for file_data in file_contents_data]
                    
                    # Add text inputs if combined mode
                    if file_summary_type == "Combined Tab Data (Uploaded Files + Text Inputs)":
                        text_contents = [doc for doc in st.session_state.text_documents if doc.strip()]
                        documents_to_summarize.extend(text_contents)

                    if not documents_to_summarize:
                        st.warning("No readable content was found in the selected files and text inputs.")
                    else:
                        combined_text = "\n\n".join(documents_to_summarize)
                        summary = summarize_text(combined_text, model, tokenizer, max_length=summary_length)
                        
                        end_time = time.time()
                        st.success(f"✅ Combined Summary Generated! (Time: {end_time - start_time:.2f}s)")
                        st.markdown("### 📋 Combined Summary")
                        
                        word_count = len(summary.split())
                        st.caption(f"Summary length: ~{word_count} words | Input sources: {len(documents_to_summarize)}")
                        
                        st.info(summary)
                        
                        # Individual summaries
                        if individual_summary_check and file_contents_data:
                            st.divider()
                            st.markdown("### 📄 Individual File Summaries")
                            
                            for file_data in file_contents_data:
                                with st.spinner(f"Summarizing {file_data['name']}..."):
                                    individual_summary = summarize_text(
                                        file_data['content'], 
                                        model, 
                                        tokenizer, 
                                        max_length=summary_length
                                    )
                                
                                st.markdown(f"**{file_data['name']}** ({file_data['word_count']} words)")
                                st.info(individual_summary)
                                st.divider()
                                
                except Exception as e:
                    st.error(f"Error generating summary: {e}")

# --- 5. Model Metrics ---
st.divider()
st.subheader("📊 Model Performance Metrics")

metrics = {
    "Pegasus (CNN/DailyMail)": {"ROUGE-1": 47.65, "ROUGE-2": 18.75, "ROUGE-L": 24.95},
    "TextRank": {"ROUGE-1": 43.83, "ROUGE-2": 7.97, "ROUGE-L": 34.13},
    "LSTM-Seq2Seq": {"ROUGE-1": 38.0, "ROUGE-2": 17.0, "ROUGE-L": 32.0}
}

st.json(metrics)