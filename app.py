import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import io
from PyPDF2 import PdfReader
from pptx import Presentation
from docx import Document
import time
import hashlib

st.set_page_config(page_title="Multi-Document Summarization", layout="wide")

st.title("🔤 Multi-Document Summarization")
st.write("Fine-tuned Pegasus Transformer Model")

# --- SUMMARY LENGTH CONTROLS ---
with st.sidebar:
    st.header("⚙️ Summary Settings")
    summary_length = st.slider(
        "Summary Length (words approx.)",
        min_value=50, max_value=500, value=150, step=50,
        help="Adjust the desired length of the summary"
    )
    st.info(f"📝 Summary will be ~{summary_length} words")

# --- TEXT EXTRACTION FUNCTIONS ---
def extract_text_from_pdf(file):
    """Extracts text content from a PDF file."""
    text = ""
    try:
        file_bytes = io.BytesIO(file.read())
        reader = PdfReader(file_bytes)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
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
    return text.strip()

# --- MODEL LOADING ---
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

def summarize_text(text, model, tokenizer, max_length=150):
    """Generate summary for a single text with adjustable length"""
    if not text.strip():
        return "No content to summarize."
    
    try:
        max_tokens = min(int(max_length * 1.3), 512)
        
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
            min_length=max(30, max_tokens // 3),
            num_beams=4,
            early_stopping=True,
            length_penalty=0.8,
            no_repeat_ngram_size=3,
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary.replace("<n>", " ")
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# Load model
model, tokenizer = load_model()

# --- FIXED FILE PROCESSING ---
def get_file_content(file):
    """Extract content from a single file with proper handling"""
    try:
        # Reset and read file content
        file.seek(0)
        file_content = file.read()
        
        # Create new file-like object
        file_obj = io.BytesIO(file_content)
        file_obj.name = file.name
        
        # Extract based on file type
        if file.name.lower().endswith('.pdf'):
            content = extract_text_from_pdf(file_obj)
        elif file.name.lower().endswith('.pptx'):
            content = extract_text_from_pptx(file_obj)
        elif file.name.lower().endswith('.docx'):
            content = extract_text_from_docx(file_obj)
        else:
            content = file_content.decode("utf-8")
        
        return content.strip() if content else ""
        
    except Exception as e:
        st.warning(f"❌ Error processing {file.name}: {e}")
        return ""

def get_all_contents(uploaded_files, text_documents):
    """Get all contents from files and text inputs"""
    all_contents = []
    
    # Process uploaded files
    if uploaded_files:
        for file in uploaded_files:
            content = get_file_content(file)
            if content:
                all_contents.append({
                    'name': file.name,
                    'content': content,
                    'type': 'file'
                })
    
    # Process text documents
    for i, text in enumerate(text_documents):
        if text and text.strip():
            all_contents.append({
                'name': f'Text Document {i+1}',
                'content': text.strip(),
                'type': 'text'
            })
    
    return all_contents

# --- STREAMLIT UI ---
if 'text_documents' not in st.session_state:
    st.session_state.text_documents = [""]  # Start with one empty document
if 'num_docs' not in st.session_state:
    st.session_state.num_docs = 1

# Tabs for different input methods
tab1, tab2 = st.tabs(["📝 Multiple Text Inputs", "📄 File Upload"])

with tab1:
    st.write("**Enter multiple documents below (one per box):**")
    
    # Add/Remove document buttons
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("➕ Add Document", key="add_doc_text"):
            st.session_state.text_documents.append("")
            st.session_state.num_docs += 1
    with col2:
        if st.button("➖ Remove Document", key="remove_doc_text") and st.session_state.num_docs > 1:
            st.session_state.text_documents.pop()
            st.session_state.num_docs -= 1
    
    # Text inputs
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
                # Get all contents based on selection
                if summary_type_text == "Current Tab (Text Inputs Only)":
                    contents = get_all_contents([], st.session_state.text_documents)
                else:
                    contents = get_all_contents(
                        st.session_state.get('uploaded_files', []), 
                        st.session_state.text_documents
                    )
                
                if not contents:
                    st.warning("Please provide some text or files to summarize.")
                else:
                    # Combine all contents for summary
                    combined_text = "\n\n".join([item['content'] for item in contents])
                    
                    # DEBUG: Show what's being summarized
                    with st.expander("🔍 Debug: Contents Being Summarized"):
                        st.write(f"**Number of documents:** {len(contents)}")
                        for item in contents:
                            st.write(f"**{item['name']}** ({item['type']}) - {len(item['content'].split())} words")
                            st.text(item['content'][:200] + "..." if len(item['content']) > 200 else item['content'])
                    
                    summary = summarize_text(combined_text, model, tokenizer, max_length=summary_length)
                    
                    end_time = time.time()
                    st.success(f"✅ Combined Summary Generated! (Time: {end_time - start_time:.2f}s)")
                    st.markdown("### 📋 Combined Summary")
                    
                    word_count = len(summary.split())
                    st.caption(f"Summary length: ~{word_count} words | Combined from {len(contents)} sources")
                    
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

    # Store in session state
    if uploaded_files is not None:
        st.session_state.uploaded_files = uploaded_files
    
    if st.session_state.get('uploaded_files'):
        st.write(f"**📁 {len(st.session_state.uploaded_files)} file(s) uploaded**")
        
        # Show file preview
        file_contents = get_all_contents(st.session_state.uploaded_files, [])
        if file_contents:
            with st.expander("📊 File Contents Preview"):
                for item in file_contents:
                    st.write(f"**{item['name']}** ({len(item['content'].split())} words)")
                    preview = item['content'][:200] + "..." if len(item['content']) > 200 else item['content']
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
                    # Get all contents based on selection
                    if file_summary_type == "Current Tab (Uploaded Files Only)":
                        contents = get_all_contents(st.session_state.uploaded_files, [])
                    else:
                        contents = get_all_contents(
                            st.session_state.uploaded_files, 
                            st.session_state.text_documents
                        )

                    if not contents:
                        st.warning("No readable content was found.")
                    else:
                        # DEBUG: Show what's being summarized
                        with st.expander("🔍 Debug: Contents Being Summarized"):
                            st.write(f"**Number of documents:** {len(contents)}")
                            for item in contents:
                                st.write(f"**{item['name']}** ({item['type']}) - {len(item['content'].split())} words")
                                st.text(item['content'][:200] + "..." if len(item['content']) > 200 else item['content'])
                        
                        # Generate combined summary
                        combined_text = "\n\n".join([item['content'] for item in contents])
                        summary = summarize_text(combined_text, model, tokenizer, max_length=summary_length)
                        
                        end_time = time.time()
                        st.success(f"✅ Combined Summary Generated! (Time: {end_time - start_time:.2f}s)")
                        st.markdown("### 📋 Combined Summary")
                        
                        word_count = len(summary.split())
                        st.caption(f"Summary length: ~{word_count} words | Combined from {len(contents)} sources")
                        
                        st.info(summary)
                        
                        # Individual summaries
                        if individual_summary_check:
                            st.divider()
                            st.markdown("### 📄 Individual File Summaries")
                            
                            file_items = [item for item in contents if item['type'] == 'file']
                            for item in file_items:
                                with st.spinner(f"Summarizing {item['name']}..."):
                                    individual_summary = summarize_text(
                                        item['content'], 
                                        model, 
                                        tokenizer, 
                                        max_length=summary_length
                                    )
                                
                                st.markdown(f"**{item['name']}** ({len(item['content'].split())} words)")
                                st.info(individual_summary)
                                st.divider()
                                
                except Exception as e:
                    st.error(f"Error generating summary: {e}")

# --- MODEL METRICS ---
st.divider()
st.subheader("📊 Model Performance Metrics")

metrics = {
    "Pegasus (CNN/DailyMail)": {"ROUGE-1": 47.65, "ROUGE-2": 18.75, "ROUGE-L": 24.95},
    "TextRank": {"ROUGE-1": 43.83, "ROUGE-2": 7.97, "ROUGE-L": 34.13},
    "LSTM-Seq2Seq": {"ROUGE-1": 38.0, "ROUGE-2": 17.0, "ROUGE-L": 32.0}
}

st.json(metrics)