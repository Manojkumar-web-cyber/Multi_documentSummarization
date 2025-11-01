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
def process_single_file(file):
    """Process a single file and return its content"""
    try:
        # Reset file pointer
        file.seek(0)
        
        if file.name.lower().endswith('.pdf'):
            content = extract_text_from_pdf(file)
        elif file.name.lower().endswith('.pptx'):
            content = extract_text_from_pptx(file)
        elif file.name.lower().endswith('.docx'):
            content = extract_text_from_docx(file)
        else:
            content = file.read().decode("utf-8")
        
        return content.strip() if content else ""
    except Exception as e:
        st.warning(f"❌ Error processing {file.name}: {e}")
        return ""

# --- STREAMLIT UI ---
if 'text_documents' not in st.session_state:
    st.session_state.text_documents = [""]
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'file_contents' not in st.session_state:
    st.session_state.file_contents = {}

# Tabs for different input methods
tab1, tab2 = st.tabs(["📝 Multiple Text Inputs", "📄 File Upload"])

with tab1:
    st.write("**Enter multiple documents below (one per box):**")
    
    # Add/Remove document buttons
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("➕ Add Document", key="add_doc_text"):
            st.session_state.text_documents.append("")
    with col2:
        if st.button("➖ Remove Document", key="remove_doc_text") and len(st.session_state.text_documents) > 1:
            st.session_state.text_documents.pop()
    
    # Text inputs
    for i in range(len(st.session_state.text_documents)):
        st.session_state.text_documents[i] = st.text_area(
            f"Document {i+1}:", 
            height=150, 
            key=f"doc_input_{i}",
            placeholder=f"Paste document {i+1} here...",
            value=st.session_state.text_documents[i]
        )

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
                # Get text contents
                text_contents = [doc for doc in st.session_state.text_documents if doc.strip()]
                
                # Get file contents if needed
                file_contents_list = []
                if summary_type_text == "Combined Tab Data (Text Inputs + Uploaded Files)" and st.session_state.uploaded_files:
                    for file in st.session_state.uploaded_files:
                        content = process_single_file(file)
                        if content:
                            file_contents_list.append(content)
                
                # COMBINE ALL CONTENTS
                all_contents = text_contents + file_contents_list
                
                if not all_contents:
                    st.warning("Please provide some text or files to summarize.")
                else:
                    # DEBUG: Show EXACT content being combined
                    with st.expander("🔍 DEBUG: Contents Being Combined"):
                        st.write(f"**Total documents to combine:** {len(all_contents)}")
                        
                        # Show text documents
                        for i, content in enumerate(text_contents):
                            st.write(f"**Text Document {i+1}** - {len(content.split())} words")
                            preview = content[:500] + "..." if len(content) > 500 else content
                            st.text(preview)
                            st.divider()
                        
                        # Show file documents  
                        for i, content in enumerate(file_contents_list):
                            st.write(f"**File Document {i+1}** - {len(content.split())} words")
                            preview = content[:500] + "..." if len(content) > 500 else content
                            st.text(preview)
                            st.divider()
                    
                    # CRITICAL FIX: Combine ALL content properly
                    combined_text = " ".join(all_contents)  # Using space instead of newlines for better continuity
                    
                    # Show combined text preview
                    with st.expander("🔍 DEBUG: Combined Text Preview"):
                        st.write(f"Combined text length: {len(combined_text)} characters, {len(combined_text.split())} words")
                        st.text(combined_text[:1000] + "..." if len(combined_text) > 1000 else combined_text)
                    
                    summary = summarize_text(combined_text, model, tokenizer, max_length=summary_length)
                    
                    end_time = time.time()
                    st.success(f"✅ Combined Summary Generated! (Time: {end_time - start_time:.2f}s)")
                    st.markdown("### 📋 Combined Summary")
                    
                    word_count = len(summary.split())
                    st.caption(f"Summary length: ~{word_count} words | Combined from {len(all_contents)} sources")
                    
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
    
    if st.session_state.uploaded_files:
        st.write(f"**📁 {len(st.session_state.uploaded_files)} file(s) uploaded**")
        
        # Process and cache file contents
        file_contents_data = []
        for file in st.session_state.uploaded_files:
            content = process_single_file(file)
            if content:
                file_contents_data.append({
                    'name': file.name,
                    'content': content,
                    'word_count': len(content.split())
                })
        
        # Show file preview
        with st.expander("📊 File Contents Preview"):
            for file_data in file_contents_data:
                st.write(f"**{file_data['name']}** - {file_data['word_count']} words")
                preview = file_data['content'][:500] + "..." if len(file_data['content']) > 500 else file_data['content']
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
                    # Get file contents
                    file_contents_list = [file_data['content'] for file_data in file_contents_data]
                    
                    # Get text contents if needed
                    text_contents = []
                    if file_summary_type == "Combined Tab Data (Uploaded Files + Text Inputs)":
                        text_contents = [doc for doc in st.session_state.text_documents if doc.strip()]
                    
                    # COMBINE ALL CONTENTS
                    all_contents = file_contents_list + text_contents

                    if not all_contents:
                        st.warning("No readable content was found.")
                    else:
                        # DEBUG: Show EXACT content being combined
                        with st.expander("🔍 DEBUG: Contents Being Combined"):
                            st.write(f"**Total documents to combine:** {len(all_contents)}")
                            
                            # Show file documents
                            for i, file_data in enumerate(file_contents_data):
                                st.write(f"**{file_data['name']}** - {file_data['word_count']} words")
                                preview = file_data['content'][:500] + "..." if len(file_data['content']) > 500 else file_data['content']
                                st.text(preview)
                                st.divider()
                            
                            # Show text documents
                            for i, content in enumerate(text_contents):
                                st.write(f"**Text Document {i+1}** - {len(content.split())} words")
                                preview = content[:500] + "..." if len(content) > 500 else content
                                st.text(preview)
                                st.divider()
                        
                        # CRITICAL FIX: Combine ALL content properly
                        combined_text = " ".join(all_contents)
                        
                        # Show combined text preview
                        with st.expander("🔍 DEBUG: Combined Text Preview"):
                            st.write(f"Combined text length: {len(combined_text)} characters, {len(combined_text.split())} words")
                            st.text(combined_text[:1000] + "..." if len(combined_text) > 1000 else combined_text)
                        
                        # Generate combined summary
                        summary = summarize_text(combined_text, model, tokenizer, max_length=summary_length)
                        
                        end_time = time.time()
                        st.success(f"✅ Combined Summary Generated! (Time: {end_time - start_time:.2f}s)")
                        st.markdown("### 📋 Combined Summary")
                        
                        word_count = len(summary.split())
                        st.caption(f"Summary length: ~{word_count} words | Combined from {len(all_contents)} sources")
                        
                        st.info(summary)
                        
                        # Individual summaries
                        if individual_summary_check:
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

# --- MODEL METRICS ---
st.divider()
st.subheader("📊 Model Performance Metrics")

metrics = {
    "Pegasus (CNN/DailyMail)": {"ROUGE-1": 47.65, "ROUGE-2": 18.75, "ROUGE-L": 24.95},
    "TextRank": {"ROUGE-1": 43.83, "ROUGE-2": 7.97, "ROUGE-L": 34.13},
    "LSTM-Seq2Seq": {"ROUGE-1": 38.0, "ROUGE-2": 17.0, "ROUGE-L": 32.0}
}

st.json(metrics)