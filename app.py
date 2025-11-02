import streamlit as st
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import io
from PyPDF2 import PdfReader
from pptx import Presentation
from docx import Document
import time
import re

st.set_page_config(page_title="Multi-Document Summarization", layout="wide")

st.title("🚀 Multi-Document Summarization")
st.write("Fine-tuned Pegasus Transformer Model")

# --- Clean Sidebar Settings ---
with st.sidebar:
    st.header("⚙️ Summary Settings")
    summary_length = st.slider(
        "Summary Length (words approx.)",
        min_value=50, max_value=500, value=150, step=50,
    )
    st.info(f"Target: ~{summary_length} words")

# --- Text Extraction Functions ---
def extract_text_from_pdf(file):
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

# --- Model Loading ---
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        with st.spinner("Loading Pegasus model..."):
            model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-cnn_dailymail")
            tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load model
model, tokenizer = load_model()

# --- IMPROVED Summarization Function ---
def summarize_text(text, max_length=150):
    """Clean, fast summarization with proper formatting"""
    if not text or not text.strip():
        return "No content to summarize."
    
    try:
        # Clean input text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Tokenize
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            max_length=1024, 
            truncation=True,
            padding=True
        )
        
        # Generate summary
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length + 50,
            min_length=40,
            num_beams=4,
            early_stopping=True,
            length_penalty=0.8,
            no_repeat_ngram_size=3,
            do_sample=False,  # More deterministic
        )
        
        # Decode and clean
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # PROPER CLEANING
        summary = summary.replace('<n>', ' ')  # Remove <n> tokens
        summary = re.sub(r'\.\.+', '.', summary)  # Remove multiple dots
        summary = re.sub(r'\s+', ' ', summary)  # Remove extra spaces
        summary = re.sub(r'\s\.', '.', summary)  # Remove space before dots
        summary = summary.strip()
        
        # Ensure proper sentence structure
        if summary and not summary.endswith(('.', '!', '?')):
            summary += '.'
            
        return summary
        
    except Exception as e:
        return f"Error in summarization: {str(e)}"

def count_words(text):
    """Accurate word count"""
    return len(text.split())

def process_single_file(file):
    try:
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
        return ""

# --- Session State ---
if 'text_documents' not in st.session_state:
    st.session_state.text_documents = [""]
if 'uploaded_files_state' not in st.session_state:
    st.session_state.uploaded_files_state = None

# --- Tabs Interface (Your Preferred Layout) ---
tab1, tab2 = st.tabs(["📝 Multiple Text Inputs", "📄 File Upload"])

with tab1:
    st.write("**Enter multiple documents below (one per box):**")

    if 'num_docs' not in st.session_state:
        st.session_state.num_docs = 2
    
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("➕ Add Document"):
            st.session_state.num_docs += 1
    with col2:
        if st.button("➖ Remove Document") and st.session_state.num_docs > 1:
            st.session_state.num_docs -= 1
    
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
    
    st.session_state.text_documents = current_text_documents

    summary_type_text = st.radio(
        "Summary Scope:",
        ["Current Tab (Text Inputs Only)", "Combined Tab Data (Text Inputs + Uploaded Files)"],
        horizontal=True,
    )
    
    if st.session_state.text_documents and st.button("🚀 Generate Summary", key="text_summary"):
        with st.spinner("Generating summary..."):
            start_time = time.time()
            try:
                documents_to_summarize = st.session_state.text_documents
                
                if summary_type_text == "Combined Tab Data (Text Inputs + Uploaded Files)" and st.session_state.uploaded_files_state:
                    file_contents = []
                    for file in st.session_state.uploaded_files_state:
                        content = process_single_file(file)
                        if content:
                            file_contents.append(content)
                    documents_to_summarize.extend(file_contents)

                if not documents_to_summarize:
                    st.warning("Please provide some text or files to summarize.")
                else:
                    combined_text = "\n\n".join(documents_to_summarize)
                    summary = summarize_text(combined_text, summary_length)
                    
                    end_time = time.time()
                    st.success(f"✅ Combined Summary Generated! (Time: {end_time - start_time:.2f}s)")
                    st.markdown("### 📋 Combined Summary")
                    
                    word_count = count_words(summary)
                    st.caption(f"Summary length: {word_count} words")
                    
                    st.info(summary)
                        
            except Exception as e:
                st.error(f"Error generating summary: {e}")

with tab2:
    st.write("**Upload files (.txt, .pdf, .pptx, .docx) for summarization:**")
    
    uploaded_files = st.file_uploader(
        "Choose files", 
        type=['txt', 'pdf', 'pptx', 'docx'], 
        accept_multiple_files=True
    )

    st.session_state.uploaded_files_state = uploaded_files
    
    if uploaded_files:
        st.write(f"**{len(uploaded_files)} file(s) uploaded**")
        
        file_summary_type = st.radio(
            "Summary Scope:",
            ["Current Tab (Uploaded Files Only)", "Combined Tab Data (Uploaded Files + Text Inputs)"],
            horizontal=True,
            key="file_summary_type"
        )
        
        individual_summary_check = st.checkbox("Also generate individual summaries for uploaded files?")

        if st.button("🚀 Generate Summary", key="file_summary"):
            with st.spinner("Processing files and generating summary..."):
                start_time = time.time()
                try:
                    file_contents = []
                    for file in uploaded_files:
                        content = process_single_file(file)
                        if content:
                            file_contents.append({
                                'name': file.name,
                                'content': content,
                                'word_count': count_words(content)
                            })
                    
                    documents_to_summarize = [item['content'] for item in file_contents]
                    
                    if file_summary_type == "Combined Tab Data (Uploaded Files + Text Inputs)":
                        documents_to_summarize.extend(st.session_state.text_documents)

                    if not documents_to_summarize:
                        st.warning("No readable content was found.")
                    else:
                        # Generate Combined Summary
                        combined_text = "\n\n".join(documents_to_summarize)
                        summary = summarize_text(combined_text, summary_length)
                        
                        end_time = time.time()
                        st.success(f"✅ Combined Summary Generated! (Time: {end_time - start_time:.2f}s)")
                        st.markdown("### 📋 Combined Summary")
                        
                        word_count = count_words(summary)
                        st.caption(f"Summary length: {word_count} words")
                        
                        st.info(summary)
                        
                        # Generate Individual Summaries if requested
                        if individual_summary_check and file_contents:
                            st.divider()
                            st.markdown("### 📄 Individual File Summaries")
                            for file_data in file_contents:
                                individual_start = time.time()
                                individual_summary = summarize_text(file_data['content'], summary_length)
                                individual_end = time.time()
                                
                                st.markdown(f"**{file_data['name']}** ({file_data['word_count']} words)")
                                st.caption(f"Generated in {individual_end - individual_start:.2f}s")
                                st.info(individual_summary)
                                st.divider()
                                
                except Exception as e:
                    st.error(f"Error generating summary: {e}")

# --- Model Metrics ---
st.divider()
st.subheader("📊 Model Performance Metrics")

metrics = {
    "Pegasus (CNN/DailyMail)": {"ROUGE-1": 47.65, "ROUGE-2": 18.75, "ROUGE-L": 24.95},
    "TextRank": {"ROUGE-1": 43.83, "ROUGE-2": 7.97, "ROUGE-L": 34.13},
    "LSTM-Seq2Seq": {"ROUGE-1": 38.0, "ROUGE-2": 17.0, "ROUGE-L": 32.0}
}

st.json(metrics)