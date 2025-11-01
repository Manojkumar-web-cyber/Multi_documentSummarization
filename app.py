import streamlit as st
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import io
from PyPDF2 import PdfReader
from pptx import Presentation
from docx import Document
import time
import re

st.set_page_config(page_title="Multi-Document Summarization", layout="wide")

st.title("🔤 Multi-Document Summarization")
st.write("**ACTUALLY USING Pegasus Transformer Model**")

# --- SIMPLE SETTINGS ---
with st.sidebar:
    st.header("⚙️ Settings")
    summary_length = st.slider("Summary Length (words)", 50, 1000, 150, 50)
    st.info(f"📝 Target: ~{summary_length} words")

# --- TEXT EXTRACTION ---
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

# --- MODEL LOADING ---
@st.cache_resource(show_spinner=False)
def load_pegasus_model():
    """Load ONLY Pegasus model - no fallbacks"""
    try:
        with st.spinner("🔄 Loading Pegasus model for summarization..."):
            model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-cnn_dailymail")
            tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
        st.success("✅ Pegasus model loaded and READY for summarization!")
        return model, tokenizer
    except Exception as e:
        st.error(f"❌ Failed to load Pegasus model: {e}")
        return None, None

# Load the model
model, tokenizer = load_pegasus_model()

def pegasus_summarize(text, max_length=150):
    """Use Pegasus model PROPERLY for summarization"""
    if not text.strip():
        return "No content to summarize."
    
    try:
        # Clean the text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize with Pegasus
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            max_length=1024, 
            truncation=True,
            padding=True
        )
        
        # Generate with Pegasus
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length + 50,  # Allow some extra tokens
            min_length=max_length // 2,
            num_beams=4,
            early_stopping=True,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
        )
        
        # Decode with Pegasus tokenizer
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Clean up
        summary = re.sub(r'\.\.+', '.', summary)
        summary = re.sub(r'\s+', ' ', summary)
        return summary.strip()
        
    except Exception as e:
        return f"Summarization error: {str(e)}"

def process_single_file(file):
    """Extract content from a single file"""
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
        st.warning(f"❌ Error processing {file.name}: {e}")
        return ""

# --- MAIN APPLICATION ---
if model is None:
    st.error("❌ Cannot continue without Pegasus model. Please check the logs.")
    st.stop()

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

st.header("📄 Upload Documents for Summarization")

uploaded_files = st.file_uploader(
    "Choose files (.txt, .pdf, .pptx, .docx)", 
    type=['txt', 'pdf', 'pptx', 'docx'], 
    accept_multiple_files=True,
    key="file_uploader"
)

if uploaded_files is not None:
    st.session_state.uploaded_files = uploaded_files

if st.session_state.uploaded_files:
    st.write(f"**📁 {len(st.session_state.uploaded_files)} file(s) ready for processing**")
    
    # Process files
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
            preview = file_data['content'][:300] + "..." if len(file_data['content']) > 300 else file_data['content']
            st.text(preview)
            st.divider()
    
    # Summary options
    col1, col2 = st.columns(2)
    with col1:
        generate_combined = st.button("🚀 Generate COMBINED Summary", use_container_width=True)
    with col2:
        generate_individual = st.button("📄 Generate INDIVIDUAL Summaries", use_container_width=True)
    
    # COMBINED SUMMARY
    if generate_combined:
        st.markdown("---")
        st.subheader("🤖 Generating COMBINED Summary with Pegasus")
        
        start_time = time.time()
        
        with st.spinner("🔄 Combining documents and using Pegasus model..."):
            try:
                # Combine ALL content from ALL documents
                all_content = []
                for file_data in file_contents_data:
                    if file_data['content']:
                        # Take meaningful content from each document (not just headers)
                        sentences = re.split(r'[.!?]+', file_data['content'])
                        meaningful_sentences = []
                        
                        for sentence in sentences:
                            sentence = sentence.strip()
                            if len(sentence.split()) > 8:  # Only proper sentences
                                meaningful_sentences.append(sentence)
                            if len(meaningful_sentences) >= 10:  # Limit per document
                                break
                        
                        if meaningful_sentences:
                            doc_content = ". ".join(meaningful_sentences)
                            all_content.append(f"DOCUMENT: {file_data['name']}. CONTENT: {doc_content}")
                
                if all_content:
                    # Create combined text for Pegasus
                    combined_text = " ".join(all_content)
                    
                    st.write("**📝 Sending this combined text to Pegasus model:**")
                    with st.expander("View combined text sent to model"):
                        st.text(combined_text[:1000] + "..." if len(combined_text) > 1000 else combined_text)
                    
                    # USE PEGASUS MODEL FOR COMBINED SUMMARY
                    combined_summary = pegasus_summarize(combined_text, summary_length)
                    
                    end_time = time.time()
                    
                    st.success(f"✅ COMBINED Summary Generated! (Time: {end_time - start_time:.2f}s)")
                    st.markdown("### 📋 Combined Summary")
                    
                    word_count = len(combined_summary.split())
                    st.caption(f"**Actual length:** {word_count} words | **Sources combined:** {len(file_contents_data)}")
                    
                    # Display the summary
                    st.info(combined_summary)
                    
                else:
                    st.warning("No meaningful content found in the documents.")
                    
            except Exception as e:
                st.error(f"Error generating combined summary: {e}")
    
    # INDIVIDUAL SUMMARIES
    if generate_individual:
        st.markdown("---")
        st.subheader("📄 Generating INDIVIDUAL Summaries with Pegasus")
        
        for file_data in file_contents_data:
            with st.spinner(f"Summarizing {file_data['name']} with Pegasus..."):
                start_time = time.time()
                
                individual_summary = pegasus_summarize(file_data['content'], summary_length)
                
                end_time = time.time()
                
                st.markdown(f"**{file_data['name']}** ({file_data['word_count']} words)")
                st.caption(f"Summary generated in {end_time - start_time:.2f}s")
                st.info(individual_summary)
                st.divider()

# --- MODEL INFO ---
st.markdown("---")
st.subheader("🤖 Model Information")
st.write("**Model:** Google Pegasus (pegasus-cnn_dailymail)")
st.write("**Type:** Transformer-based abstractive summarization")
st.write("**Status:** ✅ ACTIVE and generating summaries")
st.write("**Input:** Multiple documents → **Output:** Coherent combined summary")

st.subheader("🎯 How This Works")
st.write("""
1. **Extract text** from all uploaded documents
2. **Combine meaningful content** from each document  
3. **Send to Pegasus model** for proper abstractive summarization
4. **Generate coherent summary** that combines insights from all documents
""")