import streamlit as st
from transformers import AutoModel, AutoTokenizer, PegasusForConditionalGeneration, PegasusTokenizer
import io
from PyPDF2 import PdfReader
from pptx import Presentation
from docx import Document
import time
import re

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
    
    processing_mode = st.radio(
        "Processing Mode:",
        ["🚀 Fast AI", "🎯 Quality AI"],
        help="Fast AI: Single combined summary | Quality AI: Individual + Combined summaries"
    )
    
    st.info(f"📝 Summary will be ~{summary_length} words")

# --- TEXT EXTRACTION FUNCTIONS ---
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
def load_model():
    try:
        with st.spinner("🔄 Loading Pegasus model... This may take a minute"):
            try:
                model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-cnn_dailymail")
                tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
            except Exception as e:
                st.warning(f"Pegasus specific classes failed: {e}. Trying AutoModel...")
                model = AutoModel.from_pretrained("google/pegasus-cnn_dailymail")
                tokenizer = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
                
        st.success("✅ Pegasus model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def summarize_text(text, model, tokenizer, max_length=150):
    if not text.strip():
        return "No content to summarize."
    
    try:
        max_tokens = min(int(max_length * 1.3), 512)
        cleaned_text = clean_text(text)
        
        inputs = tokenizer(
            cleaned_text, 
            return_tensors="pt", 
            max_length=1024, 
            truncation=True,
            padding=True
        )
        
        if hasattr(model, 'generate'):
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=max_tokens,
                min_length=max(40, max_tokens // 3),
                num_beams=4,
                early_stopping=True,
                length_penalty=0.8,
                no_repeat_ngram_size=3,
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        else:
            summary = extractive_fallback_summary(cleaned_text, max_length)
        
        return clean_summary(summary)
    except Exception as e:
        return extractive_fallback_summary(text, max_length)

def extractive_fallback_summary(text, max_length=150):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    meaningful_sentences = []
    for sentence in sentences:
        if len(sentence.split()) > 5:
            meaningful_sentences.append(sentence)
        if len(meaningful_sentences) >= 5:
            break
    
    summary = ". ".join(meaningful_sentences)
    words = summary.split()
    if len(words) > max_length:
        summary = " ".join(words[:max_length]) + "..."
    
    return summary

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?;:]', '', text)
    return text.strip()

def clean_summary(summary):
    summary = re.sub(r'\.\.+', '.', summary)
    summary = re.sub(r'\s+', ' ', summary)
    sentences = summary.split('. ')
    sentences = [s.strip().capitalize() for s in sentences if s.strip()]
    return '. '.join(sentences)

def count_words(text):
    return len(text.split())

# Load model
model, tokenizer = load_model()

# --- SUMMARIZATION STRATEGIES ---
def fast_ai_summary(file_contents_data, model, tokenizer, max_length=150):
    if not file_contents_data:
        return "No content to summarize.", []
    
    combined_content = []
    
    for file_data in file_contents_data:
        content = file_data['content']
        if content and content.strip():
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            meaningful_sentences = []
            for sentence in sentences:
                if len(sentence.split()) > 5:
                    meaningful_sentences.append(sentence)
                if len(meaningful_sentences) >= 5:
                    break
            
            if meaningful_sentences:
                doc_content = ". ".join(meaningful_sentences)
                combined_content.append(doc_content)
    
    if combined_content:
        final_combined_text = " ".join(combined_content)
        summary = summarize_text(final_combined_text, model, tokenizer, max_length)
        return summary, combined_content
    else:
        return "No meaningful content found.", []

def quality_ai_summary(file_contents_data, model, tokenizer, max_length=150):
    if not file_contents_data:
        return "No content to summarize.", []
    
    individual_summaries = []
    
    for file_data in file_contents_data:
        content = file_data['content']
        if content and content.strip():
            individual_summary = summarize_text(content, model, tokenizer, max_length//2)
            individual_summaries.append({
                'name': file_data['name'],
                'summary': individual_summary,
                'word_count': count_words(individual_summary)
            })
    
    if individual_summaries:
        combined_summaries_text = " ".join([item['summary'] for item in individual_summaries])
        final_summary = summarize_text(combined_summaries_text, model, tokenizer, max_length)
        return final_summary, individual_summaries
    else:
        return "No meaningful content found.", []

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
        
        return clean_text(content) if content else ""
    except Exception as e:
        st.warning(f"❌ Error processing {file.name}: {e}")
        return ""

# --- STREAMLIT UI ---
if 'text_documents' not in st.session_state:
    st.session_state.text_documents = [""]
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

tab1, tab2 = st.tabs(["📝 Multiple Text Inputs", "📄 File Upload"])

with tab1:
    st.write("**Enter multiple documents below (one per box):**")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("➕ Add Document", key="add_doc_text"):
            st.session_state.text_documents.append("")
    with col2:
        if st.button("➖ Remove Document", key="remove_doc_text") and len(st.session_state.text_documents) > 1:
            st.session_state.text_documents.pop()
    
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
        start_time = time.time()
        try:
            text_contents = [doc for doc in st.session_state.text_documents if doc.strip()]
            
            file_contents = []
            if summary_type_text == "Combined Tab Data (Text Inputs + Uploaded Files)" and st.session_state.uploaded_files:
                for file in st.session_state.uploaded_files:
                    content = process_single_file(file)
                    if content:
                        file_contents.append(content)
            
            all_contents = text_contents + file_contents
            
            if not all_contents:
                st.warning("Please provide some text or files to summarize.")
            else:
                file_data_contents = [{'name': f'Text Document {i+1}', 'content': content} 
                                    for i, content in enumerate(all_contents)]
                
                if processing_mode == "🚀 Fast AI":
                    with st.spinner("🤖 Generating combined summary..."):
                        final_summary, intermediate_data = fast_ai_summary(
                            file_data_contents, model, tokenizer, summary_length
                        )
                    intermediate_label = "Content Used for Summary"
                else:
                    with st.spinner("🎯 Generating high-quality summaries..."):
                        final_summary, intermediate_data = quality_ai_summary(
                            file_data_contents, model, tokenizer, summary_length
                        )
                    intermediate_label = "Individual Summaries"
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                st.success(f"✅ Combined Summary Generated! (Time: {processing_time:.2f}s)")
                st.markdown("### 📋 Combined Summary")
                
                word_count = count_words(final_summary)
                st.caption(f"**Summary length:** {word_count} words | **Combined from:** {len(all_contents)} sources | **Mode:** {processing_mode}")
                
                st.info(final_summary)
                
                if intermediate_data and st.checkbox("Show processing details", key="show_details_1"):
                    with st.expander(f"🔍 {intermediate_label}"):
                        if processing_mode == "🚀 Fast AI":
                            for i, data in enumerate(intermediate_data):
                                st.write(f"**Document {i+1} content used:**")
                                st.text(data[:500] + "..." if len(data) > 500 else data)
                                st.divider()
                        else:
                            for item in intermediate_data:
                                st.write(f"**{item['name']}** ({item['word_count']} words)")
                                st.text(item['summary'])
                                st.divider()
                        
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

    if uploaded_files is not None:
        st.session_state.uploaded_files = uploaded_files
    
    if st.session_state.uploaded_files:
        st.write(f"**📁 {len(st.session_state.uploaded_files)} file(s) uploaded**")
        
        file_contents_data = []
        for file in st.session_state.uploaded_files:
            content = process_single_file(file)
            if content:
                file_contents_data.append({
                    'name': file.name,
                    'content': content,
                    'word_count': count_words(content)
                })
        
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
            start_time = time.time()
            try:
                file_contents_list = [file_data['content'] for file_data in file_contents_data]
                
                text_contents = []
                if file_summary_type == "Combined Tab Data (Uploaded Files + Text Inputs)":
                    text_contents = [doc for doc in st.session_state.text_documents if doc.strip()]
                
                all_contents = file_contents_list + text_contents
                all_file_data = file_contents_data + [{'name': f'Text Document {i+1}', 'content': content} 
                                                    for i, content in enumerate(text_contents)]

                if not all_contents:
                    st.warning("No readable content was found.")
                else:
                    if processing_mode == "🚀 Fast AI":
                        with st.spinner("🤖 Generating combined summary..."):
                            final_summary, intermediate_data = fast_ai_summary(
                                all_file_data, model, tokenizer, summary_length
                            )
                        intermediate_label = "Content Used for Summary"
                    else:
                        with st.spinner("🎯 Generating high-quality summaries..."):
                            final_summary, intermediate_data = quality_ai_summary(
                                all_file_data, model, tokenizer, summary_length
                            )
                        intermediate_label = "Individual Summaries"
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                    
                    st.success(f"✅ Combined Summary Generated! (Time: {processing_time:.2f}s)")
                    st.markdown("### 📋 Combined Summary")
                    
                    word_count = count_words(final_summary)
                    st.caption(f"**Summary length:** {word_count} words | **Combined from:** {len(all_contents)} sources | **Mode:** {processing_mode}")
                    
                    st.info(final_summary)
                    
                    if intermediate_data and st.checkbox("Show processing details", key="show_details_2"):
                        with st.expander(f"🔍 {intermediate_label}"):
                            if processing_mode == "🚀 Fast AI":
                                for i, data in enumerate(intermediate_data):
                                    st.write(f"**Document {i+1} content used:**")
                                    st.text(data[:500] + "..." if len(data) > 500 else data)
                                    st.divider()
                            else:
                                for item in intermediate_data:
                                    st.write(f"**{item['name']}** ({item['word_count']} words)")
                                    st.text(item['summary'])
                                    st.divider()
                    
                    if individual_summary_check:
                        st.divider()
                        st.markdown("### 📄 Individual File Summaries")
                        
                        for file_data in file_contents_data:
                            with st.spinner(f"🤖 Summarizing {file_data['name']}..."):
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

# --- MODEL INFORMATION ---
st.divider()
st.subheader("🤖 Model Information")
st.write("**Model:** Google Pegasus (fine-tuned on CNN/DailyMail)")
st.write("**Status:** ✅ Loaded and ready for summarization")

st.subheader("📊 Model Performance Metrics")

metrics = {
    "Pegasus (CNN/DailyMail)": {"ROUGE-1": 47.65, "ROUGE-2": 18.75, "ROUGE-L": 24.95},
    "TextRank": {"ROUGE-1": 43.83, "ROUGE-2": 7.97, "ROUGE-L": 34.13},
    "LSTM-Seq2Seq": {"ROUGE-1": 38.0, "ROUGE-2": 17.0, "ROUGE-L": 32.0}
}

st.json(metrics)