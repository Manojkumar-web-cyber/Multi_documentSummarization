import streamlit as st
from unstructured.partition.auto import partition
import tempfile
import os

def get_uploaded_files():
    """Upload multiple file types: PDF, DOCX, PPTX, TXT"""
    uploaded_files = st.file_uploader(
        "📁 Upload your documents",
        type=["pdf", "docx", "pptx", "txt"],
        accept_multiple_files=True,
        help="Supports PDF, Word documents, PowerPoint presentations, and text files"
    )
    return uploaded_files

def process_uploaded_files(uploaded_files):
    """Process multiple uploaded files using unstructured"""
    all_documents = {}
    
    if uploaded_files:
        st.info(f"📄 Loaded {len(uploaded_files)} file(s)")
        
        for file in uploaded_files:
            try:
                # Save file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp:
                    tmp.write(file.getbuffer())
                    tmp_path = tmp.name
                
                # Partition file (works for all formats)
                elements = partition(tmp_path)
                text = "\n".join([str(el) for el in elements])
                all_documents[file.name] = text
                
                # Clean up temp file
                os.remove(tmp_path)
                
                st.success(f"✅ Processed: {file.name}")
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
                continue
        
        return all_documents if all_documents else None
    else:
        st.info("Please upload documents to begin")
        return None

def main():
    st.set_page_config(page_title="Multi-Document Summarizer", layout="wide")
    st.title("🔤 Multi-Document Summarizer")
    
    uploaded_files = get_uploaded_files()
    documents = process_uploaded_files(uploaded_files)
    
    if documents:
        tab1, tab2 = st.tabs(["Combined View", "Individual Documents"])
        
        with tab1:
            st.subheader("📖 Combined Document Content")
            combined_text = "\n\n".join([f"## {name}\n{text}" for name, text in documents.items()])
            st.text_area("Combined Text:", combined_text, height=400)
        
        with tab2:
            selected_doc = st.selectbox("Select document to view:", list(documents.keys()))
            if selected_doc:
                st.subheader(f"📄 {selected_doc}")
                st.text_area("Document Content:", documents[selected_doc], height=400)

if __name__ == "__main__":
    main()
            "Summary Type:",
            ["Combined Summary", "Individual Summaries"],
            horizontal=True
        )
        
        if documents and st.button("🚀 Generate Summary", key="text_summary"):
            with st.spinner("Generating summary..."):
                try:
                    if summary_type == "Combined Summary":
                        # Combine all documents
                        combined_text = "\n\n".join(documents)
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
        st.write("**Upload text files (.txt) for summarization:**")
        
        uploaded_files = st.file_uploader(
            "Choose files", 
            type=['txt'], 
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
                        for file in uploaded_files:
                            content = file.read().decode("utf-8")
                            file_contents.append(content)
                        
                        if file_summary_type == "Combined Summary":
                            combined_text = "\n\n".join(file_contents)
                            summary = summarize_text(combined_text, model, tokenizer)
                            
                            st.success("✅ Summary Generated!")
                            st.markdown("### 📋 Combined Summary")
                            st.info(summary)
                        else:
                            st.success("✅ Summaries Generated!")
                            for idx, (file, content) in enumerate(zip(uploaded_files, file_contents), 1):
                                summary = summarize_text(content, model, tokenizer)
                                st.markdown(f"### 📄 {file.name}")
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
