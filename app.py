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
