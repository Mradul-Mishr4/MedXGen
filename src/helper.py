from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader            
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from pathlib import Path

def load_pdf_file(data):
    # Use absolute path to the Data directory
    project_root = Path("/Users/mradulmishra/Desktop/python_project/MedXGen")
    data_path = project_root / data
    
    loader = DirectoryLoader(str(data_path),
                             glob = "*.pdf",
                             loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        
    )
    
    text_chucks = text_splitter.split_documents(extracted_data)
    return text_chucks

def download_embeddings(): 
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings