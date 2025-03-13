import builtins
import os
import sys
import traceback
import hashlib
import pickle

from dotenv import load_dotenv

from datetime import datetime

from tqdm import tqdm

from docling.document_converter import DocumentConverter

from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_text_splitters import MarkdownHeaderTextSplitter

#region 0. Utils

_log_file = None

def print(log: str, log_dir: str = "./logs") -> None:
    """Print log with timestamp

    Args:
        log (str): log message
    """
    
    global _log_file
    
    if _log_file is None:
        os.makedirs(log_dir, exist_ok=True)
        log_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"
        _log_file = open(os.path.join(log_dir, log_filename), "w", encoding="utf-8")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    builtins.print(f"\033[36m[\033[1m{timestamp}\033[0m\033[36m]\033[0m {log}")
    
    _log_file.write(f"[{timestamp}] {log}\n")
    _log_file.flush()

#endregion

#region 1. Model Loading

def load_embedding_model(model_name: str, **kwargs) -> Embeddings:
    """Load embedding model

    Args:
        model_name (str): model name
        **kwargs: additional parameters for embedding model

    Returns:
        Embeddings: loaded embedding model
    """
    
    return HuggingFaceEmbeddings(
        model_name=model_name,
        **kwargs
    )

#endregion

#region 2. Main Tasks

def create_guideline_documents(data_dir: str, cache_dir: str = "./cache") -> list[Document]:
    """Create train documents

    Args:
        data_dir (str): data directory

    Returns:
        list[Document]: guideline documents
    """
    
    converter = DocumentConverter()
    
    documents = []
    
    for filename in tqdm(os.listdir(data_dir), desc="Creating guideline documents"):
        if not filename.lower().endswith(".pdf"):
            continue
        
        file_path = os.path.join(data_dir, filename)
        
        # Calculate file hash
        with open(file_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        # Create cache file path
        cache_file = os.path.join(cache_dir, f"{file_hash}.pkl")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Check cache
        if os.path.exists(cache_file):
            print(f"Loading '{filename}' from cache ...")
            with open(cache_file, "rb") as f:
                cached_docs = pickle.load(f)
                documents.extend(cached_docs)
            continue
        
        file_documents = []
        try:
            # Docling Convert
            result = converter.convert(file_path, page_range=(2, sys.maxsize))  # Skip the first page
            markdown_text = result.document.export_to_markdown()
            file_documents.append(Document(page_content=markdown_text))
        except Exception:
            # Fallback to PyPDF when Docling fails
            print(f"Docling Error: {traceback.format_exc()}")
            
            loaded = PyPDFLoader(file_path).load()
            file_documents.extend(loaded[1:] if len(loaded) > 1 else [])  # Skip the first page
        
        # Save cache
        with open(cache_file, "wb") as f:
            pickle.dump(file_documents, f)
        
        documents.extend(file_documents)
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
    )
    markdown_splits = [
        split 
        for doc in documents 
        for split in markdown_splitter.split_text(doc.page_content)
    ]
    
    return markdown_splits

def create_retriever(
    documents: list[Document],
    embedding: Embeddings,
    cache_path: str,
    path: str,
    index_name: str
):
    """Create retriever

    Args:
        documents (list[Document]): documents
        embedding (Embeddings): embedding model
        cache_path (str): cache path
        path (str): path
        index_name (str): index name
    """
    
    embedding_store = LocalFileStore(cache_path)
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(embedding, embedding_store, namespace=f"{index_name}_embed-")
    
    embedding_texts = []
    embedding_datas = []
    metadatas = []
    
    for doc in tqdm(documents, desc="Embedding documents"):
        embedding_vector = cached_embedder.embed_documents([doc.page_content])
        embedding_texts.append(doc.page_content)
        embedding_datas.extend(embedding_vector)
        metadatas.append(doc.metadata)
    
    vectorstore = FAISS.from_embeddings(zip(embedding_texts, embedding_datas), embedding, metadatas=metadatas)
    vectorstore.save_local(path, index_name)

#endregion


if __name__ == "__main__":
    # Working Directory Change (Parent Directory)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)
    
    print("🚀 Start !")
    
    # Environment Variables Loading
    load_dotenv(override=True)
    
    GUIDELINE_DATA_DIR = os.getenv("GUIDELINE_DATA_DIR", "./data/pdf")
    CACHE_PATH = os.getenv("CACHE_PATH", "./cache")
    FAISS_PATH = os.getenv("FAISS_PATH", "./faiss")
    FAISS_GUIDELINE_INDEX_NAME = os.getenv("FAISS_GUIDELINE_INDEX_NAME", "dacon_guideline")
    
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
    
    # Create Embedding Model
    print("📥 Loading embedding model ...")
    embedding = load_embedding_model(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # Create Guideline Documents
    print("📄 Create guideline documents ...")
    guideline_documents = create_guideline_documents(GUIDELINE_DATA_DIR, CACHE_PATH)
    
    # Create Guideline Retriever
    print("🔍 Create guideline retriever ...")
    create_retriever(
        guideline_documents,
        embedding,
        CACHE_PATH,
        FAISS_PATH,
        FAISS_GUIDELINE_INDEX_NAME
    )
    
    print("🏁 End !")
