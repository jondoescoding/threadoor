# PYTHON
import os
import glob
import time
from typing import List
from dotenv import load_dotenv
# LANGCHAIN
from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
# UTILIES
from constants import CHROMA_SETTINGS
from utils import get_logger

# Loading environment variables
load_dotenv()
persist_directory = os.environ.get('PERSIST_DIRECTORY')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
embeddings_model = os.environ.get('EMBEDDINGS_MODEL_NAME')
model_n_ctx = os.environ.get('MODEL_N_CTX')
chunk_size = int(os.environ.get('CHUNK_SIZE'))
chunk_overlap = int(os.environ.get('CHUNK_OVERLAP'))


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (UnstructuredEmailLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}

# Loading Logger
logger = get_logger("ingest")


def load_single_document(file_path: str) -> Document:
    """Loads a single document to be processed

    Args:
        file_path (str): where the file is located

    Raises:
        ValueError: if there a given file type which isn't supported natively an error is thrown

    Returns:
        Document: Returns a loaded document based on a specific file extension 
    """
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()[0]
    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """# Loads all documents from source documents directory

    Args:
        source_dir (str): Location of ALL the files to be analysed

    Returns:
        List[Document]: A list of the documents to be analysed
    """
    pdf_files = glob.glob(os.path.join(source_dir, "**/*.pdf"), recursive=True)

    return [load_single_document(file_path) for file_path in pdf_files]


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """# Loads all documents from source documents directory

    Args:
        source_dir (str): Location of ALL the files to be analysed

    Returns:
        List[Document]: A list of the documents to be analysed
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]
    return [load_single_document(file_path) for file_path in filtered_files]


def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """# Load documents and split in chunks

    Args:
        ignored_files (List[str], optional): List of files that have already been processed. Defaults to [].

    Returns:
        List[Document]: _description_
    """
    logger.info("Loading documents from %s", source_directory)
    
    documents = load_documents(source_directory, ignored_files)
    
    if documents == []:
        logger.info("No new documents to load")
        exit(0)
    
    logger.info("Loaded %s documents from %s", len(documents), source_directory)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    texts = text_splitter.split_documents(documents)
    
    logger.info("Split into %s chunks of text (max. %s tokens each)", len(texts), chunk_size)
    
    return texts


def main():
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
    
    # Time begins
    before = time.time()
    
    if os.path.exists(persist_directory):
        # Update and store locally vectorstore
        logger.debug("Appending to existing vectorstore at %s", persist_directory)

        db = Chroma(
            persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        
        collection = db.get()
        
        texts = process_documents([metadata['source'] for metadata in collection['metadatas']])
        
        db.add_documents(texts)
    else:
        # Create and store locally vectorstore
        logger.debug("Creating new vectorstore...")
        texts = process_documents()
        db = Chroma.from_documents(
            texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)

    # After time
    after = time.time()
    logger.debug("Took %ss (%smin) to create and store in local vectorstore", round(after - before, 2), round((after - before) / 60, 2))
    
    db.persist()
    db = None


if __name__ == "__main__":
    main()