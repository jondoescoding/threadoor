# PYTHON
import os
import glob
import time
from typing import List
from dotenv import load_dotenv
# LANGCHAIN
from langchain.document_loaders import TextLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings
from langchain.docstore.document import Document
# UTILIES
from constants import CHROMA_SETTINGS
from utils import get_logger

# Loading environment variables
load_dotenv()
persist_directory = os.environ.get('PERSIST_DIRECTORY')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
llama_embeddings_model = os.environ.get('LLAMA_EMBEDDINGS_MODEL')
model_n_ctx = os.environ.get('MODEL_N_CTX')
chunk_size = int(os.environ.get('CHUNK_SIZE'))
chunk_overlap = int(os.environ.get('CHUNK_OVERLAP'))

# Loading Logger
logger = get_logger("ingest")

def load_single_document(file_path: str) -> Document:
    """Loads a single pdf document

    Args:
        file_path (str): the file path of the document being ingested

    Returns:
        Document: the document which is to be ingested
    """
    
    return PDFMinerLoader(file_path).load()[0]


def load_documents(source_dir: str) -> List[Document]:
    """# Loads all documents from source documents directory

    Args:
        source_dir (str): Location of ALL the files to be analysed

    Returns:
        List[Document]: A list of the documents to be analysed
    """
    pdf_files = glob.glob(os.path.join(source_dir, "**/*.pdf"), recursive=True)

    return [load_single_document(file_path) for file_path in pdf_files]

if __name__ == "__main__":
    # Load documents and split in chunks
    logger.info("Loading documents from %s", source_directory)
    documents = load_documents(source_directory)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    texts = text_splitter.split_documents(documents)
    
    logger.info("Loaded %s documents from %s", len(documents), source_directory)

    logger.info("Split into %s chunks of text (max. %s tokens each)", len(texts), chunk_size)

    # Create embeddings
    llama = LlamaCppEmbeddings(model_path=llama_embeddings_model, n_ctx=model_n_ctx)
    
    # Create and store locally vectorstore
    before = time.time()
    db = Chroma.from_documents(
        texts,
        llama, 
        persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    after = time.time()
    logger.debug("Took %ss (%smin) to create and store in local vectorstore", round(after - before, 2), round((after - before) / 60, 2))
    db.persist()
    db = None