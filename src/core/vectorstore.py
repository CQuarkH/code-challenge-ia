import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

_vectorstore_instance = None
DATA_PATH = "data/info-mascotas"
CHROMA_PATH = "data/.chroma_db"

def get_vectorstore():
    global _vectorstore_instance
    
    if _vectorstore_instance is None:
        embeddings = OpenAIEmbeddings()

        if os.path.exists(CHROMA_PATH):
             print("--- Cargando Vector Store existente desde disco ---")
             _vectorstore_instance = Chroma(
                 persist_directory=CHROMA_PATH, 
                 embedding_function=embeddings
             )
        
        elif os.path.exists(DATA_PATH) and os.listdir(DATA_PATH):
            print("--- Inicializando Vector Store desde documentos fuente ---")
            
            docs = []
            
            ## hasta ahora con soporte para TXT, PDF y MD
            
            # 1. cargar PDFs
            print("   Buscando archivos PDF...")
            loader_pdf = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
            docs.extend(loader_pdf.load())

            # 2. cargar TXT
            print("   Buscando archivos TXT...")
            loader_txt = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
            docs.extend(loader_txt.load())
            
            # 3. cargar markdown (como texto plano para RAG simple)
            print("   Buscando archivos Markdown...")
            loader_md = DirectoryLoader(DATA_PATH, glob="*.md", loader_cls=TextLoader)
            docs.extend(loader_md.load())

            print(f"   Total documentos cargados: {len(docs)}")
            
            if not docs:
                print("ADVERTENCIA: No se encontraron archivos válidos (.txt, .pdf, .md).")
                return None

            # transformación
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            print(f"   Divididos en {len(splits)} fragmentos.")
            
            # creación e ingesta
            print("   Creando embeddings y persistiendo ChromaDB...")
            _vectorstore_instance = Chroma.from_documents(
                documents=splits, 
                embedding=embeddings,
                persist_directory=CHROMA_PATH
            )
            print("--- Vector Store Listo ---")
            
        else:
            print(f"ADVERTENCIA: No se encontraron documentos en {DATA_PATH} ni DB existente.")
            return None

    return _vectorstore_instance

def get_retriever():
    vs = get_vectorstore()
    return vs.as_retriever() if vs else None

def reset_vectorstore():
    global _vectorstore_instance
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    _vectorstore_instance = None
    print("Vector Store reseteado.")