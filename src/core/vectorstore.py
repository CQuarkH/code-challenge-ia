import os
import shutil
import fitz 
from rapidocr_onnxruntime import RapidOCR
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from src.core.logger import get_logger

logger = get_logger("VectorStore")
load_dotenv()

_vectorstore_instance = None
DATA_PATH = "data/info-mascotas"
CHROMA_PATH = "data/.chroma_db"

def ocr_pdf_loader(file_path: str) -> list[Document]:
    """
    Función personalizada que lee PDFs.
    Si el PDF es de texto, lo lee rápido.
    Si es de imágenes (scanned), usa RapidOCR para extraer el texto.
    """
    doc = fitz.open(file_path)
    ocr_engine = RapidOCR() # modo OCR
    extracted_docs = []

    logger.info(f"   [OCR] Procesando {os.path.basename(file_path)}...")
    
    for i, page in enumerate(doc):
        text = page.get_text()
        
        # heurística: si la página tiene muy poco texto (<50 chars), asumimos que es imagen
        if len(text.strip()) < 50:
            print(f"      - Pág {i+1}: Detectada imagen/escaneo. Aplicando OCR...")
            # convertir la página a imagen en memoria
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            
            # ejecutamos OCR
            result, _ = ocr_engine(img_bytes)
            
            if result:
                # rapidOCR devuelve una lista de tuplas, unimos el texto encontrado
                text = "\n".join([line[1] for line in result])
            else:
                text = ""
        
        # solo agregamos si logramos sacar texto
        if text.strip():
            # creamos el objeto Document con metadata
            extracted_docs.append(Document(
                page_content=text,
                metadata={"source": file_path, "page": i+1}
            ))
            
    return extracted_docs

def get_vectorstore():
    global _vectorstore_instance
    
    if _vectorstore_instance is None:
        embeddings = OpenAIEmbeddings()

        # si ya existe DB, se carga nomás
        if os.path.exists(CHROMA_PATH):
             logger.info("--- Cargando Vector Store existente desde disco ---")
             _vectorstore_instance = Chroma(
                 persist_directory=CHROMA_PATH, 
                 embedding_function=embeddings
             )
        
        # si no, se crea uno nuevo
        elif os.path.exists(DATA_PATH) and os.listdir(DATA_PATH):
            logger.info("--- 🚀 Inicializando Vector Store con MOTOR OCR ---")
            
            docs = []
            
            # 1. cargar TXT y MD
            logger.info("   Buscando archivos de texto (TXT/MD)...")
            txt_loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
            md_loader = DirectoryLoader(DATA_PATH, glob="*.md", loader_cls=TextLoader)
            docs.extend(txt_loader.load())
            docs.extend(md_loader.load())

            # 2. cargar PDFs con OCR
            pdf_files = [f for f in os.listdir(DATA_PATH) if f.lower().endswith(".pdf")]
            logger.info(f"   Buscando PDFs ({len(pdf_files)} encontrados)...")
            
            for pdf_file in pdf_files:
                full_path = os.path.join(DATA_PATH, pdf_file)
                try:
                    pdf_docs = ocr_pdf_loader(full_path)
                    docs.extend(pdf_docs)
                except Exception as e:
                    logger.error(f"   ❌ Error procesando PDF {pdf_file}: {e}")

            logger.info(f"   Total páginas/documentos procesados: {len(docs)}")
            
            if not docs:
                return None

            # transformación
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            logger.info(f"   Divididos en {len(splits)} fragmentos.")
            
            # ingesta
            _vectorstore_instance = Chroma.from_documents(
                documents=splits, 
                embedding=embeddings,
                persist_directory=CHROMA_PATH
            )
            logger.info("--- Vector Store Listo ---")
            
        else:
            return None

    return _vectorstore_instance

def get_retriever():
    vs = get_vectorstore()
    if not vs: return None
    
    # estrategia MMR
    return vs.as_retriever(search_type="mmr", search_kwargs={'k': 8, 'fetch_k': 20})

# resetear el vectorstore para testing
def reset_vectorstore():
    global _vectorstore_instance
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    _vectorstore_instance = None