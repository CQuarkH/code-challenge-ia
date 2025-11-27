import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# variable global para almacenar la instancia única del LLM (Singleton)
_llm_instance = None

def get_llm():
    """
    Implementación del Patrón Singleton para obtener la instancia del LLM.
    Asegura que solo se cree un objeto ChatOpenAI durante la vida de la aplicación.
    """
    global _llm_instance
    
    # si aún no existe la instancia, se crea.
    if _llm_instance is None:
        # carga variables de entorno si aún no se han cargado con dotenv
        load_dotenv()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # fallar rápido si no hay configuración
            raise ValueError("Error Crítico: OPENAI_API_KEY no encontrada en variables de entorno (.env)")
        
        print("--- Inicializando instancia Singleton del LLM (gpt-3.5-turbo) ---")
        # gpt-3.5-turbo porque es rápido, barato y suficiente para este desafío.
        # temperature=0 es vital para que las decisiones del router y las herramientas sean predecibles.
        _llm_instance = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            api_key=api_key
        )
        
    # devolver la instancia existente
    return _llm_instance