from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage
from src.core.llm import get_llm
from src.core.vectorstore import get_retriever
from src.state import AgentState
from src.core.logger import get_logger

logger = get_logger("RAG")

def rag_node(state: AgentState):
    """
    Estrategia RAG: Recupera contexto y responde preguntas técnicas.
    """
    logger.info("--- AGENTE RAG: Procesando consulta ---")
    
    messages = state["messages"]
    # asumir que la pregunta es el último mensaje del usuario
    question = messages[-1].content
    
    llm = get_llm()
    retriever = get_retriever()
    
    # 1. validación de seguridad: si no hay base de datos
    if not retriever:
        error_msg = "Lo siento, el sistema de conocimiento no está disponible temporalmente."
        return {"messages": [AIMessage(content=error_msg)]}

    # 2. recuperación (Retrieval)
    logger.info(f"Buscando en documentos sobre: '{question}'")
    try:
        docs = retriever.invoke(question)
    except Exception as e:
        logger.error(f"Error recuperando documentos: {e}")
        docs = []
    
    # si no encuentra nada o falla
    if not docs:
         return {"messages": [AIMessage(content="Lo siento, no encontré información específica sobre eso en mis manuales.")]}

    # formatear contexto
    context = "\n\n".join([d.page_content for d in docs])
    logger.info(f"Contexto recuperado: {len(docs)} fragmentos.")

    # 3. generación de respuesta 
    system_prompt = """Eres un asistente veterinario de la clínica 'VetCare AI'.
    Responde a la pregunta del usuario basándote EXCLUSIVAMENTE en el siguiente contexto.
    
    Reglas:
    - Si la respuesta no está en el contexto, di "No tengo información sobre eso en mis documentos".
    - Sé amable, claro y conciso.
    - No inventes tratamientos médicos.
    
    Contexto:
    {context}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])
    
    rag_chain = prompt | llm | StrOutputParser()
    
    try:
        response = rag_chain.invoke({"context": context, "question": question})
    except Exception as e:
        logger.error(f"Error generando respuesta LLM: {e}")
        response = "Tuve un problema generando la respuesta. Por favor intenta más tarde."

    logger.info("--- ✅ Respuesta generada ---")
    
    # retornamos el mensaje para que LangGraph lo añada al historial
    return {"messages": [AIMessage(content=response)]}