from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage
from src.core.llm import get_llm
from src.core.vectorstore import get_retriever
from src.state import AgentState
from src.core.logger import get_logger

logger = get_logger("RAG")

def is_veterinary_domain(question: str) -> bool:
    """
    detecta si una pregunta está relacionada con el dominio veterinario.
    retorna False para preguntas claramente fuera de tema. (TC-E05)
    """
    question_lower = question.lower()
    
    # palabras clave que indican temas veterinarios
    vet_keywords = [
        'mascota', 'perro', 'gato', 'veterinari', 'vacuna', 'enferm',
        'animal', 'cachorro', 'gatito', 'salud', 'síntoma', 'tratamiento',
        'medicamento', 'comida', 'nutrición', 'parasito', 'pulga', 'garrapata',
        'esterilización', 'castración', 'chip', 'adopción', 'pelaje', 'diente',
        'veterinaria', 'clínica', 'consulta', 'ave', 'conejo', 'hámster',
        'mascota', 'pelo', 'vómito', 'diarrea', 'comer', 'beber'
    ]
    
    # palabras que claramente indican preguntas fuera del dominio
    off_topic_indicators = [
        'capital', 'país', 'ciudad', 'historia', 'matemática', 'física',
        'receta cocina', 'cocinar', 'película', 'libro', 'música', 'deporte',
        'política', 'economía', 'presidente', 'mundial', 'fútbol',
        'humano', 'persona', 'gente', 'lasaña', 'pizza'
    ]
    
    # verificar si contiene palabras veterinarias
    has_vet_keywords = any(kw in question_lower for kw in vet_keywords)
    
    # verificar si contiene palabras claramente off-topic
    has_off_topic = any(kw in question_lower for kw in off_topic_indicators)
    
    # si tiene off-topic Y NO tiene vet keywords, es fuera de dominio
    if has_off_topic and not has_vet_keywords:
        return False
    
    # por defecto asumir que está en dominio (mejor falso positivo que negativo)
    return True

def rag_node(state: AgentState):
    """
    Estrategia RAG: Recupera contexto y responde preguntas técnicas.
    """
    logger.info("--- AGENTE RAG: Procesando consulta ---")
    
    messages = state["messages"]
    # asumir que la pregunta es el último mensaje del usuario
    question = messages[-1].content
    
    # TC-E05: pre-filtro para detectar preguntas fuera del dominio veterinario
    if not is_veterinary_domain(question):
        logger.info(f"   ⚠️ pregunta fuera de dominio detectada: '{question}'")
        off_topic_msg = """Hola! Soy el asistente veterinario de VetCare AI. 🐾

Mi especialidad es ayudarte con temas relacionados con el cuidado y la salud de tus mascotas (perros, gatos, aves, conejos, etc.).

La pregunta que hiciste parece estar fuera de mi área de conocimiento. ¿Tienes alguna consulta sobre tu mascota en la que pueda ayudarte?"""
        
        return {"messages": [AIMessage(content=off_topic_msg)]}
    
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