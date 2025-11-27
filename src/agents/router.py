from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from src.core.llm import get_llm
from src.state import AgentState
from src.core.logger import get_logger

logger = get_logger("Router")

class RouteQuery(BaseModel):
    """Decide a qué agente dirigir la consulta del usuario."""
    destination: str = Field(
        ...,
        description="Elegir 'technical_question', 'schedule_appointment' o 'escalate_to_human'."
    )

def router_node(state: AgentState):
    """
    Analiza el último mensaje y decide el siguiente paso.
    """
    logger.info("--- ROUTER: Clasificando intención ---")
    messages = state["messages"]
    last_message = messages[-1]
    user_text = last_message.content
    
    # recuperar si ya hay datos de una cita en proceso
    booking_info = state.get("booking_info", {})
    
    # si el diccionario tiene datos, significa que el usuario ya empezó a agendar.
    if booking_info and len(booking_info) > 0:
        logger.info(f"   Contexto activo detectado ({len(booking_info)} datos). Saltando clasificación y yendo a Booking.")
        return {"next_step": "schedule_appointment"}
    # --------------------------------------
    
    llm = get_llm()
    # usar function_calling para asegurar compatibilidad y precisión
    structured_llm = llm.with_structured_output(RouteQuery, method="function_calling")
    
    system_prompt = """Eres el encargado de triaje de 'VetCare AI'. Clasifica la intención:
    
    1. 'technical_question': Dudas sobre salud, cuidados, enfermedades.
    2. 'schedule_appointment': Agendar citas, ver horarios, o si el usuario da sus datos de contacto.
    3. 'escalate_to_human': Solo si el usuario está muy enojado, insulta o pide ayuda explícita.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])
    
    router = prompt | structured_llm
    
    try:
        decision = router.invoke({"question": user_text})
        destination = decision.destination
    except Exception as e:
        logger.error(f"Error en router: {e}, derivando a humano por seguridad.")
        destination = "escalate_to_human"
        
    logger.info(f"   Decisión: {destination}")
    return {"next_step": destination}