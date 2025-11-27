from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from src.core.llm import get_llm
from src.state import AgentState

# 1. definir la estructura de salida estricta
# para obligar al LLM a elegir una de las opciones sin complicaciones
class RouteQuery(BaseModel):
    """Decide a qué agente dirigir la consulta del usuario."""
    destination: str = Field(
        ...,
        description="Elegir 'technical_question' para dudas médicas/cuidados, 'schedule_appointment' para citas, o 'escalate_to_human' para quejas/ayuda humana."
    )

def router_node(state: AgentState):
    """
    Nodo del Grafo: Analiza el último mensaje y actualiza el estado 'next_step'.
    """
    print("--- ROUTER: Clasificando intención del usuario ---")
    
    # obtener el último mensaje del historial
    messages = state["messages"]
    last_message = messages[-1]
    user_text = last_message.content
    
    llm = get_llm()
    
    # hacer que el LLM devuelva JSON estructurado (function calling forzado)
    structured_llm = llm.with_structured_output(RouteQuery, method="function_calling")
    
    # prompt de sistema optimizado para clasificación
    system_prompt = """Eres el encargado de triaje de la clínica veterinaria 'VetCare AI'.
    Tu única función es clasificar la intención del usuario en una de estas categorías:
    
    1. 'technical_question': Preguntas sobre salud, alimentación, cuidados, razas, o información general de mascotas.
    2. 'schedule_appointment': Intención explícita de agendar, reservar, ver horarios o visitar la clínica.
    3. 'escalate_to_human': El usuario pide hablar con una persona, muestra enojo/frustración, o es una emergencia médica grave.
    
    Analiza el input y decide."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])
    
    # crear la cadena: prompt -> llm estructurado
    router_chain = prompt | structured_llm
    
    try:
        decision = router_chain.invoke({"question": user_text})
        destination = decision.destination
    except Exception as e:
        print(f"Error en clasificación: {e}. Derivando a humano por seguridad.")
        destination = "escalate_to_human"
        
    print(f"--- Decisión: {destination} ---")
    
    # actualizar el estado. 
    return {"next_step": destination}