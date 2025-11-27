from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, Field
from typing import Optional
from src.core.llm import get_llm
from src.state import AgentState
from src.tools.mock_api import check_availability
from src.core.logger import get_logger


logger = get_logger("Booking")

# schema que se quiere extraer para agendamiento
class BookingSchema(BaseModel):
    owner_name: Optional[str] = Field(None, description="Nombre del dueño")
    phone: Optional[str] = Field(None, description="Teléfono de contacto")
    email: Optional[str] = Field(None, description="Correo electrónico")
    pet_name: Optional[str] = Field(None, description="Nombre de la mascota")
    pet_species: Optional[str] = Field(None, description="Especie (perro, gato, etc)")
    pet_breed: Optional[str] = Field(None, description="Raza de la mascota (opcional)")
    reason: Optional[str] = Field(None, description="Motivo de la consulta")
    desired_time: Optional[str] = Field(None, description="Fecha y hora deseada para la cita (ej: mañana a las 4pm)")
    pet_age: Optional[str] = Field(None, description="Edad de la mascota")

def booking_node(state: AgentState):
    """
    Gestiona el flujo de agendamiento: Recolecta datos -> Verifica disponibilidad -> Confirma.
    """
    logger.info("--- AGENTE BOOKING: Gestionando cita ---")
    
    # recuperar estado actual
    messages = state["messages"]
    current_info = state.get("booking_info", {}) or {} # asegurar que sea dict para parseo
    last_message = messages[-1]

    # --- FASE 1: ACTUALIZACIÓN DE ESTADO (Extracción) ---
    
    # si el último mensaje es del usuario, extraer datos nuevos
    if not isinstance(last_message, (AIMessage, ToolMessage)):
        llm = get_llm()
        # modo estructurado para que actúe como un extractor de datos
        extractor = llm.with_structured_output(BookingSchema, method="function_calling")
        
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """Eres un experto extrayendo datos para citas veterinarias.
            Tu trabajo es leer el último mensaje del usuario y actualizar la información YA CONOCIDA.
            Si el usuario menciona un dato nuevo, agrégalo. Si no, mantén lo que ya tenías.
            
            Información actual conocida:
            {current_info}
            """),
            ("human", "{user_input}"),
        ])
        
        chain = extraction_prompt | extractor
        try:
            logger.info(f"   Analizando input: '{last_message.content}'")
            # extracción
            result = chain.invoke({
                "current_info": str(current_info),
                "user_input": last_message.content
            })
            
            # actualizar solo los campos que el LLM encontró
            result_dict = result.model_dump(exclude_none=True)
            if result_dict:
                logger.info(f"   📝 Datos extraídos: {result_dict}")
                current_info.update(result_dict)
            else:
                logger.info("   ⚠️ No se extrajeron datos nuevos.")
                
        except Exception as e:
            logger.error(f"Error en extracción: {e}")

    # guardar la info actualizada en el estado global inmediatamente

    # --- FASE 2: LÓGICA DE NEGOCIO Y DECISIÓN ---
    
    # campos obligatorios (validación)
    required_fields = ["owner_name", "phone", "pet_name", "pet_species", "pet_age", "reason", "desired_time"]
    missing = [f for f in required_fields if f not in current_info]
    
    # caso a: faltan datos -> preguntar nuevamente
    if missing:
        field_names_es = {
            "owner_name": "su nombre completo",
            "phone": "un teléfono de contacto",
            "pet_name": "el nombre de la mascota",
            "pet_species": "la especie (perro, gato...)",
            "pet_age": "la edad de la mascota",
            "reason": "el motivo de la consulta",
            "desired_time": "la fecha y hora deseada"
        }
        
        # tomar el primer campo faltante para no abrumar al usuario
        next_missing = missing[0]
        question = f"Para agendar, necesito {field_names_es.get(next_missing, next_missing)}. ¿Podría indicármelo?"
        
        # si existen ciertos datos se personaliza un poco la pregunta
        if current_info.get("pet_name"):
            question = f"Perfecto. Para atender a {current_info['pet_name']}, necesito {field_names_es.get(next_missing, next_missing)}."
            
        return {
            "messages": [AIMessage(content=question)],
            "booking_info": current_info # persistir los cambios
        }

    # caso b: tenemos todo -> verificar disponibilidad (tool call)
    
    # aquí simulamos la llamada a la herramienta dentro del nodo para simplificar el flujo
    # (en un grafo más complejo, la herramienta sería otro nodo, pero aquí lo haremos directo)
    
    logger.info("   ✅ Todos los datos recolectados. Verificando disponibilidad...")
    time_str = current_info["desired_time"]
    
    # llamada a la herramienta (función importada)
    is_available = check_availability.invoke({"day": "generic", "hour": time_str})
    
    if is_available:
        response = f"¡Listo! He confirmado la cita para {current_info['pet_name']} ({current_info['pet_species']}) el {time_str}. \nDatos de contacto: {current_info['owner_name']} - {current_info['phone']}.\n¡Nos vemos pronto!"
        # opcional: limpiar el estado de booking después de confirmar
        return {
            "messages": [AIMessage(content=response)],
            "booking_info": {} # limpiar para la próxima
        }
    else:
        response = f"Lo siento, verifiqué la agenda y el horario '{time_str}' NO está disponible. 😓\n¿Podría indicarme otra fecha u hora alternativa?"
        # borrar solo la hora para obligar a pedirla de nuevo
        del current_info["desired_time"]
        return {
            "messages": [AIMessage(content=response)],
            "booking_info": current_info
        }