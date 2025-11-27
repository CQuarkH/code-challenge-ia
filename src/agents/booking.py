from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, Field, EmailStr, validator, ValidationError
from typing import Optional
import re
from src.core.llm import get_llm
from src.state import AgentState
from src.tools.mock_api import check_availability
from src.core.logger import get_logger


logger = get_logger("Booking")

# schema que se quiere extraer para agendamiento
# ahora con validaciones para prevenir datos inválidos (TC-E08, TC-E09)
class BookingSchema(BaseModel):
    owner_name: Optional[str] = Field(None, description="Nombre del dueño", min_length=2)
    phone: Optional[str] = Field(None, description="Teléfono de contacto")
    email: Optional[EmailStr] = Field(None, description="Correo electrónico válido")
    pet_name: Optional[str] = Field(None, description="Nombre de la mascota", min_length=1)
    pet_species: Optional[str] = Field(None, description="Especie (perro, gato, etc)")
    pet_breed: Optional[str] = Field(None, description="Raza de la mascota (opcional)")
    reason: Optional[str] = Field(None, description="Motivo de la consulta", min_length=3)
    desired_time: Optional[str] = Field(None, description="Fecha y hora deseada para la cita (ej: mañana a las 4pm)")
    pet_age: Optional[str] = Field(None, description="Edad de la mascota")

    @validator('phone')
    def validate_phone(cls, v):
        """valida que el teléfono tenga formato numérico válido"""
        if v is None:
            return v
        
        # limpiar espacios, guiones y paréntesis comunes
        clean_phone = re.sub(r'[\s\-()]', '', v)
        
        # validar que sea numérico con posible + al inicio (internacional)
        if not re.match(r'^\+?[0-9]{7,15}$', clean_phone):
            raise ValueError('el teléfono debe contener entre 7 y 15 dígitos numéricos')
        
        return clean_phone
    
    @validator('pet_age')
    def validate_age(cls, v):
        """valida que la edad contenga al menos un número"""
        if v is None:
            return v
        
        # buscar números en el texto (ej: "5 años", "2 meses")
        numbers = re.findall(r'\d+', v)
        if not numbers:
            raise ValueError('la edad debe incluir al menos un número (ej: "3 años", "6 meses")')
        
        return v

def booking_node(state: AgentState):
    """
    Gestiona el flujo de agendamiento: Recolecta datos -> Verifica disponibilidad -> Confirma.
    """
    logger.info("--- AGENTE BOOKING: Gestionando cita ---")
    
    # recuperar estado actual
    messages = state["messages"]
    current_info = state.get("booking_info", {}) or {} # asegurar que sea dict para parseo
    last_message = messages[-1]
    
    if "status" not in current_info:
        current_info["status"] = "in_progress"

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
        
        except ValidationError as ve:
            # TC-E08, TC-E09: manejar errores de validación (email, teléfono inválidos)
            logger.warning(f"validación fallida: {ve}")
            
            # identificar el campo problemático
            error_field = ve.errors()[0]['loc'][0]
            error_msg = ve.errors()[0]['msg']
            
            # mapeo de nombres técnicos a nombres amigables
            field_names = {
                "phone": "número de teléfono",
                "email": "correo electrónico",
                "pet_age": "edad de la mascota",
                "owner_name": "nombre",
                "reason": "motivo de la consulta"
            }
            
            friendly_field = field_names.get(error_field, error_field)
            friendly_msg = f"Disculpa, el {friendly_field} que proporcionaste no tiene un formato válido.\n\n{error_msg}\n\n¿Podrías intentar de nuevo?"
            
            return {
                "messages": [AIMessage(content=friendly_msg)],
                "booking_info": current_info  # mantener lo que ya teníamos
            }
                
        except Exception as e:
            logger.error(f"Error en extracción: {e}")
            # continuar sin actualizar

    # guardar la info actualizada en el estado global inmediatamente

    # --- FASE 2: LÓGICA DE NEGOCIO Y DECISIÓN ---
    
    # campos obligatorios (validación)
    required_fields = ["owner_name", "phone", "email", "pet_name", "pet_species", "pet_age", "reason", "desired_time"]
    missing = [f for f in required_fields if f not in current_info]
    
    # caso a: faltan datos -> preguntar nuevamente
    if missing:
        field_names_es = {
            "owner_name": "su nombre completo",
            "phone": "un teléfono de contacto",
            "email": "un correo electrónico",
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
    
    # TC-E12: obtener el contador de intentos de disponibilidad
    attempts = state.get("availability_attempts", 0)
    MAX_ATTEMPTS = 3
    
    # llamada a la herramienta (función importada)
    is_available = check_availability.invoke({"day": "generic", "hour": time_str})
    
    if is_available:
        response = f"¡Listo! He confirmado la cita para {current_info['pet_name']} ({current_info['pet_species']}) el {time_str}. \nDatos de contacto: {current_info['owner_name']} - {current_info['phone']}.\n¡Nos vemos pronto!"
        # limpiar el estado de booking y resetear contador después de confirmar
        return {
            "messages": [AIMessage(content=response)],
            "booking_info": {},  # limpiar para la próxima
            "availability_attempts": 0  # resetear contador
        }
    else:
        # incrementar el contador de intentos fallidos
        new_attempts = attempts + 1
        
        # TC-E12: si supera el máximo, escalar a humano automáticamente
        if new_attempts >= MAX_ATTEMPTS:
            from src.tools.mock_api import request_human_agent
            
            logger.warning(f"   ⚠️ máximo de intentos alcanzado ({MAX_ATTEMPTS}). escalando a humano...")
            
            # preparar información para el ticket
            user_summary = f"Usuario: {current_info.get('owner_name', 'Desconocido')}, Teléfono: {current_info.get('phone', 'N/A')}, Email: {current_info.get('email', 'N/A')}"
            issue_summary = f"Problemas de disponibilidad después de {MAX_ATTEMPTS} intentos. Última hora solicitada: {time_str}"
            
            ticket_id = request_human_agent.invoke({
                "user_info": f"{user_summary} | {issue_summary}"
            })
            
            escalation_msg = f"Veo que has intentado {MAX_ATTEMPTS} horarios diferentes y ninguno está disponible. 😓\n\n"
            escalation_msg += f"He generado un ticket de atención prioritaria (**{ticket_id}**) para que un coordinador humano revise la agenda completa contigo y te ofrezca las mejores alternativas disponibles.\n\n"
            escalation_msg += "Te contactaremos pronto a tu teléfono o email. ¡Gracias por tu paciencia!"
            
            return {
                "messages": [AIMessage(content=escalation_msg)],
                "booking_info": {},  # limpiar
                "availability_attempts": 0,  # resetear
                "next_step": "end"  # terminar el flujo
            }
        
        # si aún hay intentos disponibles, continuar solicitando otra hora
        response = f"Lo siento, verifiqué la agenda y el horario '{time_str}' NO está disponible. 😓\n"
        response += f"(Intento {new_attempts}/{MAX_ATTEMPTS})\n\n"
        response += "¿Podrías indicarme otra fecha u hora alternativa?"
        
        # borrar solo la hora para obligar a pedirla de nuevo
        del current_info["desired_time"]
        return {
            "messages": [AIMessage(content=response)],
            "booking_info": current_info,
            "availability_attempts": new_attempts  # persistir el nuevo contador
        }