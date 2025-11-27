import random
from langchain_core.tools import tool

# decorador @tool para que LangChain sepa que estas son herramientas
# que el LLM podría llegar a invocar si quisieramos usar tool-calling automático.

@tool
def check_availability(day: str, hour: str) -> bool:
    """
    Verifica si una fecha y hora específicas están disponibles para una cita.
    Retorna True si está disponible, False si no.
    """
    print(f"   [SISTEMA] Consultando agenda para: {day} a las {hour}...")
    
    # simular una llamada a API con latencia o proceso
    # 80% de probabilidad de que esté libre para facilitar las pruebas
    is_available = random.random() > 0.2 
    
    if is_available:
        print("   [SISTEMA] ✅ Horario disponible.")
    else:
        print("   [SISTEMA] ❌ Horario ocupado.")
        
    return is_available

@tool
def request_human_agent(user_info: str) -> str:
    """
    Escala la conversación a un agente humano generando un ticket de soporte.
    """
    print(f"   [SISTEMA] 🚨 !!! INICIANDO PROTOCOLO DE ESCALACIÓN !!!")
    print(f"   [SISTEMA] Creando ticket para usuario con datos: {user_info}")
    
    # simular la creación del ticket
    ticket_id = f"TICKET-{random.randint(1000, 9999)}"
    print(f"   [SISTEMA] ✅ Ticket creado exitosamente: {ticket_id}")
    
    return ticket_id