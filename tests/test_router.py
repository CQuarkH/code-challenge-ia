import pytest
from langchain_core.messages import HumanMessage
from src.agents.router import router_node

## casos de prueba para el router, considerando las categorías definidas
##  technical_question, schedule_appointment, escalate_to_human

test_cases = [
    # schedule_appointment
    ("Quiero reservar una hora para mañana", "schedule_appointment"),
    ("¿Tienen disponibilidad para el martes?", "schedule_appointment"),
    
    # technical_question
    ("¿Qué tipo de alimento recomiendas para un cachorro?", "technical_question"),
    ("Mi gato está botando mucho pelo", "technical_question"),
    
    # escalate_to_human / emergencias
    ("¡Estoy harto, quiero hablar con un humano!", "escalate_to_human"),
    
    # caso que se descubrió en pruebas manuales. gpt-3.5-turbo clasifica bien las emergencias   escalando a humano.
    ("Mi perro comió chocolate y está convulsionando", "escalate_to_human"), 
]

@pytest.mark.parametrize("user_input, expected_step", test_cases)
def test_router_classification(user_input, expected_step):
    """
    Verifica que el router clasifique correctamente las intenciones.
    """
  
    mock_state = {
        "messages": [HumanMessage(content=user_input)],
        "booking_info": {},
        "next_step": ""
    }

    result = router_node(mock_state)
    
    assert result["next_step"] == expected_step, \
        f"Fallo para el input: '{user_input}'. Esperaba {expected_step}, obtuvo {result['next_step']}"
        
def test_router_break_loop_on_cancel():
    """
    Verifica que el Router permita salir del flujo de agendamiento (break loop)
    si el usuario dice una palabra clave de cancelación, incluso si hay datos previos.
    """
    # simular un estado donde el usuario ya estaba agendando (tiene datos)
    state_with_context = {
        "messages": [HumanMessage(content="Quiero cancelar todo")],
        "booking_info": {"status": "in_progress", "owner_name": "Juan"}, # contexto activo
        "next_step": ""
    }
    
    # ejecutar el router
    print("\n--- Probando Cancelación con Contexto Activo ---")
    result = router_node(state_with_context)
    
    # verificar que no se quede en agendamiento
    # no debe ser schedule_appointment (que sería el bucle infinito)
    print(f"Decisión del Router: {result['next_step']}")
    
    assert result["next_step"] == "escalate_to_human", \
        "El router debió romper el bucle y escalar/cancelar, pero forzó agendamiento."