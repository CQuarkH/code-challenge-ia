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