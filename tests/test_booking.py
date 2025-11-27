import pytest
from langchain_core.messages import HumanMessage
from src.agents.booking import booking_node

def test_booking_flow_slot_filling():
    """
    Simula una conversación completa paso a paso para verificar el llenado de slots.
    """
    # estado inicial vacío
    state = {
        "messages": [HumanMessage(content="Hola, quiero agendar una cita")],
        "booking_info": {},
        "next_step": ""
    }
    
    print("\n--- PASO 1: Inicio ---")
    result = booking_node(state)
    
    # el agente debería pedir el nombre o algún dato faltante
    print(f"Bot: {result['messages'][0].content}")
    assert len(result['booking_info']) == 0
    
    # 2. el usuario da el nombre y mascota
    state["booking_info"] = result["booking_info"] # actualizar estado simulando el grafo
    state["messages"].append(HumanMessage(content="Me llamo Carlos y mi perro es Bobby"))
    
    print("\n--- PASO 2: Usuario da Nombre y Perro ---")
    result = booking_node(state)
    print(f"Bot: {result['messages'][0].content}")
    
    # verificación: debería haber extraído Carlos y Bobby
    info = result["booking_info"]
    assert info.get("owner_name") == "Carlos"
    assert "Bobby" in info.get("pet_name", "") or "Bobby" in str(info)
    
    # 3. usuario completa lo que falta (teléfono, motivo, especie) excepto hora
    state["booking_info"] = info
    state["messages"].append(HumanMessage(content="Es un perro, tiene vómitos. Mi cel es 555-1234"))
    
    print("\n--- PASO 3: Resto de datos (menos hora) ---")
    result = booking_node(state)
    print(f"Bot: {result['messages'][0].content}")
    
    # verificación: debería pedir la hora
    assert "hora" in result['messages'][0].content.lower() or "cuándo" in result['messages'][0].content.lower()
    assert result["booking_info"].get("phone") == "555-1234"

def test_booking_availability_check():
    """Prueba que cuando tiene todo, llama a la disponibilidad"""
    
    # prepara un estado con casi todo listo
    full_info = {
        "owner_name": "Ana",
        "phone": "999",
        "pet_name": "Mishi",
        "pet_species": "Gato",
        "reason": "Vacuna"
        # falta desired_time!!!
    }
    
    state = {
        "messages": [HumanMessage(content="Quiero ir mañana a las 10am")],
        "booking_info": full_info,
        "next_step": ""
    }
    
    print("\n--- PASO FINAL: Verificación ---")
    result = booking_node(state)
    response = result['messages'][0].content
    print(f"Bot: {response}")
    
    # la respuesta debe ser confirmación o rechazo por disponibilidad
    # pero no debe ser una pregunta pidiendo datos.
    assert "confirmado" in response.lower() or "no está disponible" in response.lower()