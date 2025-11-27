# src/state.py
import operator
from typing import Annotated, TypedDict, Union, List
from langchain_core.messages import BaseMessage

# Definimos el estado global del grafo
# Usamos TypedDict para tener tipado fuerte de qué datos viajan por el sistema
class AgentState(TypedDict):
    # El historial de chat. 'operator.add' es crucial: indica a LangGraph que
    # cuando un nodo devuelve 'messages', estos se DEBEN AÑADIR a la lista existente,
    # no reemplazarla.
    messages: Annotated[List[BaseMessage], operator.add]
    
    # Blackboard para datos estructurados (específicamente para el Agente de Citas)
    # Guardamos aquí lo que vamos recolectando paso a paso (nombre, especie, etc.)
    # para no perderlo entre turnos de conversación.
    booking_info: dict
    
    # Una señal interna para que el Router sepa qué nodo ejecutar a continuación.
    next_step: str