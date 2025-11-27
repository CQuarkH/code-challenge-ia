from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage
from src.state import AgentState
from src.agents.router import router_node
from src.agents.rag import rag_node
from src.agents.booking import booking_node
from src.tools.mock_api import request_human_agent


# nodo simple de escalation, se delega a un humano
def escalation_node(state: AgentState):
    """
    Nodo que maneja la derivación a un humano.
    """
    print("--- NODO ESCALACIÓN: Creando ticket ---")
    user_msg = state["messages"][-1].content
    
    # llamar a la herramienta
    ticket_id = request_human_agent.invoke({"user_info": user_msg})
    
    response = f"Entiendo tu situación. He generado un ticket de atención urgente con ID **{ticket_id}**. Un especialista humano te contactará a la brevedad."
    
    return {
        "messages": [AIMessage(content=response)],
        "next_step": "end"
    }

# --- GRAFO ---

def create_graph():
    """
    Construye y compila el grafo de LangGraph.
    """
    # inicializar el grafo
    workflow = StateGraph(AgentState)

    # añadir los nodos o agentes
    workflow.add_node("router", router_node)
    workflow.add_node("rag_agent", rag_node)
    workflow.add_node("booking_agent", booking_node)
    workflow.add_node("human_escalation", escalation_node)

    # definir el punto de entrada mapeado hacia el router
    workflow.set_entry_point("router")

    # definir los condicionales de enrutamiento basados en 'next_step'
    workflow.add_conditional_edges(
        "router",         
        lambda x: x["next_step"],
        {
            # mapeo next_step -> nodo destino
            "technical_question": "rag_agent",
            "schedule_appointment": "booking_agent",
            "escalate_to_human": "human_escalation",
        }
    )

    # definir las transiciones de los agentes de vuelta al router
    workflow.add_edge("rag_agent", END)
    workflow.add_edge("booking_agent", END)
    workflow.add_edge("human_escalation", END)

    # compilar
    app = workflow.compile()
    return app