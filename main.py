import sys
from langchain_core.messages import HumanMessage
from src.graph.workflow import create_graph
from src.core.vectorstore import get_vectorstore

# colores para la terminal pq se ve más bonito
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"

def main():
    print(f"{BLUE}--- VetCare AI Iniciado ---{RESET}")
    
    # pre-carga de conocimiento 
    print("🧠 Cargando base de conocimientos (esto puede demorar la primera vez)...")
    try:
        # forzar el OCR al inicio, no durante el chat
        get_vectorstore() 
        print("✅ Cerebro cargado y listo.")
    except Exception as e:
        print(f"❌ Error cargando conocimientos: {e}")
        return
    # ---------------------------------------------

    print("Escribe 'salir' para terminar.\n")
    
    try:
        app = create_graph()
    except Exception as e:
        print(f"Error crítico iniciando la app: {e}")
        return

    booking_memory = {}
    chat_history = []

    while True:
        try:
            user_input = input(f"{GREEN}Tú: {RESET}")
            if user_input.lower() in ["salir", "exit", "quit"]:
                print("¡Hasta luego!")
                break
            
            # preparar el input para el grafo
            # importante pasar booking_memory para que el agente recuerde los datos entre turnos
            initial_state = {
                "messages": chat_history + [HumanMessage(content=user_input)],
                "booking_info": booking_memory,
                "next_step": ""
            }

            # ejecutar el grafo
            print(f"{BLUE}VetCare AI pensando...{RESET}")
            result = app.invoke(initial_state)
            
            # extraer la respuesta final y el estado actualizado
            last_message = result["messages"][-1].content
            booking_memory = result.get("booking_info", {}) # guardar lo que aprendió el agente para la memoria
            
            chat_history = result["messages"] # actualizar historial completo
            
            print(f"{BLUE}VetCare AI:{RESET} {last_message}\n")

        except KeyboardInterrupt:
            print("\nSalida forzada.")
            break
        except Exception as e:
            print(f"Ocurrió un error: {e}")

if __name__ == "__main__":
    main()