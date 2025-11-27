import pytest
from langchain_core.messages import HumanMessage
from src.agents.rag import rag_node
from src.core.vectorstore import get_vectorstore

# preguntas presentes en los documentos dentro de data/info-mascotas
qa_test_cases = [
    # dentro de data/info-mascotas/guia-cuidado.md
    ("¿Qué suplementos naturales recomiendas?", "calming"), 
    
    # dentro de data/info-mascotas/Tenencia-Responsable.pdf
    ("¿Cuál es la única vacuna obligatoria para caninos y felinos?", "antirrábica"),
    
    # dentro de data/info-mascotas/Tenencia-Responsable.pdf
    ("¿Quién transmite la Toxocariasis?", "perro"), 
]

@pytest.fixture(scope="module")
def setup_vectorstore():
    """Singleton del vectorstore para toda la sesión de pruebas"""
    store = get_vectorstore()
    return store

@pytest.mark.parametrize("question, expected_keyword", qa_test_cases)
def test_rag_retrieval_and_answer(setup_vectorstore, question, expected_keyword):
    """
    Prueba de integración parametrizada.
    Ejecuta el RAG con diferentes preguntas y verifica la calidad de la respuesta.
    """
    if not setup_vectorstore:
        pytest.skip("No hay documentos ni VectorStore cargado.")

    # 1. Preparar estado
    mock_state = {
        "messages": [HumanMessage(content=question)],
        "booking_info": {},
        "next_step": ""
    }
    
    # 2. Ejecutar nodo
    print(f"\n\n{'='*50}")
    print(f"🔎 PROBANDO PREGUNTA: '{question}'")
    result = rag_node(mock_state)
    
    # 3. Validaciones
    last_message = result["messages"][0]
    respuesta_texto = last_message.content
    respuesta_lower = respuesta_texto.lower()
    
    # --- LOGS PARA VER EN CONSOLA ---
    print(f"🤖 RESPUESTA AGENTE:\n{respuesta_texto}")
    print(f"{'='*50}\n")

    # Aserciones básicas
    assert len(respuesta_texto) > 20, "La respuesta es sospechosamente corta."
    
    # Aserción de seguridad (No debe decir que no sabe, a menos que sea la intención)
    assert "no tengo información" not in respuesta_lower, \
        f"El RAG no encontró info para: '{question}'"
        
    # Aserción de contenido (verifica que mencione la palabra clave esperada)
    # Esto asegura que no está alucinando sobre otro tema
    assert expected_keyword.lower() in respuesta_lower, \
        f"Se esperaba la palabra clave '{expected_keyword}' en la respuesta, pero no se encontró."