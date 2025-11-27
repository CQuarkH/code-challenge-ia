# VetCare AI - Asistente Virtual Veterinario - Elías Currihuil

**VetCare AI** es un sistema conversacional multi-agente diseñado para clínicas veterinarias. Actúa como primer punto de contacto para resolver dudas médicas generales y gestionar el agendamiento de citas, orquestado mediante **LangGraph** y potenciado por modelos de OpenAI.

---

## Guía de Inicio Rápido (Ejecución)

Sigue estos pasos para levantar el agente y ejecutar la suite de pruebas en tu entorno local.

### 1\. Prerrequisitos

- Python 3.10 o superior.
- Una API Key de OpenAI activa.

### 2\. Instalación

```bash
# 1. Clonar el repositorio y entrar al directorio
git clone https://github.com/CQuarkH/code-challenge-ia.git
cd code-challenge-ia

# 2. Crear y activar entorno virtual
python -m venv venv

# En Windows:
venv\Scripts\activate

# En Mac/Linux:
source venv/bin/activate

# 3. Instalar dependencias (Incluye motor OCR para lectura de PDFs)
pip install -r requirements.txt
```

### 3\. Configuración (.env)

Crea un archivo llamado `.env` en la raíz del proyecto y define tu llave de API:

```env
OPENAI_API_KEY=sk....
```

### 4\. Ejecutar la Aplicación

Para iniciar la interfaz de chat en consola (CLI):

```bash
python main.py
```

_Nota: Para mantener la interfaz limpia, los logs técnicos de depuración se escriben en `logs/app.log`._

### 5\. Ejecutar Tests

El proyecto cuenta con una cobertura de pruebas automatizadas con `pytest`:

```bash
pytest
```

**Qué se evalúa en los tests:**

- **Unitarios:** Clasificación de intenciones del Router y patrones Singleton.
- **Integración (RAG):** Capacidad de leer PDFs escaneados y responder preguntas médicas.
- **Flujo (Booking):** Capacidad del agente para recordar datos (Slot Filling) turno a turno.

---

## 🏗 Arquitectura y Patrones de Diseño

El sistema implementa una arquitectura modular basada en tres patrones de diseño fundamentales para garantizar escalabilidad y mantenibilidad.

### 1\. Patrón Strategy (Estrategia)

- **Ubicación:** Directorio `src/agents/`.
- **Implementación:** Cada módulo (`rag.py`, `booking.py`, `router.py`) encapsula una familia de algoritmos intercambiables.
- **Uso:** El `Router` evalúa el contexto y selecciona dinámicamente qué estrategia ejecutar. Esto permite modificar la lógica de agendamiento sin riesgo de romper la lógica de consultas médicas.

### 2\. Patrón State (Estado)

- **Ubicación:** `src/state.py` y Orquestación LangGraph.
- **Implementación:** Se define un objeto `AgentState` (TypedDict) que actúa como una pizarra compartida (_Blackboard_).
- **Uso:** Permite la persistencia de datos (como el nombre de la mascota o el historial de conversación) a través de los diferentes nodos del grafo, transformando el chatbot en una Máquina de Estados Finitos.

### 3\. Patrón Singleton (Instancia Única)

- **Ubicación:** `src/core/`.
- **Implementación:** Módulos `llm.py` y `vectorstore.py`.
- **Uso:** Garantiza que objetos pesados como la conexión a OpenAI o la carga de la base de datos vectorial (ChromaDB) se instancien una sola vez en el ciclo de vida de la aplicación, optimizando memoria y latencia.

---

## 📝 Registro de Decisiones de Arquitectura (ADRs)

### ADR-001: Orquestación con LangGraph vs. LangChain Chains

- **Contexto:** El flujo de agendamiento de citas es cíclico (Solicitar dato -\> Validar -\> Solicitar siguiente dato -\> Error -\> Repetir).
- **Decisión:** Se utilizó **LangGraph**.
- **Justificación:** Las cadenas tradicionales (Chains) son DAGs (Grafos Acíclicos Dirigidos) y no manejan bien los bucles. LangGraph permite definir flujos cíclicos y persistencia de memoria nativa, ideal para el agente de "Slot Filling".

### ADR-002: Base Vectorial ChromaDB

- **Contexto:** Necesidad de almacenamiento de embeddings para RAG.
- **Decisión:** Se utilizó **ChromaDB** (modo local).
- **Justificación:** Facilita el despliegue del prototipo sin necesidad de contenedores Docker adicionales. Permite persistencia en disco simple.

### ADR-003: Embeddings de OpenAI (`text-embedding-3-small`)

- **Contexto:** Búsqueda semántica en documentos veterinarios.
- **Decisión:** Uso de embeddings de OpenAI sobre modelos locales (HuggingFace).
- **Justificación:** Mayor fidelidad semántica en español y mejor rendimiento general para distinguir matices en preguntas médicas complejas.

### ADR-004: Modelo GPT-3.5-Turbo

- **Contexto:** Inferencia y generación de texto.
- **Decisión:** Uso de `gpt-3.5-turbo`.
- **Justificación:** Ofrece el mejor equilibrio costo-beneficio. Su latencia es lo suficientemente baja para una experiencia de chat fluida, y su capacidad de razonamiento es suficiente para la clasificación de intenciones y extracción de entidades.

---

## 🛠 Desafíos Técnicos y Soluciones

### El Problema del "PDF Ciego" (RAG + OCR)

Durante el desarrollo, el módulo RAG fallaba al responder preguntas contenidas en `Tenencia-Responsable.pdf`.

- **Diagnóstico:** El PDF no contenía capa de texto seleccionable; estaba compuesto íntegramente por imágenes escaneadas. Las librerías estándar (`pypdf`) extraían cadenas vacías.
- **Solución:** Se implementó un pipeline de ingesta híbrido en `src/core/vectorstore.py`.
  1.  El sistema intenta leer el PDF.
  2.  Si detecta páginas con bajo conteo de caracteres, activa un motor **OCR (RapidOCR + ONNX)**.
  3.  Convierte la página a imagen en memoria, extrae el texto y genera el documento vectorial.
      _Resultado:_ El sistema ahora puede "leer" documentos escaneados transparentemente.

### Persistencia en Agente de Citas (Booking Agent)

Para lograr que el agente recordara el nombre de la mascota mencionado 3 turnos atrás, se utilizó la memoria del grafo (`booking_info` en `AgentState`). El nodo de booking utiliza **Structured Output** de OpenAI para extraer entidades JSON del chat y actualizar este estado incrementalmente, sin necesidad de pedir todos los datos de nuevo.

---

## 📂 Estructura del Proyecto

```text
code-challenge-ia/
├── data/                  # Base de conocimientos (PDFs, TXT, MD)
├── logs/                  # Archivos de log generados en tiempo de ejecución
├── src/
│   ├── agents/            # Lógica de Negocio (Strategy Pattern)
│   │   ├── booking.py     # Agente de Citas (Slot Filling)
│   │   ├── rag.py         # Agente de Conocimiento
│   │   └── router.py      # Clasificador de Intención
│   ├── core/              # Infraestructura (Singleton Pattern)
│   │   ├── llm.py         # Cliente OpenAI
│   │   ├── vectorstore.py # Ingesta RAG + OCR
│   │   └── logger.py      # Configuración de logs
│   ├── graph/             # Orquestación
│   │   └── workflow.py    # Grafo LangGraph
│   ├── tools/             # Herramientas (Mock APIs)
│   └── state.py           # Definición del Estado (TypedDict)
├── tests/                 # Pruebas Automatizadas (Pytest)
├── main.py                # Punto de entrada (CLI)
└── requirements.txt       # Dependencias del proyecto
```
