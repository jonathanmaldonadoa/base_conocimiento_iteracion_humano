# Helpdesk 2.0 con RAG

Aplicación web de soporte construida con Streamlit. Clasifica consultas, busca
respuestas en una base de conocimiento mediante recuperación aumentada por
generación (RAG) y deriva a un agente humano los casos que requieren atención.

## Funcionalidades

- Creación y seguimiento de tickets durante la sesión de Streamlit.
- Búsqueda semántica en documentos Markdown con ChromaDB.
- Generación de respuestas y clasificación con modelos de OpenAI a través de
  LangChain y LangGraph.
- Escalado a una persona, con la opción de enviar una respuesta manual o usar la
  sugerencia de RAG.
- Historial del flujo y visualización de fuentes y nivel de confianza.

## Requisitos

- Python 3.12 o posterior.
- Una clave de API de OpenAI con acceso a los modelos configurados en
  [config.py](./config.py).

## Instalación

Desde la carpeta del proyecto, crea y activa un entorno virtual en PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instala las dependencias declaradas en `pyproject.toml`:

```powershell
python -m pip install --upgrade pip
python -m pip install -e .
```

Configura la clave de OpenAI en la sesión actual de PowerShell:

```powershell
$env:OPENAI_API_KEY = "tu-clave-de-openai"
```

No guardes la clave en el código ni la subas al repositorio.

## Preparar la base de conocimiento

La aplicación utiliza los archivos Markdown de `docs/`. Para crear o reconstruir
la base vectorial manualmente:

```powershell
python setup_rag.py
```

Este comando reconstruye `chroma_db` a partir de los documentos y ejecuta
búsquedas de prueba. También puedes iniciar la aplicación y usar los botones de
configuración de RAG en la barra lateral.

Los documentos de conocimiento incluidos son:

- `docs/faq.md`
- `docs/guia_resolucion_problemas.md`
- `docs/manual_usuario.md`

Puedes añadir otros archivos Markdown a `docs/` y reconstruir la base para que
sean indexados.

## Ejecutar la aplicación

Inicia Streamlit desde la carpeta del proyecto:

```powershell
streamlit run app.py
```

Abre la dirección local que muestra Streamlit en la terminal. Escribe el usuario
y la consulta para crear un ticket. Si el caso se escala, introduce una respuesta
como agente o aplica la sugerencia generada por RAG.

## Configuración y datos

Los modelos y las rutas de ChromaDB y los documentos se definen en
[config.py](./config.py). Las rutas de este proyecto están configuradas para su
ubicación local actual; si clonas o mueves el proyecto, actualízalas y ejecuta
los comandos desde la carpeta del proyecto.

ChromaDB guarda el índice en `chroma_db/`. LangGraph utiliza `helpdesk.db` como
checkpoint SQLite. Los tickets visibles en la interfaz se mantienen en la sesión
de Streamlit y no se conservan al cerrar dicha sesión.

## Estructura

```text
help_desk/
├── app.py                         # Interfaz Streamlit
├── config.py                      # Modelos y rutas
├── graph.py                       # Flujo helpdesk y escalado humano
├── rag_system.py                  # Recuperación y generación de respuestas
├── setup_rag.py                   # Indexación de documentos en ChromaDB
├── prompts.py                     # Plantillas de prompts
├── docs/                           # Base de conocimiento Markdown
├── chroma_db/                      # Índice vectorial persistente
├── pyproject.toml                  # Metadatos y dependencias
└── README.md
```
