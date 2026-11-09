# Este módulo implementa el flujo de atención de consultas de un sistema helpdesk con
# LangGraph. El estado mantiene la consulta, la respuesta y confianza obtenidas del
# sistema RAG, las fuentes consultadas, la categoría de atención, las respuestas humana
# y final, y un historial acumulable de las etapas. Primero busca información relevante
# en la base vectorial; luego, un modelo de lenguaje clasifica la consulta como
# automática o escalada usando el contexto recuperado y su confianza. Si se puede
# responder automáticamente, prepara una respuesta con la información RAG y sus fuentes.
# Si requiere atención humana, marca el caso como escalado y espera una respuesta del
# agente; al reanudar el flujo con esa respuesta, la establece como respuesta final.
# El grafo se compila con un checkpoint SQLite en helpdesk.db y una interrupción antes
# de procesar la respuesta humana, para permitir esa intervención. La función
# crear_helpdesk crea el sistema y devuelve el grafo compilado.

from typing import TypedDict, Optional, List, Annotated, Dict, Any
from operator import add
from langchain_openai import ChatOpenAI
from rag_system import VectorRAGSystem
from config import *
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from prompts import *

# Definicion del Estado
class HelpdeskState(TypedDict):
    consulta: str
    categoria: str # "automatica" o "escalada"
    respuesta_rag: Optional[str]
    confianza: float
    fuentes: List[str]
    contexto_rag: Optional[str]
    requiere_humano: bool
    respuesta_humano: Optional[str]
    respuesta_final: Optional[str]
    historial: Annotated[List[str], add]

class HelpdeskGraph:
    """Grafo del sistema Helpdesk."""

    def __init__(self):
        self.llm = ChatOpenAI(model=LANGGRAPH_QUERY_MODEL, temperature=LANGGRAPH_MODEL_TEMPERATURE)# inicializa el modelo de lenguaje para LangGraph
        self.rag = VectorRAGSystem(chroma_path=CHROMADB_PATH) # inicializa el sistema RAG con la ruta de ChromaDB
        self.graph = None

    # definir la función procesar_rag que busca el contexto de la consulta utilizando el sistema RAG
    def procesar_rag(self, state):
        """Busca el contexto de la consulta utilizando el sistema RAG."""
        consulta = state['consulta'] # obtiene la consulta del estado
        resultado = self.rag.buscar(consulta) # busca la consulta en el sistema RAG y obtiene el resultado
        return {
            "respuesta_rag": resultado["respuesta"],
            "confianza": resultado["confianza"],
            "fuentes": resultado["fuentes"],
            "contexto_rag": resultado["respuesta"],
            "historial": [
                f"RAG ejecutado con MultiQueryRetriever",
                f"Confianza: {resultado["confianza"]}",
                f"Fuentes consultadas: {len(resultado['fuentes'])}"
            ]
        }

    # definir la función clasificar_con_contexto que clasifica la consulta para decidir si responder automáticamente o escalar, usando el contexto del RAG
    def clasificar_con_contexto(self, state):
        """Clasifica la consulta para responder automaticamente o escalar con el contexto del RAG."""
        consulta = state['consulta']
        contexto_rag = state.get('contexto_rag', '')
        confianza = state.get('confianza', 0)

        prompt = ChatPromptTemplate.from_template(LANGGRAPH_PROMPT_CALIFICACION)
    
        try:
            response = self.llm.invoke(prompt.format(
                consulta=consulta,
                contexto_rag=contexto_rag,
                confianza=confianza
            ))

            content = response.content.strip().lower()

            if "automatico" in content or "automático" in content:
                categoria = "automatico"
            elif "escalado" in content:
                categoria = "escalado"
            else:
                categoria = "automatico" if confianza >= 0.60 else "escalado"

            return {
                "categoria": categoria,
                "historial": [
                    f"Clasificación con contexto: {categoria}",
                    f"Justificación: {response.content}"
                ]
            }
        except Exception as e:
            categoria = "automatico" if confianza >= 0.60 else "escalado"
            return {
                "categoria": categoria,
                "historial": [f"Error en la clasificación, usando confianza: {confianza}"]
            }

    # definir la función generar_respuesta_final que genera la respuesta final del sistema al ticket del usuario, usando la respuesta del RAG y las fuentes consultadas
    def preparar_escalado(self, state):
        """Preaparar el escalado a un humano."""
        return {
            "requiere_humano": True,
            "historial": ["Escalado a agente humano - esperando intervención."]
        }

    # definir la función procesar_respuesta_humano que procesa la respuesta del humano y la establece como respuesta final
    def procesar_respuesta_humano(self, state):
        """Procesa la respueta del humano."""
        respuesta_humano = state.get("respuesta_humano", "")

        if respuesta_humano:
            return {
                "respuesta_final": respuesta_humano,
                "historial": ["Agente humano proporcionó respuesta."]
            }
        
        return {
            "historial": ["Esperando respuesta del agente humano"]
        }

    # definir la función generar_respuesta_final que genera la respuesta final del sistema al ticket del usuario, usando la respuesta del RAG y las fuentes consultadas
    def generar_respuesta_final(self, state):
        """Genera la respueta final del sistema al ticket del usuario."""
        if state.get("respuesta_final"):
            return {
                "historial": ["Respuesta final proporcionada por agente humano."]
            }
        
        # Si no hay respueta final, la generamos con IA (usamos la respueta del sistema RAG)
        respuesta_rag = state.get("respuesta_rag", "")
        fuentes = state.get("fuentes", [])

        # Enriquecer respuesta final
        respuesta_final = respuesta_rag
        if fuentes:
            fuentes_texto = ", ".join(fuentes)
            respuesta_final += f"\n\nFuentes consultadas: {fuentes_texto}"

        return {
            "respuesta_final": respuesta_final,
            "historial": ["Respuesta final generada automaticamente."]
        }

    # Funciones de enrutamiento
    def decidir_desde_clasificacion(self, state):
        """Decide hacia donde ir despues de la clasificacion con contexto RAG."""
        categoria = state.get("categoria", "escalado")
        if categoria == "automatico":
            return "respuesta_final"
        else:
            return "escalado"
        
    def decidir_desde_humano(self, state):
        """Decide si continuar o esperar respuesta humana."""
        respuesta_humano = state.get("respuesta_humano", "")

        if respuesta_humano:
            return "procesar_humano"
        else:
            return "esperar"
        
    def crear_grafo(self):
        """Crear el grafo de LangGraph con los nodos y control de flujo."""
        graph = StateGraph(HelpdeskState)

        # Agregar nodos
        graph.add_node("rag", self.procesar_rag)
        graph.add_node("clasificar", self.clasificar_con_contexto)
        graph.add_node("escalado", self.preparar_escalado)
        graph.add_node("respuesta_final", self.generar_respuesta_final)
        graph.add_node("procesar_humano", self.procesar_respuesta_humano)

        # Definir la estructura del grafo
        graph.add_edge(START, "rag")
        graph.add_edge("rag", "clasificar")

        # Edges condicionales del grafo
        graph.add_conditional_edges(
            "clasificar",
            self.decidir_desde_clasificacion,# funcion de enrutamiento que decide si ir a respuesta_final o escalado
            {
                "respuesta_final": "respuesta_final",
                "escalado": "escalado"
            }
        )

        graph.add_conditional_edges(
            "escalado",
            self.decidir_desde_humano,
            {
                "procesar_humano": "procesar_humano",
                "esperar": END # Pausar la ejecucion del grafo hasta que responda el humano
            }
        )

        graph.add_edge("procesar_humano", END)
        graph.add_edge("respuesta_final", END)

        self.graph = graph

        return graph
    
    def compilar(self):
        """Compila el grafo con checkpointer."""
        if not self.graph:
            self.crear_grafo()

        conn = sqlite3.connect("helpdesk.db", check_same_thread=False)

        checkpointer = SqliteSaver(conn)

        compiled = self.graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["procesar_humano"]
        )

        return compiled

# Función para crear el sistema helpdesk y devolver el grafo compilado
def crear_helpdesk():
    helpdesk = HelpdeskGraph()
    return helpdesk.compilar()