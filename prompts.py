RAG_TEMPLATE = """Eres un asistente de helpdesk experto. Tu tarea es generar múltiples 
versiones de la consulta del usuario para recuperar documentos relevantes de una 
base de conocimiento de soporte técnico.

Genera 3 versiones diferentes de la consulta original, considerando:
- Sinónimos técnicos
- Diferentes formas de expresar el mismo problema
- Variaciones en terminología de helpdesk

Consulta original: {question}

Versiones alternativas:"""

RESPUESTA_LLM = """Eres un asistente de helpdesk experto. Responde a la consulta del usuario 
basándote únicamente en el contexto proporcionado de la base de conocimiento.

Instrucciones:
- Proporciona una respuesta clara, directa y útil
- Si el contexto no contiene información suficiente, dilo claramente
- Mantén un tono profesional pero amigable
- No inventes información que no esté en el contexto

Contexto de la base de conocimiento:
{contexto}

Consulta del usuario: {consulta}

Respuesta:"""

LANGGRAPH_PROMPT_CALIFICACION =   """Analiza esta consulta de helpdesk y decide si puede responderse automáticamente o necesita escalado:

CONSULTA DEL USUARIO: {consulta}

INFORMACIÓN ENCONTRADA EN LA BASE DE CONOCIMIENTO:
{contexto_rag}

CONFIANZA DE LA BÚSQUEDA: {confianza}

Criterios de decisión:
- AUTOMATICO: Si la información de la BD responde completamente la consulta, 
  tiene buena confianza (>0.6), y es un tema estándar/procedimiento conocido
  
- ESCALADO: Si la información es insuficiente, confianza baja, problema complejo/único,
  requiere acceso a sistemas internos, o involucra decisiones de negocio

Responde solo con "automatico" o "escalado" y una breve justificación (máximo 20 palabras):"""