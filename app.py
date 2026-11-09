# Esta aplicación web de Streamlit ofrece una interfaz para gestionar consultas de
# soporte como tickets y resolverlas con el flujo Helpdesk basado en LangGraph y RAG.
# Al iniciar, crea el grafo y verifica si existe la base vectorial Chroma; desde la
# barra lateral permite configurarla o reconstruirla, revisar los tickets activos y
# limpiar la lista de tickets de la sesión. El formulario permite ingresar un usuario
# y una consulta, ofrece ejemplos, genera un identificador de ticket y ejecuta el grafo
# mostrando el progreso. Para cada ticket presenta su categoría, confianza y fuentes,
# junto con la respuesta generada. Si el grafo solicita intervención humana, permite
# ingresar una respuesta o aplicar como respuesta la sugerencia de RAG, y luego reanuda
# el flujo. Al pie muestra estadísticas de tickets totales, resueltos mediante RAG o
# por una persona, y pendientes. El estado de la interfaz y los tickets se mantienen
# en la sesión de Streamlit.

import streamlit as st
import uuid
from graph import crear_helpdesk, HelpdeskState
from setup_rag import DocumentProcessor
from datetime import datetime
import os

# Configuración de página
st.set_page_config(
    page_title="Helpdesk 2.0 con RAG",
    page_icon="🎧",
    layout="wide"
)

# Inicializar 

# función para inicializar el estado de la sesión de Streamlit con el grafo y los tickets
if "helpdesk" not in st.session_state:
    st.session_state.helpdesk = crear_helpdesk()
    st.session_state.tickets = {}

def verificar_rag_setup():
    """Verifica si el sistema RAG está configurado."""
    processor = DocumentProcessor()
    return processor.chroma_path.exists()

def configurar_rag():
    """Configura el sistema RAG."""
    with st.spinner("🔧 Configurando sistema RAG..."):
        processor = DocumentProcessor()
        vectorstore = processor.setup_rag_system(force_rebuild=True)
        return vectorstore is not None

def crear_ticket_id():
    """Genera un ID único para el ticket."""
    return f"TK-{uuid.uuid4().hex[:6].upper()}"

def procesar_consulta(consulta: str, ticket_id: str):
    """Procesa una consulta nueva."""
    estado_inicial = HelpdeskState(
        consulta=consulta,
        categoria="",
        respuesta_rag=None,
        confianza=0.0,
        fuentes=[],
        requiere_humano=False,
        respuesta_humano=None,
        respuesta_final=None,
        historial=[]
    )
    
    config = {"configurable": {"thread_id": ticket_id}}
    
    # Procesar con streaming
    historial_procesamiento = []
    
    try:
        for chunk in st.session_state.helpdesk.stream(
            estado_inicial, 
            config=config, 
            stream_mode="updates"
        ):
            for nodo, salida in chunk.items():
                if "historial" in salida and salida["historial"]:
                    historial_procesamiento.extend(salida["historial"])
        
        # Obtener estado final
        estado_final = st.session_state.helpdesk.get_state(config)
        
        return estado_final.values, historial_procesamiento, config
        
    except Exception as e:
        st.error(f"Error procesando consulta: {str(e)}")
        return None, [], None

def main():
    """Aplicación principal."""
    st.title("🎧 Helpdesk 2.0 con RAG + ChromaDB")
    st.markdown("*Sistema inteligente con LangGraph y búsqueda vectorial*")
    
    # Verificar configuración RAG
    rag_configurado = verificar_rag_setup()
    
    # Sidebar con información y configuración
    with st.sidebar:
        st.header("📊 Panel de Control")
        st.metric("Tickets Activos", len(st.session_state.tickets))
        
        # Estado del sistema RAG
        st.subheader("🔍 Estado RAG")
        if rag_configurado:
            st.success("✅ ChromaDB configurado")
        else:
            st.warning("⚠️ RAG no configurado")
            if st.button("🚀 Configurar RAG"):
                if configurar_rag():
                    st.success("✅ RAG configurado exitosamente")
                    st.rerun()
                else:
                    st.error("❌ Error configurando RAG")
        
        st.subheader("🔄 Flujo del Sistema")
        st.text("""
1. 📝 Usuario envía consulta
2. 🤖 Clasificación automática
3. 🔍 Búsqueda vectorial RAG
4. 📊 Evaluación de confianza
5. 👨‍💼 Escalado si es necesario
6. ✅ Respuesta final
        """)
        
        st.subheader("⚙️ Configuración")
        if st.button("🔄 Reconfigurar RAG"):
            if configurar_rag():
                st.success("✅ RAG reconfigurado")
                st.rerun()
        
        if st.button("🗑️ Limpiar Tickets"):
            st.session_state.tickets = {}
            st.rerun()
    
    if not rag_configurado:
        st.warning("⚠️ El sistema RAG no está configurado. Usa el botón en la barra lateral para configurarlo.")
        return
    
    # Área principal
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Nueva Consulta")
        
        # Ejemplos de consultas
        with st.expander("💡 Ejemplos de consultas"):
            ejemplos = [
                "No puedo resetear mi contraseña",
                "Error 500 en la aplicación",
                "¿Cómo cancelo mi suscripción?",
                "La aplicación va muy lenta",
                "Problemas con la facturación"
            ]
            for ejemplo in ejemplos:
                if st.button(f"📋 {ejemplo}", key=f"ej_{ejemplo}"):
                    st.session_state.consulta_ejemplo = ejemplo
        
        with st.form("nueva_consulta"):
            usuario = st.text_input("👤 Usuario", placeholder="tu@email.com")
            
            consulta_inicial = st.session_state.get("consulta_ejemplo", "")
            consulta = st.text_area(
                "💬 Descripción del problema",
                value=consulta_inicial,
                placeholder="Describe tu consulta o problema aquí...",
                height=100
            )
            
            submitted = st.form_submit_button("🚀 Enviar Consulta")
            
            if submitted and consulta.strip():
                # Limpiar ejemplo usado
                if "consulta_ejemplo" in st.session_state:
                    del st.session_state.consulta_ejemplo
                
                ticket_id = crear_ticket_id()
                
                with st.spinner("🔄 Procesando consulta..."):
                    resultado, historial, config = procesar_consulta(consulta, ticket_id)
                
                if resultado:
                    # Guardar ticket
                    st.session_state.tickets[ticket_id] = {
                        "usuario": usuario,
                        "consulta": consulta,
                        "resultado": resultado,
                        "historial": historial,
                        "config": config,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    }
                    
                    st.success(f"✅ Ticket {ticket_id} creado")
                    st.rerun()
    
    with col2:
        st.subheader("🎫 Tickets Recientes")
        
        if not st.session_state.tickets:
            st.info("No hay tickets activos")
        else:
            for ticket_id, ticket_data in reversed(list(st.session_state.tickets.items())):
                with st.expander(f"🎫 {ticket_id} - {ticket_data['timestamp']}", expanded=True):
                    st.markdown(f"**👤 Usuario:** {ticket_data['usuario']}")
                    st.markdown(f"**💬 Consulta:** {ticket_data['consulta'][:100]}...")
                    
                    resultado = ticket_data['resultado']
                    
                    # Mostrar progreso del procesamiento
                    st.subheader("🔄 Procesamiento:")
                    for paso in ticket_data['historial']:
                        st.text(paso)
                    
                    # Información de categorización
                    if resultado.get('categoria'):
                        st.markdown(f"**📂 Categoría:** {resultado['categoria']}")
                    
                    # Información del RAG
                    if resultado.get('confianza', 0) > 0:
                        confidence = resultado['confianza']
                        st.markdown(f"**🎯 Confianza RAG:** {confidence:.2f}")
                        
                        # Barra de progreso visual
                        progress_color = "green" if confidence >= 0.65 else "orange" if confidence >= 0.4 else "red"
                        st.progress(confidence)
                        
                        # Mostrar fuentes consultadas
                        if resultado.get('fuentes'):
                            st.markdown(f"**📚 Fuentes:** {', '.join(resultado['fuentes'])}")
                    
                    # Human-in-the-loop
                    if resultado.get('requiere_humano') and not resultado.get('respuesta_final'):
                        st.warning("👨‍💼 Requiere intervención humana")
                        
                        # Mostrar contexto para el agente
                        if resultado.get('respuesta_rag'):
                            with st.expander("📋 Contexto para el agente"):
                                st.text(resultado['respuesta_rag'])
                        
                        respuesta_humano = st.text_area(
                            "✍️ Respuesta del agente:",
                            key=f"respuesta_{ticket_id}",
                            height=100,
                            placeholder="Escribe la respuesta para el usuario..."
                        )
                        
                        col_btn1, col_btn2 = st.columns(2)
                        
                        with col_btn1:
                            if st.button(f"💾 Enviar Respuesta", key=f"btn_{ticket_id}"):
                                if respuesta_humano.strip():
                                    # Actualizar estado con respuesta humana
                                    config = ticket_data['config']
                                    st.session_state.helpdesk.update_state(
                                        config,
                                        {"respuesta_humano": respuesta_humano}
                                    )
                                    
                                    # Continuar procesamiento
                                    for chunk in st.session_state.helpdesk.stream(None, config=config, stream_mode="updates"):
                                        for nodo, salida in chunk.items():
                                            if "historial" in salida and salida["historial"]:
                                                ticket_data['historial'].extend(salida["historial"])
                                    
                                    # Actualizar estado final
                                    estado_final = st.session_state.helpdesk.get_state(config)
                                    ticket_data['resultado'] = estado_final.values
                                    
                                    st.success("✅ Respuesta procesada")
                                    st.rerun()
                                else:
                                    st.warning("⚠️ Escribe una respuesta antes de enviar")
                        
                        with col_btn2:
                            if st.button(f"🔄 Usar RAG", key=f"rag_{ticket_id}"):
                                # Usar la respuesta RAG como base
                                respuesta_rag = resultado.get('respuesta_rag', '')
                                config = ticket_data['config']
                                st.session_state.helpdesk.update_state(
                                    config,
                                    {"respuesta_humano": respuesta_rag}
                                )
                                
                                # Continuar procesamiento
                                for chunk in st.session_state.helpdesk.stream(None, config=config, stream_mode="updates"):
                                    for nodo, salida in chunk.items():
                                        if "historial" in salida and salida["historial"]:
                                            ticket_data['historial'].extend(salida["historial"])
                                
                                estado_final = st.session_state.helpdesk.get_state(config)
                                ticket_data['resultado'] = estado_final.values
                                
                                st.success("✅ Respuesta RAG aplicada")
                                st.rerun()
                    
                    # Respuesta final
                    elif resultado.get('respuesta_final'):
                        st.success("✅ Ticket Resuelto")
                        st.markdown("**💬 Respuesta:**")
                        
                        # Formatear respuesta con fuentes
                        respuesta = resultado['respuesta_final']
                        st.info(respuesta)
                        
                        # Métricas de resolución
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.metric("🎯 Confianza", f"{resultado.get('confianza', 0):.2f}")
                        with col_m2:
                            st.metric("🔍 Fuentes", len(resultado.get('fuentes', [])))
                        with col_m3:
                            resolucion = "RAG" if not resultado.get('requiere_humano') else "Humano"
                            st.metric("🤖 Resuelto por", resolucion)
    
    # Footer con estadísticas
    st.markdown("---")
    if st.session_state.tickets:
        # Calcular estadísticas
        total_tickets = len(st.session_state.tickets)
        resueltos_rag = sum(1 for t in st.session_state.tickets.values() 
                           if t['resultado'].get('respuesta_final') and not t['resultado'].get('requiere_humano'))
        resueltos_humano = sum(1 for t in st.session_state.tickets.values() 
                              if t['resultado'].get('respuesta_final') and t['resultado'].get('requiere_humano'))
        pendientes = total_tickets - resueltos_rag - resueltos_humano
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            st.metric("📊 Total Tickets", total_tickets)
        with col_stat2:
            st.metric("🤖 Resueltos por RAG", resueltos_rag)
        with col_stat3:
            st.metric("👨‍💼 Resueltos por Humano", resueltos_humano)
        with col_stat4:
            st.metric("⏳ Pendientes", pendientes)
    
    st.markdown(
        """
        <div style='text-align: center'>
            <small>🚀 Powered by LangGraph | 🔍 ChromaDB | 🔄 Streaming | 💾 Checkpointing | 👨‍💼 Human-in-the-Loop</small>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()