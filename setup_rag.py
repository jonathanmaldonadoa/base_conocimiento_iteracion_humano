# Este script prepara y comprueba la base de conocimiento que utiliza un sistema RAG
# de soporte técnico. Carga los archivos Markdown del directorio configurado, añade a
# cada documento metadatos como el nombre de archivo, su categoría y un identificador,
# y divide el contenido en fragmentos de hasta 1000 caracteres con una superposición
# de 200 caracteres. Luego genera embeddings con el modelo de OpenAI configurado y
# almacena los fragmentos en una colección Chroma persistente. Si se solicita reconstruir
# la base, elimina primero el directorio de Chroma existente; de lo contrario, puede
# reutilizar el vectorstore disponible. Al ejecutarse como programa principal, fuerza la
# reconstrucción y prueba búsquedas de similitud con consultas de ejemplo, mostrando los
# fragmentos más relevantes y sus metadatos. La configuración y las claves necesarias
# se obtienen del módulo config.

import hashlib
from typing import List
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import * 

class DocumentProcessor:
    """Procesador de documentos para el sistema RAG."""
    
    def __init__(self, docs_path: str = "docs", chroma_path: str = "./chroma_db"):
        self.docs_path = Path(docs_path)
        self.chroma_path = Path(chroma_path)
        self.embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, # Tamaño máximo de cada chunk
            chunk_overlap=200, # Tamaño de superposición entre chunks
            length_function=len, # Función para medir la longitud del texto
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""] # Separadores para dividir el texto
        )
        
    def load_documents(self) -> List[Document]:
        """Carga documentos markdown del directorio docs."""
        print(f"📚 Cargando documentos desde {self.docs_path}")
        
        # Cargar archivos markdown
        loader = DirectoryLoader(
            str(self.docs_path),
            glob="*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        
        documents = loader.load()
        
        # Enriquecer metadatos
        for doc in documents:
            filename = Path(doc.metadata["source"]).stem # Obtener el nombre del archivo sin extensión
            # Agregar metadatos adicionales
            doc.metadata.update({
                "filename": filename,
                "doc_type": self._get_doc_type(filename),
                "doc_id": self._generate_doc_id(doc.page_content)
            })
        
        print(f"✅ Cargados {len(documents)} documentos")
        return documents
    
    def _get_doc_type(self, filename: str) -> str:
        """Determina el tipo de documento basado en el nombre."""
        if "faq" in filename.lower():
            return "faq"
        elif "manual" in filename.lower():
            return "manual"
        elif "troubleshooting" in filename.lower():
            return "troubleshooting"
        else:
            return "general"
    
    def _generate_doc_id(self, content: str) -> str:
        """Genera un ID único para el documento."""
        return hashlib.md5(content.encode()).hexdigest()[:8]

    # definir la función split_documents que divide los documentos en chunks más pequeños
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Divide documentos en chunks más pequeños."""
        print("✂️  Dividiendo documentos en chunks...")
        
        chunks = self.text_splitter.split_documents(documents)
        
        # Agregar metadatos de chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": i,
                "chunk_size": len(chunk.page_content)
            })
        
        print(f"✅ Creados {len(chunks)} chunks")
        return chunks

    # definir la función create_vectorstore que crea o actualiza el vectorstore de ChromaDB
    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """Crea el vectorstore con ChromaDB."""
        print("🔄 Creando vectorstore con ChromaDB...")

        # Actualizar la colección en el lugar para no borrar archivos que ChromaDB
        # puede mantener abiertos, especialmente en Windows.
        vectorstore = Chroma(
            persist_directory=str(self.chroma_path),
            embedding_function=self.embeddings,
            collection_name="helpdesk_knowledge"
        )
        existing_ids = vectorstore.get()["ids"]
        if existing_ids:
            vectorstore.delete(ids=existing_ids)
        vectorstore.add_documents(documents)
        
        print(f"✅ Vectorstore creado en {self.chroma_path}")
        print(f"📊 Total de vectores: {len(documents)}")
        
        return vectorstore

    # definir la función load_existing_vectorstore que carga el vectorstore existente
    def load_existing_vectorstore(self) -> Chroma:
        """Carga vectorstore existente."""
        if not self.chroma_path.exists():
            raise FileNotFoundError(f"Vectorstore no encontrado en {self.chroma_path}")
        
        vectorstore = Chroma(
            persist_directory=str(self.chroma_path),
            embedding_function=self.embeddings,
            collection_name="helpdesk_knowledge"
        )
        
        return vectorstore

    # definir la función setup_rag_system que configura el sistema RAG completo
    def setup_rag_system(self, force_rebuild: bool = False):
        """Configura el sistema RAG completo."""
        print("🚀 Configurando sistema RAG...")
        
        # Verificar si ya existe y no forzar rebuild
        if self.chroma_path.exists() and not force_rebuild:
            print("📦 Vectorstore existente encontrado")
            return self.load_existing_vectorstore()
        
        # Cargar y procesar documentos
        documents = self.load_documents()
        if not documents:
            print("⚠️  No se encontraron documentos para procesar")
            return None
        
        # Dividir documentos
        chunks = self.split_documents(documents)
        
        # Crear vectorstore
        vectorstore = self.create_vectorstore(chunks)
        
        print("✅ Sistema RAG configurado exitosamente")
        return vectorstore

    # definir la función test_search que realiza una búsqueda de prueba en el vectorstore
    def test_search(self, vectorstore: Chroma, query: str = "resetear contraseña"):
        """Prueba la funcionalidad de búsqueda."""
        print(f"\n🔍 Probando búsqueda: '{query}'")

        # Realizar búsqueda de similitud en 3 fragmentos más relevantes
        results = vectorstore.similarity_search(query, k=3)
        
        for i, doc in enumerate(results, 1):
            print(f"\n📄 Resultado {i}:")
            print(f"Tipo: {doc.metadata.get('doc_type', 'unknown')}")
            print(f"Archivo: {doc.metadata.get('filename', 'unknown')}")
            print(f"Contenido: {doc.page_content[:200]}...")
        
        return results


def main():
    """Función principal para configurar RAG."""
    print("🎧 Configuración RAG - Helpdesk 2.0")
    print("=" * 40)
    
    # Configurar procesador
    processor = DocumentProcessor(docs_path=DOCS_PATH, chroma_path=CHROMADB_PATH)
    
    # Configurar sistema RAG
    vectorstore = processor.setup_rag_system(force_rebuild=True)
    
    if vectorstore:
        # Probar búsquedas
        test_queries = [
            "resetear contraseña",
            "error 500",
            "cancelar suscripción",
            "aplicación lenta"
        ]
        
        for query in test_queries:
            processor.test_search(vectorstore, query)
    
    print("\n✅ Configuración completada")


if __name__ == "__main__":
    main()