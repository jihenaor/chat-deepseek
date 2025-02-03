import streamlit as st
import os
import logging

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

template = """
Eres un asistente especializado en procesar y responder preguntas en español. Tu tarea sera:
1. Analizar el contexto proporcionado en español
2. Entender la pregunta en español
3. Generar una respuesta clara y consisa en español

Si no encuentras la respuesta en el contexto, simplemente indica que no lo sabes.
Limita tu respuesta a tres oraciones maximo.

Pregunta: {question}
Contexto: {Context}
Respuesta (en español):
"""

pdfs_directory = 'pdfs/'
db_directory = 'vectordb'

# Nos aseguramos de que el directorio de la base de datos existe
os.makedirs(db_directory, exist_ok=True)
logging.info(f"Directorio para vectordb: '{db_directory}' verificado o creado.")

# Configuración de embeddings y vector store
model_name = "deepseek-r1:1.5b"
logging.info(f"Inicializando embeddings y vector store usando el modelo: {model_name}")
embeddings = OllamaEmbeddings(model=model_name)
vector_store = Chroma(persist_directory=db_directory, embedding_function=embeddings)

# Inicialización del modelo
logging.info(f"Inicializando modelo OllamaLLM con el modelo: {model_name}")
model = OllamaLLM(model=model_name)
# Nota: Aquí se asume que si el modelo no está disponible, OllamaLLM se encargará de descargarlo o
# levantar el error correspondiente. Se recomienda revisar la documentación del driver/modelo para más detalles.

def upload_pdf(file):
    file_path = os.path.join(pdfs_directory, file.name)
    logging.info(f"Subiendo archivo PDF: {file.name} a {file_path}")
    with open(file_path, 'wb') as f:
        f.write(file.getbuffer())
    logging.info("Archivo PDF subido correctamente.")

def load_pdf(file_path):
    logging.info(f"Cargando archivo PDF desde: {file_path}")
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    logging.info(f"Archivo PDF cargado. Se obtuvieron {len(documents)} documento(s).")
    return documents

def split_text(documents):
    logging.info("Dividiendo texto de documentos en fragmentos...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Texto dividido en {len(chunks)} fragmentos.")
    return chunks

def index_docs(documents):
    logging.info("Indexando documentos en el vector store...")
    vector_store.add_documents(documents)
    vector_store.persist()
    logging.info("Documentos indexados y vector store persistido.")

def retrieve_docs(query):
    logging.info(f"Recuperando documentos relevantes para la consulta: '{query}'")
    results = vector_store.similarity_search(query)
    logging.info(f"Se recuperaron {len(results)} documentos relacionados.")
    return results

def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    logging.info("Generando respuesta usando el modelo y el contexto proporcionado...")
    result = chain.invoke({"question": question, "context": context})
    logging.info("Respuesta generada.")
    return result

# Interfaz de Streamlit para subir archivo y realizar consultas
uploaded_file = st.file_uploader("Subir PDF", type="pdf", accept_multiple_files=False)

if uploaded_file:
    # Asegurarse que el directorio de pdfs exista
    os.makedirs(pdfs_directory, exist_ok=True)
    logging.info(f"Directorio para PDFs: '{pdfs_directory}' verificado o creado.")
    
    upload_pdf(uploaded_file)
    file_path = os.path.join(pdfs_directory, uploaded_file.name)
    documents = load_pdf(file_path)
    chunked_documents = split_text(documents)
    index_docs(chunked_documents)

    question = st.chat_input("Escribe tu pregunta aquí...")

    if question:
        st.chat_message("user").write(question)
        related_documents = retrieve_docs(question)
        answer = answer_question(question, related_documents)
        st.chat_message("assistant").write(answer)
