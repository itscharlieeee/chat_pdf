import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import platform


st.set_page_config(
    page_title="Agente RAG con OpenAI 💬",
    page_icon="🤖",
    layout="centered"
)


st.title("🤖 Generación Aumentada por Recuperación (RAG)")
st.caption(f"Versión de Python: {platform.python_version()}")
st.markdown("""
Este asistente utiliza **RAG (Retrieval-Augmented Generation)** para responder preguntas basadas en el contenido de un PDF.
Sube un documento, haz una pregunta, y deja que el modelo te dé una respuesta contextualizada. 📘✨
""")


try:
    image = Image.open('6b6dd5a171abf33c000b1ddb83bb4fe2.jpg')
    st.image(image, width=350, caption="Análisis inteligente de documentos PDF")
except Exception as e:
    st.warning(f"⚠️ No se pudo cargar la imagen: {e}")


with st.sidebar:
    st.header("💡 Instrucciones")
    st.markdown("""
    1. 🔑 Ingresa tu **clave de OpenAI**.  
    2. 📄 Sube un archivo PDF.  
    3. 💬 Escribe una pregunta sobre el contenido.  
    4. 🤯 El modelo buscará dentro del texto y te responderá.

    ---
    **Consejos:**
    - Usa documentos no muy extensos (máx. ~50 páginas).  
    - Haz preguntas específicas.  
    - Evita PDFs escaneados (sin texto seleccionable).
    """)


ke = st.text_input("🔑 Ingresa tu clave de OpenAI:", type="password")

if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.info("Por favor ingresa tu clave de API para continuar.")


pdf = st.file_uploader("📎 Carga un archivo PDF", type="pdf")

if pdf is not None and ke:
    try:
        st.markdown("### 🧾 Extrayendo texto del PDF...")
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        st.success(f"✅ Texto extraído correctamente ({len(text)} caracteres).")

        # División del texto
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.info(f"📑 Documento dividido en {len(chunks)} fragmentos.")

        # Embeddings + Base de conocimiento
        with st.spinner("🔍 Generando embeddings y construyendo base de conocimiento..."):
            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Interfaz de preguntas
        st.markdown("### 💬 Haz una pregunta sobre tu documento:")
        user_question = st.text_area("✏️ Escribe tu pregunta aquí...", placeholder="Ejemplo: ¿Cuál es el objetivo principal del documento?")

        if user_question:
            with st.spinner("🤔 Analizando tu pregunta..."):
                docs = knowledge_base.similarity_search(user_question)

                # Modelo actualizado
                llm = OpenAI(temperature=0, model_name="gpt-4o")
                chain = load_qa_chain(llm, chain_type="stuff")

                # Obtener respuesta
                response = chain.run(input_documents=docs, question=user_question)

            # Mostrar respuesta
            st.markdown("### 🎯 Respuesta:")
            st.success(response)

    except Exception as e:
        st.error(f"❌ Error al procesar el PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

elif pdf is not None and not ke:
    st.warning("⚠️ Ingresa tu clave de API antes de continuar.")
else:
    st.info("📂 Carga un archivo PDF para comenzar el análisis.")


st.markdown("---")
st.caption("Hecho con ❤️ usando Streamlit, LangChain y OpenAI.")
