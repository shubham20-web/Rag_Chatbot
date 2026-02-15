import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Advanced RAG Chatbot", layout="wide")
st.title("🤖 Advanced RAG Chatbot (Hybrid Cloud Version)")

VECTOR_PATH = "vectorstore"

# -------------------------------------------------
# API KEY INPUT (Local Version)
# -------------------------------------------------
groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

if not groq_api_key:
    st.warning("Please enter your Groq API key to continue.")
    st.stop()

# -------------------------------------------------
# CHAT MEMORY
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
st.sidebar.header("⚙️ Settings")
top_k = st.sidebar.slider("Top K Results", 1, 5, 3)

# -------------------------------------------------
# LOAD EMBEDDINGS
# -------------------------------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# -------------------------------------------------
# LOAD LLM
# -------------------------------------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"   # change if needed
)

# -------------------------------------------------
# LOAD EXISTING VECTORSTORE
# -------------------------------------------------
if os.path.exists(VECTOR_PATH):
    vectorstore = FAISS.load_local(
        VECTOR_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    vectorstore = None

# -------------------------------------------------
# MULTIPLE PDF UPLOAD
# -------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload Multiple PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    all_docs = []

    with st.spinner("Processing PDFs..."):
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name

            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            all_docs.extend(documents)

            os.remove(temp_path)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        split_docs = text_splitter.split_documents(all_docs)

        if vectorstore:
            vectorstore.add_documents(split_docs)
        else:
            vectorstore = FAISS.from_documents(split_docs, embeddings)

        vectorstore.save_local(VECTOR_PATH)

        st.success("PDFs processed and saved successfully!")

# -------------------------------------------------
# DISPLAY CHAT HISTORY
# -------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -------------------------------------------------
# CHAT INPUT
# -------------------------------------------------
if prompt := st.chat_input("Ask your question..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # If no vectorstore available
            if not vectorstore:
                response = llm.invoke(prompt).content

            else:
                retriever = vectorstore.as_retriever(
                    search_kwargs={"k": top_k}
                )

                docs = retriever.invoke(prompt)

                if docs:
                    context = "\n\n".join(
                        [doc.page_content for doc in docs]
                    )

                    final_prompt = f"""
You are a helpful AI assistant.

If the context below is relevant to the user's question,
use it to answer.

If the context is not relevant,
ignore it and answer using your general knowledge.

Context:
{context}

Question:
{prompt}
"""
                    response = llm.invoke(final_prompt).content

                else:
                    response = llm.invoke(prompt).content

            st.write(response)

            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
            