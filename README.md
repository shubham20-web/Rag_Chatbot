# 🤖 Advanced RAG Chatbot (Hybrid Cloud Version)

A production-ready Retrieval-Augmented Generation (RAG) chatbot built using:

- Streamlit
- LangChain
- FAISS
- Groq LLM (LLaMA 3.3)
- HuggingFace Embeddings

This application allows users to upload multiple PDF documents and ask questions.  
If the question is related to the uploaded PDFs, the chatbot answers using document context.  
If not, it responds using general model knowledge.

---

## 🔥 Features

- 📄 Multi-PDF Upload Support
- ⚡ Persistent FAISS Vector Database
- 🧠 Hybrid RAG (Context + General Knowledge)
- ☁️ Cloud LLM via Groq API
- 🔐 Secure API Key Management (Streamlit Secrets)
- 💬 Modern Chat UI with Memory
- 🚀 Deployable on Streamlit Cloud

---

## 🏗️ Architecture

User Question  
⬇  
FAISS Vector Search  
⬇  
Relevant Context Retrieved  
⬇  
Groq LLM Generates Response  
⬇  
Displayed in Streamlit Chat UI  

If no relevant document context is found, the LLM answers directly.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS
- **LLM**: llama-3.3-70b-versatile (Groq)
- **Framework**: LangChain

---

## 📦 Installation (Local)

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
