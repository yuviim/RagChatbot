import os
import sys
import logging
import datetime
import re
import streamlit as st
from dotenv import load_dotenv
from typing_extensions import List, TypedDict

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph.graph import START, StateGraph

# ---------------------- Setup ----------------------

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    st.error("GOOGLE_API_KEY not found in .env file.")
    st.stop()

# Setup logging
logging.basicConfig(filename="chat_logs.txt", level=logging.INFO)

# ---------------------- LangChain Setup ----------------------

llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai", google_api_key=google_api_key)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

pdf_files = ["290423-Booklet-English.pdf"]
if not os.path.exists(pdf_files[0]):
    st.error("PDF file not found. Please ensure it's in the project directory.")
    st.stop()

def load_local_documents(file_paths):
    all_documents = []
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        all_documents.extend(documents)
    return all_documents

docs = load_local_documents(pdf_files)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

vector_store = InMemoryVectorStore(embeddings)
_ = vector_store.add_documents(all_splits)

prompt_template = """You are a helpful, friendly, and patient customer support chatbot for SBI products and services. Please respond to the user's questions in a clear, concise, and empathetic manner. If the user expresses frustration, acknowledge their feelings (e.g., "I understand this can be frustrating").

Prioritize providing accurate information based on the following context. If the context does not contain the answer, please clearly state "I don't know" and do not invent information.

Under no circumstances should you provide responses that are harmful, unethical, biased, discriminatory, or promote illegal activities. If a user asks a question that falls into these categories, politely decline to answer and state that you cannot assist with such requests.

Context:
{context}

Question: {question}"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# App State
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# ---------------------- Functions ----------------------

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.format_messages(question=state["question"], context=docs_content)
    response = llm.invoke(messages)
    answer = response.content
    if state["question"].lower().startswith(("i'm frustrated", "this is annoying")):
        answer = "I understand this can be frustrating. " + answer
    return {"answer": answer}

# Graph setup
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# ---------------------- Moderation & Logging ----------------------

def moderate_input(question: str) -> bool:
    logging.warning("Using placeholder for content moderation. Implement Google Cloud Moderation for production.")
    return True

def is_prompt_injection(text: str) -> bool:
    injection_patterns = [r"(ignore previous instructions)", r"(act as)", r"(pretend to)"]
    return any(re.search(pat, text, re.IGNORECASE) for pat in injection_patterns)

def log_question(question: str):
    logging.info(f"{datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30)))} - QUESTION: {question}")

# ---------------------- Streamlit UI ----------------------

st.title("📘 Customer Support Chatbot")
st.write("Ask questions based on the uploaded document.")
st.info("⚠️ By using this chatbot, you agree that your inputs may be logged for safety and quality.")

st.sidebar.write(f"🧪 Python Path: {sys.executable}")

question = st.text_input("Your Question:")

if question:
    if is_prompt_injection(question):
        st.warning("Your question appears unsafe. Please rephrase without prompt injection.")
    elif not moderate_input(question):
        st.warning("Your question may violate safety guidelines. Please rephrase.")
    else:
        try:
            log_question(question)
            state = State(question=question, context=[], answer="")
            result = graph.invoke(state)
            st.markdown("### ✅ Answer")
            st.write(result["answer"])
        except Exception as e:
            st.error(f"🚨 Something went wrong: {str(e)}")
