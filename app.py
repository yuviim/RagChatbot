import os
import sys
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# Display current Python path (for debugging)
st.sidebar.write("Python path:", sys.executable)

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not found in .env file.")
    st.stop()

# Initialize OpenAI models
llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Load PDF documents
def load_local_documents(file_paths):
    all_documents = []
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        all_documents.extend(documents)
    return all_documents

# Load and split documents
pdf_files = ["290423-Booklet-English.pdf"]  # ✅ Ensure file is in same directory or provide full path
docs = load_local_documents(pdf_files)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Vector store
vector_store = InMemoryVectorStore(embeddings)
_ = vector_store.add_documents(all_splits)

# Prompt
prompt = hub.pull("rlm/rag-prompt")

# App state
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Step 1: Retrieve context
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

# Step 2: Generate answer
def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Build graph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Streamlit UI
st.title("📘 Customer Support chatbot")
st.write("Ask questions based on the document")

question = st.text_input("Your Question:")
if question:
    state = State(question=question, context=[], answer="")
    result = graph.invoke(state)
    st.markdown("### ✅ Answer")
    st.write(result["answer"])
