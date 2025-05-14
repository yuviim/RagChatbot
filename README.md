# Customer Support Chatbot

This is a Streamlit application for a customer support chatbot that answers questions based on a provided PDF document. It leverages LangChain for document processing, embedding generation, and language model integration with Google's Gemini Flash. LangGraph is used to orchestrate the retrieval and generation steps.

## Overview

The chatbot works by:

1.  **Loading and Processing Documents:** Reads a PDF document, splits it into smaller chunks, and generates vector embeddings for each chunk using Google's Generative AI embeddings.
2.  **Storing Embeddings:** Stores these embeddings in an in-memory vector store (ChromaDB).
3.  **Retrieving Relevant Information:** When a user asks a question, the chatbot performs a similarity search on the vector store to find the most relevant document chunks.
4.  **Generating Answers:** The retrieved document chunks are passed to the Gemini Flash language model along with the user's question to generate a concise and informative answer.
5.  **User Interface:** The application provides a simple and interactive interface using Streamlit.

## Features

* Ask questions based on the content of the uploaded PDF document.
* Basic prompt injection detection.
* Placeholder for content moderation (integration with Google Cloud Moderation recommended for production).
* Logging of user questions.

## Getting Started

### Prerequisites

* Python 3.7+
* pip (Python package installer)
* A Google Cloud Platform project with the Gemini API enabled.
* A Google API key with access to the Gemini API.

### Installation

1.  **Clone the repository (if you have it on GitHub):**
    ```bash
    git clone <your_repository_url>
    cd Chatbot-RAGLC
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    venv\Scripts\activate  # On Windows
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create a `.env` file in the project root directory and add your Google API key:**
    ```
    GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
    ```
    Replace `YOUR_GOOGLE_API_KEY` with your actual Google API key.

5.  **Place the PDF document(s) you want to use in the project root directory.** The code currently expects a file named `290423-Booklet-English.pdf`. You can modify the `pdf_files` list in the `app.py` file to include other PDF files.

### Running the Application

```bash
streamlit run app.py
