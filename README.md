# Custom Chatbot with Langchain & Flask

This project uses Langchain to create a custom chatbot that scrapes data from a website, generates embeddings, stores them in a FAISS vector store, and serves it through a Flask API.

## Setup

1. Clone the repository.
2. Create a virtual environment and activate it.
    python -m venv <env_name>
    .\myenv\Scripts\activate
4. Install dependencies:pip install -r requirements.txt
5. Create a `.env` file with your OpenAI API key: OPENAI_API_KEY=your-api-key-here
6. Run the Flask API:python app.py

