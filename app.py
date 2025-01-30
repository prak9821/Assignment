from langchain_community.document_loaders import WebBaseLoader  
from langchain_community.embeddings import OpenAIEmbeddings 
from langchain_community.vectorstores import FAISS  
from langchain_community.llms import OpenAI  
from langchain.chains import RetrievalQA
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import openai 
from time import sleep
load_dotenv()

# Get the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables")

# Step 1: Extract Data from the URL
url = 'https://brainlox.com/courses/category/technical'
loader = WebBaseLoader(url)
documents = loader.load()

# Step 2: Create Embeddings and Vector Store
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

def create_vectorstore_with_retry(documents, embeddings):
    """Try to create a vectorstore, retry on RateLimitError"""
    retries = 3  # Try up to 3 times
    for _ in range(retries):
        try:
            vectorstore = FAISS.from_documents(documents, embeddings)
            return vectorstore
        except openai.RateLimitError:  # Correct way to handle rate limit error directly
            print("Rate limit exceeded, retrying...")
            sleep(30)  # Wait for 30 seconds before retrying
    raise Exception("Exceeded maximum retries due to rate limit errors")

# Attempt to create the vectorstore
vectorstore = create_vectorstore_with_retry(documents, embeddings)

# Save the vector store locally
vectorstore.save_local('faiss_index')

# Step 3: Set up Flask API
app = Flask(__name__)

# Load the vector store from disk
vectorstore = FAISS.load_local('faiss_index', OpenAIEmbeddings(openai_api_key=openai_api_key))

# Initialize the QA chain using the vector store as a retriever
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0, openai_api_key=openai_api_key),
    chain_type="stuff", 
    retriever=vectorstore.as_retriever()
)

@app.route('/ask', methods=['POST'])
def ask():
    # Get the question from the request
    question = request.json.get('question')
    if question:
        # Get the answer using the QA chain
        answer = qa_chain.run(question)
        return jsonify({"answer": answer})
    return jsonify({"error": "No question provided"}), 400

if __name__ == '__main__':
    app.run(debug=True)
