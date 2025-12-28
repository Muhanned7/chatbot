import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "your-secret-key-here")
CORS(app, supports_credentials=True)

# Store conversation history per session
conversation_store = {}

# 1. Load and Split Manual
print("Loading PDF and creating Embeddings...")
loader = PyPDFLoader("Resume_fancy -Refined - ML-Ind (2).pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 2. Vector DB with Free Hugging Face Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. Setup Free Brain (Llama 3.2 via HF)
llm_engine = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_new_tokens=512,
    temperature=0.7,
)
chat_model = ChatHuggingFace(llm=llm_engine)

print("System ready!")

# 4. Manual RAG Function
def ask_question_with_history(question: str, session_id: str):
    
    # get or create conversation history for this session
    if session_id not in conversation_store:
        conversation_store[session_id] = []
    
    conversation_history = conversation_store[session_id]

    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)
    
    # Format context
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Create messages
    messages = [
        SystemMessage(content="You are the person whose resume is stored, A young and handsome ML Engineer. Use the following pieces of retrieved context to answer the question. " \
        "If you don't know the answer, say that you don't know. if you know the answer and its long summarize it."),
    ]
    
    # Add conversation history
    messages.extend(conversation_history[-10:])

    # Add current question with context
    messages.append(
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    )
    
    # Get response
    response = chat_model.invoke(messages)
    answer = response.content

    # Store conversation
    conversation_history.append(HumanMessage(content=question))
    conversation_history.append(AIMessage(content=answer))

    return answer

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        question = data.get('question', '')  # BUG FIX: was 'session_id'
        session_id = data.get('session_id', None)

        if not question:
            return jsonify({'error': 'No question provided'}), 400

        if not session_id:
            session_id = str(uuid.uuid4())

        answer = ask_question_with_history(question, session_id)

        return jsonify({
            'answer': answer,
            'session_id': session_id,
            'conversation_length': len(conversation_store.get(session_id, []))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    try:
        data = request.json
        session_id = data.get('session_id', None)

        if session_id and session_id in conversation_store:
            conversation_store[session_id] = []
            return jsonify({'message': 'Conversation history cleared'})
        return jsonify({'error': 'Invalid session ID'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['POST'])  # BUG FIX: was 'method' instead of 'methods'
def get_history():
    try:
        data = request.json
        session_id = data.get('session_id', None)
        
        if session_id and session_id in conversation_store:
            history = []
            messages = conversation_store[session_id]
            
            for i in range(0, len(messages), 2):
                if i + 1 < len(messages):
                    history.append({
                        'question': messages[i].content,
                        'answer': messages[i + 1].content
                    })        

            return jsonify({'history': history})
        
        return jsonify({'history': []})  # BUG FIX: was returning undefined 'history'
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy', 
        'active_sessions': len(conversation_store),
        'total_messages': sum(len(msgs) for msgs in conversation_store.values())
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)