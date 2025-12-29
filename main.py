import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from flask import Flask, request, jsonify, session
from upstash_redis import Redis as UpstashRedis
from flask_cors import CORS
import uuid
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import time
from flask_session import Session
import redis
import pickle

import logging
from logging.handlers import RotatingFileHandler

load_dotenv()


# Setup logging
if not os.path.exists('logs'):
    os.mkdir('logs')

file_handler = RotatingFileHandler('logs/app.log', maxBytes=100000, backupCount=10)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))


app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "your-secret-key-here")
CORS(app, supports_credentials=True)

# --- REDIS CONFIGURATION ---
REDIS_TCP_URL = os.getenv("REDIS_URL")              # The rediss:// one
REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL") # The https:// one
REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")
if not REDIS_REST_URL:
    raise ValueError("UPSTASH_REDIS_REST_URL not set in .env")
#redis_client = UpstashRedis(url=REDIS_URL, token=UPSTASH_REDIS_REST_TOKEN)
#TCP_URL = os.getenv("REDIS_URL")
app.config['SESSION_TYPE'] = 'redis'
'''
app.config['SESSION_REDIS'] = redis.from_url(
    f"{REDIS_TCP_URL}?ssl_cert_reqs=none",
    health_check_interval=30
)

'''
app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_REDIS'] = redis.from_url(

    REDIS_TCP_URL,

    ssl_cert_reqs=None,        # Upstash uses self-signed certificates in some regions

    health_check_interval=30,  # Keeps the connection alive

    socket_connect_timeout=10, # Prevents hanging on initial connect

    retry_on_timeout=True

)
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = True  # False for local HTTP testing
app.config['PERMANENT_SESSION_LIFETIME'] = 3600 * 24 * 7  # 7 days optional

Session(app)
redis_client = UpstashRedis(url=REDIS_REST_URL, token=REDIS_REST_TOKEN)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.info('Flask app startup')

# Store conversation history per session
# conversation_store = {}

# 1. Load and Split Manual
print("Loading PDF and creating Embeddings...")
loader = PyPDFLoader("Resume_fancy -Refined - ML-Ind (2).pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 2. Vector DB with Pinecone (hosted)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")

if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY not set in .env")

pc = Pinecone(api_key=pinecone_api_key)

# Create index if it doesn't exist
if index_name not in [idx.name for idx in pc.list_indexes()]:
    print(f"Creating Pinecone index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=384,  # all-MiniLM-L6-v2 dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    # Wait for index to be ready
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

index = pc.Index(index_name)

# Load or create vector store (idempotent - safe to run every startup)
print("Upserting documents into Pinecone...")
vectorstore = PineconeVectorStore.from_documents(
    documents=splits,
    embedding=embeddings,
    index_name=index_name
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. Setup Free Brain (Llama 3.2 via HF)
llm_engine = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_new_tokens=256,
    temperature=0.7,
)
chat_model = ChatHuggingFace(llm=llm_engine)

print("System ready!")

# 4. Manual RAG Function
def ask_question_with_history(question: str, session_id: str = None):
    # Use provided session_id or fall back to Flask's session
    if session_id is None:
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        current_session_id = session['session_id']
    else:
        current_session_id = session_id
        session['session_id'] = current_session_id  # Sync with frontend-provided ID

    # Key for Redis storage
    history_key = f"conversation:{current_session_id}"

    # Initialize Redis connection
    r = redis.from_url(REDIS_TCP_URL)

    # Load history from Redis
    serialized_history = r.get(history_key)
    if serialized_history:
        conversation_history = pickle.loads(serialized_history)
    else:
        conversation_history = []

    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Build messages
    messages = [
        SystemMessage(content="""You are a friendly, professional version of the person whose resume is in the retrieved context. You are a young ML Engineer.Answer questions about your experience, skills, education, projects, or anything mentioned in your resume using the provided context.If the question is not related to your resume or personal/professional background (e.g., general knowledge questions like 'explain quantum physics' or 'what is AI?'), respond briefly and politely with:'Sorry, I'm designed to answer questions about my experience and background as an ML Engineer. What would you like to know about my resume or skills?'Keep answers concise and engaging. Summarize long information when needed."""),
    ]
    messages.extend(conversation_history[-10:])  # Last 10 messages
    messages.append(HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"))

    # Generate response
    response = chat_model.invoke(messages)
    answer = response.content

    # Update history
    conversation_history.append(HumanMessage(content=question))
    conversation_history.append(AIMessage(content=answer))

    # Save back to Redis (with 7-day expiry)
    r.setex(history_key, 604800, pickle.dumps(conversation_history))  # 7 days

    return answer, current_session_id

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        question = data.get('question', '')
        client_session_id = data.get('session_id')  # Optional from frontend

        if not question:
            return jsonify({'error': 'No question provided'}), 400

        answer, used_session_id = ask_question_with_history(question, client_session_id)

        return jsonify({
            'answer': answer,
            'session_id': used_session_id,
        })
    except Exception as e:
        app.logger.error(f"Error in chat: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500
    

    
@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    try:
        data = request.json
        session_id = data.get('session_id') or session.get('session_id')
        if not session_id:
            return jsonify({'error': 'No session ID'}), 400

        r = redis.from_url(REDIS_TCP_URL)
        r.delete(f"conversation:{session_id}")
        return jsonify({'message': 'Conversation history cleared'})
    except Exception as e:
        app.logger.error(f"Error clearing history: {str(e)}")
        return jsonify({'error': str(e)}), 500
    

@app.route('/api/history', methods=['POST'])
def get_history():
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'session_id is required'}), 400

        # Connect to Redis
        r = redis.from_url(REDIS_TCP_URL)
        history_key = f"conversation:{session_id}"

        # Load serialized history from Redis
        serialized_history = r.get(history_key)
        if not serialized_history:
            return jsonify({'history': []})

        # Deserialize the list of LangChain messages
        conversation_history = pickle.loads(serialized_history)

        # Convert to simple question-answer pairs
        history = []
        for i in range(0, len(conversation_history), 2):
            if i + 1 < len(conversation_history):
                question_msg = conversation_history[i]
                answer_msg = conversation_history[i + 1]
                if isinstance(question_msg, HumanMessage) and isinstance(answer_msg, AIMessage):
                    history.append({
                        'question': question_msg.content,
                        'answer': answer_msg.content
                    })

        return jsonify({'history': history})

    except Exception as e:
        app.logger.error(f"Error in get_history: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to retrieve history'}), 500

@app.route('/api/health', methods=['GET'])
def health():
    try:
        r = redis.from_url(REDIS_TCP_URL)
        
        # Get all keys matching our conversation pattern
        keys = r.keys("conversation:*")
        
        active_sessions = len(keys)
        total_messages = 0
        
        if active_sessions > 0:
            # Fetch lengths of all conversation histories
            # Use pipeline for efficiency (faster than individual GETs)
            pipe = r.pipeline()
            for key in keys:
                pipe.llen(key)  # But wait — we're storing pickled lists, not Redis lists!
            
            # PROBLEM: We're using pickle.dumps() on a Python list, not Redis data structures
            # So llen() won't work. We need to deserialize each one.
            
            # Correct approach: deserialize and count
            total_messages = 0
            for key in keys:
                serialized = r.get(key)
                if serialized:
                    history = pickle.loads(serialized)
                    total_messages += len(history)
        
        return jsonify({
            'status': 'healthy',
            'active_sessions': active_sessions,
            'total_messages': total_messages,
            'redis_connected': True
        })
    
    except Exception as e:
        app.logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'redis_connected': False
        }), 500
    
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)