

https://github.com/user-attachments/assets/04e8719a-3c5a-4458-ac17-394964fe33ce

🤖 Resume-Informed RAG ChatbotAI

A production-grade Retrieval-Augmented Generation (RAG) application built to transform static PDF data into an interactive, memory-aware conversational agent. 

This system utilizes a local LLM to ensure data privacy and reduced API latency.


🌟 Key Features

Contextual Intelligence: Uses LangChain and Pinecone to perform semantic searches on PDF documents.

Persistent Conversation Memory: Integrated with Upstash Redis (via both TCP and REST) to maintain session history across user interactions.

Local LLM Execution: Runs Llama 3.2-3B natively using Transformers and PyTorch, optimized for CPU environments.

Robust Session Management: Implements Flask-Session with Redis backing and pickle serialization for complex message objects.

DevOps Ready: Fully containerized with Docker and configured for deployment on Google Cloud Platform (GCP).



🛠️ Technical Stack

Component           Technology

Backend Framework  ->   Flask (Python)

LLM Orchestration   ->  LangChain

Vector Database    ->   Pinecone (Serverless)

In-Memory Store     ->  Redis (Upstash)

Embeddings          -> HuggingFace (Sentence-Transformers)

Model      ->  Llama 3.2-3B-Instruct


🏗️ Architecture & Logic

1. Ingestion Pipeline

2. The system loads the PDF using PyPDFLoader and splits it into 1,000-character chunks with a 200-character overlap. These are then converted into 384-dimensional embeddings and upserted into Pinecone.

3. Retrieval & Memory Loop

      For every query, the system:

         Retrieves the top 3 relevant context chunks from Pinecone.

          Fetches previous conversation history from Redis.

         Feeds the System Prompt + History + Context + Query into the Llama model.
      
3. Resource Optimization

 To handle LLM execution on standard cloud hardware, the code includes:
 
 low_cpu_mem_usage=True: To prevent container OOM (Out of Memory) errors.
 
 reset_model_state(): A custom cleanup function that triggers garbage collection after each generation to maintain a low memory footprint.

🚀 Getting Started

Prerequisites

Docker

Pinecone API Key

Upstash Redis Instance

Environment Setup

Create a .env file in the root directory:

SECRET_KEY=your_secret_key

REDIS_URL=rediss://...

UPSTASH_REDIS_REST_URL=https://...

UPSTASH_REDIS_REST_TOKEN=your_token

PINECONE_API_KEY=your_key

PINECONE_INDEX_NAME=resume-bot

##Installation

Build the Docker Container:

  Bash   docker build -t rag-chatbot .

Run the Container:

  Bash    docker run -p 8080:8080 --env-file .env rag-chatbot



🚦 API Documentation
Endpoint            Method      Description
/api/chat           POST        Primary chat endpoint. Accepts question and optional session_id.
/api/history        POST        Retrieves the full interaction history for a specific session.
/api/clear-history  POST        Wipes the Redis record for a specific session.
/api/health         GET         Returns system status, active session count, and Redis health.
