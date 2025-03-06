from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import asyncio
import utils

# Initialize FastAPI app
app = FastAPI()

# Adding CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load the LLaMA model
llama_model = utils.load_llama_model()

# File to store chat history
CHAT_HISTORY_FILE = "chat_history.json"

# Executor for concurrent task execution
executor = ThreadPoolExecutor(max_workers=5)

class ChatRequest(BaseModel):
    prompt: str

# Chat history loading and saving functions
def load_chat_history():
    try:
        with open(CHAT_HISTORY_FILE, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"Error loading chat history: {e}")
        return []

def save_chat_history(chat_data):
    try:
        with open(CHAT_HISTORY_FILE, "w") as file:
            json.dump(chat_data, file, indent=4)
    except Exception as e:
        print(f"Error saving chat history: {e}")

# Main chatbot endpoint
@app.post("/chat/")
async def chat(request: ChatRequest):
    try:
        prompt = request.prompt

        folder_path = f"./corpus-"
        
        # Process prompt using RAG pipeline
        response = await asyncio.get_event_loop().run_in_executor(
            executor,
            utils.process_prompt,
            prompt,
            folder_path,
            llama_model,
        )
        
        # Save the chat history
        chat_history = load_chat_history()
        chat_history.append({"prompt": prompt, "response": response, "timestamp": str(datetime.now())})
        save_chat_history(chat_history)
        
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for getting chat history
@app.get("/history")
async def get_history():
    try:
        chat_history = load_chat_history()
        return {"history": chat_history}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error loading chat history.")

# Root endpoint to test API
@app.get("/")
async def root():
    return {"message": "Chatbot API is running with LLaMA.cpp"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # On shutdown
    yield
    print("Shutting down the app...")

# app = FastAPI(lifespan=lifespan)