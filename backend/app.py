import os
import traceback
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found")

client = Groq(api_key=api_key)

app = FastAPI(title="Kubernetes Chatbot API", version="1.0")

Instrumentator().instrument(app).expose(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    response: str


SYSTEM_PROMPT = """
You are an expert Kubernetes DevOps assistant.

Answer only Kubernetes, Docker, Linux, Helm, ArgoCD,
Ingress, Networking, Prometheus, Grafana,
Container Runtime, CI/CD and Cloud Native questions.

If the question is unrelated to Kubernetes or DevOps,
politely refuse.

Keep answers concise and technically accurate.
"""


@app.get("/")
def home():
    return {"status": "running", "service": "Kubernetes Chatbot"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    try:

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.query},
            ],
            temperature=0.3,
            max_completion_tokens=1024,
        )

        answer = completion.choices[0].message.content

        return ChatResponse(response=answer)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
