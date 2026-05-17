import os
import re
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from openai import OpenAI

app = FastAPI(title="Prophet Hacks AI Agent v1")

# Ініціалізація ключів зі змінних оточення (безпечно)
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")

client = OpenAI(api_key=OPENAI_KEY)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.1

def search_internet(query: str) -> str:
    """Функція реального пошуку в інтернеті через Tavily API"""
    if not TAVILY_KEY:
        return "Search API key is missing."
    
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_KEY,
        "query": query,
        "search_depth": "news", # шукає свіжі новини
        "max_results": 3
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            results = response.json().get("results", [])
            # Збираємо короткі описи знайдених статей в один текст
            context = "\n".join([f"- {res['title']}: {res['content']}" for res in results])
            return context
        return "Could not fetch recent data."
    except Exception:
        return "Search request failed."

SYSTEM_PROMPT = """
You are an expert superforecaster. Your job is to take a real-world forecasting question and output a well-calibrated probability (0.0 to 1.0) that the event will happen.
Analyze the request step-by-step using the provided recent web context. Consider base rates, counter-arguments, and timelines.
CRITICAL: Your final answer MUST end with the exact format: 'PROBABILITY: [value]', where [value] is a float between 0.0 and 1.0.
"""

@app.post("/v1/chat/completions")
async def forecast_endpoint(request: ChatCompletionRequest):
    if not OPENAI_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key is missing on server.")

    try:
        user_question = request.messages[-1].content
        
        # Шукаємо свіжу інформацію в Інтернеті за темою запитання
        web_context = search_internet(user_question)
        
        full_user_content = f"Question: {user_question}\n\nRecent Web Context:\n{web_context}\n\nAnalyze the context and give the final probability."
        
        response = client.chat.completions.create(
            model="gpt-4o", # Використовуємо флагманську модель для точних прогнозів
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": full_user_content}
            ],
            temperature=0.1
        )
        
        agent_thinking = response.choices[0].message.content
        
        # Шукаємо рядок 'PROBABILITY: 0.XX' у відповіді моделі
        match = re.search(r"PROBABILITY:\s*([0-9.]+)", agent_thinking)
        probability = float(match.group(1)) if match else 0.5
        
        return {
            "id": "chatcmpl-prophet",
            "object": "chat.completion",
            "created": 1715856000,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"PROBABILITY: {probability}\nReasoning:\n{agent_thinking}"
                    },
                    "finish_reason": "stop"
                }
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))