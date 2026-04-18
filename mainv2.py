import os
import io
import json
import re
import requests
import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# --- Initialization & Config ---
app = FastAPI(title="Polars RAG CodeMinds API")

ALBERT_API_KEY = os.getenv("ALBERT_API_KEY")
BASE_URL = "https://albert.api.etalab.gouv.fr/v1"
HEADERS = {"Authorization": f"Bearer {ALBERT_API_KEY}"}
COLLECTION_ID = os.getenv("ALBERT_COLLECTION_ID")

client = openai.OpenAI(
    api_key=ALBERT_API_KEY,
    base_url=BASE_URL
)

SYSTEM_PROMPT = """You are an expert Polars (Python) data engineering trainer.
Rules:
- EAGER API only (no .lazy(), no .collect())
- Load data with pl.read_parquet("file.parquet") or pl.read_csv("file.csv")
- Assign final DataFrame to `result`
- No pandas, no .apply(), no prints.
Return ONLY valid JSON array."""

# --- Schemas ---
class ChatRequest(BaseModel):
    message: str
    top_n: Optional[int] = 3


class SyntheticRequest(BaseModel):
    category: str
    n_pairs: Optional[int] = 5


class ChatResponse(BaseModel):
    response: str


# --- Internal Logic ---
def retrieve_context(query: str, top_n: int = 5):
    if not COLLECTION_ID:
        raise ValueError("COLLECTION_ID not set")

    search_payload = {
        "query": query,
        "collection_ids": [int(COLLECTION_ID)],
        "method": "semantic",
        "limit": 40,
    }

    search_res = requests.post(
        f"{BASE_URL}/search",
        headers=HEADERS,
        json=search_payload
    )
    search_data = search_res.json().get("data", [])
    candidate_chunks = [item["chunk"]["content"] for item in search_data]

    if not candidate_chunks:
        return []

    rerank_payload = {
        "model": "openweight-rerank",
        "query": query,
        "documents": candidate_chunks,
        "top_n": top_n
    }

    rerank_res = requests.post(
        f"{BASE_URL}/rerank",
        headers=HEADERS,
        json=rerank_payload
    )

    if rerank_res.status_code != 200:
        return candidate_chunks[:top_n]

    results = rerank_res.json().get("results", [])
    return [candidate_chunks[res["index"]] for res in results]


def extract_json_pairs(raw: str):
    raw = re.sub(r"```json|```", "", raw, flags=re.IGNORECASE).strip()
    try:
        pairs = json.loads(raw)
        return [p for p in pairs if isinstance(p, dict) and "code" in p]
    except Exception:
        match = re.search(r"(\[.*\])", raw, re.DOTALL)
        return json.loads(match.group(1)) if match else []


def generate_response(prompt: str, top_n: int = 3) -> str:
    context_list = retrieve_context(prompt, top_n=top_n)
    context = "\n---\n".join(context_list)

    instruction = (
        f"Polars expert context:\n{context}\n\n"
        "Rules: Eager API, result variable, no prints."
    )

    response = client.chat.completions.create(
        model="openweight-small",
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content


# --- API Endpoints ---
@app.get("/")
async def health():
    return {"status": "online", "collection_id": COLLECTION_ID}


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest):
    try:
        response = generate_response(payload.message, top_n=payload.top_n)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-synthetic")
async def generate_synthetic(req: SyntheticRequest):
    try:
        query = f"Polars patterns for {req.category}"
        context_list = retrieve_context(query, top_n=3)
        context = "\n---\n".join(context_list)

        prompt = f"Using this doc:\n{context}\n\nGenerate {req.n_pairs} pairs for {req.category}."

        response = client.chat.completions.create(
            model="openweight-large",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )

        pairs = extract_json_pairs(response.choices[0].message.content)
        return {
            "category": req.category,
            "count": len(pairs),
            "data": pairs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)