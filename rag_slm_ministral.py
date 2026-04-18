"""
   RAG_CodeMinds_Hackathon.ipynb

   Using mistralai/Ministral-3-8B-Instruct-2512 as SLM for code generation, 
   feeding to it as context RAG with scraped data from various polars documentation sources.

   - also using bge for the RAG retriever and reranker stages, 
   with a local chromadb vector store as the knowledge base.

"""

#create a RAG pipeline to create synthetic polars code data

!pip install chromadb requests -q

import os
import chromadb
import requests
from google.colab import userdata

# 1. Configuration
# Make sure your ALBERT_API_KEY is in Colab Secrets (the key icon on the left)
ALBERT_API_KEY = userdata.get('ALBERT_API_KEY').split('\n')[0].strip()
BASE_URL = "https://albert.api.etalab.gouv.fr/v1"
HEADERS = {"Authorization": f"Bearer {ALBERT_API_KEY}"}

# 2. Define your local path where the chroma.sqlite3 file is located
# Example: if you uploaded a folder named 'my_vectordb'
CHROMA_PATH = "/content/chroma_storage"

print("Environment Ready.")

def get_local_data(db_path, collection_name):
    """Extracts all text and metadata from your local .sqlite3 / HNSW index."""
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name=collection_name)
    data = collection.get(include=['documents', 'metadatas'])
    print(f"📦 Extracted {len(data['documents'])} items from {collection_name}")
    return data['documents'], data['metadatas']

docs, metas = get_local_data("/content/chroma_storage", "ma_collection")

def create_albert_collection(name, description=""):
    """Creates a sovereign collection in the ALBERT cloud."""
    response = requests.post(
        f"{BASE_URL}/collections",
        headers=HEADERS,
        json={"name": name, "description": description}
    )

    if response.status_code != 201:
        print(f"❌ Failed to create collection: {response.text}")
        response.raise_for_status()

    res_data = response.json()
    # Pydantic is picky. We ensure we get the ID and it is a pure int.
    try:
        col_id = int(res_data["id"])
        print(f"🚀 ALBERT Collection Created (ID: {col_id})")
        return col_id
    except (KeyError, ValueError, TypeError):
        print(f"⚠️ Warning: ID '{res_data.get('id')}' is not a standard integer. Attempting to use as is.")
        return res_data.get("id")

col_id = create_albert_collection("polars_code_db")

import io
import json
def ingest_to_albert(collection_id, documents, metadatas):
    url = f"{BASE_URL}/documents"
    upload_headers = {k: v for k, v in HEADERS.items() if k.lower() != "content-type"}

    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        # Filter out empty string values from metadata
        clean_meta = {k: v for k, v in meta.items() if v != "" and v is not None}

        files = {
            "file": (f"doc_{i}.txt", io.BytesIO(str(doc).encode("utf-8")), "text/plain"),
        }
        data = {
            "collection_id": int(collection_id),
            "metadata": json.dumps(clean_meta),
            "disable_chunking": "true",
        }

        res = requests.post(url, headers=upload_headers, files=files, data=data)

        if res.status_code in (200, 201):
            if i % 5 == 0:
                print(f"✅ Ingested {i+1}/{len(documents)}")
        else:
            print(f"❌ Error at index {i}: {res.status_code} {res.text}")
            break

    print("✅ Ingestion complete.")

print(f"collection_id = {col_id!r}, type = {type(col_id)}")
assert col_id is not None, "collection_id is None!"
ingest_to_albert(col_id, docs, metas)

def retrieve_and_rerank(query, collection_id, top_n=5):
    """
    Stage 1: Search the collection (Returns top 40)
    Stage 2: Rerank the search results (Returns top_n)
    """

    # --- STAGE 1: SEARCH ---
    # We use 'method': 'semantic' to pull from our collection
    search_payload = {
    "query": test_query,
    "collection_ids": [int(col_id)],
    "method": "semantic",   # essaie aussi "hybrid" si besoin
    "limit": 40,
    }

    print("search started")
    search_res = requests.post(
        f"{BASE_URL}/search",
        headers=HEADERS,
        json=search_payload,
    )

    print("search done")



    # Extracting the content from the search response
    # Structure: response['data'][i]['chunk']['content']
    search_data = search_res.json().get("data", [])
    candidate_chunks = [item["chunk"]["content"] for item in search_data]
    print(candidate_chunks)

    if not candidate_chunks:
        print("⚠️ No documents found in collection.")
        return []

    # --- STAGE 2: RERANK ---
    # We follow the exact schema you provided from the docs
    rerank_payload = {
        "model": "openweight-rerank", # Or your RERANK_MODEL_ID
        "query": query,
        "documents": candidate_chunks,
        "top_n": top_n
    }

    rerank_res = requests.post(f"{BASE_URL}/rerank", headers=HEADERS, json=rerank_payload)

    if rerank_res.status_code != 200:
        print(f"❌ Rerank Error: {rerank_res.text}")
        # Fallback: Return top candidates from search if rerank fails
        return candidate_chunks[:top_n]

    # --- FINAL MAPPING ---
    # The reranker returns indices. We map them back to our text chunks.
    results = rerank_res.json().get("results", [])
    return [candidate_chunks[res["index"]] for res in results]



import json
import re
import openai # Using the OpenAI client for Albert API

# ── API Client Setup ──────────────────────────────────────────────────────
# Replace with your actual Albert API Key and Base URL
ALBERT_API_KEY = ALBERT_API_KEY
ALBERT_BASE_URL = "https://albert.api.etalab.gouv.fr/v1" # Usually looks like this

client = openai.OpenAI(
    api_key=ALBERT_API_KEY,
    base_url=ALBERT_BASE_URL
)

# ── Config ────────────────────────────────────────────────────────────────
BENCHMARK_CATEGORIES = [
    "select", "filters", "joins", "window_functions", "aggregations", "full_pipeline",
]
PAIRS_PER_QUERY = 5

# ── System prompt ───────────────────────────────────────
SYSTEM_PROMPT = """You are an expert Polars (Python) data engineering trainer.
Your job is to generate high-quality synthetic training data for a Polars code generation model.

Rules:
- EAGER API only (no .lazy(), no .collect())
- Load data with pl.read_parquet("file.parquet") or pl.read_csv("file.csv")
- Assign final DataFrame to `result`
- No pandas, no .apply(), no prints.
Return ONLY valid JSON array."""

def build_generation_prompt(context: str, category: str, n_pairs: int) -> str:
    return f"""Using the following Polars documentation:
{context}

Generate {n_pairs} diverse training pairs for category: **{category}**
Column names: customer_id, revenue, date, product, region, quantity, price
Return ONLY the JSON array: [{"question": "...", "code": "..."}]"""

def extract_json_pairs(raw: str) -> list[dict]:
    # Strip markdown fences more aggressively for OSS models
    raw = re.sub(r"```json", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"```", "", raw).strip()
    try:
        pairs = json.loads(raw)
        return [p for p in pairs if isinstance(p, dict) and "question" in p and "code" in p and "result" in p["code"]]
    except json.JSONDecodeError:
        # Fallback: try to find the first '[' and last ']'
        match = re.search(r"(\[.*\])", raw, re.DOTALL)
        if match:
            try: return json.loads(match.group(1))
            except: return []
        return []

def generate_pairs_for_query(
    query: str,
    category: str,
    top_n: int = 3,
    n_pairs: int = PAIRS_PER_QUERY,
) -> list[dict]:

    # Stage 1 & 2: retrieve and rerank (assuming these functions exist in your env)
    best_context = retrieve_and_rerank(query, col_id, top_n=top_n)

    # Stage 3: generate pairs using Albert API (OpenAI-compatible)
    response = client.chat.completions.create(
        model="openweight-large", # Albert's name for gpt-oss-120B
        max_tokens=2048,
        temperature=0.7,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_generation_prompt(best_context, category, n_pairs)},
        ],
    )

    raw = response.choices[0].message.content
    pairs = extract_json_pairs(raw)

    for p in pairs:
        p["category"] = category

    return pairs

# ── Test cell remains the same ─────────────────────────────────────────────



import json
import re
import openai # Using the OpenAI client for Albert API

# ── API Client Setup ──────────────────────────────────────────────────────
# Replace with your actual Albert API Key and Base URL
ALBERT_API_KEY = ALBERT_API_KEY
ALBERT_BASE_URL = "https://albert.api.url/v1" # Usually looks like this

client = openai.OpenAI(
    api_key=ALBERT_API_KEY,
    base_url=ALBERT_BASE_URL
)

# ── Config ────────────────────────────────────────────────────────────────
BENCHMARK_CATEGORIES = [
    "select", "filters", "joins", "window_functions", "aggregations", "full_pipeline",
]
PAIRS_PER_QUERY = 5

# ── System prompt ───────────────────────────────────────
SYSTEM_PROMPT = """You are an expert Polars (Python) data engineering trainer.
Your job is to generate high-quality Polars code generation.

Rules:
- EAGER API only (no .lazy(), no .collect())
- Load data with pl.read_parquet("file.parquet") or pl.read_csv("file.csv")
- Assign final DataFrame to `result`
- No pandas, no .apply(), no prints.
Return ONLY valid JSON array."""

def build_generation_prompt(context: str, category: str, n_pairs: int) -> str:
    return f"""Using the following Polars documentation:
{context}

Generate {n_pairs} diverse training pairs for category: **{category}**
Column names: customer_id, revenue, date, product, region, quantity, price
Return ONLY the JSON array: [{"question": "...", "code": "..."}]"""

def extract_json_pairs(raw: str) -> list[dict]:
    # Strip markdown fences more aggressively for OSS models
    raw = re.sub(r"```json", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"```", "", raw).strip()
    try:
        pairs = json.loads(raw)
        return [p for p in pairs if isinstance(p, dict) and "question" in p and "code" in p and "result" in p["code"]]
    except json.JSONDecodeError:
        # Fallback: try to find the first '[' and last ']'
        match = re.search(r"(\[.*\])", raw, re.DOTALL)
        if match:
            try: return json.loads(match.group(1))
            except: return []
        return []

def generate_pairs_for_query(
    query: str,
    category: str,
    top_n: int = 3,
    n_pairs: int = PAIRS_PER_QUERY,
) -> list[dict]:

    # Stage 1 & 2: retrieve and rerank (assuming these functions exist in your env)
    best_context = retrieve_and_rerank(query, col_id, top_n=top_n)

    # Stage 3: generate pairs using Albert API (OpenAI-compatible)
    response = client.chat.completions.create(
        model="openweight-small", # Albert's name for gpt-oss-120B
        max_tokens=2048,
        temperature=0.7,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_generation_prompt(best_context, category, n_pairs)},
        ],
    )

    raw = response.choices[0].message.content
    pairs = extract_json_pairs(raw)

    for p in pairs:
        p["category"] = category

    return pairs

# ── Test cell remains the same ─────────────────────────────────────────────

def ask_polars_assistant(question: str, top_n: int = 3) -> str:
    """
    Retrieves context and asks the model to solve a specific Polars task.
    """

    # 1. Retrieve RAG context (using your existing functions)
    print(f"🔍 Searching documentation for: {question}...")
    context = retrieve_and_rerank(question, col_id, top_n=top_n)

    # 2. Refined Prompt for direct assistance
    instruction = f"""You are a Polars expert. Use the documentation below to solve the user's request.

--- DOCUMENTATION ---
{context}
--- END DOCUMENTATION ---

Rules:
- Use Polars EAGER API only.
- Load data with pl.read_parquet("file.parquet") or pl.read_csv("file.csv") if needed.
- The final result MUST be assigned to a variable named `result`.
- Provide ONLY the Python code, no explanation.
"""

    # 3. Call Albert API
    response = client.chat.completions.create(
        model="openweight-small",
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": question},
        ],
        temperature=0.1, # Lower temperature for accuracy over creativity
    )

    return response.choices[0].message.content

# ── Test Cell ─────────────────────────────────────────────────────────────

user_question = "How do I filter a dataframe for revenue > 100 and then calculate the mean price per product?"

print("🧠 Model is thinking...")
answer = ask_polars_assistant(user_question)

print("\n--- Generated Polars Code ---")
print(answer)

