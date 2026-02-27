# semantic-retrieval-middleware
Fast, latency-aware vector DB retrieval middleware with semantic caching and integrated re-ranker.

## How to Run Retrieval Service (Locally) - WIP

1. Install dependencies with: `pip install -r requirements.txt`

2. Set up your `.env` file at the project root. Check `.env.example` for required environmental variables.

3. Run the retrieval service with uvicorn: `uvicorn main:app --reload`

The API docs will be available at `http://localhost:8000/docs`. Use this to debug / run tests on the retrieval service.
- Also use logger to nicely print statements w/ timestamps and file locations.

## Project Notes / Planning
Necessary components:
0) pre-reqs: methods to embed sample text, define vector DB index/table schema
1) memory retrieval interface, connected to pgvector (Milvus/Pinecone TBD)
2) semantic cache
3) re-ranker
    - w.r.t: cross encoder

TODO:
- making this a fastAPI service for easier testing and lifespans
- time logger
- redis cache for persistance
- reranker + CE client

PROGRESS:
1. Baseline pre-reqs + embedding clients, ORMs - using scripts to test them in isolation
2. Memory interface to hold local state caches, but still using scripts
- optimizations via LRU and deque (max cache size)
-- TO BE DONE: 
3. Turning this into a minimal fastAPI service for easier testing across session, easier debugging
- also implementing time logging to verify the effectiveness of cache
4. Utilizing redis to keep persistant cache across session, can reset via test routes
5. Implementing reranker w/ CE model
