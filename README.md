# semantic-retrieval-middleware
Fast, latency-aware vector DB retrieval middleware with semantic caching and integrated re-ranker.

Necessary components:
0) pre-reqs: methods to embed sample text, define vector DB index/table schema
1) memory retrieval interface, connected to pgvector (Milvus/Pinecone TBD)
2) semantic cache
3) re-ranker
    - w.r.t: cross encoder

TODO:
- semantic cache + exact query cache (both in local, within MemoryInterface)
- making this a fastAPI service for easier testing and lifespans
- time logger
- redis cache for persistance
- reranker + CE client

PROGRESS:
1. Baseline pre-reqs + embedding clients, ORMs - using scripts to test them in isolation
2. Memory interface to hold local state caches, but still using scripts
-- TO BE DONE: 
3. Turning this into a minimal fastAPI service for easier testing across session, easier debugging
- also implementing time logging to verify the effectiveness of cache
4. Utilizing redis to keep persistant cache across session, can reset via test routes
5. Implementing reranker w/ CE model
