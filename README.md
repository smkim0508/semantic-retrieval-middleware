# semantic-retrieval-middleware
Fast, latency-aware vector DB retrieval middleware with semantic caching and integrated re-ranker.

Necessary components:
0) pre-reqs: methods to embed sample text, define vector DB index/table schema
1) memory retrieval interface, connected to pgvector (Milvus/Pinecone TBD)
2) semantic cache
3) re-ranker
    - w.r.t: cross encoder

TODO:
- memory retrieval methods
- cache
- time logger
- reranker + CE client
