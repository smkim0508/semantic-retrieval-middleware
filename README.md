# semantic-retrieval-middleware
Fast, latency-aware vector DB retrieval middleware with semantic caching and integrated re-ranker.

Necessary components:
1) memory retrieval interface, connected to pgvector (or Milvus if I have time)
2) semantic cache
3) re-ranker
    - w.r.t: cross encoder
