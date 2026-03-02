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
3. Turning this into a minimal fastAPI service for easier testing across session, easier debugging
- also implementing time logging to verify the effectiveness of cache
4. Utilizing redis to keep persistant cache across session, can reset via test routes
5. Implementing reranker w/ CE model
6. Added rerank, with optional param w/ dynamic threshold for limit and retrieval_sizes
7. Managing cache to be upward compatible
8. Optimizations:
- best semantic result, not the first
- unexpected bug when db doesn't have enough documents, so requested fetch size > document size, then we don't hit cached results

-- TO BE DONE
9. Rache cache client, TBD

## Real-scenario memory latency testing concerns:
- Can mock embed a bunch of random floats and load db with ~1-100M vectors
- The issue is free tiers for supabase / milvus doesn't support this...
    - Also unlike real enterprise solutions, my free tiers have data centers defaulted to EU or other far locations, so a lot of network overhead anyways
- Can verify with like ~10k vectors on supabase
- Growth in query latency is ~O(logn), so increasing index size isn't too big of a deal for query latency. Network latency costs more often.
- Verify PoC using just the baseline vs cache design

## Other Remaining Concerns
If Vector DB updates between retrievals, the cache can become stale
- Also the logic for checking whether truly exhausted or small db become stale too
-> Can resolve with indexing cache by timestamp + searching over a small region of DB and merging results.
- Not actually implemented, this is a much more sophisticated system and explores a separate concept.
- Also can use TTL for L2 staleness (somewhat resolves)

----

## Making Updates + Retrieval Freshness-aware (Managed Vector DB)
- Minimal PoC implemented with warm buffer that holds new updates (not yet flushed to managed vector db) + ground truth structured DB that is synced with any updates immediately
- Periodically flushes warm buffer to vector db
- If warm buffer is non-empty, then check buffer + db, compare with ground truth for validation
- Otherwise (in sync), take the usual 3-tiered cache hierarchy for reduced retrieval latency.

## Additional Future Considerations
- Real-scenario latency reduction can be achieved further via eager async calls (vector retrieval async call in background, and scan cache meanwhile).
- Freshness-aware managed vector db can be optimized further by making warm cache store ground truth pointers and cross-check w/ caches even when non-empty.
