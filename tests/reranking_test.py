# test to verify HuggingFace's cross encoder model

from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

if __name__ == "__main__":
    query = "fox jumping over a duck"

    # sample docs, mix of relevant and irrelevant sentences to verify reranking
    docs = [
        "The cat sat on the mat.",
        "A quick brown fox leaped over a sleeping duck.",
        "Machine learning is a subset of artificial intelligence.",
        "The fox was seen jumping above the pond where ducks rested.",
        "Stock markets closed higher on Friday.",
        "The fox jumped over the duck pond."
    ]

    pairs = [(query, doc) for doc in docs]
    scores = model.predict(pairs)

    reranked = sorted(zip(scores, docs), reverse=True)

    print(f"Query: {query}\n")
    print("Reranked results:")
    for rank, (score, doc) in enumerate(reranked, start=1):
        print(f"{rank}. [{score:.4f}] {doc}")
