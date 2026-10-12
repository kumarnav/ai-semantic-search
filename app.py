import numpy as np
from sentence_transformers import SentenceTransformer, util

def run_semantic_search():
    print("[*] Loading lightweight AI embedding model (all-MiniLM-L6-v2)...")
    # This downloads a compact, highly efficient transformer model locally
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Knowledge base documents representing different technical domains
    documents = [
        "Python is a high-level programming language widely used for web automation, scripting, and data scraping.",
        "Go (Golang) is a statically typed, compiled language designed at Google, optimized for high-concurrency cloud microservices.",
        "C++ is a high-performance general-purpose systems programming language used for engines and low-level memory management.",
        "PostgreSQL is an advanced, ACID-compliant relational database system supporting custom functions, triggers, and transactions.",
        "Vector embeddings and RAG (Retrieval-Augmented Generation) allow modern AI models to search documents based on semantic meaning rather than exact keyword matches."
    ]

    print("[*] Encoding knowledge base into vector space...")
    doc_embeddings = model.encode(documents, convert_to_tensor=True)

    # Test Queries
    queries = [
        "Which language is best for concurrent microservices?",
        "How do AI models understand contextual meaning?",
        "What tool is used for web scraping and automation?"
    ]

    print("\n==========================================")
    print("          SEMANTIC SEARCH RESULTS         ")
    print("==========================================")

    for query in queries:
        print(f"\n[QUERY]: '{query}'")
        query_embedding = model.encode(query, convert_to_tensor=True)

        # Compute cosine similarity between query and all documents
        cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
        
        # Find the best matching document index
        best_match_idx = np.argmax(cos_scores.cpu().numpy())
        best_score = cos_scores[best_match_idx].item()

        print(f"[MATCH FOUND] (Similarity Score: {best_score:.4f})")
        print(f"-> {documents[best_match_idx]}")

if __name__ == "__main__":
    run_semantic_search()