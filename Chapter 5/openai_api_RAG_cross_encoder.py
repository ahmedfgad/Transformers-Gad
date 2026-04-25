# Needs sentence_transformers library

import os
os.environ["OPENAI_API_KEY"] = "..."

# Avoid name collision with OpenAI's SDK
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Set embedding model (used for vector search)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

# (optional) choose your LLM for final generation
Settings.llm = OpenAI(model="gpt-4o")  # pick any supported chat model

# 1) Load & index documents (bi-encoder embeddings under the hood)
# You may add your resume to the folder to parse using GPT.
documents = SimpleDirectoryReader("/Users/ahmedgad/Desktop/Data").load_data()
index = VectorStoreIndex.from_documents(documents)

# 2) Cross-encoder re-ranker (re-ranks text pairs: (query, passage))
# A lightweight, popular model; for even better accuracy, try "cross-encoder/ms-marco-electra-base"
reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=5  # keep the best 5 after re-ranking
)

# 3) Build a query engine that retrieves many, then re-ranks
# similarity_top_k: how many candidates to pull from the vector index before re-ranking
# Make a query engine that first pulls the top 40 most similar chunks using vector embeddings, then re-ranks them with my cross-encoder reranker, and finally uses the top re-ranked chunks to generate an answer.
query_engine = index.as_query_engine(
    similarity_top_k=40,                  # retrieve a wider set
    node_postprocessors=[reranker],       # cross-encoder re-rank step
)

# 4) Ask questions
response = query_engine.query("What is the candidate name?")
print(response)

# Optional: See re-ranked chunks
for i, source_node in enumerate(response.source_nodes, start=1):
    print(f"\n--- Chunk {i} (Score: {source_node.score}) ---")
    print(source_node.node.get_content())
