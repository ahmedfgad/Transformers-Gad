from openai import OpenAI
client = OpenAI(api_key="...")

"""
response = client.responses.create(
    model="gpt-4.1",
    input="Write a one-sentence bedtime story about a unicorn."
)

print(response.output_text)
"""

import os
os.environ["OPENAI_API_KEY"] = "..."

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# The embedding model.
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

# Specify the model to use for the generation step.
Settings.llm = OpenAI(model="gpt-4o")

# Load the PDF.
# You may use your resume in a folder to parse using GPT.
documents = SimpleDirectoryReader("/Users/ahmedgad/Desktop/Data").load_data()

# Build RAG index
index = VectorStoreIndex.from_documents(documents)

# Query the PDF
query_engine = index.as_query_engine(similarity_top_k=2)
response = query_engine.query("What is the candidate name?")
print(response)

# These are the chunks sent to the LLM as context (in the order LlamaIndex used them)
for i, source_node in enumerate(response.source_nodes):
    print(f"\n--- Context Chunk {i} ---")
    print(f"Score: {source_node.score}")
    print(f"Node ID: {source_node.node_id}")
    print(f"Doc ID: {source_node.node.ref_doc_id}")
    print(f"File: {source_node.metadata.get('file_path')}")
    print(f"Start char: {source_node.node.start_char_idx}, End char: {source_node.node.end_char_idx}")
    print(source_node.get_content())
