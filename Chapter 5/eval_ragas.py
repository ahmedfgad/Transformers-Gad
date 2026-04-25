from ragas import evaluate
from ragas.metrics import context_precision, answer_relevancy, faithfulness, context_recall, answer_correctness

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import os
from datasets import Dataset

os.environ["OPENAI_API_KEY"] = "..."

llm = ChatOpenAI(model="gpt-4.1")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

data = {"user_input": ["What is the capital of Egypt?"],
        "retrieved_contexts": [["Cairo is the capital of Egypt.", "Egypt's capital is very crowded."]],
        "response": ["The capital of Egypt is Cairo."],
        "reference": ["Cairo"]}

dataset = Dataset.from_dict(data)

metrics = [context_precision, context_recall, answer_relevancy, faithfulness, answer_correctness]
result = evaluate(dataset=dataset,
                  metrics=metrics,
                  llm=llm,
                  embeddings=embeddings)

df = result.to_pandas()
print(df.keys())
for key in df.keys():
    print(f"{key}: {df[key][0]}")

