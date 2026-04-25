from transformers import AutoTokenizer
model_name = 'EleutherAI/gpt-neo-1.3B'
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

print("Pipeline Prediction")
# Predictions using a pipeline
from transformers import pipeline
# Load pipeline for text classification
model = pipeline("text-generation", 
                 model=model_name, 
                 tokenizer=tokenizer)
# Predict on a sample
sample = "I am feeling happy today."
prompt = f"""
Task: Classify the following sentence as either positive or negative.

{sample}

Output: 
"""

output = model(prompt,
               max_new_tokens=10,
               )
print(output)
print(output[0]["generated_text"])
