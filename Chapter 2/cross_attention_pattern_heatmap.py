import matplotlib.pyplot as plt
import seaborn as sns

import torch
import transformers

# Load model and tokenizer (BART in this case)
model_name = "facebook/bart-large-cnn"
tokenizer = transformers.BartTokenizer.from_pretrained(model_name)
# tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

# Define input and target text
input_text = "I want coffee not juice"
target_text = "I prefer coffee"

# Tokenize input and target
inputs = tokenizer(input_text, return_tensors="pt")
targets = tokenizer(target_text, return_tensors="pt")

tokens_inputs = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
tokens_inputs = [token.replace("Ġ", "") for token in tokens_inputs]
tokens_targets = tokenizer.convert_ids_to_tokens(targets["input_ids"].squeeze())
tokens_targets = [token.replace("Ġ", "") for token in tokens_targets]

model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)
# Generate outputs with attentions
with torch.no_grad():
    outputs = model(**inputs, decoder_input_ids=targets["input_ids"], output_attentions=True)

# Visualize attention for the first head
cross_attention_weights = outputs.cross_attentions[6][0, 9].cpu().numpy() # First example, first head

# Plot attention matrix
plt.figure(figsize=(10, 6))
sns.heatmap(cross_attention_weights, 
            xticklabels=tokens_inputs, 
            yticklabels=tokens_targets, 
            cmap="Blues", 
            annot=True)
# plt.title("Cross Attention Matrix (Head 0)")
plt.xlabel("Source Tokens")
plt.ylabel("Target Tokens")
plt.show()
