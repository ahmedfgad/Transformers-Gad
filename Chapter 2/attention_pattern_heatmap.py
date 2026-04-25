import torch
import transformers
import matplotlib.pyplot
import numpy

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

# Input text
text = "The quick brown fox jumped over the lazy dog while running as crazy."

# Tokenize input text
# The text is tokenized into input IDs that BERT can understand.
# It is a dictionary with these items:
    # input_ids: These are the token IDs corresponding to the tokens in the input text, based on the model's vocabulary.
        # Shape: This will be a tensor of shape (batch_size, sequence_length), where batch_size is the number of sentences you input, and sequence_length is the number of tokens (words/subwords) in the sentence.
    # attention_mask: This tensor indicates which tokens should be attended to and which should be ignored (usually padding tokens).
        # Without an attention mask, the model might mistakenly focus on padding tokens, which are not useful for the task at hand.
        # Shape: Same as input_ids — (batch_size, sequence_length).
        # It can have values of either 0 or 1. 
        # 1: Each 1 in the mask means that the corresponding token should be attended to by the model.
        # 0: A 0 in the mask means that the corresponding token should be ignored during attention calculations (usually this corresponds to padding tokens).

inputs = tokenizer(text, return_tensors="pt")

"""
text = ["The quick brown fox jumped over the lazy dog while running as crazy.",
        "While running as crazy."]
# Tokenize the text (a list of sentences)
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
"""

# Extract token IDs (input_ids) from the tokenized input
input_ids = inputs['input_ids']

# Convert token IDs back to tokens (words or subwords)
# Use [0] to access the first (and typically only) batch
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# output_attentions=True ensures attention weights are returned
model = transformers.BertModel.from_pretrained(model_name, 
                                               output_attentions=True)

# Forward pass to get attention weights
# We pass the input through the model, and it returns a tuple that includes the attention weights.
with torch.no_grad():
    # outputs = model(**inputs)
    outputs = model(input_ids=inputs['input_ids'], 
                    attention_mask=inputs['attention_mask'])

# Extract attention weights (list of attention layers/tensors)
# List of attention matrices, one for each layer
# The length of this list of tensors equal to the number of attention blocks/layers (e.g. 12 for BERT).
attentions = outputs.attentions

# Each layer has multiple attention heads. 
# Each attention head create an attention pattern of shape equal to the number of tokens extracted from the input text.
# Attention pattern (i.e. tensor) shape: [batch_size, num_heads, seq_length, seq_length]
# For example: [1, 12, 16, 16]
# There is only 1 input sequence (i.e. sentence).
# There are 12 heads for each attention block/layer.
# The number of tokens in the text is 16.

# The model’s output includes a list of attention weights, one per layer. We extract the weights for the first layer and first attention head for visualization.
# Example: Visualize attention from the first layer and the first attention head
layer_num = 0
attention_head = 0

# Get attention weights for the specified layer and head
attention = attentions[layer_num][0, attention_head].cpu().numpy()

# Plot attention pattern
matplotlib.pyplot.figure(figsize=(10, 8))
matplotlib.pyplot.imshow(attention, cmap="Blues", interpolation="nearest")
matplotlib.pyplot.colorbar()
# matplotlib.pyplot.title(f"Attention Pattern - Layer {layer_num + 1}, Head {attention_head + 1}")
matplotlib.pyplot.xlabel("Input Tokens")
matplotlib.pyplot.ylabel("Input Tokens")

# Set the x and y axis ticks to correspond to the tokens
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].tolist())
matplotlib.pyplot.xticks(numpy.arange(len(tokens)), tokens, rotation=90)
matplotlib.pyplot.yticks(numpy.arange(len(tokens)), tokens)

matplotlib.pyplot.show()