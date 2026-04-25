from datasets import load_dataset
dataset = load_dataset("ucirvine/sms_spam")

from transformers import AutoTokenizer
model_name = 'prajjwal1/bert-tiny'
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

from transformers import AutoModelForSequenceClassification
# Load the trained model.
model = AutoModelForSequenceClassification.from_pretrained("sms_model")

print("Test Sample")
# This is spam because it includes a phone number.
# sample = "You have a very good offer. Contact us soon to claim it. Reach 111111111!"
# This is ham because it does not include a phone number.
# sample = "You have a very good offer. Contact us soon to claim it. Reach us!"
sample = dataset['train']['sms'][2300]
label = dataset['train']['label'][2300]
sample_class = model.config.id2label[label]
print(sample)
print(label, sample_class)

print("Prediction")
# Evaluate
import torch
# Use the base tokenizer to encode the samples.
sample_tokenized = tokenizer.encode(sample, return_tensors="pt")
# Return the logits.
pred_logits = model(sample_tokenized).logits
print(pred_logits)
pred_label = torch.argmax(pred_logits).tolist()
pred_class = model.config.id2label[pred_label]
print(pred_label, pred_class)

print("Pipeline Prediction")
# Predictions using a pipeline
from transformers import pipeline
# Load pipeline for text classification
classifier = pipeline("text-classification", 
                      model="sms_model", 
                      tokenizer=tokenizer)
# Predict on a sample
prediction = classifier(sample)
print(prediction)
