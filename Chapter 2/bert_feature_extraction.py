import datasets
rotten_tomatoes = datasets.load_dataset("rotten_tomatoes")

import transformers
tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

def tokenize(tokenizer, data, max_length=300):
    return tokenizer(data, 
                     max_length=max_length, 
                     truncation=True, 
                     padding=True,
                     return_tensors="pt")

train_data_tokenized = tokenize(tokenizer=tokenizer, 
                                data=rotten_tomatoes['train']['text'])
test_data_tokenized = tokenize(tokenizer=tokenizer, 
                               data=rotten_tomatoes['test']['text'])


import torch
def extract_features(model, data_tokenized):
    with torch.no_grad():
        features = model(data_tokenized['input_ids'],
                         data_tokenized['attention_mask'])
        # Only get the embedding vector of the first [CLS] token.
        # This token represents the entire sequence.
        # Shape is [batch, tokens, embedding_size]
        features = features.last_hidden_state[:, 0, :]
        return features

model = transformers.AutoModel.from_pretrained("distilbert/distilbert-base-uncased")
features_train = extract_features(model, train_data_tokenized)
features_test = extract_features(model, test_data_tokenized)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
clf = RandomForestClassifier(max_depth=10, random_state=0)
clf.fit(features_train.numpy(), rotten_tomatoes['train']['label'])
y_pred = clf.predict(features_test.numpy())

score = accuracy_score(rotten_tomatoes['test']['label'], y_pred)
print(score)
