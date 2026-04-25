from datasets import load_dataset
dataset = load_dataset("ucirvine/sms_spam")

from transformers import AutoTokenizer
model_name = 'prajjwal1/bert-tiny'
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
def tokenize_function(examples):
    return tokenizer(text=examples["sms"], 
                     padding="max_length", 
                     max_length=186,
                     truncation=True)
tokenized_datasets = dataset.map(tokenize_function, batched=True)

id2label = {0: "ham",
            1: "spam"}

label2id = {v: k for k, v in id2label.items()}

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name, 
                                                           num_labels=2,
                                                           id2label=id2label,
                                                           label2id=label2id)

from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(task_type='SEQ_CLS',
                         r=4,
                         lora_alpha=32,
                         lora_dropout=0.1,
                         target_modules=['query'])

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

# Model Hyperparameters
from transformers import TrainingArguments, Trainer
training_args = TrainingArguments(output_dir="sms_fine_tuning_LoRA", 
                                  learning_rate=0.00001,
                                  num_train_epochs=50,
                                  weight_decay=0.01,
                                  eval_strategy="epoch", 
                                  save_strategy="epoch",
                                  load_best_model_at_end=True,
                                  report_to="none")

# Evaluation metrics
import numpy
import evaluate
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = numpy.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(model=peft_model,
                  args=training_args,
                  train_dataset=tokenized_datasets['train'],
                  eval_dataset=tokenized_datasets['train'],
                  compute_metrics=compute_metrics)

trainer.train()

trainer.save_model("sms_model_loRA")
# No need to save the tokenizer since we did not create any new tokens.
# tokenizer.save_pretrained("sms_model")
