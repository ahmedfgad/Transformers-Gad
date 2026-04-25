## Load the dataset
from datasets import load_dataset

dataset = load_dataset("sentence-transformers/eli5", split="train[:500]")
dataset = dataset.map(lambda x: {"question": x["question"], "answer": x["answer"]},
                      remove_columns=dataset.column_names)


## Save the responses from the Teacher
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# It has around 1.5B parameters compared to just 82M for distilGPT2.
device = torch.device("cpu")
teacher_model = AutoModelForCausalLM.from_pretrained("gpt2-xl").to(device)
teacher_tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")

def generate_teacher_response(example):
    question = example["question"]
    inputs = teacher_tokenizer(question,
                               return_tensors="pt",
                               truncation=True).to(device)
    outputs = teacher_model.generate(**inputs,
                                     max_new_tokens=50,
                                     pad_token_id=teacher_tokenizer.eos_token_id)
    decoded = teacher_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the question from the decoded output
    teacher_response = decoded[len(question):].strip()

    return {"teacher_output": teacher_response}

dataset_teacher = dataset.map(generate_teacher_response, batched=False)
dataset_teacher.save_to_disk("teacher_augmented_dataset")

print(dataset[0])
print(dataset_teacher[0])


## Prepare student for training
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

student_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
student_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# Set padding token if missing
if student_tokenizer.pad_token is None:
    student_tokenizer.pad_token = student_tokenizer.eos_token


def tokenize_student(example):
    full_text = example["question"] + " " + example["teacher_output"]
    tokenized = student_tokenizer(full_text,
                                  truncation=True,
                                  padding="max_length",
                                  max_length=512)

    input_ids = tokenized["input_ids"]

    # Calculate question token length without special tokens
    question_len = len(student_tokenizer(example["question"], add_special_tokens=False)["input_ids"])

    # Mask question tokens in labels, keep teacher output tokens
    # The question tokens are masked (i.e., set to -100 in the labels) because we don’t want the student model to learn to predict them during training. Instead, we only want it to learn to predict the teacher's response, given the question as input.
    labels = [-100] * question_len + input_ids[question_len:]
    labels = labels[:512]

    tokenized["labels"] = labels
    return tokenized


tokenized_dataset = dataset_teacher.map(tokenize_student, 
                                        remove_columns=dataset_teacher.column_names)

data_collator = DataCollatorForLanguageModeling(student_tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./student-distilled",
    per_device_train_batch_size=2,
    num_train_epochs=50,
    logging_steps=10,
    save_strategy="no",
)

trainer = Trainer(
    model=student_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=student_tokenizer,
    data_collator=data_collator,
)


## Evaluate before training
from transformers import pipeline

student_generator = pipeline("text-generation",
                             model=student_model,
                             tokenizer=student_tokenizer,
                             device=-1)

examples = dataset_teacher.select(range(5))

print("== Before training ==")
for ex in examples:
    question = ex["question"]
    print("Question:")
    print(question)

    student_output = student_generator(question,
                                       max_new_tokens=50,
                                       do_sample=False)[0]["generated_text"]
    student_response = student_output[len(question):].strip()

    teacher_output = teacher_generator(question,
                                       max_new_tokens=50,
                                       do_sample=False)[0]["generated_text"]
    teacher_response = teacher_output[len(question):].strip()

    print("Teacher Response:")
    print(teacher_response)
    print("Student Response:")
    print(student_response)

## Train the student
trainer.train()

student_model.save_pretrained("./distilled-student")
student_tokenizer.save_pretrained("./distilled-student")


## Post-training predictions (compare student and teacher)
teacher_generator = pipeline("text-generation",
                             model=teacher_model,
                             tokenizer=teacher_tokenizer,
                             device=-1)

print("== After training ==")
for ex in examples:
    question = ex["question"]
    print("Question:")
    print(question)

    student_output = student_generator(question,
                                       max_new_tokens=50,
                                       do_sample=False)[0]["generated_text"]
    student_response = student_output[len(question):].strip()

    teacher_output = teacher_generator(question,
                                       max_new_tokens=50,
                                       do_sample=False)[0]["generated_text"]
    teacher_response = teacher_output[len(question):].strip()

    print("Teacher Response:")
    print(teacher_response)
    print("Student Response:")
    print(student_response)
