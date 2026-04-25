import transformers

model_name = "t5-small"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)

input_text = "Translate from English to French: I like football!"
inputs = tokenizer(input_text, return_tensors="pt")

num_beams = 2
outputs = model.generate(**inputs, 
                         max_length=50,
                         num_beams=num_beams,
                         num_return_sequences=num_beams)

for beam in range(len(outputs)):
    generated_text = tokenizer.decode(outputs[beam], 
                                      skip_special_tokens=True)
    print(f"Beam {beam}: {generated_text}")