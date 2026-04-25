from openai import OpenAI
client = OpenAI(api_key="...")

response = client.responses.create(model="gpt-4.1",
                                   input="Write a one-sentence about PyGAD.")

print(response.output_text)
