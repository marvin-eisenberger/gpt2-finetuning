from transformers import pipeline, set_seed

generator = pipeline('text-generation', model='gpt2')

set_seed(42)

prompt = "Harry Potter was walking down Diagon Alley when he saw"

generated_text = generator(prompt, max_length=1000, do_sample=True, temperature=0.7)[0]['generated_text']

print(generated_text)
