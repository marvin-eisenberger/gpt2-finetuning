import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelWithLMHead.from_pretrained("checkpoint_0").to(device)

model.eval()

input_str = "\n"
input_ids = tokenizer.encode(input_str, return_tensors="pt").to(device)
output = model.generate(input_ids, max_length=1000, do_sample=True)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
