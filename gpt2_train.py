from transformers import AutoTokenizer, AutoModelWithLMHead
import torch
from torch.utils.data import DataLoader, Dataset

# Define a PyTorch Dataset for the tokens
class TokenDataset(Dataset):
    def __init__(self, tokens, chunk_size):
        self.tokens = tokens
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.tokens) // self.chunk_size

    def __getitem__(self, idx):
        start_idx = idx * self.chunk_size
        end_idx = start_idx + self.chunk_size
        token_chunk = self.tokens[start_idx:end_idx]
        input_ids = torch.tensor(token_chunk)
        return input_ids

# Load the tokenizer and the pre-trained GPT-2 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelWithLMHead.from_pretrained("gpt2").to(device)

# Load the Harry Potter corpus and tokenize it
with open("harrypotter.txt", "r", encoding="utf-8") as f:
    text = f.read()
    tokens = tokenizer.encode(text, add_special_tokens=False)

# Create a PyTorch DataLoader for the tokens
chunk_size = 1024
batch_size = 1
token_dataset = TokenDataset(tokens, chunk_size)
token_loader = DataLoader(token_dataset, batch_size=batch_size, shuffle=True)

# Fine-tune the model on the token loader and save checkpoints
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
for epoch in range(1):
    total_loss = 0
    for input_ids in token_loader:
        input_ids = input_ids.to(device)
        loss = model(input_ids, labels=input_ids)[0]
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch}, average loss: {total_loss / len(token_loader):.4f}")
    model.save_pretrained(f"checkpoint_{epoch}")
