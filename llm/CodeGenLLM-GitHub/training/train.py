import torch
from torch.utils.data import DataLoader, Dataset

class CodeDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size=64):
        self.tokenizer = tokenizer
        self.data = []
        for text in texts:
            ids = tokenizer.encode(text)
            for i in range(0, len(ids) - block_size):
                self.data.append((ids[i:i+block_size], ids[i+1:i+1+block_size]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x), torch.tensor(y)

def train(model, data_loader, optimizer, criterion, device):
    model.train()
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out.view(-1, out.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item():.4f}")