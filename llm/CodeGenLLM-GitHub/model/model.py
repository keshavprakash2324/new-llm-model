import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size),
        )
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        return self.norm2(x + ff_output)

class CodeGenLLM(nn.Module):
    def __init__(self, vocab_size, embed_size=256, num_heads=4, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.blocks = nn.Sequential(*[TransformerBlock(embed_size, num_heads) for _ in range(num_layers)])
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.blocks(x)
        return self.fc(x)