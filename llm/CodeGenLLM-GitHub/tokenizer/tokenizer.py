import re
from collections import Counter

class ByteTokenizer:
    def __init__(self):
        self.vocab = list(set(chr(i) for i in range(256)))
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text):
        return [self.stoi.get(ch, 0) for ch in text]

    def decode(self, tokens):
        return ''.join([self.itos.get(tok, '?') for tok in tokens])