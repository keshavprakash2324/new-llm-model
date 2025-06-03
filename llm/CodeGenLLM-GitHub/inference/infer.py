def generate(model, tokenizer, prompt, length=50, device='cpu'):
    model.eval()
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], device=device)
    for _ in range(length):
        with torch.no_grad():
            out = model(input_ids)
            next_token = out[:, -1, :].argmax(dim=-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    return tokenizer.decode(input_ids[0].tolist())