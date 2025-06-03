import gradio as gr
import torch
from tokenizer.tokenizer import ByteTokenizer
from model.model import CodeGenLLM
from inference.infer import generate

# Load model and tokenizer
vocab_size = 256
model = CodeGenLLM(vocab_size)
model.load_state_dict(torch.load("model_weights.pth", map_location=torch.device('cpu')))
model.eval()
tokenizer = ByteTokenizer()

def codegen_interface(prompt):
    return generate(model, tokenizer, prompt, length=100)

demo = gr.Interface(fn=codegen_interface, inputs="text", outputs="text", title="CodeGenLLM 🧠", description="Generate Python code from a prompt.")
demo.launch()