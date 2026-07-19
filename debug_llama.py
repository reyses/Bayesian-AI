import os
from llama_cpp import Llama, LlamaGrammar
import time

model_blob = "/mnt/c/Users/reyse/OneDrive/Desktop/Bayesian-AI/research/dojo_forge/pipeline/model.gguf"
print("Loading model...")
llm = Llama(model_path=model_blob, n_gpu_layers=-1, n_ctx=4096, n_batch=512, seed=42, temperature=0.0, logits_all=True)

genome_path = "/mnt/c/Users/reyse/OneDrive/Desktop/Bayesian-AI/research/dojo_forge/genome/GENOME.md"
if os.path.exists(genome_path):
    with open(genome_path, 'r', encoding='utf-8') as f:
        genome_text = f.read()
else:
    genome_text = ""

system_prompt = f"Decide to HOLD or EXIT based on the frame. If EXIT, provide a reason.\n\nRULES (Genome):\n{genome_text}"

print("Tokenizing system prompt...")
llm.reset()
sys_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
sys_tokens = llm.tokenize(sys_text.encode('utf-8'), add_bos=False, special=True)
sys_len = len(sys_tokens)
print(f"System tokens: {sys_len}")

print("Evaluating system prompt...")
t0 = time.time()
for i in range(0, len(sys_tokens), llm.n_batch):
    print(f"Eval chunk {i} to {i + llm.n_batch}")
    llm.eval(sys_tokens[i:i + llm.n_batch])
print(f"Done in {time.time()-t0:.2f}s")
print("SUCCESS!")
