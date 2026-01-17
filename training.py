"""
TO DO:
Make and push to git repo 
Maybe use torch saving instead of pickle
Add Comments to Code
find proper dataset for chatbot and train(!^! annyoing!)
"""

import argparse
import mmap
import pickle
import random
import time

import torch
import torch.nn as nn
from torch.nn import functional as F


def get_device():
    try: 
        import torch_directml
        device = torch_directml.device(0)
        print("\nUsing DirectML device(NPU/Radeon GPU)")
    except ImportError: 
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("\nUsing CUDA Device (NVIDA GPU)")
        else:
            device = torch.device("cpu")
            print("\nUsing CPU device")
    return device

def encode(string, string_to_int):
    return [string_to_int[c] for c in string]

def decode(indices, int_to_string):
    return ''.join(int_to_string[i] for i in indices)

def get_random_chunk(split, block_size, batch_size, string_to_int, device):
    filename = "output_train.txt" if split == 'train' else 'output_val.txt'

    input_list = []
    target_list = []

    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            if file_size <= block_size:
                raise ValueError(
                    f"File too small: file_size={file_size}, block_size={block_size}"
                )

            for _ in range(batch_size):
                start_pos = random.randint(0, file_size - block_size - 1)
                mm.seek(start_pos)

                block = mm.read(block_size + 1)
                block_text = block.decode('utf-8', errors='ignore').replace('\r', '')

                if len(block_text) < block_size + 1:
                    block_text += block_text[-1] * (block_size + 1 - len(block_text))

                encoded = [string_to_int[c] for c in block_text]

                input_seq = torch.tensor(encoded[:block_size], dtype=torch.long)
                target_seq = torch.tensor(encoded[1:block_size + 1], dtype=torch.long)

                input_list.append(input_seq)
                target_list.append(target_seq)

    input_batch = torch.stack(input_list).to(device)
    target_batch = torch.stack(target_list).to(device)

    return input_batch, target_batch

@torch.no_grad()
def estimate_loss(model, string_to_int, device, block_size, eval_iters, batch_size):
    avg_loss = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            input_seq, target_seq = get_random_chunk(split, block_size, 
                                              batch_size, string_to_int, 
                                              device)

            logits, loss = model(input_seq, target_seq)
            losses[i] = loss.item()
        avg_loss[split] = losses.mean()
    model.train()
    return avg_loss

class Head(nn.Module):

    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_sz, seq_length, embd_dim = x.shape
        key = self.key(x)
        query = self.query(x)

        attn_scores = query @ key.transpose(-2,-1) * key.shape[-1]**-0.5
        attn_scores = attn_scores.masked_fill(self.tril[:seq_length, :seq_length] == 0, float("-inf"))
        attn_scores = F.softmax(attn_scores, dim=-1)
        attn_scores = self.dropout(attn_scores)

        value = self.value(x)   
        wgt_attn_vals = attn_scores @ value
        return wgt_attn_vals
       

class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, head_size, n_embd, block_size, dropout):

        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(n_heads)])
        self.proj = nn.Linear(head_size * n_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = torch.cat([h(x) for h in self.heads], dim=-1)
        attn_out = self.dropout(self.proj(attn_out))
        return attn_out
       

class FeedForward(nn.Module):

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)
       

class Block(nn.Module):

    def __init__(self, n_embd, block_size, dropout, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        head_size = n_embd // n_head
        self.self_attn = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = self.ln1(x + self.self_attn(x))
        x = self.ln2(x + self.ffwd(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size, device, n_embd, n_head, block_size, dropout, n_layer):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, block_size, dropout, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.device = device

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        batch_sz, seq_length = index.shape        

        index = index.to(self.token_embedding_table.weight.device)
        tok_emd = self.token_embedding_table(index)
        
        pos = torch.arange(seq_length, device=index.device)
        pos_emd = self.position_embedding_table(pos)

        x = tok_emd + pos_emd
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
       
        if targets is None:
            loss = None
        else:
            batch_sz, seq_length, embd_dim = logits.shape
            logits = logits.reshape(batch_sz*seq_length, embd_dim)
            targets = targets.reshape(batch_sz*seq_length)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    @torch.no_grad()
    def generate(self, index, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            index_cond = index[:, -self.block_size:]
            logits, loss = self.forward(index_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index  

def load_model():
    print("Loading Model Parameters... ") 
    with open('model-01.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Loaded Model Parameters Successfully!")
    return model

def train_model(model, string_to_int, device, block_size, eval_iters, batch_size, learning_rate, max_iter):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, foreach=False)

    start_time = time.time()

    try: 
        for iter in range(1, max_iter+1):
            inputs, targets = get_random_chunk('train', block_size, 
                                        batch_size, string_to_int, 
                                        device)
            
            logits, loss = model(inputs, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if iter % eval_iters == 0:
                losses = estimate_loss(model, string_to_int, device, block_size, eval_iters, batch_size)
                end_time = time.time()
                elapsed_time = end_time - start_time
                avg_step_time = elapsed_time / eval_iters
                print(f"Step: {iter}, Train Loss: {losses['train']:.3f}, Val Loss: {losses['val']:.3f}, Eval Iteration Time: {elapsed_time:.3f}, Average Step Time: {avg_step_time:.3f}")
                start_time = time.time()

    except KeyboardInterrupt: 
        print("\nTraining Exited... ")
        return

    print(f"Final Loss: {loss.item()}")

def save_model(model):
    with open('model-01.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model Saved")

def main():
    print("Welcome to noaGPT Training! (Ctrl + C) to exit\n")

    parser = argparse.ArgumentParser(description="Hyperparameter Arguments")

    parser.add_argument("--block_size", type=int, default=128, help="Context length / block size")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_iter", type=int, default=10000, help="Maximum training iterations")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--eval_iters", type=int, default=20, help="Number of evaluation iterations")
    parser.add_argument("--n_embd", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--n_head", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--n_layer", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")

    args = parser.parse_args()

    print("Model Configuration: ")
    for arg in vars(args):
        print(f"  {arg:15} = {getattr(args, arg)}")

    block_size = args.block_size
    batch_size = args.batch_size
    max_iter = args.max_iter
    learning_rate = args.learning_rate
    eval_iters = args.eval_iters
    n_embd = args.n_embd
    n_head = args.n_head
    n_layer = args.n_layer
    dropout = args.dropout
    
    torch.set_default_dtype(torch.float32)

    device = get_device()
    print(f"Current Device: {device}\n")

    with open('vocab.txt', 'r', encoding='utf-8') as f:
        text = f.read()
        chars = sorted(list(set(text)))

    vocab_size = len(chars) 

    string_to_int = {ch:i for i, ch in enumerate(chars) }
    int_to_string = {i:ch for i, ch in enumerate(chars) }

    #change this depending on if mode exists yet
    model = GPTLanguageModel(vocab_size, device, n_embd, n_head, block_size, dropout, n_layer).to(device)

    #model = load_model()
    model.to(device)

    train_model(model, string_to_int, device, block_size, eval_iters, batch_size, learning_rate, max_iter)
    save_model(model)

if __name__ == "__main__":
    main()