import argparse
import pickle

import torch
import torch.nn as nn
from torch.nn import functional as F

def get_device():
    """
    Selects available compute device

    Order: 
    1. DirectML Device
    2. CUDA Device 
    3. CPU

    Returns:
        The selected device
    """

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

def encode(string: str, string_to_int: dict[str, int]) -> list[int]:
    """Encodes a string of charcters into a list of integers"""
    return [string_to_int[c] for c in string]

def decode(indices: list[int], int_to_string: dict[int, str]) -> str:
    """Decodes a list of integers into a string of characters"""
    return ''.join(int_to_string[i] for i in indices)

def load_model() -> nn.Module:
    print("Loading Model Parameters... ") 
    with open('model-01.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Loaded Model Parameters Successfully!")
    return model

class Head(nn.Module):
    """
    A single casual dot-product attention head

    Args: 
        block_size: The maximum amount of tokens in a sequence of tokens
        dropout: The probability of turning off certain nuerons
        head_size: The number of dimensions per attenttion head
        n_embd: The number of dimensions of the token embedding vector
    """

    def __init__(self, 
                 block_size: int, 
                 dropout: float, 
                 head_size: int, 
                 n_embd: int):
        
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        """
        Forward pass that calculates casual dot-product attention scores

        1. The key and query vectors are mutiplied together
        and mutiplied by the square root of the head size
        (to keep the vectors in a resonable range)
        2. Tril function is used to prevent attention scores
        from being calculated for future values
        3. Softmax is then used to convert the attention scores
        into probabilites
        4. Dropout is used to turn off some of the neurons in the model
        in order to prevent a head from focusing on the same token every time
        5. The attention scores are then mutiplied by the value vector
        to finalize the attention calculation for this head

        Args:
            x: The input tensor of shape (batch_size, seq_length, n_embd)
            
        Returns: 
            wgt_attn_vals: A tensor of shape (batch_size, seq_length, n_embd)
            that contains context-aware embeddings for each token
        """
        batch_size, seq_length, n_embd = x.shape
        key = self.key(x)
        query = self.query(x)

        attn_scores = query @ key.transpose(-2,-1) * key.shape[-1]**-0.5
        attn_scores = attn_scores.masked_fill(self.tril[:seq_length, :seq_length] == 0, 
                                              float("-inf"))
        attn_scores = F.softmax(attn_scores, dim=-1)
        attn_scores = self.dropout(attn_scores)

        value = self.value(x)   
        wgt_attn_vals = attn_scores @ value
        return wgt_attn_vals
       
class MultiHeadAttention(nn.Module):

    """
    The Multi-Head Attention Mechanism that calculates 
    the attention scores for all the heads

    Args:
        block_size: The maximum amount of tokens in a sequence of tokens
        dropout: The probability of turning off certain nuerons
        head_size: The number of dimensions per attenttion head
        n_embd: The number of dimensions of the token embedding vector
        n_heads: The number of heads in the attention mechanism 
    """

    def __init__(self, 
                 block_size: int,
                 dropout: float,
                 head_size: int,
                 n_embd: int,
                 n_heads: int ):

        super().__init__()
        self.heads = nn.ModuleList([Head(block_size, dropout, head_size, n_embd) 
                                    for _ in range(n_heads)])
        self.proj = nn.Linear(head_size * n_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Concatenates all the heads context information into each token,
        projects the tensor back to the embedding dimension, and applies
        dropout to prevent overfitting

        Args:
            x: The input tensor of shape (batch_size, seq_length, n_embd)

        Returns:
            multi_head_attention_score: The output tensor of shape 
            (batch_size, seq_length, n_embd)
            with context aware token embeddings from all heads
        """
        multi_head_attention_score = torch.cat([h(x) for h in self.heads], dim=-1)
        multi_head_attention_score = self.dropout(self.proj(multi_head_attention_score))
        return multi_head_attention_score
       

class FeedForward(nn.Module):
    """
    The Feed-Forward Mechanism

    Args:
        dropout: The probability of turning off certain nuerons
        n_embd: The number of dimensions of the token embedding vector
    """

    def __init__(self, dropout: float, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the Feed-Forward Mechanism

        1. Linear transforms the input tensor to expand the dimensionality
        and allow for the non-linear transformation 
        2. Non-linear transformation on the tensor to allow for more complex
        connections to be made
        3. Another linear transformation to compress it back to the embedding dimension
        4. Dropout to prevent overfitting

        Args:
            x: The input tensor of shape (batch_size, seq_length, n_embd)

        Returns: 
            self.net(x): A tensor of shape (batch_size, seq_length, n_embd) now with
            non-linear transformations to allow for more complex relationships 
        """
        return self.net(x)
       

class Block(nn.Module):

    """
    A single Decoder Block that containts the Multi-Head Attention
    the Residual Connection, and the Feed-Forward Mechanism

    Args: 
        block_size: The maximum amount of tokens in a sequence of tokens
        dropout: The probability of turning off certain nuerons
        n_embd: The number of dimensions of the token embedding vector
        n_heads: The number of heads in the attention mechanism 
    """


    def __init__(self, 
                 block_size: int, 
                 dropout: float, 
                 n_embd: int, 
                 n_head: int):
        
        super().__init__()
        assert n_embd % n_head == 0
        head_size = n_embd // n_head
        self.self_attn = MultiHeadAttention(block_size, dropout, head_size, n_embd, n_head)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        A forward pass through the Decoder Block

        1. Passes the input tensor through Multi-Head Attention, 
        allowing for context awareness
        2. Residual connection to prevent gradient problems and 
        keep what the network has already learned
        3. Layer Normilzation to prevent vectors getting too large
        or too small
        4. Feed-Forward to allow for complex non-linear pattern
        recognition 
        5. Another residual connection
        6. Another Layer Normilization

        Args:
            x: An input tensor of shape (block_size, seq_length, n_embd)

        Returns: 
            x: An output tensor of shape (block_size, seq_length, n_embd)
            with tokens that are context aware, normalized, and have non-linear
            relationship information
            
        """
        x = self.ln1(x + self.self_attn(x))
        x = self.ln2(x + self.ffwd(x))
        return x

class GPTLanguageModel(nn.Module):
    """
    The full model architecture

    Args:
        block_size: The maximum amount of tokens in a sequence of tokens
        device: The compute device
        dropout: The probability of turning off certain nuerons
        n_embd: The number of dimensions of the token embedding vector
        n_heads: The number of heads in the attention mechanism 
        n_layer: The number of decoder blocks
        vocab_size: The size of the model vocabulary
    """
    def __init__(self, 
                 block_size: int,
                 device: torch.device, 
                 dropout: float, 
                 n_embd:int, 
                 n_head: int,
                 n_layer:int,
                 vocab_size: int): 
        
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, block_size, dropout, n_head=n_head) 
                                      for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.device = device

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """
        Intitalizes the model weights

        Args:
            module: A module or layer of the model architecture defined 
            by the subclass nn.Module
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index: torch.Tensor, targets: torch.Tensor = None):
        """
        Calculate the logits and loss, if targets is provided, for a batch of tokens

        1. Convert token and positional indicies into vectors with
        the embedding tables and add them together
        2. Pass the tensor through the blocks of the model architecture
        3. Layer normilzation to stabalize the model
        4. A final linear layer to create the logits from the embeddings
        5. (If targets is given) reshape both logits and targets to the 
        same dimensionality and then use cross-entropy to calculate the loss

        Args:
            index: A tensor with token indicies of size (batch_size, seq_length)
            targets: Optional tensor of size (batch_size, seq_length) 
            containing target indicies for computing loss 
        """
        batch_size, seq_length = index.shape        

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
            batch_size, seq_length, embd_dim = logits.shape
            logits = logits.reshape(batch_size*seq_length, embd_dim)
            targets = targets.reshape(batch_size*seq_length)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    @torch.no_grad()
    def generate(self, index: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Generate new tokens given a starting sequence

        1. Get the last "block_size" tokens 
        2. Get the logits through a forward pass of the model and 
        focus on only the last token 
        3. Convert the logits to probabilities using softmax
        4. Sample on token from the probabilites using multinomial so
        it dosen't just pick the most likely one
        5. Add the predicted token to the end of the sequence
        6. After looping through all the new tokens, return the final sequence

        Args:
            index: A tensor with token indicies of size (batch_size, seq_length)
            max_new_tokens: The number of new tokens to generate

        Returns: 
            index: A tensor of shape (batch_size, seq_length + max_new_tokens) 
            of predicted tokens and the orignal tokens
        """

        self.eval()
        for _ in range(max_new_tokens):
            index_cond = index[:, -self.block_size:]
            logits, loss = self.forward(index_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index  
    
def main():
    print("Welcome to noaGPT! (Ctrl + C) to exit\n")

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

    try: 
        model = load_model()
    except FileNotFoundError:
        print("Model not found, please check if the specified file is correct or create a new model")
    
    model.to(device)

    while True:
        try: 
            prompt = input("Prompt:\n")
            context = torch.tensor(encode(prompt, string_to_int), dtype=torch.long, device=device)
            generated_chars = decode(model.generate(context.unsqueeze(0),
                                                     max_new_tokens=150)[0].tolist(), int_to_string)
            print(f"Completion:\n{generated_chars}")
        except KeyboardInterrupt: 
            print("Exited Model...")
            break

if __name__ == "__main__":
    main()

