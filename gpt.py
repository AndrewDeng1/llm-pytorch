import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32  # num independent sequences processed in parallel (reduced for CPU)
block_size = 64  # maximum context length for predictions (reduced for speed)
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)

eval_iters = 50  # reduced from 200 to speed up evaluation
n_embd = 128  # reduced from 384 (fewer parameters)
n_head = 4  # reduced from 6
n_layer = 3  # reduced from 6 (much faster)
dropout = 0.2

# ORIGINAL HYPERPARAMETERS
# batch_size = 64  # num independent sequences processed in parallel
# block_size = 256  # maximum context length for predictions
# max_iters = 5000
# eval_interval = 500
# learning_rate = 3e-4
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# eval_iters = 200
# n_embd = 384
# n_head = 6
# n_layer = 6
# dropout = 0.2
# -----

torch.manual_seed(67)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
# all unique characters that occur in text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encodes string into tokens (one token per char)
decode = lambda l: ''.join([itos[i] for i in l])  # decodes list of tokens back into chars

# Train-test split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))  # first 90% of data for training
train_data = data[:n]
val_data = data[n:]

# Samples a single batch of either train/validation data
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    
    # Samples batch_size number of random indices between [0, len(data)-block_size-1]
    # stored in 1D tensor
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Slices "data" tensor to get the blocks, then stacks to get (B x T)
    # or (64 x 256)
    x = torch.stack([data[i:i+block_size] for i in ix])
    
    # Shifts right to get corresponding target sequence, stacks accordingly
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    # Moves tensors to device
    x, y = x.to(device), y.to(device)
    return x, y

# Tells torch to disable gradient tracking, since only evaluating
@torch.no_grad()
def estimate_loss():
    out = {}
    
    # Sets model to evaluation mode
    model.eval()
    
    # Evaluates on training and validation data
    for split in ['train', 'val']:
        
        # Create tensor to store losses for each evaluation iteration
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            
            # sample batch
            X, Y = get_batch(split)
            
            # Pass in model and generate logits and loss
            logits, loss = model(X, Y)
            
            # Store loss
            losses[k] = loss.item()
        
        # Get mean
        out[split] = losses.mean()

    # Back to training mode
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """
    
    def __init__(self, head_size):
        super().__init__()
        
        # Initialize k, q, v projection matrices
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # Initialize tril for masked self attention
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        # Init dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # input of size (B, T, d_model)
        # output of size (B, T, d_head)
        B, T, C = x.shape
        
        # Generate K and Q matrices by applying projection
        k = self.key(x)  # (B, T, d_head)
        q = self.query(x)  # (B, T, d_head)
        
        # Compute attention scores
        # (Q*K^T)/sqrt(d_head)
        
        # Note: 1/sqrt(d_head) is to maintain the original variance of the q and k vectors
        # so softmax doesn't tend to become one-hot encoded vector
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B, T, d_head) @ (B, d_head, T) -> (B, T, T)
        
        # Gets lower triangular matrix, converts to true/false where true if 0, false otherwise
        # Then masks weight matrix by applying -inf to where it is true
        # Thus masks future tokens from contributing to attention score
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)    
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        
        # Perform weighted aggregation of the values
        # Generate V matrix
        v = self.value(x)  # (B, T, d_head)
        out = wei @ v  # (B, T, T) @ (B, T, d_head) -> (B, T, d_head)
        return out
    

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        
        # Generates list of heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

        # Final projection matrix to return back to d_model
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        # Passes x through each of heads, and concatenates result
        # along last dimension, so back to d_model
        out = torch.cat([h(x) for h in self.heads], dim=-1)

        # Applies projection and dropout
        out = self.dropout(self.proj(out))
        return out
    
    
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    
    def __init__(self, n_embd):
        super().__init__()
        
        # Defines list of layers sequentially applied
        self.net = nn.Sequential(
            
            # Linear layer expand to 4*d_model + activation function
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            
            # Compress back to d_model dimensions
            nn.Linear(4*n_embd, n_embd),
            
            # Apply dropout
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication (attention) followed by computation (ffn) """
    
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        
        # Choose d_head such that d_model evenly across all heads
        head_size = n_embd // n_head
        
        # Create MHA and FFN layers
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)

        # Layer norm, applied before x passed into MHA/FFN
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        
        # Apply MHA and FFN, with pre-norm
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
        
        
class GPTLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        # Better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx, targets=None):
        
        # idx is the batch of sequences of tokens
        B, T = idx.shape
        
        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            
            # Shallow reshape of logits tensor to (B*T, C)
            # I.e., Data itself not modified at all, but shape is (strides modified)
            # So logits becomes shape:
            # each token x probability distribution over vocabulary
            logits = logits.view(B*T, C)
            
            # targets is assumed to be (B, T) shape tensor containing ground truth tokens
            # Reshape so one long sequence of ground truth tokens
            targets = targets.view(B*T)
            
            # Cross entropy between logits and targets to determine loss
            # Note:
            # PyTorch cross entropy assumes logits is of shape 
            # (T x P), where T is every individual prediction and P is 
            # the logits representing confidence across classes 0 to P-1
            # inclusive
            # and
            # targets is of shape (T,) which contains the ground truth
            # class' index
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
    # Inference
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            # Multinormial distribution is generalized binomial distribution 
            # to any number of classes
            # Basically just samples next token from probability distribution
            # generated from softmax, across each batch
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
    
model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
    # sample a batch of data
    xb, yb = get_batch('train')
    
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))