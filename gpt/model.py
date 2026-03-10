"""
GPT - 1 Transformer model architecture 

Paper: "Improving Language Understanding by Generative Pre-Trainin" (Radford et al., 2018)
Architecture: 12- layer transformer decoder with causal(masked) self-attention
"""

import torch 
import torch.nn as nn 
from torch import Tensor # for type hinting 
from torch.utils.checkpoint import checkpoint as torch_checkpoint  # gradient checkpointing

class MultiHeadAttention(nn.Module):
    """
    Multi Head Causal Attention. 

    Splits the input into multiple attention heads, applies scaled dot product attention with causal mask,
    then concatenates and projects the result.
    
    Args:
        d_model: Dimensionality of hidden states 
        n_heads: num of parallel heads 
        dropout: dropout prob applied to attention weights
    """

    def __init__(self, d_model:int=768, n_heads:int=12, dropout:float=0.1) -> None:
        super().__init__() 

        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisble by n_heads ({n_heads})"
        )

        # parameters
        self.d_model = d_model
        self.n_heads = n_heads 
        self.d_k = d_model // n_heads # dimension per head (786 // 12 = 64)

        # Linear projections for Q, K, V and output 
        self.W_q = nn.Linear(d_model, d_model) 
        self.W_k = nn.Linear(d_model, d_model) 
        self.W_v = nn.Linear(d_model, d_model) 
        self.W_o = nn.Linear(d_model, d_model)

        # dropouts
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
    
    def forward(self, x:Tensor) -> Tensor:
        """ 
        Forward pass for causal multi-head self-attention.
        Uses PyTorch's scaled_dot_product_attention (Flash Attention) for
        memory efficiency — avoids materializing the full (seq_len x seq_len) matrix.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape 

        # project input to Q, K, V 
        # shape: (batch, seq_len, d_model)
        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x) 

        # reshape and transpose for multi-head attention 
        # (batch, seq_len, d_model) -> (batch, seq_len, n_heads, d_k) -> (batch, n_heads, seq_len, d_k) 
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1,2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1,2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1,2)

        # Fused scaled dot-product attention (Flash Attention when available)
        # is_causal=True automatically applies the causal mask
        # Replaces: manual Q@K.T, masking, softmax, dropout, @V
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )

        # Concatenate heads 
        # (batch, n_heads, seq_len, d_k) -> (batch, seq_len, n_heads, d_k) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Final Output projection + dropout 
        output = self.W_o(attn_output)
        output = self.resid_dropout(output)

        return output

class PositionWiseFFN(nn.Module): 
    """
    Position wise feed forward network. 
    Applies two linear transformations with GELU activation in between.
    FFN(x) = Linear_2(GELU(Linear_1(x)))

    Args:
        d_model: Dimensionality of model 
        d_ff: Dimensionality of inner (hideen) layer. 
    """

    def __init__(self, d_model:int = 768, d_ff:int = 3072, dropout:float = 0.1) -> None:
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff) # expand : 768 -> 3072
        self.fc2 = nn.Linear(d_ff, d_model) # compress : 3072 -> 768 
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:Tensor) -> Tensor:
        """
        Forward Pass
        Args: 
            x: Input tensor of shape (batch, seq_len, d_model)
        Returns:
            Ouput tensor of shape (batch, seq_len, d_model).
        """
        x = self.fc1(x)  # (batch, seq_len, 768) -> (batch, seq_len, 3072)
        x = self.gelu(x)      # nonlinear activation, same shape
        x = self.fc2(x)       # (batch, seq_len, 3072) -> (batch, seq_len, 768)
        x = self.dropout(x)   # regularization before residual add (done externally)
        return x

class TransformerBlock(nn.Module): 
    """
    A single transformer decoder block (post norm variant as in GPT 1) 
    
    Applies masked multi-head attention followed by postition wise feed forward network. 
    Each with residual connection and layernorm.

    Architecture: 
        h = LayerNorm(x + Multiheadattention(x)) 
        output = LayerNorm(h + PositionWiseFFN(h))

    Args: 
        d_model: Dimensionality of model's hidden state 
        n_heads: number of attention heads 
        d_ff : Inner dimensionality of feed forward network 
        dropout: Dropout probability
    """

    def __init__(
        self, 
        d_model:int = 768, 
        n_heads:int = 12, 
        d_ff:int = 3072, 
        dropout:float = 0.1
    ) -> None: 

        super().__init__()

        # sub layers 
        self.attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.ffn = PositionWiseFFN(d_model=d_model, d_ff=d_ff, dropout=dropout) 

        # Layer Norms (one per sub-layer) 
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x:Tensor) -> Tensor : 
        """
        Forward pass through on transformer decoder block. 
        Args: 
            x: Input tensor of shape (batch, seq_len, d_model)
        Returns:
            Output Tensor of shape (batch, seq_len, d_model) 
        """
        # sub layer 1: Attention + residual + norm (post-norm) 
        x = self.ln1(x + self.attn(x))

        # sub layer 2: FFN + residual + norm (post-norm)
        x = self.ln2(x + self.ffn(x))

        return x;

class GPT1(nn.Module):
    """
    Full GPT 1 paper implementation.

    Stacks token + pos embedings, N transformer decoder blocks, and a weight-tied language model head (FFN)
    
    Args: 
        vocab_size: size of vocabulary tokeniser 
        max_seq_len: maximum sequence length (pos embedding size) 
        n_layers: Numbers of transformer decoder block 
        d_model: dimensionality of our hidden state of model 
        n_heads: parallel heads for multi head self attention comp.
        d_ff: dimension of feed forward nn 
        dropout: dropout rate; dtype=float ;usually 0.1
    """
    def __init__(
        self, 
        vocab_size:int, 
        max_seq_len:int = 512, 
        n_layers:int = 12, 
        d_model:int = 768, 
        n_heads:int = 12, 
        d_ff:int = 3072, 
        dropout:float = 0.1,
        use_checkpoint:bool = False,  # gradient checkpointing to save VRAM
    ) -> None :
        super().__init__()

        self.max_seq_len = max_seq_len 
        self.use_checkpoint = use_checkpoint

        # Embeddings 
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model) # learned embeddings 
        self.emb_dropout = nn.Dropout(dropout)

        # Transformer Blocks 
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Applying base initialization to all modules 
        self.apply(self._init_weights)

        # Residual Scaling to output projections
        for dec_block in self.blocks:
            torch.nn.init.normal_(dec_block.attn.W_o.weight, mean=0.0, std=0.02 / (2*n_layers) ** 0.5) 
            torch.nn.init.normal_(dec_block.ffn.fc2.weight, mean=0.0, std=0.02 / (2*n_layers) ** 0.5)
            

    def _init_weights(self, module:nn.Module) -> None: 
        """
        Initialize weights following GPT 1 Paper specs

        - All Weights: N(0, 0.02)
        - All Bias: 0 
        - Residual Projections: (W_o in attention, fc2 in FFN)
                                additionally scaled by 1/sqrt(2* n_layers) , 
                                to counteract the effect of multiple residual connections 
        """

        if isinstance(module, nn.Linear): 
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) 
        
        elif isinstance(module, nn.Embedding): 
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        elif isinstance(module, nn.LayerNorm): 
            torch.nn.init.ones_(module.weight)  # gamma = 1
            torch.nn.init.zeros_(module.bias)   # beta = 0

    def forward(self, x:Tensor) -> Tensor:
        """
        Forward pass of the full GPT 1 Model. 

        Args:
            x : token is tensor of shape (batch, seq_len) 

        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape

        assert seq_len <= self.max_seq_len, (
            f"SEquence length {seq_len} exceeds maximum {self.max_seq_len}"
        )

        # Create pos indices [0,1,2,3, ... , seq_len - 1]
        pos_ids = torch.arange(seq_len, device=x.device)

        # Token embedding + pos embedding 
        x = self.token_emb(x) + self.pos_emb(pos_ids)
        x = self.emb_dropout(x) # dropout

        # Pass through all transformer decoder blocks 
        for dec_block in self.blocks:
            if self.use_checkpoint and self.training:
                x = torch_checkpoint(dec_block, x, use_reentrant=False)
            else:
                x = dec_block(x)

        # weight tying Language modelling head 
        # We do not use linear here coz we use weight typing, reducing param count 
        # weight tying: logits = hidden_state @ token_emb.weight.T 
        # (batch, max_seq_len, d_model) @ (d_model, vocab_size) = (batch, max_seq_len, vocab_size)
        logits = x @ self.token_emb.weight.T

        return logits 

    @torch.no_grad() 
    def generate(
        self, 
        idx: torch.Tensor, 
        max_new_tokens:int, 
        temperature:float = 1.0, 
        top_k:int | None = None
    ) -> torch.Tensor:
        """
        Take a sequence of indices idx (LongTensor of shape (b, t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time. 
        """
        for _ in range(max_new_tokens): 
            # If the sequence is growing too long we will crop it to max_seq_len
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]

            # Forward the model to get logits for index in the sequence 
            logits = self(idx_cond)

            # take logits at final step and scale by desired temperature 
            logits = logits[:, -1, :] / temperature

            # optionally crop logits to only top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            
            # Apply softmax to convert logits to normalized 
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # sample from the distribution 
            idx_next = torch.multinomial(probs, num_samples=1)

            # append sampled index to the running sequence to continue 
            idx = torch.cat((idx, idx_next), dim = 1)

        return idx
     
# checking attention 

def check_mha() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mha = MultiHeadAttention(d_model=768, n_heads=12, dropout=0.1).to(device)
    x = torch.randn(2, 10, 768, device=device)
    out = mha(x) 
    print(f"Input Shape: {x.shape}") # (2, 10, 768)
    print(f"Ouput shape: {out.shape}") # (2, 10, 768)
    print(f"Parameters:   {sum(p.numel() for p in mha.parameters()):,}") # 2,362,368

    # Verift causal masking: Output at position 0 should be independent of later positions 
    x2 = x.clone()
    x2[:, 5:, :] = torch.randn(2, 5, 768, device=device) # change positions 5-9 
    mha.eval() # disable dropout for deterministic comparision
    out1= mha(x)
    out2 = mha(x2)
    assert torch.allclose(out1[:, :5, :], out2[:, :5, :]), "Causal mask broken!"
    print("Causal mask verified!")

def check_ffn() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ffn = PositionWiseFFN(d_model=768, d_ff=3072, dropout=0.1).to(device)
    x = torch.randn(2, 10, 768, device=device)
    out = ffn(x)
    print(f"FFN Input shape:  {x.shape}")    # (2, 10, 768)
    print(f"FFN Output shape: {out.shape}")   # (2, 10, 768)
    print(f"FFN Parameters:   {sum(p.numel() for p in ffn.parameters()):,}")  # 4,722,432

    # Verify position-wise independence: changing one position shouldn't affect others
    ffn.eval()
    x2 = x.clone()
    x2[:, 0, :] = torch.randn(768, device=device)  # change only position 0
    out1 = ffn(x)
    out2 = ffn(x2)
    assert torch.allclose(out1[:, 1:, :], out2[:, 1:, :]), "FFN is not position-wise!"
    print("Position-wise independence verified!")

def check_block() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    block = TransformerBlock(d_model=768, n_heads=12, d_ff=3072, dropout=0.1).to(device)
    x = torch.randn(2, 10, 768, device=device)
    out = block(x)
    print(f"Block Input shape:  {x.shape}")    # (2, 10, 768)
    print(f"Block Output shape: {out.shape}")   # (2, 10, 768)
    print(f"Block Parameters:   {sum(p.numel() for p in block.parameters()):,}")  # 7,087,872

    # Verify causal property is preserved through the block
    block.eval()
    x2 = x.clone()
    x2[:, 5:, :] = torch.randn(2, 5, 768, device=device)
    out1 = block(x)
    out2 = block(x2)
    assert torch.allclose(out1[:, :5, :], out2[:, :5, :]), "Block causality broken!"
    print("Block causality verified!")

def check_gpt1() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size:int = 40_000 
    model = GPT1(
        vocab_size=vocab_size, 
        max_seq_len=512, 
        n_layers = 12, 
        d_model=768, 
        n_heads=12, 
        d_ff=3072, 
        dropout=0.1
    ).to(device) # GPT model to GPU

    # Fake input: batch of 2 seq, each of 10 tokens long 
    input_ids = torch.randint(0, vocab_size, (2, 10), device=device)
    logits = model(input_ids)

    print(f"GPT1 Input Shape: {input_ids.shape}")
    print(f"GPT1 Output Shape: {logits.shape}")
    print(f"GPT1 Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Verify causal property
    model.eval()
    input_ids_d = input_ids.clone()
    input_ids_d[:, 5:] = torch.randint(0, vocab_size, (2, 5), device=device)
    logits, logits_d = model(input_ids), model(input_ids_d) 
    assert torch.allclose(logits[:, :5, :], logits_d[:, :5, :], atol= 1e-5), "Causalty is Broken !!"
    print("Causalty Verified!")

if __name__ == "__main__":
    check_mha()
    check_ffn()
    check_block()
    check_gpt1()
    torch.cuda.empty_cache()