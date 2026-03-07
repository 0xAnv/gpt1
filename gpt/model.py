"""
GPT - 1 Transformer model architecture 

Paper: "Improving Language Understanding by Generative Pre-Trainin" (Radford et al., 2018)
Architecture: 12- layer transformer decoder with causal(masked) self-attention
"""

import torch 
import torch.nn as nn 
from torch import Tensor # for type hinting 

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

        # Compute scaled dot product attention scores
        # (batch, n_heads, seq_len, d_k) @ (batch, n_heads, d_k, seq_len) -> (batch, n_heads, seq_len, seq_len)
        attn_scores = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5) # normalised denom

        # Create upper triangular mask (True = positions to MASK OUT)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), 
            diagonal=1
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        # Softmax to get attention weights, then apply dropout 
        attn_weights = torch.softmax(attn_scores, dim = -1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of layers 
        # (batch, n_heads, seq_len, seq_len) @ (batch, n_heads, seq_len, d_k) -> (batch, n_heads, seq_len, d_k)
        attn_output = attn_weights @ v

        # Concatenate heads 
        # (batch, n_heads, seq_len, d_k) -> (batch, seq_len, n_heads, d_k) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Final Output projection + dropout 
        output = self.W_o(attn_output)
        output = self.resid_dropout(output)

        return output

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

if __name__ == "__main__":
    check_mha()