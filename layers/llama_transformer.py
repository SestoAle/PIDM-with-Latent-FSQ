import torch
from torch import nn
from torch.nn import functional as F

# A class that replicates the transformer architecture of LLama (3?)
# It is taken from MineWorld paper

def rotate_half(x):
    # This function rotates half the hidden dims of the input
    x1 = x[..., :x.shape[-1] // 2] # First half of the input
    x2 = x[..., x.shape[-1] // 2:] # Second half of the input
    # The, we concatenate first x2 then x1 (that's why is rotated)
    # the second half will be inverted (-1)
    return torch.cat([-x2, x1], dim=-1) 

def apply_rotary(x, cos, sin, position_ids):
    # This should be the same as standard positional embedding
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

class RotaryEmbedding(nn.Module):
    # TODO: I do not fully understand this, maybe I should read the paper
    # https://arxiv.org/abs/2104.09864
    def __init__(self, 
                 max_position_embeddings, 
                 d_model, 
                 device, 
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_position_embeddings = max_position_embeddings
        # What is rope_init_function?
        self.device = device
        self.d_model = d_model
        self.set_sin_cos_cache()

    
    def set_sin_cos_cache(self, base=100):
        t = torch.arange(0, self.max_position_embeddings).to(self.device).float()
        inv_freq = 1.0 / (base ** (torch.arange(0, self.d_model, 2).to(self.device) / self.d_model))
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.cos = emb.cos()[None, :, :]
        self.sin = emb.sin()[None, :, :]
    
    def forward(self, x, position_ids, seq_len=None):

        if seq_len == None:
            seq_len = x.shape[-2]
            
        x = apply_rotary(x, self.cos[:, :seq_len, :], self.sin[:, seq_len, :], position_ids) 

        return x

class LlamaLayerNorm(nn.Module):
    def __init__(self,
                 hidden_size,
                 eps=1e-6,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(self.hidden_size))
        self.variance_eps = eps
    
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.sqrt(variance + self.variance_eps)
        x = self.weight * x
        return x

class LlamaAttention(nn.Module):
    def __init__(self, 
                 hidden_size,
                 num_heads,
                 head_dim,
                 num_key_value_heads,
                 max_positional_embeddings,
                 device,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_group = num_heads // num_key_value_heads
        self.max_positional_embeddings = max_positional_embeddings
        self.device = device


        # TODO: understand these values
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True) 
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)

        self.rope = RotaryEmbedding(max_position_embeddings=max_positional_embeddings,
                                    d_model=hidden_size,
                                    device=device)
    
    def forward(self,
                x,
                attention_mask=None,
                position_ids=None):
        
        B, q_len, _ = x.shape
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        # In the repo, the rope is done *after* the projection. Does it make sense?
        # If so, why?
        position_ids = torch.arange(q_len).view(1, -1).repeat_interleave(B, dim=0)
        query = self.rope(query, position_ids)
        value = self.rope(value, position_ids)

        query = query.view(B, q_len, self.num_heads, self.head_dim)
        value = value.view(B, q_len, self.num_key_value_heads, self.head_dim)
        key = key.view(B, q_len, self.num_key_value_heads, self.head_dim)

        key = key.repeat_interleave(self.num_key_value_group, dim=2)
        value = value.repeat_interleave(self.num_key_value_group, dim=2)

        query, key, value = map(lambda x: x.transpose(1,2), (query, key, value))

        attn_output = F.scaled_dot_product_attention(
            query=query, 
            key=key,
            value=value, 
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False
        ).transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(B, q_len, self.hidden_size)
        output = self.o_proj(attn_output)
        return output

class LlamaMLP(nn.Module):
    def __init__(self,
                 hidden_size,
                 mid_size,
                 device,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        self.hidden_size = hidden_size
        self.mid_size = mid_size
        self.device = device

        self.gate_proj = nn.Linear(self.hidden_size, self.mid_size)
        self.up_proj = nn.Linear(self.hidden_size, self.mid_size)
        self.down_proj = nn.Linear(self.mid_size, self.hidden_size)
    
    def forward(self, x):
        gate_proj = F.silu(self.gate_proj(x))
        x = gate_proj * self.up_proj(x)
        x = self.down_proj(x)

        return x
        

class LlamaDecoder(nn.Module):
    def __init__(self, 
                 hidden_size,
                 num_heads,
                 max_positional_embeddings,
                 device,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.device = device

        self.attn = LlamaAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=hidden_size // num_heads,
            num_key_value_heads=num_heads,
            max_positional_embeddings=max_positional_embeddings,
            device=device
        )

        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            mid_size=self.hidden_size // 2,
            device=self.device
        )

        self.pre_norm = LlamaLayerNorm(
            hidden_size=hidden_size
        )

        self.post_norm = LlamaLayerNorm(
            hidden_size=hidden_size
        )
    
    def forward(
            self,
            x,
            attention_mask=None,
    ):
        residual = x
        x = self.pre_norm(x)

        x = self.attn(
            x=x,
            attention_mask=attention_mask
        )

        x = x + residual
        residual = x
        x = self.post_norm(x)
        x = self.mlp(x)
        x = x + residual

        return x