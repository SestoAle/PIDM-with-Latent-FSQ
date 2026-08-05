from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
import math

from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
import math

def modulate(x, shift, scale):
    if len(scale.shape) == 2:
        scale = scale.unsqueeze(dim=1)
    
    if len(shift.shape) == 2:
        shift = shift.unsqueeze(dim=1)

    return x * (1 + scale) + shift

class Transformer(nn.Module):
    def __init__(self, 
                 state_dim, 
                 n_head, 
                 hidden_size, 
                 n_entities, 
                 mask_value=None, 
                 mlp_layer=3, 
                 pooling='None',
                 residual=True, 
                 with_embeddings=False, 
                 with_ffn=True, 
                 post_norm=True, 
                 pre_norm=True, 
                 with_adaln=False,
                 adaln_input_size = 1,
                 is_causal=False,
                 device="cpu"
                ):
        super(Transformer, self).__init__()
        self.mask_value = mask_value
        self.with_embeddings = with_embeddings
        self.pre_norm = pre_norm
        self.n_entities = n_entities
        self.heads = n_head
        self.hidden_size = hidden_size
        self.residual = residual
        self.with_ffn = with_ffn
        self.mlp_layer = mlp_layer
        self.post_norm = post_norm
        self.pooling = pooling
        self.device = device
        self.is_causal = is_causal
        # QKV embedding layers
        self.q_emb = nn.Linear(hidden_size if with_embeddings else state_dim, hidden_size)
        self.k_emb = nn.Linear(hidden_size if with_embeddings else state_dim, hidden_size)
        self.v_emb = nn.Linear(hidden_size if with_embeddings else state_dim, hidden_size)
        # First embedding
        self.emb_layer = nn.Linear(state_dim, hidden_size)
        # Last layers
        self.a_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.mlp = None
        if self.with_ffn and self.mlp_layer >= 3:
            self.mlp = []
            for i in range(self.mlp_layer - 2):
                self.mlp.append(nn.Linear(self.hidden_size if i == 0 else self.hidden_size * 2, self.hidden_size * 2))
                self.mlp.append(nn.SiLU())
            self.mlp.append(nn.Linear(self.hidden_size * 2, self.hidden_size))
            self.mlp.append(nn.SiLU())
            self.mlp = nn.Sequential(*self.mlp)
        
        if pre_norm:
            self.pre_norm = nn.LayerNorm(hidden_size)
        if post_norm:
            self.post_norm = nn.LayerNorm(hidden_size)
        
        self.with_adaln = with_adaln
        if self.with_adaln:
            # AdaLN conditioning
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(adaln_input_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size*6, bias=False)
            )
            # Initialize the adaLN to zero
            def init_to_zero(m):
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
            
            self.adaLN_modulation.apply(init_to_zero)
    
    def forward(self, x):
        
        inp, c, mask = x
        if mask is not None:
            original_mask = mask.clone()
        else:
            original_mask = None

        BS, NE, features = inp.shape
        if mask != None or self.mask_value != None:
            if mask == None:
                mask = self.create_mask(inp, self.mask_value)
            assert np.all(np.array(mask.shape) == np.array(inp.shape[:2])), \
                f"Mask and inp should have the same first 3 dimensions. {mask.shape} -- {inp.shape}"
            mask = torch.unsqueeze(mask, dim=1).to(device=self.device)  # (BS, 1, NE)
        if self.with_embeddings:
            inp = F.tanh(self.emb_layer(inp))

        # qkv embs
        if self.pre_norm:

            a = self.pre_norm(inp)

        # AdaLN conditioning
        if self.with_adaln and c is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
            # Shift and scale prior self attention
            a = modulate(a, shift_msa, scale_msa)
        
        query = self.q_emb(a)
        key = self.k_emb(a)
        query = query.view(-1, self.n_entities, self.heads, self.hidden_size // self.heads)
        key = key.view(-1, self.n_entities, self.heads, self.hidden_size // self.heads)
        value = self.v_emb(a)
        value = value.view(-1, self.n_entities, self.heads, self.hidden_size // self.heads)
        query = torch.permute(query, (0, 2, 1, 3))
        key = torch.permute(key, (0, 2, 3, 1))
        value = torch.permute(value, (0, 2, 1, 3))
        # self attention
        logits = torch.matmul(query, key)

        if self.is_causal:
            casual_mask = torch.tril(torch.ones(NE, NE, device=self.device))
            casual_mask = casual_mask.unsqueeze(0).unsqueeze(0) # (1, 1, NE, NE)
            
            if mask is not None:
                entity_mask = mask.unsqueeze(2) # (BS, 1, 1, NE)
                mask = casual_mask * entity_mask
            else:
                mask = casual_mask

        logits /= math.sqrt(self.hidden_size / self.heads)
        softmax = self.stable_masked_softmax(logits, mask)
        att_sum = torch.matmul(softmax, value)
        out = torch.permute(att_sum, (0, 2, 1, 3))
        out = torch.reshape(out, (-1, self.n_entities, self.hidden_size))
        a = out
        att_weights = softmax


        a = F.silu(self.a_layer(a))
        
        if self.with_adaln and c is not None:
            if len(gate_msa.shape) == 2:
                gate_msa = gate_msa.unsqueeze(dim=1)
            a = gate_msa * a
        
        if self.residual:
            a = inp + a
        

        if self.mlp is not None:
            
            if self.with_adaln and c is not None:
                a = modulate(a, shift_mlp, scale_mlp)

            a = self.mlp(a)

            if self.with_adaln and c is not None:
                if len(gate_mlp.shape) == 2:
                    gate_mlp = gate_mlp.unsqueeze(dim=1)
                a = gate_mlp * a

            if self.residual:
                a = a + inp

        inp = a
        if self.post_norm:
            inp = self.post_norm(inp)
        if original_mask is not None:
            original_mask = original_mask.view(BS, NE)
        else:
            original_mask = torch.ones(BS, NE).to(device=self.device)
        if self.pooling == 'avg':
            inp = self.entity_avg_pooling_masked(inp, original_mask)
            inp = inp.view(inp.shape[0], -1, self.hidden_size)
        elif self.pooling == 'max':
            inp = self.entity_max_pooling_masked(inp, original_mask)
            inp = torch.reshape(inp.shape[0], (-1, self.hidden_size))
        else:
            original_mask = torch.unsqueeze(original_mask, dim=-1)
            inp = inp * original_mask
            inp = inp.view(BS, NE, features)

        if original_mask is not None:
            original_mask = original_mask.view(BS, NE)

        if self.with_adaln:
            return inp, c, original_mask
        else:
            return inp, None, original_mask
    
    def entity_max_pooling_masked(self, inp, mask):
        mask = torch.unsqueeze(mask, -1)
        has_unmasked_entities = torch.sign(torch.sum(mask, dim=-2, keepdim=True))
        offset = (mask - 1) * 1e9
        masked = (inp + offset) * has_unmasked_entities
        masked, _ = torch.max(masked, dim=-2)
        return masked
    
    def entity_avg_pooling_masked(self, inp, mask):
        mask = torch.unsqueeze(mask, -1)
        masked = inp * mask
        summed = torch.sum(masked, -2)
        denom = torch.sum(mask, -2) + 1e-5
        return summed / denom
    
    def stable_masked_softmax(self, logits, mask):

        if self.is_causal:
            logits = logits.masked_fill(mask == 0, -1e10)
        else:
            if mask is not None:
                mask = torch.unsqueeze(mask, 2)
                logits -= (1.0 - mask) * 1e10
        logits_mod, _ = torch.max(logits, dim=-1, keepdim=True)
        logits -= logits_mod
        unnormalized_p = torch.exp(logits)

        if mask is not None:
            unnormalized_p = unnormalized_p * mask

        normalized_p = unnormalized_p / (torch.sum(unnormalized_p, dim=-1, keepdim=True) + 1e-10)
        # if mask is not None:
        #     normalized_p *= mask
        return normalized_p
    
    def create_mask(self, inp, mask_value):
        # x = bs, NE, feature
        mask = 1 - torch.eq(inp[:, :, 0], mask_value).float()
        # mask = torch.ones(inp.shape[:-1])
        return mask

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.d_model = d_model

    def forward(self, x):
        """
        Arguments:
            x: Tensor, (batch_size, position)

        Output:
            x: Tensor, (batch_size, d_model) 
        """
        x = self.pe[x] 
        return x


if __name__ == '__main__':
    trans = Transformer(128, 8, 128, n_entities=6, pooling='max', with_embeddings=False)
    inp = torch.randn(1, 6, 128)
    mask = torch.ones(1, 6)
    out, _ = trans(inp, mask)

    pe = PositionalEncoding(256)
    x = np.arange(10).reshape(1, -1)
    y = pe(x)