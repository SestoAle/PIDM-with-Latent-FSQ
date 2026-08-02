import torch
import torch.nn as nn
import math
from layers.llama_transformer import LlamaAttention, LlamaLayerNorm, LlamaMLP
'''
    Since we are not using images, it would be good to test with LLama architecture
'''

###########################################################################################################
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

###########################################################################################################
class TimestepEmbedding(nn.Module):
    '''
        This is a way to do timestep embedding. It is not only a simple embedding layer, but something more
        First the categorical value is transformed into something similar to positional embedding, then embedding layer
    '''

    def __init__(self, d_model, frequency_embedding_size=256, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.frequency_embedding_size = frequency_embedding_size


        self.embedding_layer = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model)
        )

    def forward(self, t, dim=None, max_period=1000):

        if dim is None:
            dim = self.frequency_embedding_size

        half = dim // 2

        '''
            e ^ ( -log(max_period) * [0, dim // 2] / (dim // 2) )
        '''

        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32)/half
        ).to(t.device)

        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        
        embedding = self.embedding_layer(embedding)
        return embedding
    
###########################################################################################################
class ConditionEmbedding(nn.Module):
    '''
        Simple label embedding, with Embedding layer.
        It includes the Conditional Guidance Free (we simply have a new categorical value for non-conditioned generation)
    '''

    def __init__(self, num_classes, hidden_size, dropout_prob=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        use_cfg_embedding = dropout_prob > 0
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob

        # If classifier free guidance (so class = 0), we add a new class (class num_class + 1)
        self.embedding_layer = nn.Embedding(self.num_classes + use_cfg_embedding, self.hidden_size)
    
    def classifier_drop(self, x, force_drop_ids=None):

        if force_drop_ids is None:
            drop_ids = torch.randn(x.shape[0], device=x.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        
        labels = torch.where(drop_ids, self.num_classes, x)

        return labels
    
    def forward(self, x, use_dropout, force_drop_ids=None):
        if use_dropout:
            x = self.classifier_drop(x, force_drop_ids)
        
        x = self.embedding_layer(x)
        return x

###########################################################################################################
class DiTBlock(nn.Module):
    '''
        Attention block (the main part of the DiT model). 
        input -> layer norm (standard) -> adaLN with scale and shift from conditioning -> Self-Attention -> adaLN with scale from conditioning -> resnet with input ->
        layer norm (standard) -> adaLN with scale and shift from conditioning -> MLP -> adaLN with scale from conditioning -> resnet with input. 
    '''

    def __init__(self, hidden_size, num_heads, device="cpu", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.norm1 = LlamaLayerNorm(hidden_size, eps=1e-6)

        self.attn = LlamaAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=hidden_size // num_heads,
            num_key_value_heads=num_heads,
            max_positional_embeddings=10000,
            device=device
        )
        self.norm2 = LlamaLayerNorm(hidden_size, eps=1e-6)

        self.mlp = LlamaMLP(
            hidden_size=hidden_size,
            mid_size=hidden_size // 2,
            device=device
        )

        # Adaptive layer normalization (adaLN-Zero) conditioning
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size*6, bias=False)
        )
    
    def forward(self, x, c):
        '''
            msa = multi-head self attention
            mlp = mlp
        '''
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x

###########################################################################################################
class DiTFinalLayer(nn.Module):

    def __init__(self, hidden_size, *args, **kwargs):
        '''
            Final layer: norm layer (standard) -> adaln with conditioning with scale and shift -> mlp
            [output_channels beacuse this is for images]
        '''
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size

        # self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_final = LlamaLayerNorm(hidden_size, eps=1e-6)
        self.linear_layer = nn.Linear(hidden_size, hidden_size, bias=True)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2*hidden_size, bias=True)
        )

    def forward(self, x, c):

        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear_layer(x)
        return x

###########################################################################################################
class DiTModel(nn.Module):
    '''
        The condition and the timestep emb are summed
    '''
    
    def __init__(self, 
                 input_size = 32,
                 hidden_size = 1152,
                 depth = 28,
                 num_heads = 16,
                 mlp_ratio = 4.0,
                 class_dropout_prob = 0.1,
                 num_classes = 1000,
                 learn_sigma = True,
                 device = "cpu",
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)


        self.input_size = input_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.class_dropout_prob = class_dropout_prob
        self.num_classes = num_classes
        self.learn_sigma = learn_sigma

        # The patch embedder will return a tensor of shape (BS, hidden_size, hidden_size)
        self.x_embedding_layer = nn.Linear(self.input_size, hidden_size)
        self.t_embedding_layer = TimestepEmbedding(hidden_size)
        self.y_embedding_layer = ConditionEmbedding(self.num_classes, self.hidden_size, self.class_dropout_prob)

        self.dit_blocks = nn.ModuleList(
            [DiTBlock(self.hidden_size, self.num_heads, device) for _ in range(self.depth)]
        )

        self.final_layer = DiTFinalLayer(self.hidden_size)
    
    def forward(self, x, t, y, train=True):

        x = self.x_embedding_layer(x)
        t = self.t_embedding_layer(t)
        y = self.y_embedding_layer(y, train)
        c = t + y

        for dit_block in self.dit_blocks:
            x = dit_block(x, c)
        
        x = self.final_layer(x, c)

        return x

###########################################################################################################
if __name__ == "__main__":

    '''
        To run this, copy the file in the root
    '''

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Complete DiT Architecture
    dit = DiTModel(
        input_size=32,
        hidden_size=256,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=False,
        device=device
    )
    x = torch.randn(128, 10, 32)
    t = torch.randint(0, 100, (128,)).view(-1)
    c = torch.randint(0, 1000, (128,)).view(-1)

    output = dit(x, t, c)
    print(f"Output of the DiT {output}")


    import ipdb; ipdb.set_trace()
    