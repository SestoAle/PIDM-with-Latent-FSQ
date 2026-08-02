import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import Attention, PatchEmbed, Mlp
import numpy as np
'''
    Since we are not using images, it would be good to test with LLama architecture
'''

###########################################################################################################
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


###########################################################################################################
# TODO: This is just copy and paste. I do not understand why we need to do this if we already have 
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

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
class LabelEmbedding(nn.Module):
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

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=True
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_size = int(hidden_size * mlp_ratio)
        # TODO: what is this?
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_size,
            act_layer=approx_gelu,
            drop=0.0
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

    def __init__(self, hidden_size, patch_size, output_channels, *args, **kwargs):
        '''
            Final layer: norm layer (standard) -> adaln with conditioning with scale and shift -> mlp
            [output_channels beacuse this is for images]
        '''
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.output_channels = output_channels

        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear_layer = nn.Linear(hidden_size, patch_size * patch_size * output_channels, bias=True)

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
    
    def __init__(self, 
                 input_size = 32,
                 patch_size = 2,
                 in_channels = 4,
                 hidden_size = 1152,
                 depth = 28,
                 num_heads = 16,
                 mlp_ratio = 4.0,
                 class_dropout_prob = 0.1,
                 num_classes = 1000,
                 learn_sigma = True,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)


        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.class_dropout_prob = class_dropout_prob
        self.num_classes = num_classes
        self.learn_sigma = learn_sigma

        # The patch embedder will return a tensor of shape (BS, hidden_size, hidden_size)
        self.x_embedding_layer = PatchEmbed(self.input_size, self.patch_size, self.in_channels, self.hidden_size, bias=True)
        self.t_embedding_layer = TimestepEmbedding(hidden_size)
        self.y_embedding_layer = LabelEmbedding(self.num_classes, self.hidden_size, self.class_dropout_prob)

        self.num_patches = self.x_embedding_layer.num_patches

        # Ok this is standard positional embedding (sin/cos) for _tokens_. Since they are 2D patches, we need
        # to unroll them. The positional embedding for T is not for the transformer, but rather the diffusion model
        self.pos_embedder = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size, requires_grad=True))

        self.dit_blocks = nn.ModuleList(
            [DiTBlock(self.hidden_size, self.num_heads, self.mlp_ratio) for _ in range(self.depth)]
        )

        self.final_layer = DiTFinalLayer(self.hidden_size, self.patch_size, self.out_channels)
        # self.initialize_weight()

        # Let's initialuze only the pos embedding
        # TODO: btw, why we have pos embedding here and in the time embedding, but they are different?
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embedder.shape[-1], int(self.x_embedding_layer.num_patches ** 0.5))
        self.pos_embedder.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
    
    def unpacthify(self, x):
        '''
            From patches to images
        '''

        c = self.out_channels
        p = self.x_embedding_layer.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)

        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc -> nchpwq", x)
        imgs = x.reshape((x.shape[0], c, h * p, h * p))
        return imgs
    
    def forward(self, x, t, y, train=True):

        x = self.x_embedding_layer(x) + self.pos_embedder
        t = self.t_embedding_layer(t)
        y = self.y_embedding_layer(y, train)
        c = t + y

        for dit_block in self.dit_blocks:
            x = dit_block(x, c)
        
        x = self.final_layer(x, c)
        x = self.unpacthify(x)

        return x

###########################################################################################################
if __name__ == "__main__":

    # Timestep embedding
    t_emb = TimestepEmbedding(
        d_model = 128
    )
    x = torch.randint(0, 10, (100,)).view(-1)
    time_output = t_emb(x)
    print(f"timestep embedding output: {time_output}")

    # Labels embedding
    class_emb = LabelEmbedding(
        num_classes=10, 
        hidden_size=256, 
        dropout_prob=0.3
    )
    x = torch.randint(0, 10, (100,)).view(-1)
    class_output = class_emb(x, use_dropout=True)
    print(f"Class embedding output {class_output}")

    # Single DiT block
    dit_block = DiTBlock(
        hidden_size=256, 
        num_heads=8
    )
    x = torch.randn((100, 10, 256))
    dit_block_output = dit_block(x, class_output)
    print(f"DiT block output {dit_block_output}")

    # Complete DiT Architecture
    dit = DiTModel(
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=256,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=False
    )
    x = torch.randn(128, 4, 32, 32)
    t = torch.randint(0, 100, (128,)).view(-1)
    c = torch.randint(0, 1000, (128,)).view(-1)

    output = dit(x, t, c)
    print(f"Output of the DiT {output}")


    import ipdb; ipdb.set_trace()
    