import torch
import torchvision
import numpy as np

from torch.nn import functional as F
from torch import nn
from layers.transformer import Transformer
from einops import rearrange
from layers.fsq_latent import FSQLatent

from torch.distributions import Categorical

#######################################################################################
# This I copied from the original repo
class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time

#######################################################################################
class FSQEncoder(torch.nn.Module):
        
    def __init__(
                 self,
                 input_size : int,
                 output_size : int, 
                 L : int,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.input_size      = input_size
        self.output_size     = output_size
        self.L               = L

        self.fsq_latent = FSQLatent(
            input_size = self.input_size,
            output_size = self.output_size,
            L = L,
            with_rescale=True
        )
    
    # If we have values between 0-L, shift and scale to have -1, 1
    def shift_and_scale(self, x):
        x = x - (self.L/2)
        x = x / (self.L/2)
        return x
    
    def reshift_and_rescale(self, x):
        x = x * (self.L/2)
        x = x + (self.L/2)
        x = x.long()
        return x


    def forward(self, x):
        emb = self.fsq_latent(x)

        return emb

#######################################################################################
class MLPEncoder(torch.nn.Module):
        
    def __init__(
                 self,
                 input_size : int,
                 output_size : int, 
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.input_size      = input_size
        self.output_size     = output_size

        self.linear_latent = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size),
        )
    
    def forward(self, x):
        emb = self.linear_latent(x)
        return emb

#######################################################################################
class LeWorldModel(nn.Module):

    def __init__(self, 
                 action_dim : int,
                 max_seq_length : int,
                 lr : float = 1e-4,
                 lambd_sigreg : float = 0.09,
                 lambd_reconstruction : float = 1.0,
                 lambd_latent_l1 : float = 1.0,
                 lambd_reward : float = 0.1,
                 reward_priority_fraction : float = 0.25,
                 reward_high_quantile : float = 0.9,
                 validation_fraction : float = 0.1,
                 validation_seed : int = 423,
                 autoregressive_rollout_length : int = 5,
                 teacher_forcing_probability : float = 0.5,
                 autoregressive_loss_weight : float = 1.0,
                 encoder_hidden_dim : int = 192,
                 num_heads : int = 4,
                 num_decoder_layers : int = 6,
                 device : str = "cpu", 
                 feature_base : bool = False,
                 # With terminal and reward output
                 with_reward_prediction : bool = False,
                 with_terminal_prediction : bool = False,
                 # If this is a state-only model
                 with_action : bool = True,
                 # If it is feature base, we have FSQ hyperparameters
                 fsq_output_size : int = 256,
                 fsq_input_size : int = 256,
                 fsq_encoder : bool = True,
                 L : int = 50,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.action_dim                 = action_dim
        self.device                     = device
        self.encoder_hidden_dim         = encoder_hidden_dim
        self.num_decoder_layers         = num_decoder_layers
        self.max_seq_length             = max_seq_length
        self.lambd_sigreg               = lambd_sigreg
        self.lambd_reconstruction       = lambd_reconstruction
        self.lambd_latent_l1            = lambd_latent_l1
        self.lambd_reward               = lambd_reward
        self.reward_priority_fraction   = reward_priority_fraction
        self.reward_high_quantile       = reward_high_quantile
        self.validation_fraction        = validation_fraction
        self.validation_seed            = validation_seed
        self.autoregressive_rollout_length = autoregressive_rollout_length
        self.teacher_forcing_probability = teacher_forcing_probability
        self.autoregressive_loss_weight = autoregressive_loss_weight
        self.lr                         = lr
        self.feature_base               = feature_base
        self.fsq_output_size            = fsq_output_size
        self.fsq_input_size             = fsq_input_size
        self.fsq_L                      = L
        self.with_terminal_prediction   = with_terminal_prediction
        self.with_reward_prediction     = with_reward_prediction
        self.fsq_encoder                = fsq_encoder
        self.with_action                = with_action 

        # Sigreg is only for continuous latent space
        self.sigreg                 = SIGReg()

        # ----- Encoder ------
        # The encoder is a vision transformer base
        # For now let's take it from pytorch, later we can implement our own
        if not self.feature_base:
            self.encoder    = torchvision.models.vit_b_16()
            # From the original ViT, we need to replace the head (because it is the classification head,
            # We need a projection head)
            self.projection_head    = nn.Sequential(
                nn.Linear(in_features=self.encoder.heads[0].in_features, out_features=self.encoder_hidden_dim),
                nn.BatchNorm1d(self.encoder_hidden_dim)
            )
            self.encoder.heads      = self.projection_head
        else:
            if self.fsq_encoder:
                # We need an FSQ encoder here
                self.encoder            = FSQEncoder(input_size=self.fsq_input_size, output_size=self.fsq_output_size, L=self.fsq_L)
                self.encoder_emb        = nn.Embedding(num_embeddings=self.fsq_L, embedding_dim=self.encoder_hidden_dim)
                self.cls_tkn            = nn.Parameter(torch.zeros(self.encoder_hidden_dim))
                self.position_embedding = nn.Parameter(torch.randn(1, self.fsq_output_size + 1, self.encoder_hidden_dim) * 0.02)
                self.encoder_trans      = Transformer(
                    state_dim=self.encoder_hidden_dim,
                    n_head=8,
                    hidden_size=self.encoder_hidden_dim,
                    pooling="None",
                    with_embeddings=False,
                    post_norm=True,
                    pre_norm=True,
                    n_entities=self.fsq_output_size + 1,
                    device=device
                )
                self.reconstruction_head = nn.Sequential(
                    nn.Linear(self.fsq_output_size, self.encoder_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.encoder_hidden_dim, self.fsq_input_size)
                )
            else:
                # For ablation, we are gonna use a MLP based encoder similar to the FSQ encoder
                self.encoder    = MLPEncoder(input_size=self.fsq_input_size, output_size=self.fsq_output_size)

        # ----- Predictor -----
        # The predictor is a transformer with 6 layers and 16 heads. Pretty big transformer
        self.predictor        = nn.Sequential(
            *[Transformer(
                state_dim=self.encoder_hidden_dim,
                n_head=num_heads,
                n_entities=max_seq_length,
                hidden_size=self.encoder_hidden_dim,
                pooling="None",
                with_embeddings=False,
                post_norm=True,
                pre_norm=True,
                with_adaln=self.with_action,
                adaln_input_size=action_dim,
                is_causal=True,
                device=device
            ) for _ in range(self.num_decoder_layers)],
        )

        if not self.fsq_encoder:
            # Plus the same projection head that we have for the encoder
            self.predictor_head     = nn.Sequential(
                nn.Linear(in_features=self.encoder_hidden_dim, out_features=self.encoder_hidden_dim),
                nn.LayerNorm(self.encoder_hidden_dim)
            )
        else:
            # A projection over 8 possible values for dim
            self.predictor_head    = nn.Sequential(
                nn.Linear(in_features=self.encoder_hidden_dim, out_features=self.fsq_L*self.fsq_output_size)
            )

        if self.with_reward_prediction:
            self.reward_head = nn.Sequential(
                nn.Linear(
                    in_features=self.encoder_hidden_dim,
                    out_features=self.encoder_hidden_dim,
                ),
                nn.SiLU(),
                nn.Linear(
                    in_features=self.encoder_hidden_dim,
                    out_features=1,
                ),
            )
        
        if self.with_terminal_prediction:
            self.terminal_head  = nn.Sequential(
                nn.Linear(in_features=self.encoder_hidden_dim, out_features=1),
                nn.Sigmoid()
            )

        self.optimizer  = torch.optim.AdamW(self.parameters(), lr=self.lr) 

        self.to(device=device)

#######################################################################################
    def predictor_fwd(self, x, deterministic=True):
        state_seq, action_seq, _ = x
        # The state sequences is already embedded
        bs, seq = state_seq.shape[:2]

        if self.fsq_encoder:
            # Embed the FSQ codes
            if len(state_seq.shape) > 2:
                state_seq = rearrange(state_seq, "bs seq f -> (bs seq) f")

            # Encode the sequence with FSQ
            emb = self.encoder_emb(state_seq.long())

            # Time-based transformer
            # Add a CLS token
            cls_tkns = torch.repeat_interleave(self.cls_tkn.view(1, -1), emb.shape[0], dim=0).view(emb.shape[0], 1, -1)
            emb = torch.concat([cls_tkns, emb], dim=1)
            # Add positional encoding
            emb = emb + self.position_embedding
            emb, _, _ = self.encoder_trans([emb, None, None])
            emb = emb[:, 0, :]
            emb = rearrange(emb, "(bs seq) h -> bs seq h", bs=bs, seq=seq)
        else:
            emb = state_seq

        # This may be unnecessary, but just to be sure
        action_seq = action_seq if self.with_action else None
        emb, _, _ = self.predictor([emb, action_seq, None])
        bs = emb.shape[0]
        emb = rearrange(emb, "bs t f -> (bs t) f")
        predicted = self.predictor_head(emb)
        # If we have the fsq_encoder, we are gonna have a categorical distribution
        predicted_logits = None
        if self.fsq_encoder:
            predicted_logits = rearrange(predicted, "bs (d l) -> bs d l", d=self.fsq_output_size, l=self.fsq_L)
            predicted_probs = F.softmax(predicted_logits, dim=-1)
            if deterministic:
                predicted = torch.argmax(predicted_probs, dim=-1)
            else:
                dist = Categorical(predicted_probs)
                predicted = dist.sample()

        predicted = rearrange(predicted, "(bs t) f -> bs t f", bs=bs)
        if predicted_logits is not None:
            predicted_logits = rearrange(predicted_logits, "(bs t) d l -> bs t d l", bs=bs, d=self.fsq_output_size, l=self.fsq_L)

        predicted_reward = None
        if self.with_reward_prediction:
            predicted_reward = self.reward_head(emb)
            predicted_reward = rearrange(
                predicted_reward, "(bs t) f -> bs t f", bs=bs
            )
        
        predicted_terminal = None
        if self.with_terminal_prediction:
            predicted_terminal = self.terminal_head(emb)
            predicted_terminal = rearrange(
                predicted_terminal, "(bs t) f -> bs t f", bs=bs
            )

        return predicted, predicted_logits, predicted_reward, predicted_terminal

#######################################################################################
    def encoder_fwd(self, x, return_quantized=False):
        bs  = x.shape[0]

        if not self.feature_base:
            # flatten for embeddin
            x   = rearrange(x, "bs seq c w h -> (bs seq) c w h")
            emb = self.encoder(x)
            emb = rearrange(emb, "(bs seq) h -> bs seq h", bs=bs)
        else:
            # FSQ based encoder
            if len(x.shape) > 2:
                x = rearrange(x, "bs seq f -> (bs seq) f")
            quantized_values = self.encoder(x)

            if self.fsq_encoder:
                code_indices = self.encoder.reshift_and_rescale(quantized_values.detach())
                code_indices = rearrange(code_indices, "(bs seq) h -> bs seq h", bs=bs )
                quantized_values = rearrange(quantized_values, "(bs seq) h -> bs seq h", bs=bs)

                if return_quantized:
                    return code_indices, quantized_values

                return code_indices

            emb = rearrange(quantized_values, "(bs seq) h -> bs seq h", bs=bs)

        if return_quantized:
            return emb, None
        
        return emb

#######################################################################################
    def set_dataset(self, dataset):
        # We assume the dataset is a dict of states, actions, rewards, next_states
        # TODO: This does not take into account the ending of sequences. 
        # Not sure it really matters for this small project, but it is something
        # to keep in mind

        self.dataset                = dataset

        for key in self.dataset.keys():
            
            if torch.is_tensor(self.dataset[key]):
                self.dataset[key] = self.dataset[key].to(self.device)
                continue

            if not isinstance(self.dataset[key], np.ndarray):
                self.dataset[key]   = np.asarray(self.dataset[key])
            
#######################################################################################
    def train_step(self, states_seq, actions_seq, rewards_seq=None, terminals_seq=None):
    
        latents, quantized_values                                                   = self.encoder_fwd(states_seq, return_quantized=True)
        predicted, predicted_logits, predicted_rewards, predicted_terminals         = self.predictor_fwd([latents[:, :-1], actions_seq[:, :-1], None], deterministic=False)
        if rewards_seq is not None:
            rewards_seq                                                                 = rewards_seq[:, :-1]
        if terminals_seq is not None:
            terminals_seq                                                               = terminals_seq[:, :-1]
    
        labels      = latents[:, 1:]
    
        if self.fsq_encoder:
            categorical_logits  = rearrange(predicted_logits, "bs t f d -> (bs t f) d")
            categorical_loss    = F.cross_entropy(categorical_logits, labels.reshape(-1))
    
            pred_loss           = categorical_loss
    
            reconstructed_states    = self.reconstruction_head(quantized_values)
            reconstruction_loss     = F.smooth_l1_loss(reconstructed_states, states_seq.float())
            sigreg_loss             = 0
        else:
            pred_loss   = (labels - predicted).pow(2).mean()
            sigreg_loss = self.sigreg(predicted.transpose(0, 1))
            reconstruction_loss = 0
    
        reward_loss = 0
        if self.with_reward_prediction:
            rewards_seq     = rewards_seq.reshape(-1, 1)
            reward_loss     = (rewards_seq - predicted_rewards) .pow(2).mean()
            
        terminal_loss = 0
        if self.with_terminal_prediction:
            terminals_seq       = terminals_seq.reshape(-1, 1).float()
            terminal_loss       = F.binary_cross_entropy(predicted_terminals, terminals_seq) 
    
        total_loss  = (
            pred_loss
            + self.lambd_sigreg * sigreg_loss
            + self.lambd_reconstruction * reconstruction_loss
            + reward_loss
            + terminal_loss
        )
        loss_dict   = dict(pred_loss=pred_loss, sigreg_loss=sigreg_loss)
        if self.fsq_encoder:
            loss_dict["categorical_loss"] = categorical_loss
            loss_dict["reconstruction_loss"] = reconstruction_loss
        if self.with_reward_prediction:
            loss_dict["rew_loss"] = reward_loss
            
        if self.with_terminal_prediction:
            loss_dict["term_loss"] = terminal_loss
    
        return total_loss, loss_dict  

#######################################################################################
    def train_epoch(self, batch_size):
        dataset_length  = self.dataset["states"].shape[0] - self.max_seq_length - 1
        num_batches     = int(np.ceil(dataset_length / batch_size)) 

        random_indices  = np.random.choice(np.arange(dataset_length), replace=False, size=np.arange(dataset_length).shape)
        
        losses = dict(total_loss=0)
        for mini_b in range(num_batches):
            mini_b_indices              = random_indices[mini_b * batch_size: mini_b * batch_size + batch_size]

            mini_b_states_sequences         = torch.tensor(np.asarray([self.dataset["states"][b:b+self.max_seq_length+1] for b in mini_b_indices])).to(self.device)
            mini_b_actions_sequences        = torch.tensor(np.asarray([self.dataset["actions"][b:b+self.max_seq_length+1] for b in mini_b_indices])).to(self.device)
            mini_b_rewards_sequences        = None
            if self.with_reward_prediction:
                mini_b_rewards_sequences    = torch.tensor(np.asarray([self.dataset["rewards"][b:b+self.max_seq_length+1] for b in mini_b_indices])).to(self.device)
                mini_b_rewards_sequences    = mini_b_rewards_sequences.unsqueeze(-1)
            mini_b_terminals_sequences      = None
            # TODO: This probably needs to be balanced, the states with False are much more than states with True
            if self.with_terminal_prediction:
                mini_b_terminals_sequences  = torch.tensor(np.asarray([self.dataset["terminals"][b:b+self.max_seq_length+1] for b in mini_b_indices])).to(self.device)
                mini_b_terminals_sequences  = mini_b_terminals_sequences.unsqueeze(-1)

            total_loss, loss_dict       = self.train_step(mini_b_states_sequences, mini_b_actions_sequences, mini_b_rewards_sequences, mini_b_terminals_sequences)
            for key in loss_dict.keys():
                if key not in losses:
                    losses[key] = 0 
                
                losses[key] += loss_dict[key]

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            losses["total_loss"] += total_loss
        
        for key in losses.keys():
            losses[key] /= num_batches
        
        return losses

#######################################################################################
    def validate_epoch(self, batch_size):
        if len(self.validation_sequence_starts) == 0:
            return None

        was_training = self.training
        teacher_forcing_probability = self.teacher_forcing_probability
        self.eval()
        self.teacher_forcing_probability = 0.0
        try:
            with torch.inference_mode():
                metrics = self._run_batches(
                    self.validation_sequence_starts,
                    batch_size,
                    training=False,
                )
        finally:
            self.teacher_forcing_probability = teacher_forcing_probability
            self.train(was_training)
        return metrics

#######################################################################################
    def save_model(self, name=None, folder='saved'):
        torch.save(self.state_dict(), '{}/{}'.format(folder, name))
        print("Model saved succesfully!")

#######################################################################################
    def load_model(self, name=None, folder='saved'):
        checkpoint = torch.load(
            '{}/{}'.format(folder, name),
            map_location=torch.device(self.device)
        )
        try:
            self.load_state_dict(checkpoint)
        except RuntimeError:
            current_state = self.state_dict()
            compatible_checkpoint = {
                key: value for key, value in checkpoint.items()
                if key in current_state
                and current_state[key].shape == value.shape
            }
            incompatible_keys = {
                key for key in set(checkpoint) | set(current_state)
                if not key.startswith("reward_head.")
                and (
                    key not in checkpoint
                    or key not in current_state
                    or (
                        key in checkpoint
                        and key in current_state
                        and checkpoint[key].shape != current_state[key].shape
                    )
                )
            }
            if incompatible_keys:
                raise
            self.load_state_dict(compatible_checkpoint, strict=False)
            print(
                "Loaded the shared world model; initialized the new "
                "MLP reward head from scratch."
            )
        print("Model loaded succesfully!")

#######################################################################################
if __name__ == "__main__":

    device          = "cuda"
    world_model     = LeWorldModel(
        action_dim=2,
        max_seq_length=3,
        device="cuda"
    )

    dataset = dict()
    dataset["states"] = torch.randn(100, 3, 224, 224).to(device)
    dataset["actions"] = torch.randn(100, 2).to(device)

    world_model.set_dataset(dataset)

    print("Start training")
    for epoch in range(10):
        losses = world_model.train_epoch(batch_size=4)
        print(f"At epoch {epoch}:")
        for key in losses.keys():
            print(f"    -{key}: {losses[key]}")
