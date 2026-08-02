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
            nn.Tanh()
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
                 num_heads : int = 16,
                 num_decoder_layers : int = 6,
                 device : str = "cpu", 
                 feature_base : bool = False,
                 # With terminal and reward output
                 with_reward_prediction : bool = False,
                 with_terminal_prediction : bool = False,
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
        if self.autoregressive_rollout_length < 1:
            raise ValueError("autoregressive_rollout_length must be at least 1")
        if not 0.0 <= self.teacher_forcing_probability <= 1.0:
            raise ValueError("teacher_forcing_probability must be between 0 and 1")
        if not 0.0 <= self.reward_priority_fraction <= 1.0:
            raise ValueError("reward_priority_fraction must be between 0 and 1")
        if not 0.0 <= self.reward_high_quantile <= 1.0:
            raise ValueError("reward_high_quantile must be between 0 and 1")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")

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
                with_adaln=True,
                adaln_input_size=action_dim,
                is_causal=True,
                device=device
            ) for _ in range(self.num_decoder_layers)],
        )

        if not self.fsq_encoder:
            # Plus the same projection head that we have for the encoder
            self.predictor_head     = nn.Sequential(
                nn.Linear(in_features=self.encoder_hidden_dim, out_features=self.encoder_hidden_dim),
                nn.BatchNorm1d(self.encoder_hidden_dim)
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
        bs, seq = state_seq.shape[:2]

        # Embed the FSQ codes
        if len(state_seq.shape) > 2:
            state_seq = rearrange(state_seq, "bs seq f -> (bs seq) f")
        emb = self.encoder_emb(state_seq.long())
        # Add a CLS token
        cls_tkns = torch.repeat_interleave(self.cls_tkn.view(1, -1), emb.shape[0], dim=0).view(emb.shape[0], 1, -1)
        emb = torch.concat([cls_tkns, emb], dim=1)
        # Add positional encoding
        emb = emb + self.position_embedding
        emb, _ = self.encoder_trans([emb, None, None])
        emb = emb[:, 0, :]
        emb = rearrange(emb, "(bs seq) h -> bs seq h", bs=bs, seq=seq)

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
                # level_values = torch.arange(
                #     self.fsq_L,
                #     device=predicted_logits.device,
                #     dtype=predicted_logits.dtype
                # )
                # level_values = self.encoder.shift_and_scale(level_values)
                # predicted = (predicted_probs * level_values.view(1, 1, -1)).sum(dim=-1)
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
                code_indices = self.encoder.reshift_and_rescale(
                    quantized_values.detach()
                )
                code_indices = rearrange(
                    code_indices,
                    "(bs seq) h -> bs seq h",
                    bs=bs
                )
                quantized_values = rearrange(
                    quantized_values,
                    "(bs seq) h -> bs seq h",
                    bs=bs
                )

                if return_quantized:
                    return code_indices, quantized_values

                return code_indices

            emb = rearrange(quantized_values, "(bs seq) h -> bs seq h", bs=bs)

        return emb

#######################################################################################
    def set_dataset(self, dataset):
        # We assume the dataset is a dict of states, actions, rewards, next_states
        # But we will not use everything (e.g. the rewards)

        self.dataset                = dataset

        for key in self.dataset.keys():
            
            if torch.is_tensor(self.dataset[key]):
                self.dataset[key] = self.dataset[key].to(self.device)
                continue

            if not isinstance(self.dataset[key], np.ndarray):
                self.dataset[key]   = np.asarray(self.dataset[key])

        sequence_length = self.max_seq_length + self.autoregressive_rollout_length
        dataset_length = self.dataset["states"].shape[0]
        has_next_states = "next_states" in self.dataset
        num_windows = (
            dataset_length - sequence_length + 2
            if has_next_states
            else dataset_length - sequence_length + 1
        )
        valid_starts = np.arange(max(0, num_windows))
        if "terminals" in self.dataset:
            terminals = self.dataset["terminals"]
            if torch.is_tensor(terminals):
                terminals = terminals.detach().cpu().numpy()
            terminals = np.asarray(terminals).reshape(-1).astype(bool)
            valid_starts = np.asarray([
                start for start in valid_starts
                # A terminal is valid on the final modeled transition, but
                # never before it: no prediction may cross an episode boundary.
                if not terminals[start:start + sequence_length - 2].any()
            ])
        self.validation_sequence_starts = np.asarray([], dtype=np.int64)
        if (
            self.validation_fraction > 0
            and "terminals" in self.dataset
        ):
            episode_ids = np.cumsum(np.concatenate([
                np.asarray([False]),
                terminals[:-1],
            ]))
            unique_episodes = np.unique(episode_ids)
            if len(unique_episodes) > 1:
                rng = np.random.default_rng(self.validation_seed)
                num_validation_episodes = max(
                    1,
                    int(round(
                        len(unique_episodes) * self.validation_fraction
                    ))
                )
                num_validation_episodes = min(
                    num_validation_episodes,
                    len(unique_episodes) - 1,
                )
                validation_episodes = rng.choice(
                    unique_episodes,
                    size=num_validation_episodes,
                    replace=False,
                )
                validation_transition_mask = np.isin(
                    episode_ids, validation_episodes
                )
                validation_start_mask = validation_transition_mask[
                    valid_starts
                ]
                self.validation_sequence_starts = valid_starts[
                    validation_start_mask
                ]
                valid_starts = valid_starts[~validation_start_mask]

        self.valid_sequence_starts = valid_starts
        if len(self.valid_sequence_starts) == 0:
            raise ValueError(
                f"Dataset has no episode-safe windows of length {sequence_length}"
            )

        self.terminal_sequence_starts = np.asarray([], dtype=np.int64)
        self.high_reward_sequence_starts = np.asarray([], dtype=np.int64)
        if "rewards" in self.dataset:
            rewards = self.dataset["rewards"]
            if torch.is_tensor(rewards):
                rewards = rewards.detach().cpu().numpy()
            rewards = np.asarray(rewards).reshape(-1)
            training_reward_indices = (
                np.arange(len(rewards))
                if len(self.validation_sequence_starts) == 0
                else np.setdiff1d(
                    np.arange(len(rewards)),
                    np.concatenate([
                        np.arange(
                            start,
                            start + sequence_length - 1,
                        )
                        for start in self.validation_sequence_starts
                    ]),
                )
            )
            high_reward_threshold = np.quantile(
                np.abs(rewards[training_reward_indices]),
                self.reward_high_quantile
            )
            final_transition_indices = (
                self.valid_sequence_starts + sequence_length - 2
            )
            self.high_reward_sequence_starts = self.valid_sequence_starts[
                np.abs(rewards[final_transition_indices])
                >= high_reward_threshold
            ]
            if "terminals" in self.dataset:
                self.terminal_sequence_starts = self.valid_sequence_starts[
                    terminals[final_transition_indices]
                ]
            
#######################################################################################
    def train_step(self, states_seq, actions_seq, rewards_seq=None, terminals_seq=None):
        states_seq = states_seq.float()
        actions_seq = actions_seq.float()
        if rewards_seq is not None:
            rewards_seq = rewards_seq.float()
        if terminals_seq is not None:
            terminals_seq = terminals_seq.float()

        if self.fsq_encoder:
            latents, quantized_values = self.encoder_fwd(
                states_seq,
                return_quantized=True
            )
        else:
            latents = self.encoder_fwd(states_seq)
            quantized_values = None

        one_step_latents = latents[:, :self.max_seq_length + 1]
        one_step_actions = actions_seq[:, :self.max_seq_length]
        predicted, predicted_logits, predicted_rewards, predicted_terminals = self.predictor_fwd(
            [one_step_latents[:, :-1], one_step_actions, None],
            deterministic=False
        )
        one_step_rewards_seq = None
        if rewards_seq is not None:
            one_step_rewards_seq = rewards_seq[:, :self.max_seq_length]
        one_step_terminals_seq = None
        if terminals_seq is not None:
            one_step_terminals_seq = terminals_seq[:, :self.max_seq_length]

        labels = one_step_latents[:, 1:]

        if self.fsq_encoder:
            target_indices      = labels.detach()
            labels              = rearrange(target_indices, "bs t f -> (bs t f)").long()
            categorical_logits  = rearrange(predicted_logits, "bs t f d -> (bs t f) d")
            categorical_loss    = F.cross_entropy(categorical_logits, labels)

            with torch.no_grad():
                predicted_indices = predicted_logits.argmax(dim=-1)
                predicted_values = self.encoder.shift_and_scale(
                    predicted_indices.float()
                )
                target_values = self.encoder.shift_and_scale(
                    target_indices.float()
                )
                latent_mse_metric = F.mse_loss(
                    predicted_values,
                    target_values
                )

            pred_loss = categorical_loss
            autoregressive_loss = predicted_logits.new_zeros(())
            autoregressive_correct = predicted_logits.new_zeros(())
            autoregressive_reward_predictions = []
            autoregressive_reward_targets = []
            code_context = latents[:, :self.max_seq_length].detach()
            action_context = actions_seq[:, :self.max_seq_length]

            for rollout_step in range(self.autoregressive_rollout_length):
                _, rollout_logits, rollout_rewards, _ = self.predictor_fwd(
                    [code_context, action_context, None],
                    deterministic=True
                )
                next_logits = rollout_logits[:, -1]
                target_codes = latents[
                    :, self.max_seq_length + rollout_step
                ].detach().long()
                autoregressive_loss = autoregressive_loss + F.cross_entropy(
                    rearrange(next_logits, "bs f d -> (bs f) d"),
                    rearrange(target_codes, "bs f -> (bs f)")
                )
                predicted_codes = next_logits.argmax(dim=-1)
                autoregressive_correct = autoregressive_correct + (
                    predicted_codes == target_codes
                ).float().mean()
                if self.with_reward_prediction:
                    autoregressive_reward_predictions.append(
                        rollout_rewards[:, -1, 0]
                    )
                    reward_index = (
                        self.max_seq_length - 1 + rollout_step
                    )
                    autoregressive_reward_targets.append(
                        rewards_seq[:, reward_index].reshape(-1)
                    )

                if rollout_step + 1 < self.autoregressive_rollout_length:
                    teacher_mask = (
                        torch.rand(
                            target_codes.shape[0], 1,
                            device=target_codes.device
                        ) < self.teacher_forcing_probability
                    )
                    next_codes = torch.where(
                        teacher_mask, target_codes, predicted_codes
                    )
                    code_context = torch.cat(
                        [code_context[:, 1:], next_codes.unsqueeze(1)], dim=1
                    )
                    next_action_index = self.max_seq_length + rollout_step
                    action_context = torch.cat(
                        [
                            action_context[:, 1:],
                            actions_seq[:, next_action_index].unsqueeze(1)
                        ],
                        dim=1
                    )

            autoregressive_loss = (
                autoregressive_loss / self.autoregressive_rollout_length
            )
            autoregressive_code_accuracy = (
                autoregressive_correct / self.autoregressive_rollout_length
            )
            pred_loss = (
                categorical_loss
                + self.autoregressive_loss_weight * autoregressive_loss
            )
            # predicted_probs     = F.softmax(predicted_logits, dim=-1)
            # level_values        = torch.arange(
            #     self.fsq_L,
            #     device=predicted_logits.device,
            #     dtype=predicted_logits.dtype
            # )
            # level_values        = self.encoder.shift_and_scale(level_values)
            # expected_latents    = (predicted_probs * level_values.view(1, 1, 1, -1)).sum(dim=-1)
            # latent_l1_loss      = F.l1_loss(expected_latents, target_latents)
            # latent_mse_metric   = F.mse_loss(expected_latents.detach(), target_latents.detach())
            # pred_loss           = categorical_loss + self.lambd_latent_l1 * latent_l1_loss

            reconstructed_states = self.reconstruction_head(quantized_values)
            reconstruction_loss = F.smooth_l1_loss(reconstructed_states, states_seq.float())
            sigreg_loss         = 0
        else:
            pred_loss   = (labels - predicted).pow(2).mean()
            sigreg_loss = self.sigreg(predicted.transpose(0, 1))
            reconstruction_loss = 0

        reward_loss = 0
        if self.with_reward_prediction:
            reward_targets = one_step_rewards_seq.reshape(-1)
            reward_predictions = predicted_rewards.reshape(-1)
            one_step_reward_loss = F.mse_loss(
                reward_predictions, reward_targets
            )
            autoregressive_reward_predictions = torch.stack(
                autoregressive_reward_predictions, dim=1
            )
            autoregressive_reward_targets = torch.stack(
                autoregressive_reward_targets, dim=1
            )
            autoregressive_reward_loss = F.mse_loss(
                autoregressive_reward_predictions,
                autoregressive_reward_targets,
            )
            all_reward_predictions = torch.cat(
                [
                    reward_predictions,
                    autoregressive_reward_predictions.reshape(-1),
                ]
            )
            all_reward_targets = torch.cat(
                [
                    reward_targets,
                    autoregressive_reward_targets.reshape(-1),
                ]
            )
            reward_loss = F.mse_loss(
                all_reward_predictions, all_reward_targets
            )
            with torch.no_grad():
                def reward_metrics(predictions, targets):
                    predictions = predictions.reshape(-1)
                    targets = targets.reshape(-1)
                    centered_predictions = (
                        predictions - predictions.mean()
                    )
                    centered_targets = targets - targets.mean()
                    correlation = (
                        centered_predictions * centered_targets
                    ).mean() / (
                        centered_predictions.square().mean().sqrt()
                        * centered_targets.square().mean().sqrt()
                        + 1e-8
                    )
                    mse = F.mse_loss(predictions, targets)
                    return mse, correlation

                (
                    one_step_reward_mse,
                    one_step_reward_correlation,
                ) = reward_metrics(
                    reward_predictions, reward_targets
                )
                (
                    autoregressive_reward_mse,
                    autoregressive_reward_correlation,
                ) = reward_metrics(
                    autoregressive_reward_predictions,
                    autoregressive_reward_targets,
                )
                reward_mse_metric, reward_correlation = reward_metrics(
                    all_reward_predictions, all_reward_targets
                )
        
        terminal_loss = 0
        if self.with_terminal_prediction:
            terminals_seq = one_step_terminals_seq.reshape(-1, 1).float()
            terminal_loss = F.binary_cross_entropy(
                predicted_terminals.reshape(-1, 1), terminals_seq
            )


        total_loss  = (
            pred_loss
            + self.lambd_sigreg * sigreg_loss
            + self.lambd_reconstruction * reconstruction_loss
            + self.lambd_reward * reward_loss
            + terminal_loss
        )
        loss_dict   = dict(pred_loss=pred_loss, sigreg_loss=sigreg_loss)
        if self.fsq_encoder:
            loss_dict["categorical_loss"] = categorical_loss
            loss_dict["latent_mse_metric"] = latent_mse_metric
            loss_dict["reconstruction_loss"] = reconstruction_loss
            loss_dict["autoregressive_loss"] = autoregressive_loss
            loss_dict["autoregressive_code_accuracy"] = autoregressive_code_accuracy
        if self.with_reward_prediction:
            loss_dict["rew_loss"] = reward_loss
            loss_dict["one_step_reward_loss"] = one_step_reward_loss
            loss_dict["autoregressive_reward_loss"] = autoregressive_reward_loss
            loss_dict["one_step_reward_mse"] = one_step_reward_mse
            loss_dict["one_step_reward_correlation"] = one_step_reward_correlation
            loss_dict["autoregressive_reward_mse"] = autoregressive_reward_mse
            loss_dict["autoregressive_reward_correlation"] = autoregressive_reward_correlation
            loss_dict["reward_mse_metric"] = reward_mse_metric
            loss_dict["reward_correlation"] = reward_correlation
        
        if self.with_terminal_prediction:
            loss_dict["term_loss"] = terminal_loss

        return total_loss, loss_dict 
    
#######################################################################################
    def _get_sequences(self, key, indices, sequence_length):
        values = self.dataset[key]
        if key == "states" and "next_states" in self.dataset:
            next_states = self.dataset["next_states"]
            sequences = []
            for start in indices:
                prefix = values[start:start + sequence_length - 1]
                final_next_state = next_states[
                    start + sequence_length - 2
                ]
                if torch.is_tensor(values):
                    sequence = torch.cat(
                        [prefix, final_next_state.unsqueeze(0)], dim=0
                    )
                else:
                    sequence = np.concatenate(
                        [prefix, final_next_state[None]], axis=0
                    )
                sequences.append(sequence)
        else:
            sequences = [
                values[start:start + sequence_length - 1]
                for start in indices
            ]
        if torch.is_tensor(values):
            return torch.stack(sequences).to(self.device)
        return torch.as_tensor(
            np.asarray(sequences), device=self.device
        )

#######################################################################################
    def _run_batches(self, indices, batch_size, training):
        sequence_length = (
            self.max_seq_length + self.autoregressive_rollout_length
        )
        num_batches = int(np.ceil(len(indices) / batch_size))
        losses = dict(total_loss=0)

        for mini_b in range(num_batches):
            mini_b_indices = indices[
                mini_b * batch_size:mini_b * batch_size + batch_size
            ]
            states = self._get_sequences(
                "states", mini_b_indices, sequence_length
            )
            actions = self._get_sequences(
                "actions", mini_b_indices, sequence_length
            )
            rewards = None
            if self.with_reward_prediction:
                rewards = self._get_sequences(
                    "rewards", mini_b_indices, sequence_length
                ).unsqueeze(-1)
            terminals = None
            if self.with_terminal_prediction:
                terminals = self._get_sequences(
                    "terminals", mini_b_indices, sequence_length
                ).unsqueeze(-1)

            total_loss, loss_dict = self.train_step(
                states, actions, rewards, terminals
            )
            for key, value in loss_dict.items():
                if key not in losses:
                    losses[key] = 0
                losses[key] += (
                    value.detach() if torch.is_tensor(value) else value
                )
            losses["total_loss"] += total_loss.detach()

            if training:
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        return {
            key: value / num_batches for key, value in losses.items()
        }

#######################################################################################
    def train_epoch(self, batch_size):
        num_batches = int(np.ceil(
            len(self.valid_sequence_starts) / batch_size
        ))
        num_samples = len(self.valid_sequence_starts)
        terminal_samples = (
            int(num_samples * self.reward_priority_fraction / 2)
            if len(self.terminal_sequence_starts) > 0 else 0
        )
        high_reward_samples = (
            int(num_samples * self.reward_priority_fraction / 2)
            if len(self.high_reward_sequence_starts) > 0 else 0
        )
        regular_samples = (
            num_samples - terminal_samples - high_reward_samples
        )
        sampled_indices = [
            np.random.choice(
                self.valid_sequence_starts,
                size=regular_samples,
                replace=regular_samples > len(self.valid_sequence_starts),
            )
        ]
        if terminal_samples:
            sampled_indices.append(np.random.choice(
                self.terminal_sequence_starts,
                size=terminal_samples,
                replace=terminal_samples > len(self.terminal_sequence_starts),
            ))
        if high_reward_samples:
            sampled_indices.append(np.random.choice(
                self.high_reward_sequence_starts,
                size=high_reward_samples,
                replace=high_reward_samples > len(self.high_reward_sequence_starts),
            ))
        random_indices = np.concatenate(sampled_indices)
        np.random.shuffle(random_indices)
        return self._run_batches(random_indices, batch_size, training=True)

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
