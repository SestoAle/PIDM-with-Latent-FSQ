import torch.nn as nn
from utils.utils import *
from torch.distributions import Categorical, Beta
from torch import einsum
from einops import reduce

class CategoricalMasked(Categorical):
    def __init__(self, logits: torch.Tensor, mask=None):
        self.mask = mask.bool()
        self.batch, self.nb_action = logits.size()
        if mask is None:
            super(CategoricalMasked, self).__init__(logits=logits)
        else:
            self.mask_value = torch.tensor(
                torch.finfo(logits.dtype).min, dtype=logits.dtype
            ).to(logits.device)
            logits = torch.where(self.mask, logits, self.mask_value)
            super(CategoricalMasked, self).__init__(logits=logits)

    def entropy(self):
        if self.mask is None:
            return super().entropy()
        # Elementwise multiplication
        p_log_p = einsum("ij,ij->ij", self.logits, self.probs)
        # Compute the entropy with possible action only
        p_log_p = torch.where(
            self.mask,
            p_log_p,
            torch.tensor(0, dtype=p_log_p.dtype, device=p_log_p.device),
        )
        return -reduce(p_log_p, "b a -> b", "sum", b=self.batch, a=self.nb_action)


eps = 1e-13

class Policy(nn.Module):
    def __init__(self, state_dim, embedding_arch, action_size=4, action_type='discrete', distribution_type='beta',
                 max_action_value=1, min_action_value=-1,
                 **kwargs):
        super(Policy, self).__init__()

        # Policy hyperparameters
        self.action_size = action_size
        self.state_dim = state_dim
        self.max_action_value = max_action_value
        self.min_action_value = min_action_value
        self.action_type = action_type
        self.distribution_type = distribution_type

        # Layers specification
        self.embedding_l = embedding_arch(state_dim)

        if self.action_type == 'discrete':
            if type(action_size) == list:
                self.action_layers = []
                self.action_l = nn.ModuleList([nn.Linear(self.embedding_l.output_dim, a_s) for a_s in action_size])
            else:
                self.action_l = nn.Linear(self.embedding_l.output_dim, self.action_size)
        elif self.action_type == 'continuous':
            if self.distribution_type == 'beta':
                self.alpha_l = nn.Linear(self.embedding_l.output_dim, self.action_size)
                self.beta_l = nn.Linear(self.embedding_l.output_dim, self.action_size)

    def forward(self, inputs):
        state = torch.reshape(inputs, (-1, self.state_dim)).float()
        emb = self.embedding_l(state)
        if self.action_type == 'discrete':
            if type(self.action_size) == list:
                logits = []
                x = []
                for l in self.action_l:
                    out = l(emb)
                    logits.append(out)
                    x = F.softmax(out)
            else:
                logits = self.action_l(emb)
                x = F.softmax(logits)
        elif self.action_type == 'continuous':
            alpha = F.softplus(self.alpha_l(emb)) + 1
            beta = F.softplus(self.beta_l(emb)) + 1
            x = torch.cat([alpha, beta], dim=1)
            logits = x
        return x, logits, emb

class Value(nn.Module):
    def __init__(self, state_dim, embedding_arch, action_size, action_masking=False, **kwargs):
        super(Value, self).__init__()
        self.state_dim = state_dim
        self.action_masking = action_masking
        self.action_size = action_size
        # Layers specification
        self.embedding_l = embedding_arch(state_dim)
        self.q1_l = nn.Linear(self.embedding_l.output_dim, 1)

    def forward(self, state):
        if self.action_masking:
            if type(self.action_size) == list or type(self.action_size) == np.ndarray:
                mask_size = np.sum(self.action_size)
            else:
                mask_size = self.action_size
            state = state[:, :-mask_size]
        state = torch.reshape(state, (-1, self.state_dim))
        q1 = self.embedding_l(state)
        q1 = self.q1_l(q1)

        return q1

class PPOAgent(nn.Module):
    def __init__(self, state_dim, policy_embedding, critic_embedding, p_lr=7e-6, v_lr=7e-5, batch_size=32,
                 p_num_itr=20, v_num_itr=2, v_batch_size=32, previous_act=False, device='cpu',
                 # Actions
                 distribution='beta', action_type='continuous', action_size=[45, 5], action_min_value=-1,
                 action_max_value=1, frequency_mode='episodes',
                 epsilon=0.2, c1=0.5, c2=0.01, discount=0.99, lmbda=1.0, name='ppo', memory=10, norm_reward=False,
                 model_name='agent', action_masking=False,
                 **kwargs):
        super(PPOAgent, self).__init__()
        # Model parameters
        self.p_lr = p_lr
        self.v_lr = v_lr
        self.batch_size = batch_size
        self.v_batch_size = v_batch_size
        self.p_num_itr = p_num_itr
        self.v_num_itr = v_num_itr
        self.name = name
        self.norm_reward = norm_reward
        self.model_name = model_name
        self.frequency_mode = frequency_mode
        self.state_dim = state_dim
        self.device = device

        # Whether to use action masking
        self.action_masking = action_masking

        # Functions that define input and network specifications
        # Whether to use the previous actions or not.
        # Typically this is done with LSTM
        self.previous_act = previous_act

        # PPO hyper-parameters
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.discount = discount
        self.lmbda = lmbda
        # Action hyper-parameters
        # Types permitted: 'discrete' or 'continuous'. Default: 'discrete'
        self.action_type = action_type if action_type == 'continuous' or action_type == 'discrete' else 'discrete'
        self.action_size = action_size
        self.multi_branches = False
        if self.action_type == "discrete" and type(self.action_size) is list:
            self.multi_branches = True
            self.branches = len(self.action_size)

        # min and max values for continuous actions
        self.action_min_value = action_min_value
        self.action_max_value = action_max_value
        # Distribution type for continuous actions
        self.distribution_type = distribution if distribution == 'gaussian' or distribution == 'beta' else 'gaussian'

        self.buffer = dict()
        self.clear_buffer()
        self.memory = memory

        self.policy = Policy(state_dim, policy_embedding, self.action_size, self.action_type, self.distribution_type,
                             self.action_max_value, self.action_min_value).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.p_lr)

        self.critic = Value(state_dim, critic_embedding, action_size=self.action_size,
                             action_masking=self.action_masking).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.v_lr)

    def forward(self, inputs, deterministic=False, temperature=1, train=True):

        if self.action_masking:
            # The mask is the first action_size part of the input
            if self.multi_branches:
                mask_size = np.sum(self.action_size)
            else:
                mask_size = self.action_size            
            mask = inputs[:, -mask_size:]
            inputs = inputs[:, :-mask_size]
        if self.action_type == 'discrete':
            if deterministic:
                # TODO: multi branches
                # Get the most probable action
                probs, _, embs = self.policy(inputs)
                action = torch.argmax(probs)
            else:
                # Sample an action from probability distribution
                probs, logits, embs = self.policy(inputs)

                if self.multi_branches:
                    action = []
                    logprob = []
                    dist = []
                    action_masking_index = 0
                    for i, logit in enumerate(logits):
                        if self.action_masking:
                            d = CategoricalMasked(logits=logit, mask=mask[:, action_masking_index : action_masking_index + self.action_size[i]])
                            action_masking_index += self.action_size[i]
                        else:
                            d = Categorical(logits=logit)
                        a = d.sample()
                        lp = d.log_prob(a)
                        action.append(a)
                        logprob.append(lp)
                        dist.append(d)
                else:
                    logits /= temperature
                    if self.action_masking:
                        dist = CategoricalMasked(logits=logits, mask=mask)
                    else:
                        dist = Categorical(logits=logits)
                    action = dist.sample()
                    logprob = dist.log_prob(action)
        elif self.action_type == 'continuous':
            if self.distribution_type == 'beta':
                probs, logits, embs = self.policy(inputs)
                alpha = probs[:, :self.action_size]
                beta = probs[:, self.action_size:]
                dist = Beta(alpha, beta)
                # Sample an action from beta distribution
                action = dist.sample()
                logprob = dist.log_prob(action)
                # If there are more than 1 continuous actions, do the mean of log_probs
                if self.action_size > 1:
                    logprob = torch.sum(logprob, dim=1)
                # Standardize the action between min value and max value
                action = self.action_min_value + (
                        self.action_max_value - self.action_min_value) * action
                
                if deterministic:        
                    action = beta / (alpha + beta)
                    action = self.action_min_value + (
                        self.action_max_value - self.action_min_value) * action
        # action, all_xts, all_epsilons, all_ts, logprob/cfm, probs, dist
        return action, None, None, None, logprob, probs, dist

    # Clear the memory buffer
    def clear_buffer(self):

        self.buffer['episode_lengths'] = []
        self.buffer['states'] = []
        self.buffer['actions'] = []
        self.buffer['old_probs'] = []
        self.buffer['states_n'] = []
        self.buffer['rewards'] = []
        self.buffer['terminals'] = []

    # Add a transition to the buffer
    def add_to_buffer(self, state, state_n, action, reward, old_prob, terminals, epsilons, taus):
        # If we store more than memory episodes, remove the last episode
        if self.frequency_mode == 'episodes':
            if len(self.buffer['episode_lengths']) + 1 >= self.memory + 1:
                idxs_to_remove = self.buffer['episode_lengths'][0]
                del self.buffer['states'][:idxs_to_remove]
                del self.buffer['actions'][:idxs_to_remove]
                del self.buffer['old_probs'][:idxs_to_remove]
                del self.buffer['states_n'][:idxs_to_remove]
                del self.buffer['rewards'][:idxs_to_remove]
                del self.buffer['terminals'][:idxs_to_remove]
                del self.buffer['episode_lengths'][0]

        # If we store more than memory timesteps, remove the last timestep
        elif self.frequency_mode == 'timesteps':
            if (len(self.buffer['states']) + 1 > self.memory):
                del self.buffer['states'][0]
                del self.buffer['actions'][0]
                del self.buffer['old_probs'][0]
                del self.buffer['states_n'][0]
                del self.buffer['rewards'][0]
                del self.buffer['terminals'][0]

        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['old_probs'].append(old_prob)
        self.buffer['states_n'].append(state_n)
        self.buffer['rewards'].append(reward)
        self.buffer['terminals'].append(terminals)
        
        # If its terminal, update the episode length count (all states - sum(previous episode lengths)
        if self.frequency_mode == 'episodes':
            if terminals == 1 or terminals == 2:
                self.buffer['episode_lengths'].append(
                    int(len(self.buffer['states']) - np.sum(self.buffer['episode_lengths'])))
        else:
            self.buffer['episode_lengths'] = []
            for i, t in enumerate(self.buffer['terminals']):
                if t == 1 or t == 2:
                    self.buffer['episode_lengths'].append(
                        int(i + 1 - np.sum(self.buffer['episode_lengths'])))

    def add_batch_to_buffer(self, states, states_n, actions, rewards, old_probs, terminals):
        n = len(states)
        if self.frequency_mode == 'timesteps':
            overflow = len(self.buffer['states']) + n - self.memory
            if overflow > 0:
                del self.buffer['states'][:overflow]
                del self.buffer['actions'][:overflow]
                del self.buffer['old_probs'][:overflow]
                del self.buffer['states_n'][:overflow]
                del self.buffer['rewards'][:overflow]
                del self.buffer['terminals'][:overflow]
        self.buffer['states'].extend(states)
        self.buffer['actions'].extend(actions)
        self.buffer['old_probs'].extend(old_probs)
        self.buffer['states_n'].extend(states_n)
        self.buffer['rewards'].extend(rewards)
        self.buffer['terminals'].extend(terminals)

        # If its terminal, update the episode length count (all states - sum(previous episode lengths)
        if self.frequency_mode == 'episodes':
            if terminals == 1 or terminals == 2:
                self.buffer['episode_lengths'].append(
                    int(len(self.buffer['states']) - np.sum(self.buffer['episode_lengths'])))
        else:
            self.buffer['episode_lengths'] = []
            for i, t in enumerate(self.buffer['terminals']):
                if t == 1 or t == 2:
                    self.buffer['episode_lengths'].append(
                        int(i + 1 - np.sum(self.buffer['episode_lengths'])))

    
    # Change rewards in buffer to discounted rewards
    def compute_discounted_reward(self):

        discounted_rewards = []
        discounted_reward = 0
        # The discounted reward can be computed in reverse
        for (terminal, reward, i) in zip(reversed(self.buffer['terminals']), reversed(self.buffer['rewards']),
                                         reversed(range(len(self.buffer['rewards'])))):
            if terminal == 1:
                discounted_reward = 0
                # state = self.obs_to_state([self.buffer['states_n'][i]])
                # feed_dict = self.create_state_feed_dict(state)
                # discounted_reward = self.sess.run([self.value], feed_dict)[0]
            elif terminal == 2:
                state = self.buffer['states_n'][i]
                with torch.no_grad():
                    discounted_reward = self.critic(torch.from_numpy(state).to(self.device).float()).cpu().numpy().reshape(-1)

            discounted_reward = reward + (self.discount * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)

        # Normalizing reward
        if self.norm_reward:
            discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (
                    np.std(discounted_rewards) + eps)

        return discounted_rewards

    # Change rewards in buffer to discounted rewards or GAE rewards (if lambda == 1, gae == discounted)
    def compute_gae(self, v_values):

        rewards = []
        gae = 0

        # The gae rewards can be computed in reverse
        for (terminal, reward, i) in zip(reversed(self.buffer['terminals']), reversed(self.buffer['rewards']),
                                         reversed(range(len(self.buffer['rewards'])))):
            m = 1
            if terminal == 1:
                m = 0
                gae = 0
                v_next = 0
            elif terminal == 2:
                # Truncated: bootstrap with critic value of the actual next state,
                # but reset gae so it does not propagate across env boundaries
                m = 0
                gae = 0
                with torch.no_grad():
                    s_n = torch.from_numpy(self.buffer['states_n'][i]).unsqueeze(0).to(self.device).float()
                    v_next = self.critic(s_n).item()
            else:
                v_next = v_values[i + 1]

            delta = reward + self.discount * v_next - v_values[i]
            gae = delta + self.discount * self.lmbda * m * gae
            discounted_reward = gae + v_values[i]

            rewards.insert(0, discounted_reward)

        # Normalizing
        if self.norm_reward:
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + eps)

        return rewards

    # Critic loss
    def critic_loss(self, q_values, rewards):
        return F.mse_loss(q_values, rewards)

    # Policy loss
    def policy_loss(self, rewards, actions, dist, baseline_values, oldprob):
        baseline_values = baseline_values.reshape(-1, 1)
        oldprob = oldprob.reshape(-1, 1)

        # Advantage (reward - baseline)
        advantages = (rewards - baseline_values).reshape(-1, 1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + eps)

        # L_clip loss
        if self.action_type == 'discrete':
            if self.multi_branches:
                logprob_action = []
                for i, d in zip(range(len(dist)), dist):
                    tmp_a = actions[:, i].view(-1)
                    tmp_l = d.log_prob(tmp_a).view(-1, 1)
                    logprob_action.append(tmp_l)
                logprob_action = torch.cat(logprob_action, dim=1)
                logprob_action = torch.sum(logprob_action, dim=1).view(-1, 1)

                oldprob = oldprob.view(-1, len(self.action_size))
                oldprob = torch.sum(oldprob, dim=1).view(-1, 1)
            else:
                logprob_action = dist.log_prob(actions.reshape(-1)).reshape((-1, 1))
        else:
            # Inverse normalization actions between min_value and max_value
            # Beta Distribution
            if self.distribution_type == 'beta':
                actions = (actions - self.action_min_value) / (
                                    self.action_max_value - self.action_min_value)
                actions = torch.clamp(actions, 0 + eps, 1 - eps)

                logprob_action = dist.log_prob(actions)
                if self.action_size > 1:
                    logprob_action = torch.sum(logprob_action, dim=1)
                logprob_action = logprob_action.reshape(-1, 1)
        
        ratio = torch.exp(logprob_action - oldprob)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        clip_loss = torch.minimum(surr1, surr2)

        # Entropy Bonus
        if self.multi_branches:
            entr_loss = 0
            for d in dist:
                entr_loss += d.entropy()
        else:
            entr_loss = dist.entropy()
        # If there are more than 1 continuous actions, do the mean of entropies
        if self.action_type == 'continuous' and self.action_size > 1:
            entr_loss = torch.sum(entr_loss, dim=1)
        entr_loss = entr_loss.reshape(-1, 1)

        total_loss = - torch.mean(clip_loss + self.c2 * (entr_loss + eps))
        return total_loss

    # Update the model
    def update(self):
        self.train()
        losses = []
        v_losses = []


        # Compute GAE using the current critic BEFORE any parameter updates,
        # so advantages are consistent with the policy that collected the data.
        with torch.no_grad():
            states_all = torch.from_numpy(np.asarray(self.buffer['states'])).to(self.device).float()
            v_values = self.critic(states_all).detach().cpu().numpy()
        v_values = np.append(v_values, 0)
        discounted_rewards = self.compute_gae(v_values)

        # Train the value function
        batch_size = np.minimum(self.v_batch_size, len(self.buffer["states"]))
        for it in range(self.v_num_itr):
            mini_batch_idxs = random.sample(range(len(self.buffer['states'])), batch_size)
            states_mini_batch = [self.buffer['states'][id] for id in mini_batch_idxs]
            states_mini_batch = torch.from_numpy(np.asarray(states_mini_batch)).to(self.device).float()
            rewards_mini_batch = [discounted_rewards[id] for id in mini_batch_idxs]
            rewards_mini_batch = torch.from_numpy(np.asarray(rewards_mini_batch)).to(self.device).float()
            rewards_mini_batch = rewards_mini_batch.reshape(-1, 1)

            critic_loss = self.critic_loss(self.critic(states_mini_batch), rewards_mini_batch)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
            self.critic_optimizer.step()

            v_losses.append(critic_loss.detach().cpu())

        # Get policy batch size based on batch_size
        batch_size = np.minimum(self.batch_size, len(self.buffer["states"]))
        # Train the policy
        for it in range(self.p_num_itr):
            # Take a mini-batch of batch_size experience
            mini_batch_idxs = random.sample(range(len(self.buffer['states'])), batch_size)

            states_mini_batch = [self.buffer['states'][id] for id in mini_batch_idxs]
            states_mini_batch = torch.from_numpy(np.asarray(states_mini_batch)).to(self.device).float()
            actions_mini_batch = [self.buffer['actions'][id] for id in mini_batch_idxs]
            actions_mini_batch = torch.from_numpy(np.asarray(actions_mini_batch)).to(self.device).float()
            old_probs_mini_batch = [self.buffer['old_probs'][id] for id in mini_batch_idxs]
            old_probs_mini_batch = torch.from_numpy(np.asarray(old_probs_mini_batch)).to(self.device).float()
            rewards_mini_batch = [discounted_rewards[id] for id in mini_batch_idxs]
            rewards_mini_batch = torch.from_numpy(np.asarray(rewards_mini_batch)).to(self.device).float().reshape(-1, 1)
            # Get the baseline values
            v_values_mini_batch = [v_values[id] for id in mini_batch_idxs]
            v_values_mini_batch = torch.from_numpy(np.asarray(v_values_mini_batch)).to(self.device).float()

            if self.action_type == "continuous":
                actions_mini_batch = actions_mini_batch.view(-1, self.action_size)

            pi_actions, _, _, _, logprob, probs, dist = self.forward(states_mini_batch)

            p_loss = self.policy_loss(rewards_mini_batch, actions_mini_batch, dist, v_values_mini_batch, old_probs_mini_batch)
            # Optimize the actor
            self.policy_optimizer.zero_grad()
            p_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.policy_optimizer.step()
            losses.append(p_loss.detach().cpu())
        
        return np.mean(losses)

    def save_model(self, name=None, folder='saved'):
        torch.save(self.critic.state_dict(), '{}/{}_value'.format(folder, name))
        torch.save(self.critic_optimizer.state_dict(), '{}/{}_value_optimizer'.format(folder, name))

        torch.save(self.policy.state_dict(), '{}/{}_policy'.format(folder, name))
        torch.save(self.policy_optimizer.state_dict(), '{}/{}_policy_optimizer'.format(folder, name))

        # Input to the model
        if False:
            x = torch.zeros(1, self.state_dim).to(self.device)
            # Export the model
            torch.onnx.export(self.policy,  # model being run
                              x,  # model input (or a tuple for multiple inputs)
                              "{}/{}.onnx".format(folder, self.model_name),
                              # where to save the model (can be a file or file-like object)
                              export_params=True,  # store the trained parameter weights inside the model file
                              opset_version=9,  # the ONNX version to export the model to
                              do_constant_folding=True,  # whether to execute constant folding for optimization
                              input_names=['X'],  # the model's input names
                              output_names=['Y']  # the model's output names
                              )

    def load_model(self, name=None, folder='saved'):
        self.critic.load_state_dict(torch.load('{}/{}_value'.format(folder, self.model_name), map_location=torch.device(self.device)))
        self.critic_optimizer.load_state_dict(torch.load('{}/{}_value_optimizer'.format(folder, self.model_name), map_location=torch.device(self.device)))

        self.policy.load_state_dict(torch.load('{}/{}_policy'.format(folder, name), map_location=torch.device('cpu')))
        self.policy_optimizer.load_state_dict(torch.load('{}/{}_policy_optimizer'.format(folder, name), map_location=torch.device(self.device)))
        #print("PPO agent loaded succesfully!")