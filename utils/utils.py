import numpy as np
import json
import colorama, textwrap
import random
from copy import deepcopy
import torch
import torch.nn.functional as F

colorama.init(convert=True)

# Exponential annealing for any values, e.g. dscount increasing
def exponential_decay(initial_value, decay_rate, decay_steps, step):
    return initial_value * (decay_rate ** (step / decay_steps))

# Normalize values between -1 and 1
def fc_normalize(arr, max_val, min_val):  
        diff = max_val - min_val  
        normalized = ((arr - min_val) / diff) * 2 - 1  
        normalized = np.clip(normalized, -1, 1)
        return normalized  

def fc_compute_actions(data, with_target_direction=True):
    # Get the movement components
    actions = []
    min_magnitudes = []
    max_magnitudes = []
    angle_in_data = False
    with_sprint = False

    cancel_id = 99999
    i = 0
    for trajectory in data:
        i += 1
        if not with_target_direction:
            actionX = [-(next_state[5] - state[5]) for state, next_state in zip(trajectory["states"][:-1], trajectory["states"][1:])]
            actionZ = [next_state[6] - state[6] for state, next_state in zip(trajectory["states"][:-1], trajectory["states"][1:])]
            magnitudes = np.sqrt(np.sum(np.column_stack([actionX, actionZ])**2, axis=1))
        else:
            if len(trajectory["agent_actions"][:]) > 2:
                angle_in_data = True
                with_sprint = True
                try:
                    angles = np.asarray(trajectory["agent_actions"])[:, 0]
                    magnitudes = np.asarray(trajectory["agent_actions"])[:, 1]
                    is_sprints = np.asarray(trajectory["agent_actions"])[:, 2]
                except:
                    cancel_id = i - 1
                    continue
            else:
                try:
                    actionX = -np.asarray(trajectory["agent_actions"])[:, 0]
                    actionZ = np.asarray(trajectory["agent_actions"])[:, 1]
                except Exception as e:
                    cancel_id = i - 1
                    continue
                magnitudes = np.ones((len(actionX), ))

        try:
            dones = np.zeros(len(trajectory["states"][:-1]))
            dones[-1] = 1
        except Exception as e:
            cancel_id = i - 1
            continue
        
        if not angle_in_data:
            # Get the angle and normalize it to -1, 1
            angles = np.arctan2(actionZ, actionX)/np.pi
        else:
            # Normalize the angles between -1, 1
            angles = angles / np.pi - 1

        # Save the max and min magnitude 
        max_magnitudes.append(np.max(magnitudes))
        min_magnitudes.append(np.min(magnitudes))

        if with_sprint:
            actions = [[a, m, s] for a, m, s in zip(angles, magnitudes, is_sprints)]
        else:
            actions = [[a, m] for a, m in zip(angles, magnitudes)]
        trajectory["actions"] = actions
        trajectory["dones"] = dones

    if cancel_id < 99999:
        del data[cancel_id]

    data = [t for t in data if len(t["states"]) > 1] 
    return data


def fc_compute_actions_from_vector(direction, with_target_direction=False):
    # Get the angle and normalize it to 3.14
    angles = np.arctan2(direction[1], direction[0])/np.pi

    # Get the magnitude of the movement
    magnitudes = np.sqrt(np.sum(np.column_stack([direction[0], direction[1]])**2, axis=1))
    return angles, magnitudes


def fc_continuous_to_discrete(action, num_discrete_actions = 8):
        angle, _ = action
        if not (-1 <= angle <= 1):
            import ipdb; ipdb.set_trace()   
            raise ValueError("Invalid angle value. Angle should be within the range [-1, 1].")
        if num_discrete_actions < 1:
            raise ValueError("Invalid number of discrete actions. It should be a positive integer.")

        segment_length = 2 / num_discrete_actions
        discrete_action = round((angle + 1) / segment_length)

        # Ensure the highest angle value maps to the last discrete action
        if discrete_action == num_discrete_actions:
            discrete_action -= 1

        return discrete_action

def fc_continuous_to_discrete_actions(data):
    for trajectory in data:
        actions = []
        for action in trajectory["actions"]:
            discrete_action = fc_continuous_to_discrete(action)
            actions.append(discrete_action)
        trajectory["actions"] = actions

def fc_add_timestep_obs(data, norm=1):
    for trajectory in data:
        trajectory["states"] = np.concatenate([trajectory["states"], np.arange(len(trajectory["states"])).reshape(-1, 1)/norm], axis=1)
        
def color_string(description, value=None):
    ret_string = colorama.Fore.LIGHTGREEN_EX + "{}".format(description)
    if value is None:
        ret_string += colorama.Fore.RESET
    else:
        value = textwrap.fill(str(value), width=80, subsequent_indent=(2 + len(description))*" ")
        ret_string += " " + colorama.Fore.LIGHTRED_EX + "[{}]".format(value) + colorama.Fore.RESET
    return ret_string

def product(xs, empty=1):
    result = None
    for x in xs:
        if result is None:
            result = x
        else:
            result *= x

    if result is None:
        result = empty

    return result

class RunningStat(object):
        def __init__(self, shape=()):
            self._n = 0
            self._M = np.zeros(shape)
            self._S = np.zeros(shape)

        def push(self, x):
            x = np.asarray(x)
            assert x.shape == self._M.shape
            self._n += 1
            if self._n == 1:
                self._M[...] = x
            else:
                oldM = self._M.copy()
                self._M[...] = oldM + (x - oldM)/self._n
                self._S[...] = self._S + (x - oldM)*(x - self._M)

        @property
        def n(self):
            return self._n

        @property
        def mean(self):
            return self._M

        @property
        def var(self):
            if self._n >= 2:
                return self._S/(self._n - 1)
            else:
                return np.square(self._M)

        @property
        def std(self):
            return np.sqrt(self.var)

        @property
        def shape(self):

            return self._M.shape

class LimitedRunningStat(object):
    def __init__(self, len=1000):
        self.values = np.array(np.zeros(len))
        self.n_values = 0
        self.i = 0
        self.len = len

    def push(self, x):
        self.values[self.i] = x
        self.i = (self.i + 1) % len(self.values)
        if self.n_values < len(self.values):
            self.n_values += 1

    @property
    def n(self):
        return self.n_values

    @property
    def mean(self):
        return np.mean(self.values[:self.n_values])

    @property
    def var(self):
        return np.var(self.values[:self.n_values])

    @property
    def std(self):
        return np.std(self.values[:self.n_values])

class DynamicRunningStat(object):

    def __init__(self):
        self.current_rewards = list()
        self.next_rewards = list()

    def push(self, x):
        self.next_rewards.append(x)

    def reset(self):
        self.current_rewards = self.next_rewards
        self.next_rewards = list()

    @property
    def n(self):
        return len(self.current_rewards)

    @property
    def mean(self):
        return np.mean(np.asarray(self.current_rewards))

    @property
    def std(self):
        return np.std(np.asarray(self.current_rewards))


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def shape_list(x):
    '''
        deal with dynamic shape in tensorflow cleanly
    '''
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]

def stable_masked_softmax(logits, mask):

    #  Subtract a big number from the masked logits so they don't interfere with computing the max value
    if mask is not None:
        mask = tf.expand_dims(mask, 2)
        logits -= (1.0 - mask) * 1e10

    #  Subtract the max logit from everything so we don't overflow
    logits -= tf.reduce_max(logits, axis=-1, keepdims=True)
    unnormalized_p = tf.exp(logits)

    #  Mask the unnormalized probibilities and then normalize and remask
    if mask is not None:
        unnormalized_p *= mask
    normalized_p = unnormalized_p / (tf.reduce_sum(unnormalized_p, axis=-1, keepdims=True) + 1e-10)
    if mask is not None:
        normalized_p *= mask
    return normalized_p

def entity_avg_pooling_masked(x, mask):
    '''
        Masks and pools x along the second to last dimension. Arguments have dimensions:
            x:    batch x time x n_entities x n_features
            mask: batch x time x n_entities
    '''
    mask = tf.expand_dims(mask, -1)
    masked = x * mask
    summed = tf.reduce_sum(masked, -2)
    denom = tf.reduce_sum(mask, -2) + 1e-5
    return summed / denom

def entity_max_pooling_masked(x, mask):
    '''
        Masks and pools x along the second to last dimension. Arguments have dimensions:
            x:    batch x time x n_entities x n_features
            mask: batch x time x n_entities
    '''
    mask = tf.expand_dims(mask, -1)
    has_unmasked_entities = tf.sign(tf.reduce_sum(mask, axis=-2, keepdims=True))
    offset = (mask - 1) * 1e9
    masked = (x + offset) * has_unmasked_entities
    return tf.reduce_max(masked, -2)

# Boltzmann transformation to probability distribution
def boltzmann(probs, temperature = 1.):
    sum = np.sum(np.power(probs, 1/temperature))
    new_probs = []
    for p in probs:
        new_probs.append(np.power(p, 1/temperature) / sum)

    return np.asarray(new_probs)

# Very fast np.random.choice
def multidimensional_shifting(num_samples, sample_size, elements, probabilities):
    # replicate probabilities as many times as `num_samples`
    replicated_probabilities = np.tile(probabilities, (num_samples, 1))
    # get random shifting numbers & scale them correctly
    random_shifts = np.random.random(replicated_probabilities.shape)
    random_shifts /= random_shifts.sum(axis=1)[:, np.newaxis]
    # shift by numbers & find largest (by finding the smallest of the negative)
    shifted_probabilities = random_shifts - replicated_probabilities
    return np.argpartition(shifted_probabilities, sample_size, axis=1)[:, :sample_size]

def tf_normalize(value, tmin, tmax, rmin=-1, rmax=1):
    return (((value - rmin) / (rmax - rmin))*(tmax - tmin)) + tmin

# Copyright 2022 Div Garg. All rights reserved.
# Standalone IQ-Learn algorithm. See LICENSE for licensing terms.
# Full IQ-Learn objective with other divergences and options
def iq_loss(agent, current_Q, current_v, next_v, batch):
    # args = agent.args
    args = dict(loss="v0")
    gamma = agent.discount
    alpha = 0.5
    obs, next_obs, action, done, is_expert = batch

    # keep track of value of initial states
    v0 = agent.get_v(obs[is_expert.squeeze(1), ...]).mean()

    #  calculate 1st term for IQ loss
    #  -E_(ρ_expert)[Q(s, a) - γV(s')]
    y = (1 - done) * gamma * next_v
    reward = (current_Q - y)[is_expert]

    phi_grad = 1
    loss = -(phi_grad * reward).mean()

    # calculate 2nd term for IQ loss, we show different sampling strategies
    if args["loss"] == "value_expert":
        # sample using only expert states (works offline)
        # E_(ρ)[Q(s,a) - γV(s')]
        value_loss = (current_v - y)[is_expert].mean()
        loss += value_loss

    elif args["loss"] == "value":
        # sample using expert and policy states (works online)
        # E_(ρ)[V(s) - γV(s')]
        value_loss = (current_v - y).mean()
        loss += value_loss

    elif args["loss"] == "v0":
        # alternate sampling using only initial states (works offline but usually suboptimal than `value_expert` startegy)
        # (1-γ)E_(ρ0)[V(s0)]
        v0_loss = (1 - gamma) * v0
        loss += v0_loss

    y = (1 - done) * gamma * next_v
    reward = current_Q - y
    chi2_loss = 1 / (4 * alpha) * (reward ** 2).mean()
    loss += chi2_loss

    return loss

# This class will create the dataset for the Decision Transformer algorithm
class DTDemDataset:
    def __init__(self, trajectories, context_len, rtg_scale, state_norm=False, gamma=.99):

        # Trajectories must be a list of *episodes dict*, each of one should be a dict:
        # states => list of states of the trajectory_i
        # actions => list of actions of the trajectory_i
        # rewards => list of actions of the rewards_i
        self.trajectories = trajectories
        self.context_len = context_len

        # Compute the minimum length of the trajectories, and if state_norm=True normalize the *states*
        # In the original paper they scale the running reward, we set scale=1
        min_len = 10 ** 6
        states = []
        for traj in self.trajectories:
            traj_len = len(traj['states'])
            min_len = min(min_len, traj_len)
            states.append(traj['states'])
            traj['returns_to_go'] = self.discount_cumsum(traj['rewards'], gamma) / rtg_scale

        if state_norm:
            states = np.concatenate(states, axis=0)
            self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

            # normalize states
            for traj in self.trajectories:
                traj['states'] = (traj['states'] - self.state_mean) / self.state_std

    def get_state_stats(self):
        return self.state_mean, self.state_std

    def __len__(self):
        return len(self.trajectories)

    # Compute discount cumulative reward. In the original paper, gamma=1
    def discount_cumsum(self, x, gamma):
        disc_cumsum = np.zeros_like(x)
        disc_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            disc_cumsum[t] = x[t] + gamma * disc_cumsum[t + 1]
        return disc_cumsum

    # This will do the dataset magic. It creates sequence of context_len transitions to create the conext.
    # The sequences are created sampling a random index to slice the trajectories (if this is > context_len)
    def get_item(self, idx):
        traj = self.trajectories[idx]
        traj_len = len(traj['states'])

        # If the trajectory is less than context_len, we need to pad it and create a *mask*
        # (we can do it as we are using transformers)

        if traj_len >= self.context_len:
            si = random.randint(0, traj_len - self.context_len)

            states = traj['states'][si: si + self.context_len]
            actions = traj['actions'][si: si + self.context_len]
            returns_to_go = traj['returns_to_go'][si: si + self.context_len]
            timesteps = np.arange(si, si + self.context_len)

            traj_mask = np.ones(self.context_len)
        else:
            padding_len = self.context_len - traj_len

            states = deepcopy(traj['states'])
            # TODO: this works only if you have global_in. For now it is okay but correct this
            for p in range(padding_len):
                states.append(dict(global_in=np.zeros(np.shape(states[0]['global_in']))))

            actions = traj['actions']
            actions = np.concatenate([actions, np.zeros(([padding_len] + list(actions.shape[1:])))], axis=0)

            returns_to_go = traj['returns_to_go']
            returns_to_go = np.concatenate([returns_to_go, np.zeros(([padding_len] + list(returns_to_go.shape[1:])))],
                                      axis=0)

            timesteps = np.arange(0, self.context_len)
            traj_mask = np.concatenate([np.ones(traj_len), np.zeros(padding_len)], axis=0)

        return timesteps, states, actions, returns_to_go, traj_mask

    def get_minibatch(self, batch_size):
        random_indices = np.random.choice(len(self.trajectories), batch_size, replace=False)
        timesteps = []
        states = []
        actions = []
        returns_to_go = []
        traj_mask = []

        for idx in random_indices:
            i_timesteps, i_states, i_actions, i_rtg, i_tmask = self.get_item(idx)
            timesteps.append(i_timesteps)
            states.append(i_states)
            actions.append(i_actions)
            returns_to_go.append(i_rtg)
            traj_mask.append(i_tmask)

        return timesteps, states, actions, returns_to_go, traj_mask

