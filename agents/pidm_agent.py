import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW

###########################################################################################
class PIDMAgent(nn.Module):

    def __init__(self, 
                 state_size : int, 
                 action_size : int, 
                 policy_arch : nn.Module, 
                 lr : float,
                 device : str = "cpu",
                 only_real_states : bool = False,
                 **kwargs):
        super(PIDMAgent, self).__init__()


        self.action_size = action_size
        self.state_size = state_size
        self.lr = lr
        self.only_real_states = only_real_states

        self.policy = policy_arch(state_size)
        self.action_head = nn.Linear(self.policy.output_dim, action_size)
        self.optimizer = AdamW(self.parameters(), self.lr)
        self.device = device

        self.to(device)

###########################################################################################
    def forward(self, x):
        if self.only_real_states:
            x = x[0]
        else:
            x = torch.concatenate(x, dim=-1).float()
        emb = self.policy(x)
        action = F.tanh(self.action_head(emb))

        return action

###########################################################################################
    def set_dataset(self, states, actions, next_states):

        # This will probably be already tensors, because they will come
        # from the world model
        self.states         = states
        self.actions        = actions
        self.next_states    = next_states

        assert self.states.shape[0] == self.actions.shape[0] == self.next_states.shape[0], "We need equal number of states, actions, and next states"

###########################################################################################
    def train_step(self, batch):
        mb_states, mb_actions, mb_next_states = batch

        predicted_actions = self.forward([mb_states, mb_next_states])
        mse_loss = F.mse_loss(predicted_actions, mb_actions)

        self.optimizer.zero_grad()
        mse_loss.backward()
        self.optimizer.step()

        return mse_loss
    
###########################################################################################
    def train_epoch(self, batch_size):

        dataset_length = self.states.shape[0]
        num_batches = int(np.ceil(dataset_length / batch_size))
        random_indices = np.random.choice(np.arange(dataset_length), dataset_length, False)

        losses = []

        for mb in range(num_batches):
            mb_indices = random_indices[mb * batch_size : mb * batch_size + batch_size]

            mb_states = self.states[mb_indices]
            mb_actions = self.actions[mb_indices]
            mb_next_states = self.next_states[mb_indices]

            loss = self.train_step([mb_states, mb_actions, mb_next_states])
            losses.append(loss.detach().cpu().numpy())

        epoch_loss = np.mean(losses)
        loss = dict(total_loss=epoch_loss)
        return loss

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
        self.load_state_dict(checkpoint)