import gymnasium as gym
import numpy as np

class GymEnv:
    def __init__(self, max_episode_timesteps, continuous=True, save_trajectories=False, visualize_inference=False):

        self.continuous = continuous
        self._max_episode_timesteps = max_episode_timesteps
        self.ep = 0
        self.save_trajectories = save_trajectories
        self.visualize_inference = visualize_inference

        if self.save_trajectories:
            self.trajectories = []
            self.new_trajectory = None

        self.env = gym.make("LunarLanderContinuous-v3", render_mode="human" if self.visualize_inference else None)

        self.state_dim  = 8
        self.action_dim = 2

    def reset(self, seed=None):

        self.ep_reward = 0
        self.current_timestep = 0

        state, info = self.env.reset(seed=seed)

        if self.save_trajectories:

            if self.new_trajectory is not None:
                self.trajectories.append(self.new_trajectory)

            self.new_trajectory = {
                "states": [state],
                "actions": [],
                "rewards": [],
                "terminals": [],
                "next_states": []
            }

        # state = self.normalize_obs(state)

        self.ep += 1
        return state

    def step(self, actions):
        state, reward, done, interrupted, info = self.env.step(actions)
        if self.visualize_inference:
            self.env.render()

        # state = self.normalize_obs(state)

        self.current_timestep += 1
        if self.current_timestep >= self._max_episode_timesteps:
            done = True
        
        if interrupted:
            done = True

        self.ep_reward += reward

        if self.save_trajectories:
            self.new_trajectory["actions"].append(actions)
            self.new_trajectory["rewards"].append(reward)
            self.new_trajectory["terminals"].append(done)
            self.new_trajectory["next_states"].append(state)
            if not done:
                self.new_trajectory["states"].append(state)

        return state, reward, done, dict()

    def entropy(self, probs):
        if self.continuous:
            return 0
        entr = 0
        for p in probs:
            entr += (p * np.log(p))
        return -entr

    def set_config(self, config):
        self.config = config

    def close(self):
        self.env.close()
    
    def observations_space(self):
        return self.env.observation_space.shape[0]
    
    def action_space(self):
        return self.env.action_space.shape[0]
    
    def normalize_obs(self, obs):
        # Based on LunarLander documentation
        mins = np.asarray(self.env.observation_space.low)
        maxs = np.asarray(self.env.observation_space.high)

        normed_obs = (2 * (obs - mins) / (maxs - mins)) -1
        return normed_obs