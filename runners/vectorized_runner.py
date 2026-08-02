import os
import numpy as np
import json
from utils.utils import NumpyEncoder
import time
import pickle
import torch
from copy import deepcopy

class Runner:
    def __init__(self, should_stop, agent, frequency, env, 
                 
                 # These are specific to the vectorized environment
                 num_envs,
                 
                 save_frequency=3000, logging=100, total_episode=1e10, curriculum=None,
                 frequency_mode='episodes', random_actions=None, curriculum_mode='steps', evaluate=False,
                 callback_function=None, motivation=None, temperature=1,
                 # IRL
                 reward_model=None, fixed_reward_model=False, dems_name='', reward_frequency=30, demonstrations_name='None',
                 # Adversarial Play
                 adversarial_play=False, double_agent=None,
                 **kwargs):

        # Thread handling
        self.should_stop = should_stop

        # Vectorized environment
        self.num_envs = num_envs

        # Runner objects and parameters
        self.agent = agent
        self.curriculum = curriculum
        self.current_curriculum_step = 0
        if self.curriculum is not None:
            self.current_curriculum_step = self.curriculum["current_step"]
        self.total_episode = total_episode
        self.frequency = frequency
        self.frequency_mode = frequency_mode
        self.random_actions = random_actions
        self.logging = logging
        self.save_frequency = save_frequency
        self.env = env
        self.curriculum_mode = curriculum_mode
        self.device = agent.device
        
        # If we want to evaluate it, and what temperature we should use
        self.evaluate = evaluate
        self.temperature = temperature

        # TODO: pass this as an argument
        self.motivation_frequency = 1

        # For alternating between motivation and imitation reward
        self.alternate_frequency = 0
        self.alternate_count = 0
        self.alternate_turn = 0

        # If we want to use intrinsic motivation
        # Right now only RND is available
        self.motivation = motivation

        # Function to call at the end of each episode.
        # It takes the agent, the runner and the env as input arguments
        self.callback_function = callback_function

        # Recurrent
        self.recurrent = False

        # Objects and parameters for IRL
        self.reward_model = reward_model
        self.fixed_reward_model = fixed_reward_model
        self.dems_name = dems_name
        self.reward_frequency = reward_frequency
        self.demonstrations_name = demonstrations_name

        # Adversarial play
        self.adversarial_play = adversarial_play
        self.double_agent = double_agent
        # If adversarial play, save the first version of the main agent and load it to the double agent
        if self.adversarial_play:
            self.agent.save_model(name=self.agent.model_name + '_0', folder='saved/adversarial')
            self.double_agent.load_model(name=self.agent.model_name + '_0', folder='saved/adversarial')

        # Global runner statistics
        # total episode
        self.ep = 0
        self.ep_before_training = 0
        self.ep_before_motivation_training = 0
        self.ep_before_reward_training = 0
        self.ep_before_logging = 0
        self.ep_before_saving = 0
        # total steps
        self.total_step = 0
        # Initialize history
        # History to save model statistics
        self.history = {
            "episode_rewards": [],
            "episode_timesteps": [],
            "mean_entropies": [],
            "std_entropies": [],
            "reward_model_loss": [],
            "env_rewards": [],
            "info": []
        }


        # Initialize reward model
        if self.reward_model is not None:
            if not self.fixed_reward_model:
                # Ask for demonstrations
                answer = None
                while answer != 'y' and answer != 'n':
                    answer = input('Do you want to create new demonstrations? [y/n] ')
                # Before asking for demonstrations, set the curriculum of the environment
                config = self.set_curriculum(self.curriculum, self.history, self.curriculum_mode)
                self.env.set_config(config)
                if answer == 'y':
                    dems, vals = self.reward_model.create_demonstrations(env=self.env, dems_name=dems_name)
                elif answer == 'p':
                    dems, vals = self.reward_model.create_demonstrations(env=self.env, with_policy=True)
                else:
                    print('Loading demonstrations...')
                    dems, vals = self.reward_model.load_demonstrations(self.dems_name)

                # Set demonstrations for the environment
                # self.env.set_demonstrations(dems)

                print('Demonstrations loaded! We have ' + str(len(dems['obs'])) + " timesteps in these demonstrations")
                #print('and ' + str(len(vals['obs'])) + " timesteps in these validations.")

                # Getting initial experience from the environment to do the first training epoch of the reward model
                self.get_experience(env, self.reward_frequency, random=True)
                self.reward_model.update()

        # For curriculum training
        self.start_training = 0
        self.current_curriculum_change = 0


        # If a saved model with the model_name already exists, load it (and the history attached to it)
        if os.path.exists('{}/{}_policy'.format('saved', agent.model_name)):
            answer = None
            while answer != 'y' and answer != 'n':
                answer = input("There's already an agent saved with name {}, "
                               "do you want to continue training? [y/n] ".format(agent.model_name))

            if answer == 'y':
                old_history = self.load_model(agent.model_name, agent) 
                self.history = old_history if old_history is not None else self.history
                self.ep = len(self.history['episode_timesteps'])
                self.total_step = np.sum(self.history['episode_timesteps'])
                #self.agent.total_itr = self.agent.reset_steps * self.agent.replay_ratio
                # self.random_actions = None


        # Decaying weight of the motivation/inverse reinforcement learning model
        self.last_episode_for_decaying = 0
        # if self.motivation is not None:
        #     self.motivation.motivation_weight = 0.8
        #     self.min_motivation_weight = 0.2

    def run(self):

        self.trajectories = []
        traj = None
        # Trainin loop
        # Start training
        start_time = time.time()

        while self.ep <= self.total_episode and not self.should_stop.is_set():
            # Reset the episode

            step = np.zeros((self.num_envs, 1))

            # Set actual curriculum
            config = self.set_curriculum(self.curriculum, self.history, self.curriculum_mode)
            if self.start_training == 0:
                print(config)
            self.start_training = 1

            self.env.set_config(config)

            # This is a vectorized runner, meaning we are using a vectorized environment.
            # The vectorized environment will return NUM_ENVS of everything.


            state = self.env.reset()

            done = False
            # Total reward of the episode
            episode_reward = np.zeros((self.num_envs, 1))
            # Total reward of the environment, in case of IRL it can be different from the actual reward of the agent
            env_episode_reward = np.zeros((self.num_envs, 1))

            # Save local entropies
            local_entropies = []

            # The tmp buffer will be used in case we use Episodes as frequency
            self.tmp_buffer = {
                "states"        : [[] for _ in range(self.num_envs)],
                "actions"       : [[] for _ in range(self.num_envs)],
                "rewards"       : [[] for _ in range(self.num_envs)],
                "states_n"      : [[] for _ in range(self.num_envs)],
                "logprobs"      : [[] for _ in range(self.num_envs)],
                "dones"         : [[] for _ in range(self.num_envs)]
            }

            # If recurrent, initialize hidden state
            if self.recurrent:
                raise NotImplementedError("Not implemented for vectorized environment")
                internal = (np.zeros([1, self.agent.recurrent_size]), np.zeros([1, self.agent.recurrent_size]))
                v_internal = (np.zeros([1, self.agent.recurrent_size]), np.zeros([1, self.agent.recurrent_size]))

            # Episode loop
            while True:

                if self.should_stop.is_set():
                    break
                
                # Evaluation - Execute step
                if not self.recurrent:
                    action, _, epsilons, taus, logprob, probs, dist = self.agent(torch.from_numpy(state).to(self.agent.device).float())
                else:
                    raise NotImplementedError("Not implemented for vectorized environment")
                    action, _, epsilons, taus, logprob, probs, internal_n, v_internal_n = self.agent.eval_recurrent([state], internal, v_internal)

                action = action.detach().cpu().numpy()
                logprob = logprob.detach().cpu().numpy() if logprob is not None else None
                probs = probs.detach().cpu().numpy() if probs is not None else None
                epsilons = epsilons.detach().cpu().numpy() if epsilons is not None else None
                taus = taus.detach().cpu().numpy() if taus is not None else None

                # All these should be (NUM_ENVS, -1)
                if self.random_actions is not None and self.total_step < self.random_actions:
                    raise NotImplementedError("Not implemented yest for VecEnv")
                    if self.agent.action_type == 'discrete':
                        action = np.asarray(np.random.randint(self.num_envs, self.agent.action_size))
                    else:
                        action = np.asarray(np.random.uniform(-1, 1, (self.num_envs, self.agent.action_size)))

                local_entropies.append(self.env.entropy(dist))
                
                state_n = None
                while state_n is None:
                    state_n, reward, done, info = self.env.step(action)
                
                step += 1
                self.total_step += 1

                # Intrinsic Motivation
                # Add the next state to the motivation buffer
                # - The intrinsic reward will be added later -
                if self.motivation is not None:
                    raise NotImplementedError("Not implemented for vectorized environment")

                    motivation_reward = self.motivation.eval([state_n])
                    self.motivation.add_to_buffer(state_n)

                # Inverse Reinforcement Learning
                # Add the next state to the IRL buffer
                # - The intrinsic reward will be added later -
                if self.reward_model is not None:
                    raise NotImplementedError("Not implemented for vectorized environment")

                    irl_reward = self.reward_model.eval([state], [state_n],
                                                            [action])
                    # print(irl_reward)
                    self.reward_model.add_to_policy_buffer([state], [state_n], [action])


                # Get the cumulative reward
                episode_reward += reward

                # Update memory
                if not self.recurrent:

                    if self.frequency_mode == "episodes":
                        # Add everything to the tmp buffer
                        for i in range(self.num_envs):
                            self.tmp_buffer["states"][i].append(deepcopy(state[i]))
                            self.tmp_buffer["states_n"][i].append(deepcopy(state_n[i]))
                            self.tmp_buffer["actions"][i].append(deepcopy(action[i]))
                            self.tmp_buffer["rewards"][i].append(deepcopy(reward[i]))
                            self.tmp_buffer["logprobs"][i].append(deepcopy(logprob[i]))
                            self.tmp_buffer["dones"][i].append(deepcopy(done[i]))

                            if done[i] == 1 or done[i] == 2:
                                # At this point we will update the history
                                self.history['episode_rewards'].append(deepcopy(episode_reward[i]))
                                self.history['episode_timesteps'].append(deepcopy(step[i]))
                                # TODO this will be added once we know it is working
                                self.history['mean_entropies'].append(0)
                                self.history['std_entropies'].append(0)
                                self.history['env_rewards'].append(deepcopy(env_episode_reward[i]))
                                # TODO also this needs to be implemented
                                self.history['info'].append(None)

                                self.ep += 1
                                self.ep_before_training += 1
                                self.ep_before_motivation_training += 1
                                self.ep_before_reward_training += 1
                                self.ep_before_logging += 1
                                self.ep_before_saving += 1
                                step[i] = 0
                                episode_reward[i] = 0
                                for j in range(len(self.tmp_buffer["states"][i])):
                                    self.agent.add_to_buffer(
                                        self.tmp_buffer["states"][i][j],
                                        self.tmp_buffer["states_n"][i][j],
                                        self.tmp_buffer["actions"][i][j],
                                        self.tmp_buffer["rewards"][i][j],
                                        self.tmp_buffer["logprobs"][i][j],
                                        self.tmp_buffer["dones"][i][j],
                                        None,
                                        None,
                                    )
                                
                                self.tmp_buffer["states"][i]    = []
                                self.tmp_buffer["states_n"][i]  = []
                                self.tmp_buffer["actions"][i]   = []
                                self.tmp_buffer["rewards"][i]   = []
                                self.tmp_buffer["logprobs"][i]  = []
                                self.tmp_buffer["dones"][i]     = []

                    else:
                        for i in range(self.num_envs):
                            self.tmp_buffer["states"][i].append(deepcopy(state[i]))
                            self.tmp_buffer["states_n"][i].append(deepcopy(state_n[i]))
                            self.tmp_buffer["actions"][i].append(deepcopy(action[i]))
                            self.tmp_buffer["rewards"][i].append(deepcopy(reward[i]))
                            self.tmp_buffer["logprobs"][i].append(deepcopy(logprob[i]))
                            self.tmp_buffer["dones"][i].append(deepcopy(done[i]))

                            if done[i] == 1 or done[i] == 2:
                                # At this point we will update the history
                                self.history['episode_rewards'].append(deepcopy(episode_reward[i]))
                                self.history['episode_timesteps'].append(deepcopy(step[i]))
                                # TODO this will be added once we know it is working
                                self.history['mean_entropies'].append(0)
                                self.history['std_entropies'].append(0)
                                self.history['env_rewards'].append(deepcopy(env_episode_reward[i]))
                                # TODO also this needs to be implemented
                                self.history['info'].append(None)

                                self.ep += 1
                                self.ep_before_training += 1
                                self.ep_before_motivation_training += 1
                                self.ep_before_reward_training += 1
                                self.ep_before_logging += 1
                                self.ep_before_saving += 1
                                step[i] = 0
                                episode_reward[i] = 0

                        # If we have enough samples, add them to the buffer
                        if self.total_step > 0 and self.total_step % self.frequency == 0:
                            for i in range(self.num_envs):
                                self.tmp_buffer["dones"][i][-1] = 1 if self.tmp_buffer["dones"][i][-1] == 1 else 2

                                self.agent.add_batch_to_buffer(
                                    self.tmp_buffer["states"][i],
                                    self.tmp_buffer["states_n"][i],
                                    self.tmp_buffer["actions"][i],
                                    self.tmp_buffer["rewards"][i],
                                    self.tmp_buffer["logprobs"][i],
                                    self.tmp_buffer["dones"][i],
                                )

                                self.tmp_buffer["states"][i]    = []
                                self.tmp_buffer["states_n"][i]  = []
                                self.tmp_buffer["actions"][i]   = []
                                self.tmp_buffer["rewards"][i]   = []
                                self.tmp_buffer["logprobs"][i]  = []
                                self.tmp_buffer["dones"][i]     = []
                else:
                    raise NotImplementedError("Not implemented for vectorized environment")
                    try:
                        self.agent.add_to_buffer(state, state_n, action, reward, logprob, done,
                                                 internal.c[0], internal.h[0], v_internal.c[0], v_internal.h[0])
                    except Exception as e:
                        zero_state = np.reshape(internal[0], [-1,])
                        self.agent.add_to_buffer(state, state_n, action, reward, logprob, done,
                                                 zero_state, zero_state, zero_state, zero_state)
                    internal = internal_n
                    v_internal = v_internal_n

                state = state_n

                # If frequency timesteps are passed, update the policy
                if not self.evaluate and self.frequency_mode == 'timesteps' and \
                        self.total_step > 0 and self.total_step % self.frequency == 0:
                    if self.random_actions is not None:
                        if self.total_step > self.random_actions:
                            self.agent.update()
                    else:

                        if self.motivation is not None:
                            # TODO: move this into a function
                            # Normalize observation of the motivation buffer
                            # self.motivation.normalize_buffer()
                            # Compute intrinsic rewards
                            intrinsic_rews = self.motivation.eval(self.agent.buffer['states_n'])

                            # Normalize rewards
                            intrinsic_rews -= np.mean(intrinsic_rews)
                            intrinsic_rews /= np.std(intrinsic_rews)
                            intrinsic_rews *= self.motivation.motivation_weight
                            self.agent.buffer['rewards'] = list(
                                intrinsic_rews + np.asarray(self.agent.buffer['rewards']))

                        if self.reward_model is not None:
                            # Compute intrinsic rewards
                            intrinsic_rews = self.reward_model.eval(self.agent.buffer['states'],
                                                                    self.agent.buffer['states_n'],
                                                                    self.agent.buffer['actions'])

                            # Normalize rewards
                            intrinsic_rews -= np.mean(intrinsic_rews)
                            intrinsic_rews /= np.std(intrinsic_rews)
                            intrinsic_rews *= self.reward_model.reward_model_weight
                            self.agent.buffer['rewards'] = list(
                                intrinsic_rews + np.asarray(self.agent.buffer['rewards']))

                        # Train the agent
                        self.agent.update()

                    # If frequency episodes are passed, update the policy
                    if not self.evaluate and self.frequency_mode == 'timesteps' and \
                            self.total_step > 0 and self.total_step % self.motivation_frequency == 0:


                        # If we use intrinsic motivation, update also intrinsic motivation
                        if self.motivation is not None:
                            self.update_motivation()

                    # If frequency episodes are passed, update the policy
                    if not self.evaluate and self.frequency_mode == 'timesteps' and \
                            self.total_step > 0 and self.total_step % self.reward_frequency == 0:

                        # If we use intrinsic motivation, update also intrinsic motivation
                        if self.reward_model is not None and not self.fixed_reward_model:
                            self.update_reward_model()

                # Logging information
                if self.ep_before_logging > 0 and self.ep_before_logging >= self.logging:
                    print('Mean of {} episode reward after {} episodes: {}'.
                          format(self.logging, self.ep, np.mean(self.history['episode_rewards'][-self.logging:])))

                    if self.reward_model is not None:
                        print('Mean of {} environment episode reward after {} episodes: {}'.
                                format(self.logging, self.ep, np.mean(self.history['env_rewards'][-self.logging:])))

                    print('The agent made a total of {} steps'.format(self.total_step * self.num_envs))

                    if self.callback_function is not None:
                        self.callback_function(self.agent, self.env, self)

                    self.timer(start_time, time.time())
                    self.ep_before_logging = 0

                # If frequency episodes are passed, update the policy
                if not self.evaluate and self.frequency_mode == 'episodes' and \
                        self.ep_before_motivation_training > 0 and self.ep_before_motivation_training >= self.motivation_frequency:

                    # If we use intrinsic motivation, update also intrinsic motivation
                    if self.motivation is not None:
                        self.update_motivation()
                    
                    self.ep_before_motivation_training = 0

                if not self.evaluate and self.frequency_mode == 'episodes' and \
                        self.ep_before_reward_training > 0 and self.ep_before_reward_training >= self.reward_frequency:

                    # If we use intrinsic motivation, update also intrinsic motivation
                    if self.reward_model is not None and not self.fixed_reward_model:
                        self.update_reward_model()
                    
                    self.ep_before_reward_training = 0

                # Save model and statistics
                if not self.evaluate and self.ep_before_saving > 0 and self.ep_before_saving >= self.save_frequency:
                    self.save_model(self.history, self.agent.model_name, self.curriculum, self.agent)
                    self.ep_before_saving = True

                # If frequency episodes are passed, update the policy
                if not self.evaluate and self.frequency_mode == 'episodes' and \
                        self.ep_before_training > 0 and self.ep_before_training >= self.frequency:

                    if self.random_actions is not None:
                        if self.total_step <= self.random_actions:
                            self.motivation.clear_buffer()
                            continue

                    if self.motivation is not None:
                        # Normalize observation of the motivation buffer
                        # self.motivation.normalize_buffer()
                        # Compute intrinsic rewards
                        intrinsic_rews = self.motivation.eval(self.agent.buffer['states_n'])

                        # Normalize rewards
                        # intrinsic_rews -= self.motivation.r_norm.mean
                        # intrinsic_rews /= self.motivation.r_norm.std
                        intrinsic_rews -= np.mean(intrinsic_rews)
                        intrinsic_rews /= np.std(intrinsic_rews)
                        intrinsic_rews *= self.motivation.motivation_weight
                        if self.alternate_frequency > 0:
                            if self.alternate_turn == 0:
                                self.agent.buffer['rewards'] = list(intrinsic_rews)
                        else:
                            self.agent.buffer['rewards'] = list(intrinsic_rews)

                    if self.reward_model is not None:

                        # Compute intrinsic rewards
                        intrinsic_rews = self.reward_model.eval(self.agent.buffer['states'], self.agent.buffer['states_n'],
                                                                self.agent.buffer['actions'])

                        # Normalize rewards
                        # intrinsic_rews -= self.reward_model.r_norm.mean
                        # intrinsic_rews /= self.reward_model.r_norm.std

                        #intrinsic_rews = (intrinsic_rews - np.min(intrinsic_rews)) / (np.max(intrinsic_rews) - np.min(intrinsic_rews))
                        intrinsic_rews -= np.mean(intrinsic_rews)
                        intrinsic_rews /= np.std(intrinsic_rews)
                        if self.last_episode_for_decaying > 0:
                            intrinsic_rews *= (1 - self.motivation.motivation_weight)
                        else:
                            intrinsic_rews *= self.reward_model.reward_model_weight

                        if self.alternate_frequency > 0:
                            if self.alternate_turn == 1:
                                self.agent.buffer['rewards'] = list(intrinsic_rews + np.asarray(self.agent.buffer['rewards']))
                        else:
                            self.agent.buffer['rewards'] = list(intrinsic_rews + np.asarray(self.agent.buffer['rewards']))

                    self.agent.update()
                    self.ep_before_training = 0
                    # For alternating between motivation and imitation learning
                    if self.alternate_frequency > 0:
                        self.alternate_count += 1
                        if self.alternate_count % self.alternate_frequency == 0:
                            self.alternate_turn = (self.alternate_turn + 1) % 2

                    # Decaying the motivation weight
                    if self.last_episode_for_decaying > 0:
                        if self.ep < self.last_episode_for_decaying:
                            self.motivation.motivation_weight -= ((0.8 - self.min_motivation_weight) / self.last_episode_for_decaying)
                    
                    # To avoid being off-policy, we can break here
                    break




    def save_model(self, history, model_name, curriculum, agent):

        # Save statistics as json
        json_str = json.dumps(history, cls=NumpyEncoder)
        f = open("arrays/{}.json".format(model_name), "w")
        f.write(json_str)
        f.close()

        # Save curriculum as json
        json_str = json.dumps(curriculum, cls=NumpyEncoder)
        f = open("arrays/{}_curriculum.json".format(model_name), "w")
        f.write(json_str)
        f.close()

        # Save the tf model
        agent.save_model(name=model_name, folder='saved')

        # If we use intrinsic motivation, save the motivation model
        if self.motivation is not None:
            self.motivation.save_model(name=model_name, folder='saved')

        # If we use IRL, save the reward model
        if self.reward_model is not None and not self.fixed_reward_model:
            self.reward_model.save_model('{}_{}'.format(model_name, self.ep))

        print('Model saved with name: {}'.format(model_name))

    def load_model(self, model_name, agent):
        agent.load_model(name=model_name, folder='saved')

        # Load intrinsic motivation for testing
        if self.motivation is not None:
            self.motivation.load_model(name=model_name, folder='saved')

        # # Load reward motivation for testing
        # if self.reward_model is not None:
        #     self.reward_model.load_model(name=model_name)
        
        try:
            with open("arrays/{}.json".format(model_name)) as f:
                history = json.load(f)
        except Exception as e:
            print("There is no history for this model, using a new one")
            return None
        
        try:
            with open("arrays/{}_curriculum.json".format(model_name)) as f:
                curriculum = json.load(f)
            self.current_curriculum_step = curriculum['current_step']
        except Exception as e:
            print("There is no curriculum for this model, using a new one")

        if curriculum is not None:
            self.current_curriculum_step = curriculum['current_step']

        return history

    # Update curriculum for DeepCrawl
    def set_curriculum(self, curriculum, history, mode='steps'):

        total_timesteps = np.sum(history['episode_timesteps'])
        total_episodes = len(history['episode_timesteps'])

        if curriculum == None:
            return None

        if mode == 'episodes':
            lessons = np.cumsum(curriculum['thresholds'])
            curriculum_step = 0

            for (index, l) in enumerate(lessons):

                if total_episodes > l:
                    curriculum_step = index + 1

        if mode == 'steps':
            lessons = np.cumsum(curriculum['thresholds'])

            curriculum_step = 0

            for (index, l) in enumerate(lessons):

                if total_timesteps > l:
                    curriculum_step = index + 1
        
        if mode == 'success':
            if self.current_curriculum_step >= len(curriculum["thresholds"]) :
                curriculum_step = self.current_curriculum_step
            else:
                current_lesson = curriculum["thresholds"][self.current_curriculum_step]
                current_lesson_threshold = curriculum["threshold_episodes"][self.current_curriculum_step]
                current_episode = len(history['episode_timesteps'])
                success_rate = np.asarray([info["success_rate"] for info in  self.history["info"]]) == 1

                if current_episode - self.current_curriculum_change >= current_lesson_threshold:
                    current_success_rate = np.mean(success_rate[-current_lesson_threshold:])
                    if current_success_rate > current_lesson:
                        self.current_curriculum_step += 1
                        self.current_curriculum_change = current_episode
                
                curriculum_step = self.current_curriculum_step

        parameters = curriculum['parameters']
        config = {}

        for (par, value) in parameters.items():
            config[par] = value[curriculum_step]

        # If Adversarial play
        if self.adversarial_play:
            if curriculum_step > self.current_curriculum_step:
                # Save the current version of the main agent
                self.agent.save_model(name=self.agent.model_name + '_' + str(curriculum_step), folder='saved/adversarial')
                # Load the weights of the current version of the main agent to the double agent
                self.double_agent.load_model(name=self.agent.model_name + '_' + str(curriculum_step), folder='saved/adversarial')

        self.current_curriculum_step = curriculum_step
        self.curriculum["current_step"] = curriculum_step

        # If parameters of the algorithm are in the curriculum
        if "algo_parameters" in curriculum:
            algo_parameters = curriculum["algo_parameters"]
            for (par, value) in algo_parameters.items():
                setattr(self.agent, par, value[self.current_curriculum_step])
        
        return config

    # Update intrinsic motivation
    # Update its statistics AND train the model. We print also the model loss
    def update_motivation(self):
        loss = self.motivation.update()
        #print('Mean motivation loss = {}'.format(loss))

    # Update reward model
    # Update its statistics AND train the model. We print also the model loss
    def update_reward_model(self):
        loss, _ = self.reward_model.update()
        #print('Mean reward loss = {}'.format(loss))

    # For IRL, get initial experience from environment, the agent act in the env without update itself
    def get_experience(self, env, num_discriminator_exp=None, verbose=False, random=False):

        if num_discriminator_exp == None:
            num_discriminator_exp = self.frequency

        # For policy update number
        for ep in range(num_discriminator_exp):
            states = []
            state = env.reset()
            step = 0
            # While the episode si not finished
            reward = 0
            while True:
                step += 1
                if random:
                    num_actions = self.agent.action_size
                    action = np.random.randint(0, num_actions)
                else:
                    action, _, c_probs = self.agent.eval([state])
                state_n, terminal, step_reward = env.execute(actions=action)


                self.reward_model.add_to_policy_buffer([state], [state_n], [action])

                state = state_n
                reward += step_reward
                if terminal or step >= env._max_episode_timesteps:
                    break

            if verbose:
                print("Reward at the end of episode " + str(ep + 1) + ": " + str(reward))

    # Method for count time after each episode
    def timer(self, start, end):
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Time passed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
