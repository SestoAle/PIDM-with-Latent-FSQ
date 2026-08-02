from agents.ppo_agent import PPOAgent
from agents.sac_agent import SACAgent
from runners.parallel_runner import Runner as PRunner
from runners.runner import Runner as SRunner
from architectures.mlp_based_policy import PolicyEmbedding, CriticEmbedding
from envs.gym_env import GymEnv

import argparse
import torch
import threading
import signal
import pickle
import sys
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse arguments for training
parser = argparse.ArgumentParser()
parser.add_argument('-mn', '--model-name', help="The name of the policy", default='test')
parser.add_argument('-al', '--algorithm_name', help="We can choose between two algorithms, ppo and sac", default='sac', choices=["ppo", "sac"])
parser.add_argument('-sf', '--save-frequency', help="How mane episodes after save the model", default=1000)
parser.add_argument('-lg', '--logging', help="How many episodes after logging statistics", default=100)
parser.add_argument('-mt', '--max-timesteps', help="Max timestep per episode", default=1000)
parser.add_argument('-pl', '--parallel', help="How many environments to simulate in parallel. Default is 1", type=int, default=1)
parser.add_argument('-fs', '--fixed-seed', help="If we want to use a fixed seed", default=None, type=int)

# In case we want to use SMP
parser.add_argument('-dp', '--with-diffusion-prior', help="Whether to use the diffusion prior as reward model", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('-dn', '--diffusion-prior-name', help="The name of the pre-trained diffusion prior that we want to use as reward model.", default=None)
parser.add_argument('-hz', '--horizon', help="The horizon of the model", type=int, default=10)
parser.add_argument('-hs', '--hidden-size', help="The hidden size of the diffusion prior", type=int, default=512)
parser.add_argument('-ds', '--denoising-steps', help="The number of the denoising steps", type=int, default=50)
parser.add_argument('-ts', '--denoising-timesteps', help="The K set of the denoising steps that we use to compute the ensemble.", default=[22, 15, 8])

# For evaluation and eventually collecting data
parser.add_argument('-ev', '--evaluate', help="Whether to train or evaluate the agent", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('-ns', '--num-samples-to-save', help='The number of transitions we want to save to train the world model', default=15000, type=int)
parser.add_argument('-st', '--save-trajectories', help='If we want to save the trajectories to train the world mdoel', action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()


eps = 1e-12

def callback(agent, envs, runner):
    if args.save_trajectories:
        num_states = np.sum([len(traj["states"]) for traj in envs.trajectories])
        print(f"{num_states}/{args.num_samples_to_save}")
        if np.sum([len(traj["states"]) for traj in envs.trajectories]) > args.num_samples_to_save:
            all_trajectories = []
            all_trajectories.extend(envs.trajectories)

            print("Saving trajectories...")
            savename = f"datasets/dataset{f'_fixed_seed_{args.fixed_seed}' if args.fixed_seed is not None else ''}.pkl" 
            a_file = open(savename, "wb")
            pickle.dump(all_trajectories, a_file)
            a_file.close()
            print("Trajectories saved")
            sys.exit()

    return

def init_env(envs_list, l_index, max_timesteps):
    env = GymEnv(
        max_episode_timesteps=max_timesteps,
        save_trajectories=args.save_trajectories,
    )
    envs_list[l_index] = env
    return

def signal_handler(signum, frame):
    signal.signal(signal.SIGINT, original_sigint)
    print("KeyboardInterrupt: Killing the training process. Please wait...")
    should_stop.set()
    signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    should_stop = threading.Event()
    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal_handler)
    # RL arguments
    model_name = args.model_name
    save_frequency = int(args.save_frequency) 
    logging = int(args.logging)
    max_episode_timestep = int(args.max_timesteps)

    curriculum = None

    # Total episode of training
    total_episode = 10e6
    # Name of the algorithm
    algorithm_name = args.algorithm_name
    # Units of training (episodes or timesteps)
    frequency_mode = 'timesteps' 
    # Frequency of training (in episode or timesteps)
    frequency = 50 if frequency_mode == "episodes" else 1024
    frequency = frequency if not args.evaluate else 1e10 
    # Memory of the agent (in episodes or timesteps)
    # For this project, the main algorithm is gonna be SAC so the memory
    # becomes an hyperparameter
    memory = 1e6 
    # Learning rate
    lr = 3e-4
    # Random initial action
    random_actions = None
    # Action type of the policy
    action_type = "continuous"
    action_masking = False

    evaluate = args.evaluate 

    # Open the environment with all the desired flags
    # If parallel, create more environments
    envs = [None] * args.parallel
    threads = []
    for i in range(args.parallel):
        task_index = i
        t = threading.Thread(target=init_env, args=(envs, task_index, max_episode_timestep))
        t.start()
        threads.append(t)

    for thr in threads:
        thr.join()
    
    # Get the state and action specs
    state_size = 8
    action_size = 2


    # Create agent
    # The policy embedding and the critic embedding for the PPO agent are defined in the architecture file
    # You can change those architectures, the agent class will manage the action layers and the value layers
    if algorithm_name == "ppo":
        agent = PPOAgent(state_dim=state_size, policy_embedding=PolicyEmbedding, 
                 critic_embedding=CriticEmbedding, action_type=action_type, action_size=action_size,
                 model_name=model_name, p_lr=lr, v_batch_size=4096, v_num_itr=50, memory=memory, batch_size=4096,
                 c2=0.01, discount=0.99, v_lr=lr, frequency_mode=frequency_mode, distribution='beta', lmbda=1.0,
                 action_min_value=-1, action_max_value=1, p_num_itr=50, device=device, action_masking=action_masking)
    elif algorithm_name == "sac":
        agent = SACAgent(state_dim=state_size, policy_embedding=PolicyEmbedding, critic_embedding=CriticEmbedding,
                         discount=0.99, p_lr=lr, v_lr=lr, frequency_mode=frequency_mode, memory=memory,
                         policy_freq=1, alpha=0.2, tau=0.005, batch_size=256, num_itr=256, action_size=action_size,
                         max_action_value=1, min_action_value=-1, device=device, name=model_name) 
    else:
        print(f"No algorithm with name {algorithm_name}")
    
    # Create runner
    # This class manages the evaluation of the policy and the collection of experience in a parallel setting
    # (not vectorized)
    if args.parallel < 2:
        runner = SRunner(should_stop, agent=agent, frequency=frequency, env=envs[0], save_frequency=save_frequency,
                        logging=logging, total_episode=total_episode, curriculum=curriculum, demonstrations_name="dems",
                        frequency_mode=frequency_mode, curriculum_mode='episodes', callback_function=callback, 
                        random_actions=random_actions, evaluation=args.evaluate,
                        timesteps_set=args.denoising_timesteps)
    else:
        runner = PRunner(should_stop, agent=agent, frequency=frequency, envs=envs, save_frequency=save_frequency,
                        logging=logging, total_episode=total_episode, curriculum=curriculum,
                        frequency_mode=frequency_mode, curriculum_mode='episodes', random_actions=random_actions,
                        callback_function=callback, evaluation=args.evaluate, 
                        timesteps_set=args.denoising_timesteps)

    runner.run()

