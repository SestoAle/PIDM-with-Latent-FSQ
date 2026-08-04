import argparse
import torch
import pickle
import os
import numpy as np

from world_model.leworldmodel import LeWorldModel
from agents.pidm_agent import PIDMAgent
from architectures.mlp_based_pidm import PolicyEmbedding
from envs.gym_env import GymEnv

from colorama import Fore, Style, init

init(autoreset=True)

#######################################################################################
def create_env(seed, visualize_inference):
    env = GymEnv(
        max_episode_timesteps=400,
        save_trajectories=False,
        visualize_inference=visualize_inference
    )
    return env

#######################################################################################
def create_agent(state_size, action_size, policy_arch, lr):
    agent = PIDMAgent(
        state_size=state_size,
        action_size=action_size,
        policy_arch=policy_arch,
        lr=lr
    )

    return agent

#######################################################################################
def create_world_model(action_size, sequence_length, lr, device, fsq_input_size, fsq_output_size, L, model_name, mlp_encoder=False):
    model = LeWorldModel(
        action_dim=action_size,
        max_seq_length=sequence_length,
        encoder_hidden_dim=fsq_output_size,
        lr = lr,
        device=device,

        # This world model will be feature based
        with_reward_prediction=False,
        with_terminal_prediction=False,
        feature_base=True,
        fsq_input_size=fsq_input_size,
        fsq_output_size=fsq_output_size,
        L=L,
        fsq_encoder=not mlp_encoder
    )

    model.load_model(model_name)

    return model

#######################################################################################
def load_dataset(dataset_path, model):

    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    new_dataset = {
        "states": np.asarray(dataset["states"]),
        "actions": np.asarray(dataset["actions"]),
    }

    print(f"This dataset has a total of {new_dataset["states"].shape[0]} transitions")

    model.set_dataset(new_dataset)

#######################################################################################
def train(epochs_number, batch_size, model, model_name):
    for epoch in range(epochs_number):
        losses = model.train_epoch(batch_size=batch_size)
        print(f"At epoch {epoch}:")
        for key in losses.keys():
            print(f"    -{key}: {losses[key]}")
        
        print("Saving the model now...")
        model.save_model(name=model_name)

#######################################################################################
def check_if_model_exists(model_name, model):
    loaded = False
    evaluate = False
    if os.path.exists(rf"saved/{model_name}"):
        answer = None
        while answer != 'y' and answer != 'n':
            answer = input("A pre-trained model exists with the same name. Do you want to load it? [y/n] ")
        
        if answer == "y":
            model.load_model(model_name)
            loaded = True

        if loaded:
            answer = None
            while answer != 'y' and answer != 'n':
                answer = input("You loaded a model. Do you want to evaluate it? [y/n] ")
            
            if answer == "y":
                evaluate = True
    
    return loaded, evaluate

#######################################################################################
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse arguments for training
    parser = argparse.ArgumentParser()


    # Arguments for both the WM and the Agent
    parser.add_argument('-mn', '--model-name', help="The name with which we want to save the world model", default="lelunar")
    parser.add_argument('-dn', '--dataset-name', help="The name of the precollected dataset with which we train the world model", default="datasets/dataset.pkl")
    parser.add_argument('-as', '--action-size', help="The action dimension of the env", default=2, type=int)
    parser.add_argument('-is', '--input-size', help="The state dimension of the env", default=8, type=int)
    parser.add_argument('-ed', '--encoder-dim', help="The dimension of the encoder, in this case an FSQ encoder", default=64, type=int)
    parser.add_argument('-ld', '--levels-dim', help="The dimension of levels for FSQ", default=8, type=int)
    parser.add_argument('-sl', '--sequence-length', help="The max sequence length of the world model", default=4, type=int)
    parser.add_argument('-bs', '--batch-size', help="The batch size during training", default=1024, type=int)
    parser.add_argument('-en', '--epochs-number', help="The number of epochs during training", default=5, type=int)
    parser.add_argument('-lr', '--learning-rate', help="The learning rate used during training", default=1e-3, type=float)

    args = parser.parse_args()

    print("")
    print("####")
    print(Fore.CYAN + "Creating the agent.." + Style.RESET_ALL)
    print("...")
    agent = create_agent(
        state_size=args.encoder_dim,
        action_size=args.action_size,
        policy_arch=PolicyEmbedding,
        lr=args.learning_rate
    )
    print(Fore.CYAN + "For this agent, we need a world model. Loading it now" + Style.RESET_ALL)
    model = create_model(args.action_size, args.sequence_length, args.learning_rate, fsq_input_size=args.input_size, fsq_output_size=args.encoder_dim, L=args.levels_dim, device=device)
    loaded, evaluate = check_if_model_exists(args.model_name, model)
    print(Fore.GREEN + "Model created!" + Style.RESET_ALL)
    print("####")

    print(Fore.CYAN + "Loading dataset..." + Style.RESET_ALL)
    print("...")
    load_dataset(args.dataset_name, model)
    print(Fore.GREEN + "Dataset loaded!" + Style.RESET_ALL)
    print("####")

    if not evaluate:
        print(Fore.RED + "Start training!" + Style.RESET_ALL)
        train(args.epochs_number, args.batch_size, model, args.model_name)
        print(Fore.GREEN + "Model trained!" + Style.RESET_ALL)
        print("####")

    else:
        print(Fore.GREEN + "Start evaluate!" + Style.RESET_ALL)
        env = create_env(seed=args.fixed_seed, visualize_inference=args.visualize_inference)
        model.eval()


        with torch.inference_mode():

            evaluate_world_model(
                world_model=model,
                horizon=args.sequence_length,
                env=env
            )
            import ipdb; ipdb.set_trace()