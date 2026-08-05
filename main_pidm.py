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
def create_env():
    env = GymEnv(
        max_episode_timesteps=400,
        save_trajectories=False,
    )
    return env

#######################################################################################
def evaluate_agent(env, agent, world_model, horizon, eval_episodes=100):
    with torch.no_grad():
        for episode in range(eval_episodes):
            running_latent = torch.zeros(horizon, world_model.encoder_hidden_dim).to(device)

            state = env.reset()
            done = False
            episode_reward = 0

            while not done:
                # Run the world model to estimate the next state
                # Encode the state
                state = torch.from_numpy(state).to(device)
                # Update the running input and action
                running_latent = torch.roll(running_latent, -1, 0)
                encoded_state = world_model.encoder_fwd(state.view(1, 1, -1)).squeeze(0)
                running_latent[-1] = encoded_state
                predicted_next_state = world_model.predictor_fwd([running_latent.view(1, horizon, -1), None, None])[0][-1, -1].unsqueeze(dim=0)

                # Run the agent
                action = agent([encoded_state, predicted_next_state])
                action = action.detach().cpu().numpy()[0]

                next_state, reward, done, _ = env.step(action)
                state = next_state
                episode_reward += reward

            print(f"Episode reward at episode {episode}: {episode_reward}")

#######################################################################################
def create_agent(state_size, action_size, policy_arch, lr, device):
    agent = PIDMAgent(
        state_size=state_size,
        action_size=action_size,
        policy_arch=policy_arch,
        lr=lr,
        device=device
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
        with_action=False,
        feature_base=True,
        fsq_input_size=fsq_input_size,
        fsq_output_size=fsq_output_size,
        L=L,
        fsq_encoder=not mlp_encoder
    )

    model.load_model(model_name)

    return model

#######################################################################################
def load_and_set_dataset(dataset_path, world_model, agent, world_model_epochs, world_model_batch_size):

    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    new_dataset = {
        "states": np.asarray(dataset["states"])[:1000],
        "next_states": np.asarray(dataset["states_n"])[:1000],
        "actions": np.asarray(dataset["actions"])[:1000],
        "rewards": np.asarray(dataset["rewards"])[:1000],
        "terminals": np.asarray(dataset["terminals"])[:1000]
    }

    print(f"This dataset has a total of {new_dataset["states"].shape[0]} transitions")

    world_model.set_dataset(new_dataset)

    # We need to fine-tune the world model for what we have now 
    for epoch in range(world_model_epochs):
        loss = world_model.train_epoch(world_model_batch_size)["total_loss"].detach().cpu().numpy()
        print(f"Loss of the world model at epoch {epoch}: {loss}")

    # Now, we need to encode the states and next states and set them to the agent
    encoded_states = world_model.encoder_fwd(torch.from_numpy(new_dataset["states"]).to(device)).squeeze()
    # TODO: do we want to train with predicted or ground truth states? I would say ground
    # truth for now. Probably a mix of those would be nice
    encoded_next_states = world_model.encoder_fwd(torch.from_numpy(new_dataset["next_states"]).to(device)).squeeze()

    # Set the dataset of the agent
    agent.set_dataset(
        states=encoded_states,
        actions=torch.from_numpy(new_dataset["actions"]).to(device),
        next_states=encoded_next_states
    )

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
    parser.add_argument('-mn', '--model-name', help="The name with which we want to save the agent", default="pidm_lunar")
    parser.add_argument('-wm', '--world-model-name', help="The name of the world model we already trained", required=True)
    parser.add_argument('-dn', '--dataset-name', help="The name of the precollected dataset with which we train the world model", default="datasets/dataset.pkl")
    parser.add_argument('-as', '--action-size', help="The action dimension of the env", default=2, type=int)
    parser.add_argument('-is', '--input-size', help="The state dimension of the env", default=8, type=int)
    parser.add_argument('-ed', '--encoder-dim', help="The dimension of the encoder, in this case an FSQ encoder", default=64, type=int)
    parser.add_argument('-ld', '--levels-dim', help="The dimension of levels for FSQ", default=8, type=int)
    parser.add_argument('-sl', '--sequence-length', help="The max sequence length of the world model", default=4, type=int)
    parser.add_argument('-bs', '--batch-size', help="The batch size during training", default=1024, type=int)
    parser.add_argument('-en', '--epochs-number', help="The number of epochs during training", default=5, type=int)
    parser.add_argument('-lr', '--learning-rate', help="The learning rate used during training", default=1e-3, type=float)
    parser.add_argument('-wb', '--world-model-batch-size', help="The batch size during training of the world model", default=1024, type=int)
    parser.add_argument('-wr', '--world-model-learning-rate', help="The learning rate used during training for the world model", default=1e-3, type=float)
    parser.add_argument('-ww', '--world-model-epochs-number', help="Number of fine-tuning epochs for the world model", default=5, type=int)

    args = parser.parse_args()

    print("")
    print("####")
    print(Fore.CYAN + "Creating the agent.." + Style.RESET_ALL)
    print("...")
    agent = create_agent(
        state_size=args.encoder_dim * 2, # Because we have state and next state
        action_size=args.action_size,
        policy_arch=PolicyEmbedding,
        lr=args.learning_rate,
        device=device
    )
    loaded, evaluate = check_if_model_exists(args.model_name, agent)
    print(Fore.CYAN + "For this agent, we need a world model. Loading it now" + Style.RESET_ALL)
    world_model = create_world_model(
        action_size=args.action_size, 
        sequence_length=args.sequence_length, 
        lr=args.learning_rate, 
        model_name=args.world_model_name,
        fsq_input_size=args.input_size, 
        fsq_output_size=args.encoder_dim, 
        L=args.levels_dim, 
        device=device
    )
    print(Fore.GREEN + "Models created!" + Style.RESET_ALL)
    print("####")

    if not evaluate:
        print(Fore.CYAN + "Loading dataset..." + Style.RESET_ALL)
        print("...")
        load_and_set_dataset(
            dataset_path=args.dataset_name,
            world_model=world_model,
            agent=agent,
            world_model_batch_size=args.world_model_batch_size,
            world_model_epochs=args.world_model_epochs_number
        )

        print(Fore.GREEN + "Dataset loaded!" + Style.RESET_ALL)
        print("####")

    if not evaluate:
        print(Fore.RED + "Start training!" + Style.RESET_ALL)
        train(
            epochs_number=args.epochs_number, 
            batch_size=args.batch_size, 
            model=agent, 
            model_name=args.model_name
        )
        print(Fore.GREEN + "Model trained!" + Style.RESET_ALL)
        print("####")

    print(Fore.GREEN + "Start evaluate!" + Style.RESET_ALL)
    env = create_env()
    agent.eval()

    evaluate_agent(
        env=env,
        agent=agent,
        world_model=world_model,
        horizon=args.sequence_length,
        eval_episodes=100
    )