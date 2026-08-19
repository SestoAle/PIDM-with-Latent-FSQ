import argparse
import torch
import pickle
import os
import numpy as np
from collections import deque

from world_model.leworldmodel import LeWorldModel
from colorama import Fore, Style, init
from torch.nn import functional as F
from envs.gym_env import GymEnv

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
def create_model(action_size, sequence_length, prediction_horizon, lr, device, fsq_input_size, fsq_output_size, L, mlp_encoder=False):
    model = LeWorldModel(
        action_dim=action_size,
        max_seq_length=sequence_length,
        prediction_horizon=prediction_horizon,
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

    return model

#######################################################################################
def evaluate_world_model(world_model, env, horizon, policy=None, num_episodes=100):
    k = world_model.prediction_horizon
    with torch.no_grad():
        for e in range(num_episodes):
            state = env.reset()
            done = False
            accuracies = []
            pending_predictions = deque()

            running_latent = torch.zeros(horizon, world_model.encoder_hidden_dim).to(device)
            running_action = torch.zeros(horizon, env.action_dim).to(device)

            while not done:

                # If we have a policy, we are gonna run the policy.
                # Otherwise, a random action
                if policy is not None:
                    action = policy(state)[0]
                else:
                    action = np.random.randn(env.action_dim)

                # Run the real environment 
                next_state, reward, done, _ = env.step(action)
                # Run the world model to estimate the next state
                # Encode the state
                state = torch.from_numpy(state).to(device)
                action = torch.from_numpy(action).to(device)
                # Update the running input and action
                running_latent = torch.roll(running_latent, -1, 0)
                running_action = torch.roll(running_action, -1, 0)
                encoded_state = world_model.encoder_fwd(state.view(1, 1, -1))
                running_latent[-1] = encoded_state
                running_action[-1] = action
                predicted_next_state = world_model.predictor_fwd([running_latent.view(1, horizon, -1), running_action.view(1, horizon, -1), None])[0][-1, -1]
                pending_predictions.append(predicted_next_state)

                # Encode the real next state
                next_state = torch.from_numpy(next_state).to(device)
                encoded_next_state = world_model.encoder_fwd(next_state.view(1, 1, -1))[-1, -1]

                # Check how different the predicted and the real are
                if len(pending_predictions) >= k:
                    due_prediction = pending_predictions.popleft()
                    accuracy = torch.sum(torch.stack([a == b for a, b in zip(due_prediction, encoded_next_state)])) / world_model.encoder_hidden_dim
                    accuracies.append(accuracy.cpu().numpy())
                state = next_state.cpu().numpy()

            print(f"Average {k}-step accuracy for episode {e}: {np.mean(accuracies)}")

#######################################################################################
def load_dataset(dataset_path, model):

    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    if isinstance(dataset, list):
        new_dataset = dict(states=[], actions=[], rewards=[], terminals=[])
        for trajectory in dataset:
            for key in new_dataset:
                new_dataset[key].extend(trajectory[key])
        new_dataset = {key: np.asarray(value) for key, value in new_dataset.items()}
    else:
        new_dataset = {
            "states": np.asarray(dataset["states"]),
            "actions": np.asarray(dataset["actions"]),
            "rewards": np.asarray(dataset["rewards"]),
            "terminals": np.asarray(dataset["terminals"])
        }

    # Standardize the reward
    new_dataset["rewards"] = (new_dataset["rewards"] - np.mean(new_dataset["rewards"])) / (np.std(new_dataset["rewards"]) + 1e-6)

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
    parser.add_argument('-mn', '--model-name', help="The name with which we want to save the world model", default="lelunar")
    parser.add_argument('-dn', '--dataset-name', help="The name of the precollected dataset with which we train the world model", default="datasets/dataset.pkl")
    parser.add_argument('-as', '--action-size', help="The action dimension of the env", default=2, type=int)
    parser.add_argument('-is', '--input-size', help="The state dimension of the env", default=8, type=int)
    parser.add_argument('-ed', '--encoder-dim', help="The dimension of the encoder, in this case an FSQ encoder", default=16, type=int)
    parser.add_argument('-ld', '--levels-dim', help="The dimension of levels for FSQ", default=12, type=int)
    parser.add_argument('-fs', '--fixed-seed', help="If we want to use a fixed seed", default=423, type=int)
    parser.add_argument('-sl', '--sequence-length', help="The max sequence length of the world model", default=8, type=int)
    parser.add_argument('-k', '--prediction-horizon', help="How many steps ahead the world model predicts", default=1, type=int)
    parser.add_argument('-bs', '--batch-size', help="The batch size during training", default=1024, type=int)
    parser.add_argument('-en', '--epochs-number', help="The number of epochs during training", default=5, type=int)
    parser.add_argument('-lr', '--learning-rate', help="The learning rate used during training", default=1e-4, type=float)
    parser.add_argument('-vi', '--visualize-inference', help="If we want to see the agent in the environment", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-wf', '--without-fsq', help="If we want to run an ablation without fsq", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()

    print("")
    print("####")
    print(Fore.CYAN + "Creating the model.." + Style.RESET_ALL)
    print("...")
    model = create_model(
        action_size=args.action_size, 
        sequence_length=args.sequence_length, 
        prediction_horizon=args.prediction_horizon,
        lr=args.learning_rate, 
        fsq_input_size=args.input_size, 
        fsq_output_size=args.encoder_dim, 
        L=args.levels_dim, 
        device=device,
        mlp_encoder=args.without_fsq
        )

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
