import argparse
import torch
import pickle
import os
import numpy as np

from world_model.leworldmodel import LeWorldModel
from agents.pidm_agent import PIDMAgent
from architectures.mlp_based_pidm import PolicyEmbedding
from architectures.mlp_based_policy import PolicyEmbedding as SacPolicy, CriticEmbedding as SacCritic
from envs.gym_env import GymEnv
from agents.sac_agent import SACAgent
from sklearn.metrics.pairwise import pairwise_distances

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
def evaluate_agent(env, agent, world_model, horizon, eval_episodes=100, only_real_states=False):

    episode_rewards = []
    with torch.no_grad():
        for episode in range(eval_episodes):
            if world_model is not None:
                running_latent = torch.zeros(horizon, world_model.encoder_hidden_dim).to(device)

            state = env.reset()
            done = False
            episode_reward = 0
            mean_accuracy = 0
            accuracies = []
            step = 0

            while not done:
                # Run the world model to estimate the next state
                # Encode the state
                state = torch.from_numpy(state).to(device)
                step += 1

                if world_model is not None:
                    # Update the running input and action
                    running_latent = torch.roll(running_latent, -1, 0)
                    encoded_state = world_model.encoder_fwd(state.view(1, 1, -1)).squeeze(0)
                    running_latent[-1] = encoded_state
                    predicted_next_state = world_model.predictor_fwd([running_latent.view(1, horizon, -1), None, None])[0][-1, -1].unsqueeze(dim=0)
                    og_predicted_next_state = predicted_next_state.clone()

                    # Run the agent
                    if world_model.fsq_encoder:
                        encoded_state = world_model.encoder.shift_and_scale(encoded_state)
                        predicted_next_state = world_model.encoder.shift_and_scale(predicted_next_state)
                if only_real_states:
                    action = agent([state.view(1, -1), None])
                else:
                    action = agent([encoded_state, predicted_next_state])
                action = action.detach().cpu().numpy()[0]

                
                next_state, reward, done, _ = env.step(action)

                # Check how different the predicted and the real are
                encoded_next_state = world_model.encoder_fwd(torch.from_numpy(next_state).to(device).view(1, 1, -1)).squeeze(0)
                accuracy = (torch.sum(torch.stack([a == b for a, b in zip(og_predicted_next_state, encoded_next_state)])) / world_model.encoder_hidden_dim).item()
                mean_accuracy += accuracy
                accuracies.append(accuracy)

                state = next_state
                episode_reward += reward

            print(f"Mean accuracy for the world model: at episode {episode}: {mean_accuracy/step}")
            print(f"Episode reward at episode {episode}: {episode_reward}")
            if episode_reward < 0:
                import ipdb; ipdb.set_trace()
            episode_rewards.append(episode_reward)

    print(f"Final performance over {eval_episodes} episodes: {np.mean(episode_rewards)}")

#######################################################################################
def create_agent(state_size, action_size, policy_arch, lr, device, only_real_states=False):
    agent = PIDMAgent(
        state_size=state_size,
        action_size=action_size,
        policy_arch=policy_arch,
        lr=lr,
        device=device,
        only_real_states=only_real_states
    )

    return agent

#######################################################################################
def get_next_state_through_search(state, dataset):
    # Given a state, get the closest next state in the dataset.
    # Similar to what the microsoft paper does

    # The dataset is a list of transitions
    all_states = np.asarray(dataset["states"])
    feature_size = all_states.shape[1]

    state = np.asarray(state).reshape(1, feature_size)

    # Get the distance
    distances = pairwise_distances(state, all_states)
    min_distance_index = np.argmin(distances)
    closest_next_state = dataset["next_states"][min_distance_index]

    return closest_next_state

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
def load_and_set_dataset(
        dataset_path, 
        world_model, 
        agent, 
        world_model_epochs, 
        world_model_batch_size,
        horizon,
        only_real_states=False,
        ):

    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    # Dataset of demonstrations work differently than replay buffer
    # the dataset contains list of episodes
    # We need to flatten it

    new_dataset = dict(states=[], next_states=[], actions=[], rewards=[], terminals=[])
    for trajectory in dataset:
        new_dataset["states"].extend(trajectory["states"])
        new_dataset["next_states"].extend(trajectory["next_states"])
        new_dataset["actions"].extend(trajectory["actions"])
        new_dataset["rewards"].extend(trajectory["rewards"])
        new_dataset["terminals"].extend(trajectory["terminals"])

    new_dataset["states"] = np.asarray(new_dataset["states"])
    new_dataset["next_states"] = np.asarray(new_dataset["next_states"])
    new_dataset["actions"] = np.asarray(new_dataset["actions"])
    new_dataset["rewards"] = np.asarray(new_dataset["rewards"])
    new_dataset["terminals"] = np.asarray(new_dataset["terminals"])

    print(f"This dataset has a total of {new_dataset["states"].shape[0]} transitions")

    import ipdb; ipdb.set_trace()
    get_next_state_through_search(new_dataset["states"][234], new_dataset)

    if world_model is not None:
        world_model.set_dataset(new_dataset)

        world_model.train()
        # We need to fine-tune the world model for what we have now 
        for epoch in range(world_model_epochs):
            losses = world_model.train_epoch(world_model_batch_size)
            print(
              f"Epoch {epoch}: "
              f"total={losses['total_loss'].item():.4f}, "
            )
        world_model.eval()

        with torch.no_grad():
            # Now, we need to encode the states and next states and set them to the agent
            encoded_states = world_model.encoder_fwd(torch.from_numpy(new_dataset["states"]).to(device)).squeeze()
            # TODO: do we want to train with predicted or ground truth states? I would say ground
            # truth for now. Probably a mix of those would be nice
            # TODO: here we need to build the state sequence, how?
            # TODO: Do we do overlapping sequences?
            # TODO: Apparently, the state do not change that much in a sequence of 4 states

            state_sequences = torch.zeros(encoded_states.shape[0]-horizon, horizon, encoded_states.shape[-1]).to(device)
            for i in range(0, len(encoded_states) - horizon):
                state_sequences[i] = encoded_states[i:i+horizon]

            predicted_next_states = world_model.predictor_fwd([state_sequences, None, None])[0][:, -1, :]
            if world_model.fsq_encoder:
                encoded_states = world_model.encoder.shift_and_scale(encoded_states)
                predicted_next_states = world_model.encoder.shift_and_scale(predicted_next_states)
            encoded_next_states = predicted_next_states

            encoded_states = encoded_states[horizon-1:-1]
        actions = torch.from_numpy(new_dataset["actions"]).to(device)[horizon-1:-1]

    # encoded_next_states = world_model.encoder_fwd(torch.from_numpy(new_dataset["next_states"]).to(device)).squeeze()
    # encoded_next_states = world_model.encoder.shift_and_scale(encoded_next_states)
    # actions = torch.from_numpy(new_dataset["actions"]).to(device)

    # Set the dataset of the agent

    # ABLATION: Train with only real states
    if only_real_states:
        agent.set_dataset(
            states=torch.from_numpy(new_dataset["states"]).to(device),
            actions=torch.from_numpy(new_dataset["actions"]).to(device),
            next_states=torch.from_numpy(new_dataset["next_states"]).to(device)
        )
    else:
        agent.set_dataset(
            states=encoded_states,
            actions=actions,
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
    parser.add_argument('-ed', '--encoder-dim', help="The dimension of the encoder, in this case an FSQ encoder", default=16, type=int)
    parser.add_argument('-ld', '--levels-dim', help="The dimension of levels for FSQ", default=14, type=int)
    parser.add_argument('-sl', '--sequence-length', help="The max sequence length of the world model", default=8, type=int)
    parser.add_argument('-bs', '--batch-size', help="The batch size during training", default=32, type=int)
    parser.add_argument('-en', '--epochs-number', help="The number of epochs during training", default=100, type=int)
    parser.add_argument('-lr', '--learning-rate', help="The learning rate used during training", default=1e-3, type=float)
    parser.add_argument('-wb', '--world-model-batch-size', help="The batch size during training of the world model", default=32, type=int)
    parser.add_argument('-wr', '--world-model-learning-rate', help="The learning rate used during training for the world model", default=5e-5, type=float)
    parser.add_argument('-ww', '--world-model-epochs-number', help="Number of fine-tuning epochs for the world model", default=5, type=int)

    # Ablations
    parser.add_argument('-rs', '--only-real-states', help="Wether to use only real states as input to the IL agent", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-wf', '--without-fsq', help="If we want to run an ablation without fsq", action=argparse.BooleanOptionalAction, default=False)


    args = parser.parse_args()

    print("")
    print("####")
    print(Fore.CYAN + "Creating the agent.." + Style.RESET_ALL)
    print("...")
    agent = create_agent(
        state_size= args.input_size if args.only_real_states else args.encoder_dim * 2, # Because we have state and next state
        action_size=args.action_size,
        policy_arch=PolicyEmbedding,
        lr=args.learning_rate,
        device=device,
        only_real_states=args.only_real_states
    )
    loaded, evaluate = check_if_model_exists(args.model_name, agent)
    print(Fore.CYAN + "For this agent, we need a world model. Loading it now" + Style.RESET_ALL)
    world_model = None
    if not args.only_real_states:
        world_model = create_world_model(
            action_size=args.action_size, 
            sequence_length=args.sequence_length, 
            lr=args.world_model_learning_rate, 
            model_name=args.world_model_name,
            fsq_input_size=args.input_size, 
            fsq_output_size=args.encoder_dim, 
            L=args.levels_dim, 
            mlp_encoder=args.without_fsq,
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
            world_model_epochs=args.world_model_epochs_number,
            horizon=args.sequence_length,
            only_real_states=args.only_real_states
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
        eval_episodes=100,
        only_real_states=args.only_real_states
    )