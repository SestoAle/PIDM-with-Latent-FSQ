import argparse
import torch
import pickle
import os
import numpy as np
from collections import deque
from copy import deepcopy

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
def evaluate_agent(
        env,
        agent,
        world_model,
        horizon,
        multi_horizon_length,
        prediction_horizon=1,
        dataset=None,
        eval_episodes=100,
        only_real_states=False,
        deterministic_wm=True,
        with_reconstruction=False

        ):

    episode_rewards = []
    with torch.no_grad():
        for episode in range(eval_episodes):
            if world_model is not None:
                running_latent = torch.zeros(horizon, world_model.fsq_output_size).to(device)
                pending_predictions = deque()

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

                if dataset is not None:
                    predicted_next_state, _ = get_next_state_through_search(
                        state.cpu().numpy(), 
                        dataset, 
                        prediction_horizon=prediction_horizon
                    )
                    predicted_next_state = torch.from_numpy(predicted_next_state).to(device).view(1, -1)
                    encoded_state = state.view(1, -1)

                if world_model is not None:
                    # Update the running input and action
                    encoded_state = world_model.encoder_fwd(state.view(1, 1, -1)).squeeze(0)
                    if step == 1:
                        running_latent[:] = encoded_state
                    else:
                        running_latent = torch.roll(running_latent, -1, 0)
                        running_latent[-1] = encoded_state
                    predicted_next_state = world_model.predictor_fwd([running_latent.view(1, horizon, -1), None, None], deterministic=deterministic_wm)[0][-1, -1].unsqueeze(dim=0)
                    og_predicted_next_state = predicted_next_state.clone()
                    pending_predictions.append(og_predicted_next_state)

                    if world_model.fsq_encoder:
                        encoded_state = world_model.encoder.shift_and_scale(encoded_state)
                        predicted_next_state = world_model.encoder.shift_and_scale(predicted_next_state)

                if only_real_states:
                    action = agent([state.view(1, -1), None])
                elif with_reconstruction:
                    reconstructed_state = world_model.reconstructor_fwd(encoded_state)
                    reconstructed_next_state = world_model.reconstructor_fwd(predicted_next_state)
                    action = agent([reconstructed_state, reconstructed_next_state])
                else:
                    if multi_horizon_length > 1:
                        # We need to pass the one-hot encoding of the 1-step horizon
                        one_hot_horizon = torch.zeros(1, multi_horizon_length).to(device)
                        one_hot_horizon[0, 0] = 1
                        action = agent([encoded_state, predicted_next_state, one_hot_horizon])
                    else:
                        action = agent([encoded_state, predicted_next_state])

                action = action.detach().cpu().numpy()[0]
                
                next_state, reward, done, _ = env.step(action)

                # Check how different the predicted and the real are
                if world_model is not None:
                    encoded_next_state = world_model.encoder_fwd(torch.from_numpy(next_state).to(device).view(1, 1, -1)).squeeze(0)
                    if len(pending_predictions) >= world_model.prediction_horizon:
                        due_prediction = pending_predictions.popleft()
                        accuracy = (torch.sum(torch.stack([a == b for a, b in zip(due_prediction, encoded_next_state)])) / world_model.fsq_output_size).item()
                        mean_accuracy += accuracy
                        accuracies.append(accuracy)

                state = next_state
                episode_reward += reward

            if world_model is not None:
                mean_world_model_accuracy = mean_accuracy / len(accuracies) if accuracies else float("nan")
                print(f"Mean accuracy for the world model: at episode {episode}: {mean_world_model_accuracy}")
            print(f"Episode reward at episode {episode}: {episode_reward}")
            # if episode_reward < 0:
            #     import ipdb; ipdb.set_trace()
            episode_rewards.append(episode_reward)

    print(f"Final performance over {eval_episodes} episodes: {np.mean(episode_rewards)}")
    return np.mean(episode_rewards)

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
def get_next_state_through_search(state, dataset, normalization=True, prediction_horizon=1):
    # Given a state, get the closest next state in the dataset.
    # Similar to what the microsoft paper does
    # TODO: In here, we do not care about cross-episode transitions
    # for lunar lander there is no problem, but for more complex environments
    # it could be a problem

    # The dataset is a list of transitions
    all_states = np.asarray(dataset["states"])
    feature_size = all_states.shape[1]

    if normalization:

        all_states_mean = np.mean(all_states, axis=0)
        all_states_std = np.std(all_states, axis=0) 

        all_states = (all_states - all_states_mean) / (all_states_std + 1e-6)

    state = np.asarray(state).reshape(1, feature_size)
    if normalization:
        state = (state - all_states_mean) / (all_states_std + 1e-6)

    # Get the distance
    distances = pairwise_distances(state, all_states)
    min_distance_index = np.argmin(distances.reshape(-1))
    if prediction_horizon == 1:
        closest_next_state = dataset["next_states"][min_distance_index]
    else:
        # Here we need to use the sate + t. For now, we do not care about cross-episode problems
        closest_next_state = dataset["states"][min(min_distance_index + prediction_horizon, dataset["states"].shape[0] - 1)]
    closest_state = dataset["states"][min_distance_index]

    return closest_next_state, closest_state

#######################################################################################
def create_world_model(action_size, sequence_length, prediction_horizon, lr, device, fsq_input_size, fsq_output_size, encoder_dim, L, model_name, mlp_encoder=False):
    model = LeWorldModel(
        action_dim=action_size,
        max_seq_length=sequence_length,
        prediction_horizon=prediction_horizon,
        encoder_hidden_dim=encoder_dim,
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
        world_model_prediction_horizon=1,
        deterministic_wm=True,
        with_reconstruction=False,
        agent_horizons=[1]
        ):

    # TODO: This method has a bunch of sub-optimal code and inconsistencies (e.g. cross-boundary next states)
    # The current setting works fine, but for a real implementation, do not use this method.

    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

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

    if only_real_states:
        agent.set_dataset(
            states=torch.from_numpy(new_dataset["states"]).to(device),
            actions=torch.from_numpy(new_dataset["actions"]).to(device),
            next_states=torch.from_numpy(new_dataset["next_states"]).to(device)
        )
        return new_dataset

    # Change the next_states key with a dict of multi-horizons
    max_horizon = np.max(agent_horizons)

    # Just save this for future reference, especially for multi-horizon
    all_states = deepcopy(new_dataset["states"])
    new_dataset["states"] = new_dataset["states"][:-max_horizon]
    new_dataset["actions"] = new_dataset["actions"][:-max_horizon]
    new_dataset["rewards"] = new_dataset["rewards"][:-max_horizon]
    new_dataset["terminals"] = new_dataset["terminals"][:-max_horizon]
    new_dataset["next_states"] = new_dataset["next_states"][:-max_horizon]

    print(f"This dataset has a total of {new_dataset["states"].shape[0]} transitions")

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

            sequence_starts = world_model.get_valid_sequence_starts()
            state_sequences = torch.stack([
                encoded_states[start:start+horizon]
                for start in sequence_starts
            ])
            current_indices = torch.from_numpy(sequence_starts + horizon - 1).to(device)

            predicted_next_states = world_model.predictor_fwd([state_sequences, None, None], deterministic=deterministic_wm)[0][:, -1, :]
            if world_model.fsq_encoder:
                encoded_states = world_model.encoder.shift_and_scale(encoded_states)
                predicted_next_states = world_model.encoder.shift_and_scale(predicted_next_states)
            encoded_next_states = predicted_next_states
            encoded_states = encoded_states[current_indices]

            # If we do multi-horizon, we will use gt-forcing
            # For how I created it, I need to replace the old new_states dataset, even if multi-horizon
            if len(agent_horizons) > 1:
                encoded_next_states = []
                for idx in current_indices:
                    for h in agent_horizons:
                        next_state_h = all_states[idx + h]
                        encoded_next_state_h = world_model.encoder_fwd(torch.from_numpy(next_state_h).to(device).view(1, -1)).to(device).squeeze()
                        encoded_next_states.append(world_model.encoder.shift_and_scale(encoded_next_state_h).cpu().numpy())
                encoded_next_states = np.asarray(encoded_next_states)

        actions = torch.from_numpy(new_dataset["actions"]).to(device)[current_indices]

    # Set the dataset of the agent
    if world_model is None:
        states=torch.from_numpy(new_dataset["states"]).to(device)
        actions=torch.from_numpy(new_dataset["actions"]).to(device)
        if world_model_prediction_horizon == 1:
            next_states=torch.from_numpy(new_dataset["next_states"]).to(device)
        else:
            next_states=states[world_model_prediction_horizon:, :]
            states=states[:-world_model_prediction_horizon, :]
            actions=actions[:-world_model_prediction_horizon, :]
        agent.set_dataset(
            states      = states,
            actions     = actions,
            next_states = next_states
        )
    else:
        if with_reconstruction:
            with torch.no_grad():
                encoded_states = world_model.reconstructor_fwd(encoded_states)
                encoded_next_states = world_model.reconstructor_fwd(encoded_next_states)


        if len(agent_horizons) > 1:
            agent.set_multi_horizons_dataset(
                states=encoded_states,
                actions=actions,
                next_states=encoded_next_states,
                horizons=agent_horizons
            )
        else:
            agent.set_dataset(
                states=encoded_states,
                actions=actions,
                next_states=encoded_next_states
            )

    return new_dataset

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
    parser.add_argument('-wm', '--world-model-name', help="The name of the world model we already trained")
    parser.add_argument('-dn', '--dataset-name', help="The name of the precollected dataset with which we train the world model", default="datasets/dataset.pkl")
    parser.add_argument('-as', '--action-size', help="The action dimension of the env", default=2, type=int)
    parser.add_argument('-is', '--input-size', help="The state dimension of the env", default=8, type=int)
    parser.add_argument('-ed', '--encoder-dim', help="The dimension of the encoder", default=128, type=int)
    parser.add_argument('-os', '--fsq-output-size', help="The dimension of the FSQ encoder", default=16, type=int)
    parser.add_argument('-ld', '--levels-dim', help="The dimension of levels for FSQ", default=12, type=int)
    parser.add_argument('-sl', '--sequence-length', help="The max sequence length of the world model", default=8, type=int)
    parser.add_argument('-ph', '--prediction-horizon', help="How many steps ahead the world model predicts", default=1, type=int)
    parser.add_argument('-ah', '--agent-horizons', help="The horizons we want to use for our agent, it needs to be a list", default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=int, nargs="+")
    parser.add_argument('-bs', '--batch-size', help="The batch size during training", default=32, type=int)
    parser.add_argument('-en', '--epochs-number', help="The number of epochs during training", default=100, type=int)
    parser.add_argument('-lr', '--learning-rate', help="The learning rate used during training", default=1e-4, type=float)
    parser.add_argument('-wb', '--world-model-batch-size', help="The batch size during training of the world model", default=32, type=int)
    parser.add_argument('-wr', '--world-model-learning-rate', help="The learning rate used during training for the world model", default=5e-5, type=float)
    parser.add_argument('-ww', '--world-model-epochs-number', help="Number of fine-tuning epochs for the world model", default=5, type=int)
    # Number of experiments 
    parser.add_argument('-ne', '--number-of-experiments', help="Number of experiment for stability evaluation", default=1, type=int)

    # Ablations
    parser.add_argument('-rs', '--only-real-states', help="Wether to use only real states as input to the IL agent", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-wf', '--without-fsq', help="If we want to run an ablation without fsq", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-ws', '--with-search', help="If we want to use search instead of a world model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-rc', '--with-reconstruction', help="If we want to reconstruct the next state", action=argparse.BooleanOptionalAction, default=False)


    args = parser.parse_args()

    if args.world_model_name is None and not args.only_real_states and not args.with_search:
        raise ValueError("world-model-name is required unless using only-real-states or with-search")
    performance_across_experiments = []
    for i in range(args.number_of_experiments):
        print("")
        print("####")
        print(Fore.CYAN + "Creating the agent.." + Style.RESET_ALL)
        print("...")

        input_size = args.fsq_output_size * 2

        if args.only_real_states:
            input_size = args.input_size

        if args.with_search:
            input_size = args.input_size*2

        if args.with_reconstruction:
            input_size = args.input_size*2

        if len(args.agent_horizons) > 1 and not args.only_real_states:
            input_size += len(args.agent_horizons)

        agent = create_agent(
            state_size=input_size,
            action_size=args.action_size,
            policy_arch=PolicyEmbedding,
            lr=args.learning_rate,
            device=device,
            only_real_states=args.only_real_states
        )
        if args.number_of_experiments > 1:
            laoded = evaluate = False
        else:
            loaded, evaluate = check_if_model_exists(args.model_name, agent)
        world_model = None
        if not args.only_real_states and not args.with_search:
            print(Fore.CYAN + "For this agent, we need a world model. Loading it now" + Style.RESET_ALL)
            world_model = create_world_model(
                action_size=args.action_size, 
                sequence_length=args.sequence_length, 
                prediction_horizon=args.prediction_horizon,
                lr=args.world_model_learning_rate, 
                model_name=args.world_model_name,
                fsq_input_size=args.input_size, 
                fsq_output_size=args.fsq_output_size, 
                encoder_dim=args.encoder_dim,
                L=args.levels_dim, 
                mlp_encoder=args.without_fsq,
                device=device
            )
        elif args.with_search:
            print(Fore.CYAN + "For this agent, we use dataset search instead of a world model" + Style.RESET_ALL)
        print(Fore.GREEN + "Models created!" + Style.RESET_ALL)
        print("####")

        print(Fore.CYAN + "Loading dataset..." + Style.RESET_ALL)
        print("...")
        dataset = load_and_set_dataset(
            dataset_path=args.dataset_name,
            world_model=world_model,
            agent=agent,
            world_model_batch_size=args.world_model_batch_size,
            world_model_epochs=args.world_model_epochs_number,
            horizon=args.sequence_length,
            only_real_states=args.only_real_states,
            world_model_prediction_horizon=args.prediction_horizon,
            agent_horizons=args.agent_horizons,
            with_reconstruction=args.with_reconstruction
        )

        if not args.with_search:
            dataset = None

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

        preformance = evaluate_agent(
            env=env,
            agent=agent,
            world_model=world_model,
            horizon=args.sequence_length,
            dataset=dataset,
            multi_horizon_length=len(args.agent_horizons),
            eval_episodes=100,
            only_real_states=args.only_real_states,
            prediction_horizon=args.prediction_horizon,
            with_reconstruction=args.with_reconstruction
        )

        performance_across_experiments.append(preformance)

    print(f"Final performance across experiments: {np.mean(performance_across_experiments)} +- {np.std(performance_across_experiments)}")
