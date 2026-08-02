import argparse
import torch
import pickle
import os
import numpy as np

from world_model.leworldmodel import LeWorldModel
from einops import rearrange
from colorama import Fore, Style, init
from torch.nn import functional as F
from torch.distributions import Categorical
from envs.gym_env import GymEnv
from main_distill_policy import load_student
from architectures.fsq_based_student import PolicyEmbedding
from architectures.mlp_based_policy import (
    PolicyEmbedding as SACPolicyEmbedding,
    CriticEmbedding as SACCriticEmbedding,
)
from agents.sac_agent import SACAgent

from main_rm import create_reward_model 
from main_rm import load_dataset as load_reward_model_dataset
from main_rm import train as train_reward_model

init(autoreset=True)

#######################################################################################
def create_env(seed, visualize_inference):
    env = GymEnv(
        max_episode_timesteps=400,
        save_trajectories=False,
        visualize_inference=visualize_inference,
    )
    return env

#######################################################################################
def run_student(student, world_model, state):
    with torch.no_grad():
        # Utils for running the student with the prior
        prior = world_model.encoder_fwd(state).squeeze()
        action = student(prior).cpu().numpy()
        return action

#######################################################################################
def load_sac_value_agent(
        policy_name,
        state_dim,
        action_dim,
        device,
        folder="saved",
    ):
    """Load the SAC policy and twin critic for standalone planning."""
    if policy_name is None:
        raise ValueError(
            "--policy-name is required for critic-advantage planning"
        )

    policy_path = os.path.join(folder, f"{policy_name}_policy")
    critic_path = os.path.join(folder, f"{policy_name}_critic")
    if not os.path.exists(policy_path):
        raise FileNotFoundError(
            f"SAC policy checkpoint not found: {policy_path}"
        )
    if not os.path.exists(critic_path):
        raise FileNotFoundError(
            f"SAC critic checkpoint not found: {critic_path}"
        )

    agent = SACAgent(
        state_dim=state_dim,
        policy_embedding=SACPolicyEmbedding,
        critic_embedding=SACCriticEmbedding,
        discount=0.99,
        p_lr=1e-4,
        v_lr=1e-4,
        frequency_mode="timesteps",
        memory=1,
        policy_freq=1,
        alpha=0.2,
        tau=0.005,
        batch_size=1,
        num_itr=1,
        action_size=action_dim,
        max_action_value=1,
        min_action_value=-1,
        device=device,
        name=policy_name,
    )
    agent.policy.load_state_dict(
        torch.load(policy_path, map_location=device)
    )
    agent.critic.load_state_dict(
        torch.load(critic_path, map_location=device)
    )
    agent.critic_target.load_state_dict(agent.critic.state_dict())
    agent.policy.eval()
    agent.critic.eval()
    agent.critic_target.eval()
    print("SAC policy and twin critic loaded correctly")
    return agent

#######################################################################################
def evaluate_one_step_world_model(
      env,
      model,
      distilled_policy,
      device,
      num_episodes=10,
  ):
      model.eval()
      distilled_policy.eval()

      code_accuracies = []
      latent_mses = []
      state_mses = []

      context_length = model.max_seq_length
      action_size = model.action_dim

      with torch.inference_mode():
          for episode in range(num_episodes):
              state = env.reset(seed=episode)
              done = False

              running_codes = torch.zeros(
                  1,
                  env._max_episode_timesteps + context_length,
                  model.fsq_output_size,
                  device=device,
              )

              running_actions = torch.zeros(
                  1,
                  env._max_episode_timesteps + context_length,
                  action_size,
                  device=device,
              )

              step = context_length

              while not done:
                  state_tensor = torch.as_tensor(
                      state,
                      dtype=torch.float32,
                      device=device,
                  ).view(1, 1, -1)

                  current_codes = model.encoder_fwd(state_tensor)
                  running_codes[0, step - 1] = current_codes

                  code_context = running_codes[
                      :,
                      step - context_length:step,
                  ]

                  action_context = running_actions[
                      :,
                      step - context_length:step,
                  ]

                  action = distilled_policy(
                      current_codes[:, -1].float()
                  )

                  predictor_actions = torch.cat(
                      [
                          action_context[:, 1:],
                          action.unsqueeze(1),
                      ],
                      dim=1,
                  )

                  predicted_codes, _, _, _ = model.predictor_fwd(
                      [code_context, predictor_actions, None],
                      deterministic=True,
                  )

                  predicted_codes = predicted_codes[:, -1]

                  next_state, _, done, _ = env.step(
                      action[0].cpu().numpy()
                  )

                  next_state_tensor = torch.as_tensor(
                      next_state,
                      dtype=torch.float32,
                      device=device,
                  ).view(1, 1, -1)

                  real_next_codes = model.encoder_fwd(
                      next_state_tensor
                  )[:, -1]

                  code_accuracy = (
                      predicted_codes == real_next_codes
                  ).float().mean()

                  predicted_values = model.encoder.shift_and_scale(
                      predicted_codes.float()
                  )
                  real_values = model.encoder.shift_and_scale(
                      real_next_codes.float()
                  )

                  latent_mse = F.mse_loss(
                      predicted_values,
                      real_values,
                  )

                  predicted_next_state = model.reconstruction_head(
                      predicted_values
                  )

                  state_mse = F.mse_loss(
                      predicted_next_state,
                      next_state_tensor[:, -1],
                  )

                  code_accuracies.append(code_accuracy.item())
                  latent_mses.append(latent_mse.item())
                  state_mses.append(state_mse.item())

                  running_actions[0, step] = action[0]
                  state = next_state
                  step += 1

              print(
                  f"Episode {episode}: "
                  f"code accuracy={np.mean(code_accuracies):.4f}, "
                  f"latent MSE={np.mean(latent_mses):.4f}, "
                  f"state MSE={np.mean(state_mses):.4f}"
              )

      results = {
          "code_accuracy": np.mean(code_accuracies),
          "latent_mse": np.mean(latent_mses),
          "state_mse": np.mean(state_mses),
      }

      print(f"Final one-step world-model metrics: {results}")
      return results

#######################################################################################
def evaluate_distilled_policy_in_real_env(
        env,
        model,
        distilled_policy,
        device,
        num_episodes=100,
    ):
    """Evaluate the distilled policy using freshly encoded real states."""
    model.eval()
    distilled_policy.eval()

    episode_rewards = []
    episode_lengths = []

    with torch.inference_mode():
        for episode in range(num_episodes):
            state = env.reset(seed=episode)
            done = False
            episode_reward = 0.0
            episode_length = 0

            while not done:
                state_tensor = torch.as_tensor(
                    state,
                    dtype=torch.float32,
                    device=device,
                ).view(1, 1, -1)

                current_codes = model.encoder_fwd(state_tensor)
                action = distilled_policy(
                    current_codes[:, -1].float()
                )

                state, reward, done, _ = env.step(
                    action[0].cpu().numpy()
                )
                episode_reward += reward
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            print(
                f"Real-policy episode {episode}: "
                f"reward={episode_reward:.2f}, "
                f"steps={episode_length}"
            )

    results = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_episode_length": np.mean(episode_lengths),
    }

    print(
        "Final real-environment distilled-policy metrics: "
        f"reward={results['mean_reward']:.2f} "
        f"+/- {results['std_reward']:.2f}, "
        f"mean steps={results['mean_episode_length']:.1f}"
    )

    return results

#######################################################################################
def evaluate_distilled_policy_in_world_model(
        env,
        model,
        distilled_policy,
        device,
        num_episodes=10,
        max_rollout_steps=None,
    ):
    """Evaluate a student policy using only autoregressively imagined FSQ codes."""
    model.eval()
    distilled_policy.eval()

    context_length = model.max_seq_length
    action_size = model.action_dim
    max_steps = max_rollout_steps or env._max_episode_timesteps

    episode_results = []
    metrics_by_depth = {
        "code_accuracy": [],
        "latent_mse": [],
        "state_mse": [],
    }

    with torch.inference_mode():
        for episode in range(num_episodes):
            real_state = env.reset(seed=episode)
            initial_state = torch.as_tensor(
                real_state,
                dtype=torch.float32,
                device=device,
            ).view(1, 1, -1)

            initial_codes = model.encoder_fwd(initial_state)

            imagined_codes = torch.zeros(
                1,
                context_length,
                model.fsq_output_size,
                device=device,
            )
            imagined_codes[:, -1] = initial_codes[:, -1]

            imagined_actions = torch.zeros(
                1,
                context_length,
                action_size,
                device=device,
            )

            episode_code_accuracies = []
            episode_latent_mses = []
            episode_state_mses = []
            episode_reward = 0.0
            done = False
            depth = 0

            while not done and depth < max_steps:
                action = distilled_policy(
                    imagined_codes[:, -1].float()
                )

                predictor_actions = torch.cat(
                    [
                        imagined_actions[:, 1:],
                        action.unsqueeze(1),
                    ],
                    dim=1,
                )

                predicted_codes, _, _, _ = model.predictor_fwd(
                    [imagined_codes, predictor_actions, None],
                    deterministic=True,
                )
                predicted_codes = predicted_codes[:, -1]

                real_next_state, reward, done, _ = env.step(
                    action[0].cpu().numpy()
                )
                episode_reward += reward

                real_next_state_tensor = torch.as_tensor(
                    real_next_state,
                    dtype=torch.float32,
                    device=device,
                ).view(1, 1, -1)
                real_next_codes = model.encoder_fwd(
                    real_next_state_tensor
                )[:, -1]

                code_accuracy = (
                    predicted_codes == real_next_codes
                ).float().mean()

                predicted_values = model.encoder.shift_and_scale(
                    predicted_codes.float()
                )
                real_values = model.encoder.shift_and_scale(
                    real_next_codes.float()
                )
                latent_mse = F.mse_loss(
                    predicted_values,
                    real_values,
                )

                decoded_next_state = model.reconstruction_head(
                    predicted_values
                )
                state_mse = F.mse_loss(
                    decoded_next_state,
                    real_next_state_tensor[:, -1],
                )

                episode_code_accuracies.append(code_accuracy.item())
                episode_latent_mses.append(latent_mse.item())
                episode_state_mses.append(state_mse.item())

                for key, value in (
                    ("code_accuracy", code_accuracy.item()),
                    ("latent_mse", latent_mse.item()),
                    ("state_mse", state_mse.item()),
                ):
                    while len(metrics_by_depth[key]) <= depth:
                        metrics_by_depth[key].append([])
                    metrics_by_depth[key][depth].append(value)

                imagined_codes = torch.cat(
                    [
                        imagined_codes[:, 1:],
                        predicted_codes.unsqueeze(1),
                    ],
                    dim=1,
                )
                imagined_actions = predictor_actions
                depth += 1

            result = {
                "episode_reward": episode_reward,
                "steps": depth,
                "code_accuracy": np.mean(episode_code_accuracies),
                "latent_mse": np.mean(episode_latent_mses),
                "state_mse": np.mean(episode_state_mses),
            }
            episode_results.append(result)

            print(
                f"Autoregressive episode {episode}: "
                f"reward={result['episode_reward']:.2f}, "
                f"steps={result['steps']}, "
                f"code accuracy={result['code_accuracy']:.4f}, "
                f"latent MSE={result['latent_mse']:.4f}, "
                f"state MSE={result['state_mse']:.4f}"
            )

    depth_results = {
        key: np.asarray(
            [np.mean(values) for values in depth_values],
            dtype=np.float32,
        )
        for key, depth_values in metrics_by_depth.items()
    }

    aggregate_results = {
        "episode_reward": np.mean(
            [result["episode_reward"] for result in episode_results]
        ),
        "code_accuracy": np.mean(
            [result["code_accuracy"] for result in episode_results]
        ),
        "latent_mse": np.mean(
            [result["latent_mse"] for result in episode_results]
        ),
        "state_mse": np.mean(
            [result["state_mse"] for result in episode_results]
        ),
        "by_depth": depth_results,
        "episodes": episode_results,
    }

    print(
        "Final autoregressive world-model metrics: "
        f"reward={aggregate_results['episode_reward']:.2f}, "
        f"code accuracy={aggregate_results['code_accuracy']:.4f}, "
        f"latent MSE={aggregate_results['latent_mse']:.4f}, "
        f"state MSE={aggregate_results['state_mse']:.4f}"
    )

    return aggregate_results

#######################################################################################
def evaluate_horizon_world_model(
        env,
        model,
        distilled_policy,
        device,
        horizon=10,
        num_episodes=10,
        window_stride=None,
        gamma=0.99,
    ):
    """Measure free-running model error over planning-length real trajectories."""
    model.eval()
    distilled_policy.eval()

    context_length = model.max_seq_length
    action_size = model.action_dim
    window_stride = horizon if window_stride is None else window_stride
    metrics_by_depth = {
        "code_accuracy": [[] for _ in range(horizon)],
        "full_code_accuracy": [[] for _ in range(horizon)],
        "latent_mse": [[] for _ in range(horizon)],
        "state_mse": [[] for _ in range(horizon)],
    }
    reward_predictions_by_depth = [[] for _ in range(horizon)]
    reward_targets_by_depth = [[] for _ in range(horizon)]
    predicted_returns_by_depth = [[] for _ in range(horizon)]
    real_returns_by_depth = [[] for _ in range(horizon)]
    reward_mean = getattr(model, "reward_mean", 0.0)
    reward_std = getattr(model, "reward_std", 1.0)
    total_windows = 0

    with torch.inference_mode():
        for episode in range(num_episodes):
            state = env.reset(seed=episode)
            states = [np.asarray(state, dtype=np.float32)]
            actions = []
            rewards = []
            done = False

            # First record one real-policy trajectory. These same actions are then
            # supplied to every imagined rollout, isolating dynamics-model error.
            while not done:
                state_tensor = torch.as_tensor(
                    state, dtype=torch.float32, device=device
                ).view(1, 1, -1)
                state_codes = model.encoder_fwd(state_tensor)
                action = distilled_policy(state_codes[:, -1].float())
                state, reward, done, _ = env.step(action[0].cpu().numpy())
                actions.append(action[0].detach())
                rewards.append(float(reward))
                states.append(np.asarray(state, dtype=np.float32))

            states_tensor = torch.as_tensor(
                np.asarray(states), dtype=torch.float32, device=device
            ).unsqueeze(0)
            real_codes = model.encoder_fwd(states_tensor)[0]

            episode_windows = 0
            last_start = len(actions) - horizon
            for start in range(0, last_start + 1, window_stride):
                first_context_state = max(0, start - context_length + 1)
                context_codes = torch.zeros(
                    1, context_length, model.fsq_output_size, device=device
                )
                real_context = real_codes[first_context_state:start + 1]
                context_codes[:, -len(real_context):] = real_context

                context_actions = torch.zeros(
                    1, context_length, action_size, device=device
                )
                real_context_actions = torch.stack(
                    actions[first_context_state:start + 1]
                )
                context_actions[:, -len(real_context_actions):] = (
                    real_context_actions
                )

                predicted_return = 0.0
                real_return = 0.0
                for depth in range(horizon):
                    predicted_codes, _, predicted_rewards, _ = model.predictor_fwd(
                        [context_codes, context_actions, None],
                        deterministic=True,
                    )
                    predicted_codes = predicted_codes[:, -1]
                    predicted_reward = (
                        predicted_rewards[:, -1, 0].item() * reward_std
                        + reward_mean
                    )
                    real_reward = rewards[start + depth]
                    predicted_return += gamma ** depth * predicted_reward
                    real_return += gamma ** depth * real_reward
                    reward_predictions_by_depth[depth].append(
                        predicted_reward
                    )
                    reward_targets_by_depth[depth].append(real_reward)
                    predicted_returns_by_depth[depth].append(
                        predicted_return
                    )
                    real_returns_by_depth[depth].append(real_return)
                    target_codes = real_codes[start + depth + 1].unsqueeze(0)

                    code_matches = predicted_codes == target_codes
                    predicted_values = model.encoder.shift_and_scale(
                        predicted_codes.float()
                    )
                    target_values = model.encoder.shift_and_scale(
                        target_codes.float()
                    )
                    decoded_state = model.reconstruction_head(predicted_values)
                    target_state = states_tensor[:, start + depth + 1]

                    metrics_by_depth["code_accuracy"][depth].append(
                        code_matches.float().mean().item()
                    )
                    metrics_by_depth["full_code_accuracy"][depth].append(
                        code_matches.all(dim=-1).float().mean().item()
                    )
                    metrics_by_depth["latent_mse"][depth].append(
                        F.mse_loss(predicted_values, target_values).item()
                    )
                    metrics_by_depth["state_mse"][depth].append(
                        F.mse_loss(decoded_state, target_state).item()
                    )

                    if depth + 1 < horizon:
                        next_action = actions[start + depth + 1].view(
                            1, 1, -1
                        )
                        context_codes = torch.cat(
                            [context_codes[:, 1:], predicted_codes.unsqueeze(1)],
                            dim=1,
                        )
                        context_actions = torch.cat(
                            [context_actions[:, 1:], next_action], dim=1
                        )

                episode_windows += 1
                total_windows += 1

            print(
                f"Horizon evaluation episode {episode}: "
                f"steps={len(actions)}, windows={episode_windows}"
            )

    if total_windows == 0:
        raise RuntimeError(
            f"No episode was long enough for a horizon-{horizon} rollout"
        )

    results = {
        key: np.asarray(
            [np.mean(depth_values) for depth_values in values],
            dtype=np.float32,
        )
        for key, values in metrics_by_depth.items()
    }
    results["num_windows"] = total_windows

    def correlation(predictions, targets):
        predictions = np.asarray(predictions)
        targets = np.asarray(targets)
        if (
            len(predictions) < 2
            or predictions.std() < 1e-8
            or targets.std() < 1e-8
        ):
            return np.nan
        return np.corrcoef(predictions, targets)[0, 1]

    results["reward_mae"] = np.asarray([
        np.mean(np.abs(
            np.asarray(predictions) - np.asarray(targets)
        ))
        for predictions, targets in zip(
            reward_predictions_by_depth, reward_targets_by_depth
        )
    ], dtype=np.float32)
    results["reward_correlation"] = np.asarray([
        correlation(predictions, targets)
        for predictions, targets in zip(
            reward_predictions_by_depth, reward_targets_by_depth
        )
    ], dtype=np.float32)
    results["return_mae"] = np.asarray([
        np.mean(np.abs(
            np.asarray(predictions) - np.asarray(targets)
        ))
        for predictions, targets in zip(
            predicted_returns_by_depth, real_returns_by_depth
        )
    ], dtype=np.float32)
    results["return_correlation"] = np.asarray([
        correlation(predictions, targets)
        for predictions, targets in zip(
            predicted_returns_by_depth, real_returns_by_depth
        )
    ], dtype=np.float32)

    print(
        f"Planning-horizon world-model metrics "
        f"({total_windows} windows, horizon={horizon}):"
    )
    for depth in range(horizon):
        print(
            f"  depth {depth + 1}: "
            f"code accuracy={results['code_accuracy'][depth]:.4f}, "
            f"full code accuracy={results['full_code_accuracy'][depth]:.4f}, "
            f"latent MSE={results['latent_mse'][depth]:.4f}, "
            f"state MSE={results['state_mse'][depth]:.4f}, "
            f"reward MAE={results['reward_mae'][depth]:.3f}, "
            f"reward correlation={results['reward_correlation'][depth]:.4f}, "
            f"return correlation={results['return_correlation'][depth]:.4f}"
        )

    return results

#######################################################################################
def evaluate_real_environment_cem(
        env,
        model,
        distilled_policy,
        device,
        action_size,
        random_actions=64,
        horizon=5,
        action_chunk=1,
        num_eval_episode=10,
        cem_iterations=2,
        elite_fraction=0.1,
        residual_bound=0.05,
        residual_std=0.02,
        residual_penalty=1.0,
        uncertainty_penalty=0.1,
        improvement_margin=0.1,
        gamma=0.99,
    ):
    """Oracle CEM evaluation whose candidate rollouts use the real simulator.

    This is a diagnostic upper bound, not a deployable controller. Each
    candidate resets a second environment with the episode seed, replays the
    executed action prefix, and then evaluates policy-plus-residual actions for
    ``horizon`` real steps. Action selection intentionally matches the current
    planner: execute the first action of the best sampled sequence encountered
    across all CEM iterations, with a zero-residual student fallback.
    """
    if distilled_policy is None:
        raise ValueError("Real-environment CEM requires a distilled policy")
    if horizon % action_chunk != 0:
        raise ValueError("Horizon must be divisible by action_chunk")

    model.eval()
    distilled_policy.eval()
    action_space = env.env.action_space
    action_low = torch.as_tensor(
        action_space.low, dtype=torch.float32, device=device
    )
    action_high = torch.as_tensor(
        action_space.high, dtype=torch.float32, device=device
    )
    num_chunks = horizon // action_chunk
    num_elites = max(1, int(random_actions * elite_fraction))
    alpha = 0.1

    oracle_env = GymEnv(
        max_episode_timesteps=env._max_episode_timesteps,
        save_trajectories=False,
        visualize_inference=False,
    )

    def student_action(state):
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=device
        ).view(1, 1, -1)
        codes = model.encoder_fwd(state_tensor)
        action = distilled_policy(codes[:, -1].float())[0]
        return torch.clamp(action, action_low, action_high)

    def build_model_context(state_prefix, action_prefix):
        context_length = model.max_seq_length
        code_context = torch.zeros(
            1,
            context_length,
            model.fsq_output_size,
            device=device,
        )
        action_context = torch.zeros(
            1, context_length, action_size, device=device
        )
        recent_states = state_prefix[-context_length:]
        state_tensor = torch.as_tensor(
            np.asarray(recent_states),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        recent_codes = model.encoder_fwd(state_tensor)
        code_context[:, -recent_codes.shape[1]:] = recent_codes
        recent_actions = action_prefix[-context_length:]
        if recent_actions:
            action_tensor = torch.as_tensor(
                np.asarray(recent_actions),
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            action_context[:, -action_tensor.shape[1]:] = action_tensor
        return code_context, action_context

    def score_model_residuals(
            residual_sequences, code_context, action_context
        ):
        num_candidates = residual_sequences.shape[0]
        current_codes = code_context.repeat(num_candidates, 1, 1)
        current_actions = action_context.repeat(num_candidates, 1, 1)
        predicted_return = torch.zeros(
            num_candidates, 1, device=device
        )
        predicted_uncertainty = torch.zeros_like(predicted_return)
        residual_cost = torch.zeros_like(predicted_return)
        candidate_action = torch.clamp(
            distilled_policy(current_codes[:, -1].float())
            + residual_sequences[:, 0],
            action_low.view(1, action_size),
            action_high.view(1, action_size),
        )
        for depth in range(horizon):
            current_actions = torch.cat(
                [current_actions[:, 1:], candidate_action.unsqueeze(1)],
                dim=1,
            )
            predicted_codes, predicted_logits, predicted_rewards, _ = (
                model.predictor_fwd(
                    [current_codes.float(), current_actions, None]
                )
            )
            next_codes = predicted_codes[:, -1]
            next_logits = predicted_logits[:, -1]
            discount = gamma ** depth
            predicted_return += (
                discount * predicted_rewards[:, -1]
            )
            predicted_uncertainty += discount * (
                Categorical(logits=next_logits)
                .entropy()
                .mean(dim=-1, keepdim=True)
                / np.log(model.fsq_L)
            )
            residual_cost += (
                residual_penalty
                * discount
                * residual_sequences[:, depth]
                .square()
                .mean(dim=-1, keepdim=True)
            )
            current_codes = torch.cat(
                [current_codes[:, 1:], next_codes.unsqueeze(1)], dim=1
            )
            if depth + 1 < horizon:
                candidate_action = torch.clamp(
                    distilled_policy(next_codes.float())
                    + residual_sequences[:, depth + 1],
                    action_low.view(1, action_size),
                    action_high.view(1, action_size),
                )
        model_score = (
            predicted_return
            - uncertainty_penalty * predicted_uncertainty
            - residual_cost
        )
        return (
            model_score[:, 0].cpu().numpy(),
            predicted_return[:, 0].cpu().numpy(),
        )

    def score_teacher_forced_rewards(
            trajectory_states, trajectory_actions,
            code_context, action_context,
        ):
        current_codes = code_context.clone()
        current_actions = action_context.clone()
        predicted_return = 0.0
        for depth, (next_state, action) in enumerate(
            zip(trajectory_states[1:], trajectory_actions)
        ):
            action_tensor = torch.as_tensor(
                action, dtype=torch.float32, device=device
            ).view(1, 1, -1)
            current_actions = torch.cat(
                [current_actions[:, 1:], action_tensor], dim=1
            )
            _, _, predicted_rewards, _ = model.predictor_fwd(
                [current_codes.float(), current_actions, None]
            )
            predicted_return += (
                gamma ** depth * predicted_rewards[:, -1, 0].item()
            )
            next_state_tensor = torch.as_tensor(
                next_state, dtype=torch.float32, device=device
            ).view(1, 1, -1)
            next_codes = model.encoder_fwd(next_state_tensor)
            current_codes = torch.cat(
                [current_codes[:, 1:], next_codes], dim=1
            )
        return predicted_return

    def correlation(first, second):
        first = np.asarray(first, dtype=np.float64)
        second = np.asarray(second, dtype=np.float64)
        if (
            first.size < 2
            or np.std(first) < 1e-8
            or np.std(second) < 1e-8
        ):
            return float("nan")
        return float(np.corrcoef(first, second)[0, 1])

    def rank_correlation(first, second):
        first_ranks = np.empty(len(first), dtype=np.float64)
        second_ranks = np.empty(len(second), dtype=np.float64)
        first_ranks[np.argsort(first)] = np.arange(len(first))
        second_ranks[np.argsort(second)] = np.arange(len(second))
        return correlation(first_ranks, second_ranks)

    def score_residual_sequence(seed, action_prefix, residuals):
        candidate_state = oracle_env.reset(seed=seed)
        prefix_done = False
        for prefix_action in action_prefix:
            candidate_state, _, prefix_done, _ = oracle_env.step(
                np.asarray(prefix_action, dtype=np.float32)
            )
            if prefix_done:
                raise RuntimeError(
                    "Oracle replay terminated before reaching the decision state"
                )

        score = 0.0
        discounted_reward = 0.0
        candidate_actions = []
        trajectory_states = [np.asarray(candidate_state).copy()]
        for depth in range(horizon):
            policy_action = student_action(candidate_state)
            candidate_action = torch.clamp(
                policy_action + residuals[depth],
                action_low,
                action_high,
            )
            candidate_action_np = candidate_action.cpu().numpy()
            candidate_actions.append(candidate_action_np.copy())
            candidate_state, reward, done, _ = oracle_env.step(
                candidate_action_np
            )
            trajectory_states.append(np.asarray(candidate_state).copy())
            discount = gamma ** depth
            discounted_reward += discount * float(reward)
            score += (
                discount * float(reward)
                - residual_penalty
                * discount
                * residuals[depth].square().mean().item()
            )
            if done:
                break
        return (
            score,
            discounted_reward,
            candidate_actions,
            trajectory_states,
        )

    episode_rewards = []
    episode_acceptances = []
    episode_advantages = []
    candidate_rank_correlations = []
    teacher_forced_rank_correlations = []
    elite_overlaps = []
    model_selection_regrets = []
    model_selection_advantages = []
    try:
        with torch.inference_mode():
            for episode in range(num_eval_episode):
                state = env.reset(seed=episode)
                done = False
                episode_reward = 0.0
                action_prefix = []
                state_prefix = [np.asarray(state).copy()]
                accepted_decisions = []
                predicted_advantages = []

                while not done:
                    code_context, action_context = build_model_context(
                        state_prefix, action_prefix
                    )
                    zero_residuals = torch.zeros(
                        horizon, action_size, device=device
                    )
                    (
                        baseline_score,
                        baseline_reward,
                        baseline_actions,
                        _,
                    ) = score_residual_sequence(
                        episode, action_prefix, zero_residuals
                    )

                    residual_mean = torch.zeros(
                        num_chunks, action_size, device=device
                    )
                    residual_sigma = torch.full_like(
                        residual_mean, residual_std
                    )
                    best_score = None
                    best_reward = None
                    best_actions = None
                    best_residuals = None

                    for _ in range(cem_iterations):
                        chunk_residuals = (
                            residual_mean.unsqueeze(0)
                            + residual_sigma.unsqueeze(0)
                            * torch.randn(
                                random_actions,
                                num_chunks,
                                action_size,
                                device=device,
                            )
                        ).clamp(-residual_bound, residual_bound)
                        # Keep the exact student rollout in both the evaluated
                        # population and any subsequent elite update.
                        chunk_residuals[0].zero_()
                        residual_sequences = torch.repeat_interleave(
                            chunk_residuals,
                            repeats=action_chunk,
                            dim=1,
                        )
                        # Always compare against the exact student rollout.

                        model_scores, model_predicted_returns = (
                            score_model_residuals(
                                residual_sequences,
                                code_context,
                                action_context,
                            )
                        )

                        scores = []
                        candidate_rewards = []
                        candidate_actions = []
                        teacher_forced_returns = []
                        for candidate in range(random_actions):
                            (
                                score,
                                candidate_reward,
                                actions,
                                trajectory_states,
                            ) = (
                                score_residual_sequence(
                                    episode,
                                    action_prefix,
                                    residual_sequences[candidate],
                                )
                            )
                            scores.append(score)
                            candidate_rewards.append(candidate_reward)
                            candidate_actions.append(actions)
                            teacher_forced_returns.append(
                                score_teacher_forced_rewards(
                                    trajectory_states,
                                    actions,
                                    code_context,
                                    action_context,
                                )
                            )

                        scores_array = np.asarray(scores)
                        rewards_array = np.asarray(candidate_rewards)
                        teacher_forced_array = np.asarray(
                            teacher_forced_returns
                        )
                        candidate_rank_correlations.append(
                            rank_correlation(model_scores, scores_array)
                        )
                        teacher_forced_rank_correlations.append(
                            rank_correlation(
                                teacher_forced_array, rewards_array
                            )
                        )
                        model_elites = set(
                            np.argsort(model_scores)[-num_elites:].tolist()
                        )
                        oracle_elites = set(
                            np.argsort(scores_array)[-num_elites:].tolist()
                        )
                        elite_overlaps.append(
                            len(model_elites & oracle_elites) / num_elites
                        )
                        model_choice = int(np.argmax(model_scores))
                        model_selection_regrets.append(
                            float(scores_array.max() - scores_array[model_choice])
                        )
                        model_selection_advantages.append(
                            float(
                                scores_array[model_choice] - baseline_score
                            )
                        )

                        scores_tensor = torch.as_tensor(
                            scores, dtype=torch.float32, device=device
                        )
                        elite_indices = torch.topk(
                            scores_tensor,
                            k=num_elites,
                            largest=True,
                        ).indices
                        elite_chunks = chunk_residuals[elite_indices]
                        new_mean = elite_chunks.mean(dim=0)
                        new_sigma = elite_chunks.std(
                            dim=0, unbiased=False
                        ).clamp_min(1e-3)
                        residual_mean = (
                            alpha * residual_mean
                            + (1.0 - alpha) * new_mean
                        ).clamp(-residual_bound, residual_bound)
                        residual_sigma = (
                            alpha * residual_sigma
                            + (1.0 - alpha) * new_sigma
                        ).clamp(1e-3, residual_bound)

                        best_index = int(scores_tensor.argmax().item())
                        candidate_score = scores[best_index]
                        if best_score is None or candidate_score > best_score:
                            best_score = candidate_score
                            best_reward = candidate_rewards[best_index]
                            best_actions = candidate_actions[best_index]
                            best_residuals = residual_sequences[
                                best_index
                            ].clone()

                    accepted = (
                        best_score >= baseline_score + improvement_margin
                    )
                    if accepted:
                        selected_action = best_actions[0]
                        selected_reward = best_reward
                    else:
                        selected_action = baseline_actions[0]
                        selected_reward = baseline_reward
                        best_residuals = zero_residuals

                    predicted_advantages.append(
                        float(selected_reward - baseline_reward)
                    )
                    accepted_decisions.append(float(accepted))
                    state, reward, done, _ = env.step(selected_action)
                    action_prefix.append(np.asarray(selected_action).copy())
                    state_prefix.append(np.asarray(state).copy())
                    episode_reward += float(reward)

                episode_rewards.append(episode_reward)
                episode_acceptances.append(np.mean(accepted_decisions))
                episode_advantages.append(np.mean(predicted_advantages))
                print(
                    f"Real-environment CEM episode {episode}: "
                    f"reward={episode_reward:.2f}, "
                    f"oracle horizon advantage="
                    f"{episode_advantages[-1]:.3f}, "
                    f"acceptance={episode_acceptances[-1]:.3f}"
                )
    finally:
        oracle_env.close()

    print(
        "Final real-environment CEM metrics: "
        f"reward={np.mean(episode_rewards):.2f} +/- "
        f"{np.std(episode_rewards):.2f}, "
        f"mean true advantage={np.mean(episode_advantages):.3f}, "
        f"acceptance={np.mean(episode_acceptances):.3f}"
    )
    print("Candidate-ranking diagnostics:")
    print(
        "  autoregressive model/oracle rank correlation="
        f"{np.nanmean(candidate_rank_correlations):.4f}"
    )
    print(
        "  teacher-forced reward/oracle rank correlation="
        f"{np.nanmean(teacher_forced_rank_correlations):.4f}"
    )
    print(
        f"  top-{num_elites} elite overlap="
        f"{np.mean(elite_overlaps):.4f}"
    )
    print(
        "  world-model top-1 oracle regret="
        f"{np.mean(model_selection_regrets):.4f}, "
        "world-model top-1 oracle advantage over student="
        f"{np.mean(model_selection_advantages):.4f}"
    )
    return episode_rewards

#######################################################################################
def evaluate_wm(env, 
                model, 
                reward_model,
                action_size, 
                random_actions, 
                horizon, 
                dataset_path, 
                device, 
                max_episode_timesteps,
                action_chunk, 
                distilled_policy=None,
                num_eval_episode=100,
                cem_iterations=3,
                elite_fraction=0.1,
                residual_bound=0.05,
                residual_std=0.02,
                residual_penalty=1.0,
                uncertainty_penalty=0.1,
                improvement_margin=0.1,
                value_agent=None,
                terminal_value_weight=1.0,
                planner="cem",
                mppi_temperature=1.0,
                mppi_iterations=1,
                planning_objective="reward",
                critic_advantage_weight=1.0):
    gamma = 0.99
    alpha = 0.1
    if planner not in {"cem", "mppi"}:
        raise ValueError("planner must be either 'cem' or 'mppi'")
    if mppi_temperature <= 0:
        raise ValueError("mppi_temperature must be positive")
    if mppi_iterations < 1:
        raise ValueError("mppi_iterations must be at least one")
    if planning_objective not in {
        "reward", "critic_advantage", "hybrid"
    }:
        raise ValueError(
            "planning_objective must be 'reward', "
            "'critic_advantage', or 'hybrid'"
        )
    if (
        planning_objective in {"critic_advantage", "hybrid"}
        and value_agent is None
    ):
        raise ValueError(
            "critic-advantage planning requires a SAC value_agent"
        )

    # Get the original dataset, and get a state with high reward. Use that as goal

    # The dataset is in trajectory form
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    
    episode_rewards = []

    avg_reward = np.mean(dataset["rewards"])
    std_reward = np.std(dataset["rewards"]) + 1e-6

    if value_agent is not None:
        value_agent.policy.eval()
        value_agent.critic_target.eval()


    rewards = np.asarray(dataset["rewards"])
    diagnostic_actual_rewards = []
    diagnostic_predicted_rewards = []
    diagnostic_true_next_rewards = []
    diagnostic_actual_returns = []
    diagnostic_predicted_returns = []
    diagnostic_true_next_returns = []
    diagnostic_predicted_advantages = []
    diagnostic_residual_magnitudes = []
    diagnostic_acceptances = []
    
    index_with_max_reward = np.argmax(rewards)
    goal_state = dataset["states"][index_with_max_reward]
    goal_state = np.expand_dims(np.asarray(goal_state), 0)

    action_space = env.env.action_space
    action_low = torch.as_tensor(action_space.low, dtype=torch.float32, device=device)
    action_high = torch.as_tensor(action_space.high, dtype=torch.float32, device=device)
    action_center = (action_high + action_low) / 2.0
    action_scale = (action_high - action_low) / 2.0
    num_elites = max(1, int(random_actions * elite_fraction))

    def rollout_action_sequences(action_sequences, initial_input, initial_action, action_residuals=None, distilled_policy=None):
        current_input = initial_input.unsqueeze(0).repeat(action_sequences.shape[0], 1, 1)
        current_action = initial_action.unsqueeze(0).repeat(action_sequences.shape[0], 1, 1)

        current_reward = 0
        current_uncertainty = 0
        current_residual_cost = 0
        current_critic_advantage = 0
        first_predicted_reward = 0

        for i in range(action_sequences.shape[1]):
            next_action = action_sequences[:, i, :].unsqueeze(1)
            if planning_objective in {"critic_advantage", "hybrid"}:
                current_codes = current_input[:, -1].long()
                if i == 0:
                    decoded_states = current_state[:, -1].float().repeat(
                        current_codes.shape[0], 1
                    )
                else:
                    current_latents = model.encoder.shift_and_scale(
                        current_codes.float()
                    )
                    decoded_states = model.reconstruction_head(
                        current_latents
                    ).float()
                baseline_actions = torch.clamp(
                    distilled_policy(current_codes.float()),
                    action_low.view(1, action_size),
                    action_high.view(1, action_size),
                )
                candidate_q1, candidate_q2 = value_agent.critic_target(
                    decoded_states, next_action[:, 0]
                )
                baseline_q1, baseline_q2 = value_agent.critic_target(
                    decoded_states, baseline_actions
                )
                step_advantage = (
                    torch.minimum(candidate_q1, candidate_q2)
                    - torch.minimum(baseline_q1, baseline_q2)
                ) / std_reward
                current_critic_advantage += gamma**i * step_advantage
            current_action = torch.cat([current_action[:, 1:], next_action], dim=1)
            predicted_state, predicted_logits, predicted_reward, predicted_terminal = model.predictor_fwd([current_input.float(), current_action, None])
            predicted_reward = predicted_reward[:, -1]
            next_logits = predicted_logits[:, -1]
            normalized_entropy = (
                Categorical(logits=next_logits).entropy().mean(dim=-1, keepdim=True)
                / np.log(model.fsq_L)
            )
            if i == 0:
                # first_predicted_reward = predicted_reward.view(-1, predicted_state.shape[1], 1)[:, -1]
                first_predicted_reward = predicted_reward.view(-1, 1)
            predicted_state = predicted_state[:, -1, :].unsqueeze(1)

            if distilled_policy is not None and i+1 < action_residuals.shape[1]:
                predicted_action = distilled_policy(predicted_state.squeeze())
                # Perturb
                predicted_action = predicted_action + action_residuals[:, i+1] 
                predicted_action = torch.clamp(
                    predicted_action,
                    action_low.view(1, action_size),
                    action_high.view(1, action_size),
                )
                # Replace in the sequence
                # And we do not care about the last action
                action_sequences[:, i+1, :] = predicted_action
            
            current_input = torch.cat([current_input[:, 1:], predicted_state], dim=1)
            current_reward += gamma**i * predicted_reward
            current_uncertainty += gamma**i * normalized_entropy
            if action_residuals is not None:
                current_residual_cost += (
                    gamma**i
                    * action_residuals[:, i].square().mean(
                        dim=-1, keepdim=True
                    )
                )

        terminal_value = torch.zeros_like(current_reward)
        if value_agent is not None and terminal_value_weight != 0.0:
            # SAC's critic operates on continuous environment states. Decode
            # the final imagined FSQ code, follow the deterministic SAC policy,
            # and conservatively bootstrap with the smaller twin-Q estimate.
            terminal_codes = current_input[:, -1].long()
            terminal_latents = model.encoder.shift_and_scale(
                terminal_codes.float()
            )
            terminal_states = model.reconstruction_head(
                terminal_latents
            ).float()
            terminal_actions = value_agent(
                terminal_states,
                deterministic=True,
            )[0]
            terminal_q1, terminal_q2 = value_agent.critic_target(
                terminal_states,
                terminal_actions,
            )
            # World-model rewards are standardized, whereas SAC Q-values are
            # trained in raw reward units. A constant offset does not affect
            # candidate ranking, so only the scale conversion is required.
            terminal_value = (
                terminal_value_weight
                * gamma ** action_sequences.shape[1]
                * torch.minimum(terminal_q1, terminal_q2)
                / std_reward
            )

        reward_return = current_reward + terminal_value
        if planning_objective == "reward":
            objective_return = reward_return
        elif planning_objective == "critic_advantage":
            objective_return = (
                critic_advantage_weight * current_critic_advantage
            )
        else:
            objective_return = (
                reward_return
                + critic_advantage_weight * current_critic_advantage
            )

        conservative_score = (
            objective_return
            - uncertainty_penalty * current_uncertainty
            - residual_penalty * current_residual_cost
        )
        return (
            conservative_score,
            objective_return,
            first_predicted_reward,
            current_uncertainty,
            current_residual_cost,
        )

    # For now let's just do an infinite loop over the episodes
    for e in range(num_eval_episode):
        current_state = env.reset(seed=e)
        current_state = torch.from_numpy(np.expand_dims(np.asarray(current_state), [0,1])).to(device=device)
        context_length = model.max_seq_length
        step = context_length

        running_latent  = torch.zeros(1, max_episode_timesteps + context_length, model.fsq_output_size).to(device)
        running_actions = torch.zeros(1, max_episode_timesteps + context_length, action_size).to(device) 
        done            = False
        ep_reward       = 0
        episode_actual_rewards = []
        episode_predicted_rewards = []
        episode_true_next_rewards = []
        episode_predicted_advantages = []
        episode_residual_magnitudes = []
        episode_acceptances = []
        mppi_action_mean = torch.zeros(
            horizon // action_chunk, action_size, device=device
        )

        while not done:
            encoded_current = model.encoder_fwd(current_state)
            running_latent[0, step-1] = encoded_current

            current_input   = running_latent[0, step - context_length:step]
            current_action  = running_actions[0, step - context_length:step]

            action_mean = action_center.view(1, action_size).repeat(int(np.ceil(horizon/action_chunk)), 1)
            action_std = action_scale.view(1, action_size).repeat(int(np.ceil(horizon/action_chunk)), 1)

            # Initialize the mean and std to lower values if we use a pre-trained distilled policy
            if distilled_policy is not None:
                if planner == "mppi":
                    action_mean = mppi_action_mean.clone()
                else:
                    action_mean = torch.zeros_like(action_mean).to(device)
                action_std = (
                    torch.ones_like(action_std).to(device) * residual_std
                )

            baseline_score = None
            baseline_reward = None
            baseline_sequence = None
            baseline_step_reward = None
            if distilled_policy is not None:
                baseline_residuals = torch.zeros(
                    1, horizon, action_size, device=device
                )
                baseline_actions = torch.zeros_like(baseline_residuals)
                baseline_actions[:, 0] = torch.clamp(
                    distilled_policy(current_input[-1]).view(1, -1),
                    action_low.view(1, action_size),
                    action_high.view(1, action_size),
                )
                (
                    baseline_score,
                    baseline_reward,
                    baseline_step_reward,
                    _,
                    _,
                ) = rollout_action_sequences(
                    baseline_actions,
                    current_input,
                    current_action,
                    action_residuals=baseline_residuals,
                    distilled_policy=distilled_policy,
                )
                baseline_sequence = baseline_actions[0].clone()
                baseline_score = baseline_score.item()
                baseline_reward = baseline_reward.item()

            best_sequence = None
            best_score = None
            best_predicted_reward = None
            best_residuals = None
            planner_iterations = (
                cem_iterations if planner == "cem" else mppi_iterations
            )
            for _ in range(planner_iterations):
                # Here is CEM iterations. With action_chunk we need to repeat the action for chunk times (actually, horizon/chunk)
                
                # If we use a pre-trained distilled policy, all of these we will not use. But for retro-compatibility, let's keep them
                # The overhead should be relatively small
                action_sequences = action_mean.unsqueeze(0) + action_std.unsqueeze(0) * torch.randn(random_actions, int(np.ceil(horizon/action_chunk)), action_size, device=device)
                action_sequences = torch.clamp(action_sequences, action_low.view(1, 1, action_size), action_high.view(1, 1, action_size))
                action_sequences = action_sequences.unsqueeze(2)
                action_sequences = torch.repeat_interleave(action_sequences, repeats=action_chunk, dim=2).view(action_sequences.shape[0], -1, action_sequences.shape[-1])

                # In case we have the pre-trained distilled model, we need to initialize the action sequences with a predticet action (plus noise)
                action_residuals = None
                if distilled_policy is not None:
                    action_residuals = action_mean.unsqueeze(0) + action_std.unsqueeze(0) * torch.randn(random_actions, int(np.ceil(horizon/action_chunk)), action_size, device=device) 
                    action_residuals = action_residuals.clamp(
                        -residual_bound, residual_bound
                    )
                    action_residuals = action_residuals.unsqueeze(2)
                    action_residuals = torch.repeat_interleave(action_residuals, repeats=action_chunk, dim=2).view(action_sequences.shape[0], -1, action_sequences.shape[-1])
                    # The unmodified student rollout is always a candidate.
                    action_residuals[0].zero_()

                    predicted_action = distilled_policy(current_input[-1]).view(1, -1)
                    # Repeat
                    predicted_action = torch.repeat_interleave(predicted_action, random_actions, 0)
                    # Perturb
                    predicted_action = predicted_action + action_residuals[:, 0] 
                    predicted_action = torch.clamp(
                        predicted_action,
                        action_low.view(1, action_size),
                        action_high.view(1, action_size),
                    )
                    action_sequences[:, 0, :] = predicted_action

                # Now repeat for action chunk times
                (
                    candidate_scores,
                    predicted_rewards,
                    step_rewards,
                    _,
                    _,
                ) = rollout_action_sequences(
                    action_sequences,
                    current_input,
                    current_action,
                    action_residuals=action_residuals,
                    distilled_policy=distilled_policy,
                )

                num_chunks = horizon // action_chunk
                if planner == "cem":
                    elite_idxs = torch.topk(
                        candidate_scores.view(-1),
                        k=num_elites,
                        largest=True,
                    ).indices
                    if distilled_policy is not None:
                        elites = action_residuals[elite_idxs]
                    else:
                        elites = action_sequences[elite_idxs]

                    elites = elites.view(
                        elites.shape[0],
                        num_chunks,
                        action_chunk,
                        action_size,
                    )[:, :, 0, :]

                    new_mean = elites.mean(dim=0)
                    new_std = elites.std(
                        dim=0, unbiased=False
                    ).clamp_min(1e-3)
                    action_mean = alpha * action_mean + (1-alpha)*new_mean
                    action_std = alpha * action_std + (1-alpha)*new_std
                    if distilled_policy is not None:
                        action_mean = action_mean.clamp(
                            -residual_bound, residual_bound
                        )
                        action_std = action_std.clamp(
                            min=1e-3, max=residual_bound
                        )

                    idx_max_score = torch.argmax(candidate_scores)
                    candidate_score = candidate_scores[idx_max_score].item()
                    if best_score is None or candidate_score > best_score:
                        best_score = candidate_score
                        best_predicted_reward = predicted_rewards[
                            idx_max_score
                        ].item()
                        best_sequence = action_sequences[
                            idx_max_score
                        ].clone()
                        if action_residuals is not None:
                            best_residuals = action_residuals[
                                idx_max_score
                            ].clone()
                        step_reward = step_rewards[idx_max_score]
                else:
                    if distilled_policy is None:
                        raise ValueError(
                            "Policy-guided MPPI requires a distilled policy"
                        )
                    sampled_chunks = action_residuals.view(
                        random_actions,
                        num_chunks,
                        action_chunk,
                        action_size,
                    )[:, :, 0, :]
                    stabilized_scores = (
                        candidate_scores.view(-1)
                        - candidate_scores.max()
                    ) / mppi_temperature
                    weights = torch.softmax(stabilized_scores, dim=0)
                    action_mean = torch.sum(
                        weights.view(-1, 1, 1) * sampled_chunks,
                        dim=0,
                    ).clamp(-residual_bound, residual_bound)

            if planner == "mppi":
                # MPPI executes the weighted mean residual, never the most
                # optimistic individual sample. Re-evaluate that mean sequence
                # before applying the same student fallback used by CEM.
                mean_residuals = torch.repeat_interleave(
                    action_mean.unsqueeze(0),
                    repeats=action_chunk,
                    dim=1,
                )
                mean_sequence = torch.zeros_like(mean_residuals)
                mean_sequence[:, 0] = torch.clamp(
                    distilled_policy(current_input[-1]).view(1, -1)
                    + mean_residuals[:, 0],
                    action_low.view(1, action_size),
                    action_high.view(1, action_size),
                )
                (
                    mean_score,
                    mean_reward,
                    mean_step_reward,
                    _,
                    _,
                ) = rollout_action_sequences(
                    mean_sequence,
                    current_input,
                    current_action,
                    action_residuals=mean_residuals,
                    distilled_policy=distilled_policy,
                )
                best_score = mean_score.item()
                best_predicted_reward = mean_reward.item()
                step_reward = mean_step_reward
                # Match the existing CEM representation: [horizon, action],
                # with no candidate batch dimension.
                best_sequence = mean_sequence[0].clone()
                best_residuals = mean_residuals[0].clone()

                # Warm-start the next MPC decision. Action chunks cannot be
                # shifted by one simulator step without breaking their repeat
                # structure, so only the unchunked case is warm-started.
                if action_chunk == 1:
                    mppi_action_mean = torch.cat(
                        [
                            action_mean[1:],
                            torch.zeros_like(action_mean[-1:]),
                        ],
                        dim=0,
                    )
                else:
                    mppi_action_mean.zero_()

            accepted_search = True
            if (
                baseline_score is not None
                and best_score < baseline_score + improvement_margin
            ):
                accepted_search = False
                best_sequence = baseline_sequence
                best_residuals = torch.zeros_like(best_residuals)
                best_predicted_reward = baseline_reward
                step_reward = baseline_step_reward
                if planner == "mppi":
                    mppi_action_mean.zero_()

            step_reward = step_reward * std_reward + avg_reward
            action = best_sequence[0]
            action = torch.clamp(action, action_low, action_high)
            running_actions[0, step] = action
            
            current_state, reward, done, _  = env.step(action.detach().cpu().numpy())
            current_state = torch.from_numpy(np.expand_dims(np.asarray(current_state), [0,1])).to(device=device)

            encoded_next_state = model.encoder_fwd(current_state)
            predicted_step_reward = step_reward.item()

            episode_actual_rewards.append(float(reward))
            episode_predicted_rewards.append(predicted_step_reward)
            if reward_model is not None:
                true_next_reward = reward_model(
                    encoded_current[:, -1, :],
                    action.view(1, -1),
                    encoded_next_state[:, -1, :],
                ).item()
                true_next_reward = (
                    true_next_reward * std_reward + avg_reward
                )
                episode_true_next_rewards.append(true_next_reward)
            if baseline_reward is not None:
                predicted_advantage = std_reward * (
                    best_predicted_reward - baseline_reward
                )
                episode_predicted_advantages.append(predicted_advantage)
                episode_acceptances.append(float(accepted_search))
            if best_residuals is not None:
                episode_residual_magnitudes.append(
                    best_residuals.square().mean().sqrt().item()
                )

            ep_reward += reward
            step += 1
        
        print(f"Episode reward at episode {e}: {ep_reward}")
        if episode_predicted_advantages:
            print(
                "  Planner diagnostics: "
                f"predicted CEM advantage="
                f"{np.mean(episode_predicted_advantages):.3f}, "
                f"selected residual RMS="
                f"{np.mean(episode_residual_magnitudes):.4f}, "
                f"search acceptance="
                f"{np.mean(episode_acceptances):.3f}, "
                f"one-step reward MAE="
                f"{np.mean(np.abs(np.asarray(episode_predicted_rewards) - np.asarray(episode_actual_rewards))):.3f}"
            )

        diagnostic_actual_rewards.extend(episode_actual_rewards)
        diagnostic_predicted_rewards.extend(episode_predicted_rewards)
        diagnostic_true_next_rewards.extend(episode_true_next_rewards)
        diagnostic_predicted_advantages.extend(episode_predicted_advantages)
        diagnostic_residual_magnitudes.extend(episode_residual_magnitudes)
        diagnostic_acceptances.extend(episode_acceptances)

        if len(episode_actual_rewards) >= horizon:
            discounts = gamma ** np.arange(horizon)
            for start in range(len(episode_actual_rewards) - horizon + 1):
                diagnostic_actual_returns.append(
                    np.dot(
                        episode_actual_rewards[start:start + horizon],
                        discounts,
                    )
                )
                diagnostic_predicted_returns.append(
                    np.dot(
                        episode_predicted_rewards[start:start + horizon],
                        discounts,
                    )
                )
                if episode_true_next_rewards:
                    diagnostic_true_next_returns.append(
                        np.dot(
                            episode_true_next_rewards[start:start + horizon],
                            discounts,
                        )
                    )

        episode_rewards.append(ep_reward)

    def correlation(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        if len(x) < 2 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    print("Final planner diagnostics:")
    print(
        "  one-step reward correlation: "
        f"{correlation(diagnostic_predicted_rewards, diagnostic_actual_rewards):.4f}"
    )
    print(
        f"  horizon-{horizon} reward correlation: "
        f"{correlation(diagnostic_predicted_returns, diagnostic_actual_returns):.4f}"
    )
    print(
        f"  {planner.upper()} search: "
        f"mean predicted advantage={np.mean(diagnostic_predicted_advantages):.4f}, "
        f"mean selected residual RMS={np.mean(diagnostic_residual_magnitudes):.4f}, "
        f"acceptance rate={np.mean(diagnostic_acceptances):.4f}"
    )

    return episode_rewards

#######################################################################################
def create_model(
    action_size, sequence_length, lr, device, fsq_input_size, fsq_dim,
    hidden_dim, L, mlp_encoder=False, wm_rollout_length=5,
    wm_teacher_forcing=0.5, wm_autoregressive_weight=1.0,
    wm_reward_weight=0.1, wm_reward_priority_fraction=0.25,
    wm_reward_high_quantile=0.9, wm_validation_fraction=0.1,
    wm_validation_seed=423
):
    model = LeWorldModel(
        action_dim=action_size,
        max_seq_length=sequence_length,
        encoder_hidden_dim=hidden_dim,
        lr = lr,
        device=device,

        # This world model will be feature based
        with_reward_prediction=True,
        with_terminal_prediction=False,
        feature_base=True,
        fsq_input_size=fsq_input_size,
        fsq_output_size=fsq_dim,
        L=L,
        fsq_encoder=not mlp_encoder,
        autoregressive_rollout_length=wm_rollout_length,
        teacher_forcing_probability=wm_teacher_forcing,
        autoregressive_loss_weight=wm_autoregressive_weight,
        lambd_reward=wm_reward_weight,
        reward_priority_fraction=wm_reward_priority_fraction,
        reward_high_quantile=wm_reward_high_quantile,
        validation_fraction=wm_validation_fraction,
        validation_seed=wm_validation_seed
    )

    return model

#######################################################################################
def load_dataset(dataset_path, model):

    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    new_dataset = {
        "states": np.asarray(dataset["states"]),
        "next_states": np.asarray(dataset["states_n"]),
        "actions": np.asarray(dataset["actions"]),
        "rewards": np.asarray(dataset["rewards"]),
        "terminals": np.asarray(dataset["terminals"])
    }

    # Standardize the reward and retain its scale for online evaluation.
    model.reward_mean = float(np.mean(new_dataset["rewards"]))
    model.reward_std = float(np.std(new_dataset["rewards"]) + 1e-6)
    new_dataset["rewards"] = (
        new_dataset["rewards"] - model.reward_mean
    ) / model.reward_std

    print(f"This dataset has a total of {new_dataset["states"].shape[0]} transitions")

    model.set_dataset(new_dataset)
    print(
        f"Training windows: {len(model.valid_sequence_starts)}, "
        f"validation windows: {len(model.validation_sequence_starts)}"
    )

#######################################################################################
def train(epochs_number, batch_size, model, model_name):
    best_validation_correlation = -float("inf")
    for epoch in range(epochs_number):
        losses = model.train_epoch(batch_size=batch_size)
        print(f"At epoch {epoch}:")
        for key in losses.keys():
            print(f"    -{key}: {losses[key]}")

        validation_losses = model.validate_epoch(batch_size=batch_size)
        if validation_losses is not None:
            print(f"Validation at epoch {epoch}:")
            for key, value in validation_losses.items():
                print(f"    -val_{key}: {value}")
            validation_correlation = float(
                validation_losses["reward_correlation"]
            )
            if validation_correlation > best_validation_correlation:
                best_validation_correlation = validation_correlation
                print("Saving the best validation model now...")
                model.save_model(name=f"{model_name}_best")
        
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
    parser.add_argument('-dm', '--distilled-model-name', help="The name of the distilled policy if we want to use the distilled policy", default=None)
    parser.add_argument('-pn', '--policy-name', help="Saved SAC policy name, required by critic-advantage planning", default=None)
    parser.add_argument('-dn', '--dataset-name', help="The name of the precollected dataset with which we train the world model", default="datasets/dataset.pkl")
    parser.add_argument('-as', '--action-size', help="The action dimension of the env", default=2, type=int)
    parser.add_argument('-is', '--input-size', help="The state dimension of the env", default=8, type=int)
    parser.add_argument('-fd', '--fsq-dim', help="Number of independently quantized FSQ coordinates", default=16, type=int)
    parser.add_argument('-hd', '--hidden-dim', help="Width of categorical embeddings and transformers", default=64, type=int)
    parser.add_argument('-ld', '--levels-dim', help="The dimension of levels for FSQ", default=8, type=int)
    parser.add_argument('-fs', '--fixed-seed', help="If we want to use a fixed seed", default=423, type=int)
    parser.add_argument('-sl', '--sequence-length', help="The max sequence length of the world model", default=4, type=int)
    parser.add_argument('-bs', '--batch-size', help="The batch size during training", default=32, type=int)
    parser.add_argument('-en', '--epochs-number', help="The number of epochs during training", default=5, type=int)
    parser.add_argument('-lr', '--learning-rate', help="The learning rate used during training", default=5e-4, type=float)
    parser.add_argument('--wm-rollout-length', help="Number of autoregressive training steps", default=5, type=int)
    parser.add_argument('--wm-teacher-forcing', help="Probability of feeding the true code during autoregressive training", default=0.5, type=float)
    parser.add_argument('--wm-autoregressive-weight', help="Weight of the autoregressive categorical loss", default=1.0, type=float)
    parser.add_argument('--wm-reward-weight', help="Weight of the joint world-model reward loss", default=0.1, type=float)
    parser.add_argument('--wm-reward-priority-fraction', help="Fraction of windows balanced across terminal and high-reward endings", default=0.25, type=float)
    parser.add_argument('--wm-reward-high-quantile', help="Absolute-reward quantile defining high-reward windows", default=0.9, type=float)
    parser.add_argument('--wm-validation-fraction', help="Fraction of complete episodes reserved for validation", default=0.1, type=float)
    parser.add_argument('--wm-validation-seed', help="Seed used for the episode-level validation split", default=423, type=int)
    parser.add_argument('-ac', '--action-chunk', help="Wether to use action chunk for the CEM algorithm", default=1, type=int)
    parser.add_argument('-hz', '--horizon', help="The horizon of the CEM algorithm", default=5, type=int)
    parser.add_argument('-ra', '--random-actions', help="How many candidates rollout the CEM algorithm will evaluate", default=256, type=int)
    parser.add_argument('--cem-iterations', help="Number of CEM distribution updates", default=3, type=int)
    parser.add_argument('--cem-elite-fraction', help="Fraction of candidates used to update CEM", default=0.1, type=float)
    parser.add_argument('--residual-bound', help="Hard absolute bound on policy action residuals", default=0.05, type=float)
    parser.add_argument('--residual-std', help="Initial standard deviation of policy residuals", default=0.02, type=float)
    parser.add_argument('--residual-penalty', help="Penalty on squared policy residuals", default=1.0, type=float)
    parser.add_argument('--uncertainty-penalty', help="Penalty on normalized categorical prediction entropy", default=0.1, type=float)
    parser.add_argument('--improvement-margin', help="Minimum conservative score improvement required to leave the student policy", default=0.1, type=float)
    parser.add_argument('--planner', choices=['cem', 'mppi'], help="Learned-model test-time planner", default='cem')
    parser.add_argument('--mppi-temperature', help="Softmax temperature for MPPI trajectory weights", default=1.0, type=float)
    parser.add_argument('--mppi-iterations', help="Number of MPPI sampling and mean-update iterations", default=1, type=int)
    parser.add_argument('--planning-objective', choices=['reward', 'critic_advantage', 'hybrid'], help="Planner score objective; critic modes require a SAC value agent", default='reward')
    parser.add_argument('--critic-advantage-weight', help="Weight of normalized SAC critic advantages", default=1.0, type=float)
    parser.add_argument('--real-env-cem-episodes', help="Number of expensive oracle real-simulator CEM episodes; zero disables it", default=0, type=int)
    parser.add_argument('--real-env-cem-candidates', help="Candidates per iteration for oracle real-simulator CEM", default=64, type=int)
    parser.add_argument('-vi', '--visualize-inference', help="If we want to see the agent in the environment", action=argparse.BooleanOptionalAction, default=False)

    # RM specific argument
    parser.add_argument('-rm', '--rm-model-name', help="The name of the model with which we save it", type=str, default="reward_model")
    parser.add_argument('-rs', '--rm-batch-size', help="The batch size during training", default=32, type=int)
    parser.add_argument('-rn', '--rm-epochs-number', help="The number of epochs during training", default=1000, type=int)
    parser.add_argument('-rr', '--rm-learning-rate', help="The learning rate used during training", default=1e-4, type=float)
    parser.add_argument('-rl', '--rm-num-hidden-layer', help="Number of hidden layers of the reward model", default=5, type=int)
    parser.add_argument('-rh', '--rm-hidden-size', help="Size of the hidden layers of the reward model", default=512, type=int)
    parser.add_argument('-rp', '--rm-latent-prob', help="The probability to sample a latent distribution", default=0.3, type=float)
    parser.add_argument('--rm-ranking-loss-weight', help="Weight of the pairwise reward-ranking loss", default=0.1, type=float)

    args = parser.parse_args()

    print("")
    print("####")
    print(Fore.CYAN + "Creating the model.." + Style.RESET_ALL)
    print("...")
    model = create_model(
        args.action_size, args.sequence_length, args.learning_rate,
        fsq_input_size=args.input_size, fsq_dim=args.fsq_dim,
        hidden_dim=args.hidden_dim, L=args.levels_dim, device=device,
        wm_rollout_length=args.wm_rollout_length,
        wm_teacher_forcing=args.wm_teacher_forcing,
        wm_autoregressive_weight=args.wm_autoregressive_weight,
        wm_reward_weight=args.wm_reward_weight,
        wm_reward_priority_fraction=args.wm_reward_priority_fraction,
        wm_reward_high_quantile=args.wm_reward_high_quantile,
        wm_validation_fraction=args.wm_validation_fraction,
        wm_validation_seed=args.wm_validation_seed
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

        amp_enabled = device.type == "cuda"

        # Load the distilled policy in case we want to use it
        distilled_policy = None
        if args.distilled_model_name is not None:
           distilled_policy = PolicyEmbedding(state_dim=args.fsq_dim, output_dim=args.action_size, levels=args.levels_dim).to(device)
           load_student(student=distilled_policy, model_name=args.distilled_model_name)

        value_agent = None
        if args.planning_objective in {"critic_advantage", "hybrid"}:
            value_agent = load_sac_value_agent(
                policy_name=args.policy_name,
                state_dim=args.input_size,
                action_dim=args.action_size,
                device=device,
            )

        assert args.horizon % args.action_chunk == 0, "Horizon should be divisible for the number of the action chunks"

        evaluate_distilled_policy_in_real_env(
            env=env,
            model=model,
            distilled_policy=distilled_policy,
            device=device
        )

        evaluate_one_step_world_model(
            env=env,
            model=model,
            distilled_policy=distilled_policy,
            device=device
        )

        evaluate_distilled_policy_in_world_model(
            env=env,
            model=model,
            distilled_policy=distilled_policy,
            device=device
        )

        evaluate_horizon_world_model(
            env=env,
            model=model,
            distilled_policy=distilled_policy,
            device=device,
            horizon=args.horizon
        )

        if args.real_env_cem_episodes > 0:
            evaluate_real_environment_cem(
                env=env,
                model=model,
                distilled_policy=distilled_policy,
                device=device,
                action_size=args.action_size,
                random_actions=args.real_env_cem_candidates,
                horizon=args.horizon,
                action_chunk=args.action_chunk,
                num_eval_episode=args.real_env_cem_episodes,
                cem_iterations=args.cem_iterations,
                elite_fraction=args.cem_elite_fraction,
                residual_bound=args.residual_bound,
                residual_std=args.residual_std,
                residual_penalty=args.residual_penalty,
                uncertainty_penalty=args.uncertainty_penalty,
                improvement_margin=args.improvement_margin,
            )

        with torch.inference_mode():
            episode_rewards = evaluate_wm(
                env=env,
                model=model,
                reward_model=None,
                action_size=model.action_dim,
                random_actions=args.random_actions,
                horizon=args.horizon,
                dataset_path=args.dataset_name,
                device=device,
                max_episode_timesteps=env._max_episode_timesteps,
                action_chunk=args.action_chunk,
                distilled_policy=distilled_policy,
                cem_iterations=args.cem_iterations,
                elite_fraction=args.cem_elite_fraction,
                residual_bound=args.residual_bound,
                residual_std=args.residual_std,
                residual_penalty=args.residual_penalty,
                uncertainty_penalty=args.uncertainty_penalty,
                improvement_margin=args.improvement_margin,
                planner=args.planner,
                mppi_temperature=args.mppi_temperature,
                mppi_iterations=args.mppi_iterations,
                planning_objective=args.planning_objective,
                critic_advantage_weight=args.critic_advantage_weight,
                value_agent=value_agent
            )
            print(f"Average reward of the approach: {np.mean(episode_rewards)}")
