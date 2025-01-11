from collections import defaultdict
from typing import Callable
import numpy as np
import torch

def update_history_obs(obs, history_obs, history_window):
    if len(history_obs) > 0:
        history_obs = history_obs[1:]
    history_obs.append(obs)
    while len(history_obs) < history_window:
        history_obs.append(obs)
    return history_obs

def evaluate(n: int, sample_fn: Callable, history_window: int, action_dim: int, eval_envs):
    """
    Evaluate the agent on the evaluation environments for at least n episodes.

    Args:
        n: The minimum number of episodes to evaluate.
        sample_fn: The function to call to sample actions from the agent by passing in the observations
        eval_envs: The evaluation environments.

    Returns:
        A dictionary containing the evaluation results.
    """

    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, info = eval_envs.reset()
        eps_count = 0
        history_obs = []
        history_obs = update_history_obs(obs, history_obs, history_window)
        if action_dim > 0:
            action_history = np.zeros((obs.shape[0], action_dim))
        while eps_count < n:
            if action_dim > 0:
                action = sample_fn(np.concatenate([np.concatenate(history_obs, axis=-1), action_history], axis=-1))
            else:
                action = sample_fn(np.concatenate(history_obs, axis=-1))
            obs, _, _, truncated, info = eval_envs.step(action)
            history_obs = update_history_obs(obs, history_obs, history_window)
            # note as there are no partial resets, truncated is True for all environments at the same time
            if truncated.any():
                if isinstance(info["final_info"], dict):
                    for k, v in info["final_info"]["episode"].items():
                        eval_metrics[k].append(v.float().cpu().numpy())
                else:
                    for final_info in info["final_info"]:
                        for k, v in final_info["episode"].items():
                            eval_metrics[k].append(v)
                eps_count += eval_envs.num_envs
    for k in eval_metrics.keys():
        eval_metrics[k] = np.stack(eval_metrics[k])
    return eval_metrics
