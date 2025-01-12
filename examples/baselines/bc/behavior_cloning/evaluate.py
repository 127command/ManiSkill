from collections import defaultdict
from typing import Callable
import numpy as np
import torch

def evaluate(n: int, sample_fn: Callable, eval_envs, action_dim, history_action=False, delta_action=False, update_action=False, gamma=0.8, delta_action_num=5):
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
        last_action = np.zeros((obs.shape[0], action_dim))
        action_count = 0
        while eps_count < n:
            if history_action:
                input_state = np.concatenate([obs, last_action], axis=-1)
            else:
                input_state = obs

            if update_action:
                total_action = sample_fn(input_state, action_count == 0)
            else:
                total_action = sample_fn(input_state)

            if delta_action and action_count != 0:
                # gamma = (np.cos(np.pi * action_count / delta_action_num) + 1) * 0.5 * (max_gamma - min_gamma) + min_gamma
                action = (total_action[:, action_dim:2*action_dim] + last_action) * gamma + total_action[:, :action_dim] *  (1-gamma)
            else:
                action = total_action[:, :action_dim]
            action_count = (action_count + 1) % delta_action_num
            last_action = action.copy()
            obs, _, _, truncated, info = eval_envs.step(action)
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
