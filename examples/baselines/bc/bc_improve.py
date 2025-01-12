import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import tyro
from mani_skill.utils import gym_utils
from mani_skill.utils.io_utils import load_json
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from behavior_cloning.evaluate import evaluate
from behavior_cloning.make_env import make_eval_envs


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    env_id: str = "PegInsertionSide-v0"
    """the id of the environment"""
    demo_path: str = "data/ms2_official_demos/rigid_body/PegInsertionSide-v0/trajectory.state.pd_ee_delta_pose.h5"
    """the path of demo dataset (pkl or h5)"""
    num_demos: Optional[int] = None
    """number of trajectories to load from the demo dataset"""
    total_iters: int = 1_000_000
    """total timesteps of the experiment"""
    batch_size: int = 1024
    """the batch size of sample from the replay memory"""

    # Behavior cloning specific arguments
    lr: float = 3e-4
    """the learning rate for the actor"""
    normalize_states: bool = False
    """if toggled, states are normalized to mean 0 and standard deviation 1"""

    # Environment/experiment specific arguments
    max_episode_steps: Optional[int] = None
    """Change the environments' max_episode_steps to this value. Sometimes necessary if the demonstrations being imitated are too short. Typically the default
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning agents can learn faster."""
    log_freq: int = 1000
    """the frequency of logging the training metrics"""
    eval_freq: int = 1000
    """the frequency of evaluating the agent on the evaluation environments"""
    save_freq: Optional[int] = None
    """the frequency of saving the model checkpoints. By default this is None and will only save checkpoints based on the best evaluation metrics."""
    num_eval_episodes: int = 100
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 10
    """the number of parallel environments to evaluate the agent on"""
    sim_backend: str = "cpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""
    num_dataload_workers: int = 0
    """the number of workers to use for loading the training data in the torch dataloader"""
    control_mode: str = "pd_joint_delta_pos"
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""

    # additional tags/configs for logging purposes to wandb and shared comparisons with other algorithms
    demo_type: Optional[str] = None

    chunk_size: int = 5


# taken from here
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Segmentation/MaskRCNN/pytorch/maskrcnn_benchmark/data/samplers/iteration_based_batch_sampler.py
class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


class ManiSkillDataset(Dataset):
    def __init__(
        self,
        dataset_file: str,
        device,
        load_count=-1,
        normalize_states=False,
        chunk_size=0,
    ) -> None:
        self.dataset_file = dataset_file
        # for details on how the code below works, see the
        # quick start tutorial
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]

        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]

        self.next_observations = []
        self.observations = []
        self.actions = []
        self.dones = []
        self.total_frames = 0
        self.device = device
        self.chunk_size = chunk_size
        assert self.chunk_size > 0
        if load_count is None:
            load_count = len(self.episodes)
        print(f"Loading {load_count} episodes")

        for eps_id in tqdm(range(load_count)):
            eps = self.episodes[eps_id]
            trajectory = self.data[f"traj_{eps['episode_id']}"]
            trajectory = load_h5_data(trajectory)

            # we use :-1 here to ignore the last observation as that
            # is the terminal observation which has no actions
            obs = trajectory["obs"][:-1]
            next_obs = trajectory["obs"][1:]
            action = trajectory["actions"]
            done = trajectory["success"].reshape(-1, 1)

            history_action = np.concatenate([np.zeros_like(action[:1]), action[:-1].copy()], axis=0)
            delta_action = action - history_action
            obs = np.concatenate([obs, history_action], axis=-1)
            action = np.concatenate([action, delta_action], axis=-1)

            self.observations.append(obs)
            self.next_observations.append(next_obs)
            self.actions.append(action)
            self.dones.append(done)

        self.observations = np.vstack(self.observations)
        self.next_observations = np.vstack(self.next_observations)
        self.actions = np.vstack(self.actions)
        self.dones = np.vstack(self.dones)
        assert self.observations.shape[0] == self.actions.shape[0]
        assert self.next_observations.shape[0] == self.actions.shape[0]
        assert self.dones.shape[0] == self.actions.shape[0]
        self.action_dim = self.actions.shape[-1] // 2
        # self.delta_observations = self.next_observations - self.observations

        self.action_mean = None
        self.action_std = None
        self.obs_mean = None
        self.obs_std = None
        self.delta_obs_mean = None
        self.delta_obs_std = None
        self.normalized = normalize_states
        if normalize_states:
            print("Normalizing actions")
            self.action_mean, self.action_std = self.get_state_stats("action")
            print("Stats shape", self.action_mean.shape, self.action_std.shape)
            self.actions = (self.actions - self.action_mean) / self.action_std
            self.obs_mean, self.obs_std = self.get_state_stats("obs")
            # print("Normalizing delta obs")
            # self.delta_obs_mean, self.delta_obs_std = self.get_state_stats("delta obs")
            # print("Stats shape", self.delta_obs_mean.shape, self.delta_obs_std.shape)
            # self.delta_observations = (self.delta_observations - self.delta_obs_mean) / self.delta_obs_std

    def get_state_stats(self, state_name):
        if not self.normalized:
            return None, None
        if state_name == "action":
            if self.action_mean is not None and self.action_std is not None:
                return self.action_mean, self.action_std
            return np.mean(self.actions, axis=0), np.std(self.actions, axis=0)
        if state_name == "obs":
            if self.obs_mean is not None and self.obs_std is not None:
                return self.obs_mean, self.obs_std
            return np.mean(self.observations, axis=0), np.std(self.observations, axis=0)
        if state_name == "delta obs":
            if self.delta_obs_mean is not None and self.delta_obs_std is not None:
                return self.delta_obs_mean, self.delta_obs_std
            return np.mean(self.delta_observations, axis=0), np.std(self.delta_observations, axis=0) + 1e-8

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        action = torch.from_numpy(self.actions[idx]).float().to(device=self.device)
        obs = torch.from_numpy(self.observations[idx]).float().to(device=self.device)
        done = torch.from_numpy(self.dones[idx]).to(device=self.device)
        # delta_obs = torch.from_numpy(self.delta_observations[idx]).float().to(device=self.device)
        future_actions_list = []
        index = idx
        for i in range(self.chunk_size):
            if index + 1 < self.actions.shape[0] and self.dones[index] < 1e-3:
                index += 1
            future_actions_list.append(torch.from_numpy(self.actions[index][..., self.action_dim:]))
        future_actions = torch.concat(future_actions_list, dim=-1).to(device=self.device)
        return obs, action, done, future_actions


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, action_mean, action_std, obs_mean, obs_std, chunk_size):
        super(Actor, self).__init__()

        self.action_mean = action_mean
        self.action_std = action_std
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.chunk_size = chunk_size
        assert action_mean.shape[0] == action_dim * 2
        assert action_std.shape[0] == action_dim * 2
        assert obs_mean.shape[0] == state_dim + action_dim
        assert obs_std.shape[0] == state_dim + action_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim * (2 + chunk_size)),
        )

        self.update_gate = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        assert state.shape[-1] == self.state_dim + self.action_dim
        state_dim = self.state_dim
        action_dim = self.action_dim
        actions = self.net(state[..., :state_dim]) # [action, delta_action]
        # sigma = torch.sigmoid(self.update_gate(state))
        sigma = torch.clip(self.update_gate(state), min=-0.5, max=0.5) + 0.5
        delta_action = (state[..., state_dim:] + (actions[..., action_dim:action_dim*2].clone().detach()*self.action_std[action_dim:]+self.action_mean[action_dim:]) - self.action_mean[:action_dim]) / self.action_std[:action_dim]
        a_hat = sigma * actions[..., 0:action_dim].clone().detach() + (1-sigma) * delta_action

        return a_hat, actions


def save_ckpt(run_name, tag):
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    torch.save(
        {
            "actor": actor.state_dict(),
        },
        f"runs/{run_name}/checkpoints/{tag}.pt",
    )


if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    if args.demo_path.endswith(".h5"):
        import json

        json_file = args.demo_path[:-2] + "json"
        with open(json_file, "r") as f:
            demo_info = json.load(f)
            if "control_mode" in demo_info["env_info"]["env_kwargs"]:
                control_mode = demo_info["env_info"]["env_kwargs"]["control_mode"]
            elif "control_mode" in demo_info["episodes"][0]:
                control_mode = demo_info["episodes"][0]["control_mode"]
            else:
                raise Exception("Control mode not found in json")
            assert (
                control_mode == args.control_mode
            ), f"Control mode mismatched. Dataset has control mode {control_mode}, but args has control mode {args.control_mode}"

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode="state",
        render_mode="rgb_array",
    )
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        video_dir=f"runs/{run_name}/videos" if args.capture_video else None,
    )

    if args.track:
        import wandb

        config = vars(args)
        config["eval_env_cfg"] = dict(
            **env_kwargs,
            num_envs=args.num_eval_envs,
            env_id=args.env_id,
            env_horizon=gym_utils.find_max_episode_steps_value(envs),
        )
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            save_code=True,
            group="BehaviorCloning",
            tags=["behavior_cloning"],
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    ds = ManiSkillDataset(
        args.demo_path,
        device=device,
        load_count=args.num_demos,
        normalize_states=args.normalize_states,
        chunk_size=args.chunk_size,
    )

    obs, _ = envs.reset(seed=args.seed)

    sampler = RandomSampler(ds)
    batchsampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    itersampler = IterationBasedBatchSampler(batchsampler, args.total_iters)
    dataloader = DataLoader(
        ds, batch_sampler=itersampler, num_workers=args.num_dataload_workers
    )
    state_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]
    action_mean, action_std = None, None
    obs_mean, obs_std = None, None
    if args.normalize_states:
        action_mean, action_std = ds.get_state_stats("action")
        action_mean = torch.from_numpy(action_mean).to(device)
        action_std = torch.from_numpy(action_std).to(device)
        obs_mean, obs_std = ds.get_state_stats("obs")
        obs_mean = torch.from_numpy(obs_mean).to(device)
        obs_std = torch.from_numpy(obs_std).to(device)
    actor = Actor(
        state_dim, action_dim, action_mean, action_std, obs_mean, obs_std, args.chunk_size
    )
    actor = actor.to(device=device)
    optimizer = optim.Adam(actor.parameters(), lr=args.lr)
    lr_scheduler = lr_scheduler.MultiStepLR(optimizer, [int(args.total_iters * 0.7)], gamma=0.1, last_epoch=-1)

    best_eval_metrics = defaultdict(float)

        # obs_mean, obs_std = ds.get_state_stats("obs")
        # obs_mean = torch.from_numpy(obs_mean).to(device)
        # obs_std = torch.from_numpy(obs_std).to(device)
    losses = []
    beta = 0.2
    for iteration, batch in tqdm(enumerate(dataloader)):
        log_dict = {}
        obs, total_action, _, future_actions = batch
        action = total_action[..., 0:action_dim]
        delta_action = total_action[..., action_dim:2*action_dim]
        obs += torch.randn_like(obs) * obs_std * 0.1
        a_hat, pred_actions = actor(obs)

        optimizer.zero_grad()
        a_hat_loss = F.mse_loss(a_hat, action)
        action_loss = F.mse_loss(pred_actions[..., 0:action_dim], action)
        delta_action_loss = F.mse_loss(pred_actions[..., action_dim:action_dim*2], delta_action)
        future_action_loss = F.mse_loss(pred_actions[..., action_dim*2:], future_actions)
        # loss = a_hat_loss + (action_loss + delta_action_loss) * beta
        loss = (a_hat_loss + action_loss + delta_action_loss) / 3.0 + future_action_loss * 0.3
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        losses.append(loss.item())

        if (iteration + 1) % args.log_freq == 0:
            print(f"Iteration {iteration}, loss: {np.mean(losses[-100:])}")
            writer.add_scalar(
                "charts/learning_rate", optimizer.param_groups[0]["lr"], iteration
            )
            writer.add_scalar("losses/total_loss", loss.item(), iteration)

        if (iteration + 1) % args.eval_freq == 0 and iteration > 0:
            actor.eval()
            # print(actor.beta)

            def sample_fn(obs, update=False):
                if isinstance(obs, np.ndarray):
                    obs = torch.from_numpy(obs).float().to(device)
                # obs = (obs - obs_mean) / obs_std
                a_hat, pred_actions = actor(obs)
                if update:
                    action = pred_actions[..., :action_dim]
                else:
                    action = a_hat[..., :action_dim]
                if args.normalize_states:
                    action = action * action_std[:action_dim] + action_mean[:action_dim]
                if args.sim_backend == "cpu":
                    action = action.cpu().numpy()
                return action

            with torch.no_grad():
                eval_metrics = evaluate(args.num_eval_episodes, sample_fn, envs, envs.single_action_space.shape[0], True, False, True)
            actor.train()
            print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval/{k}", eval_metrics[k], iteration)
                print(f"{k}: {eval_metrics[k]:.4f}")

            save_on_best_metrics = ["success_once", "success_at_end"]
            for k in save_on_best_metrics:
                if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                    best_eval_metrics[k] = eval_metrics[k]
                    save_ckpt(run_name, f"best_eval_{k}_{eval_metrics[k]:.4f}_{str(iteration)}")
                    print(
                        f"New best {k}_rate: {eval_metrics[k]:.4f}. Saving checkpoint."
                    )

        if args.save_freq is not None and iteration % args.save_freq == 0:
            save_ckpt(run_name, str(iteration))
    envs.close()
    if args.track:
        wandb.finish()
