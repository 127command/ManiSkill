"""
Code for a minimal environment/task with just a robot being loaded. We recommend copying this template and modifying as you need.

At a high-level, ManiSkill tasks can minimally be defined by what agents/actors are 
loaded, how agents/actors are randomly initialized during env resets, how goals are randomized and parameterized in observations, and success conditions

Environment reset is comprised of running two functions, `self.reconfigure` and `self.initialize_episode`, which is auto
run by ManiSkill. As a user, you can override a number of functions that affect reconfiguration and episode initialization.

Reconfiguration will reset the entire environment scene and allow you to load/swap assets and agents.

Episode initialization will reset the poses of all actors, articulations, and agents,
in addition to initializing any task relevant data like a goal

See comments for how to make your own environment and what each required function should do. If followed correctly you can easily build a 
task that can simulate on the CPU and be parallelized on the GPU without having to manage GPU memory and parallelization apart from some 
code that need to be written in batched mode (e.g. reward, success conditions)

For a minimal implementation of a simple task, check out
mani_skill2/envs/tasks/push_cube.py which is annotated with comments to explain how it is implemented
"""

from collections import OrderedDict
from typing import Any, Dict

import numpy as np
import torch

from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.sapien_utils import (  # import various useful utilities for working with sapien
    look_at,
)


class CustomEnv(BaseEnv):
    # in the __init__ function you can pick a default robot your task should use e.g. the panda robot
    def __init__(self, *args, robot_uid="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uid=robot_uid, **kwargs)

    """
    Reconfiguration Code

    below are all functions involved in reconfiguration during environment reset called in the same order. As a user
    you can change these however you want for your desired task. These functions will only ever be called once in general. In CPU simulation,
    for some tasks these may need to be called multiple times if you need to swap out object assets. In GPU simulation these will only ever be called once.
    """

    def _register_sensors(self):
        # To customize the sensors that capture images/pointclouds for the environment observations,
        # simply define a CameraConfig as done below for Camera sensors. You can add multiple sensors by returning a list
        pose = look_at(
            eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1]
        )  # look_at is a utility to get the pose of a camera that looks at a target

        # to see what all the sensors capture in the environment for observations, run env.render_cameras() which returns an rgb array you can visualize
        return [
            CameraConfig("base_camera", pose.p, pose.q, 128, 128, np.pi / 2, 0.01, 10)
        ]

    def _register_render_cameras(self):
        # this is just like _register_sensors, but for adding cameras used for rendering when you call env.render() when render_mode="rgb_array" or env.render_rgb_array()
        pose = look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose.p, pose.q, 512, 512, 1, 0.01, 10)

    def _setup_sensors(self):
        # default code here will setup all sensors. You can add additional code to change the sensors e.g.
        # if you want to randomize camera positions
        return super()._setup_sensors()

    def _setup_lighting(self):
        # default code here will setup all lighting. You can add additional code to change the lighting e.g.
        # if you want to randomize lighting in the scene
        return super()._setup_lighting()

    def _load_agent(self):
        # this code loads the agent into the current scene. You can usually ignore this function by deleting it or calling the inherited
        # BaseEnv._load_agent function
        super()._load_agent()

    def _load_actors(self):
        # here you add various objects (called actors). If your task was to push a ball, you may add a dynamic sphere object on the ground
        pass

    def _load_articulations(self):
        # here you add various articulations. If your task was to open a drawer, you may add a drawer articulation into the scene
        pass

    """
    Episode Initialization Code

    below are all functions involved in episode initialization during environment reset called in the same order. As a user
    you can change these however you want for your desired task.
    """

    def _initialize_actors(self):
        pass

    def _initialize_task(self):
        # we highly recommend to generate some kind of "goal" information to then later include in observations
        # goal can be parameterized as a state (e.g. target pose of a object)
        pass

    """
    Modifying observations, goal parameterization, and success conditions for your task

    the code below all impact some part of `self.step` function
    """

    def _get_obs_extra(self):
        # should return an OrderedDict of additional observation data for your tasks
        # this will be included as part of the observation in the "extra" key when obs_mode="state_dict" or any of the visual obs_modes
        # and included as part of a flattened observation when obs_mode="state"
        return OrderedDict()

    def evaluate(self, obs: Any):
        # should return a dictionary containing "success": bool indicating if the environment is in success state or not. The value here is also what the sparse reward is
        # for the task. You may also include additional keys which will populate the info object returned by self.step.
        # note that as everything is batched, you must return a batched array of self.num_envs booleans (or 0/1 values) as done in the example below
        return {"success": torch.zeros(self.num_envs, device=self.device)}

    def compute_dense_reward(self, obs: Any, action: np.ndarray, info: Dict):
        # you can optionally provide a dense reward function by returning a scalar value here. This is used when reward_mode="dense"
        # note that as everything is batched, you must return a batch of of self.num_envs rewards as done in the example below
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs: Any, action: np.ndarray, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward