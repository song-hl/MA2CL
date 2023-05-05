import cv2
from gym.spaces.box import Box
from .drone_envs.FlockAviary import FlockAviary
from .drone_envs.LeaderFollowerAviary import LeaderFollowerAviary
from .drone_envs.MeetupAviary import MeetupAviary

from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    ActionType,
    ObservationType,
)
import numpy as np
from .multiagentenv import MultiAgentEnv
from gym_pybullet_drones.utils.enums import DroneModel

import pybullet as p
from PIL import Image
import os
from pathlib import Path

class DroneEnv(MultiAgentEnv):
    def __init__(self, env_args):
        env_fn = {
            "flock": FlockAviary,
            "leader_follower": LeaderFollowerAviary,
            "meetup": MeetupAviary,
        }
        initial_xyz = self.initial_xyzs(env_args["is_initial_xyz"], env_args["num_agents"], env_args["map_name"], env_args["action_type"])
        assert hasattr(ActionType, env_args["action_type"])
        assert hasattr(ObservationType, env_args["observation_type"])
        self.env = env_fn[env_args["map_name"]](
            num_drones=env_args["num_agents"],
            freq=240,
            aggregate_phy_steps=5,
            obs=ObservationType[env_args["observation_type"]],
            act=ActionType[env_args["action_type"]],
            initial_xyzs=initial_xyz,
            neighbourhood_radius=0.1
        )
        self.n_agents = self.env.NUM_DRONES
        shp = self.env.observation_space[0].shape
        self.n_actions = self.env.action_space[0].shape[0]
        self.observation_space = [Box(0, 255, shp[::-1], np.uint8)] * self.n_agents
        self.action_space = [self.env.action_space[i] for i in range(self.n_agents)]
        self.use_camera_state = env_args["use_camera_state"]
        if self.use_camera_state :
            self.CLIENT = p.connect(p.DIRECT)
            self.VID_WIDTH = int(96)
            self.VID_HEIGHT = int(72)
            self.FRAME_PER_SEC = 24
            self.CAPTURE_FREQ = int(self.env.SIM_FREQ/self.FRAME_PER_SEC)
            self.CAM_VIEW = p.computeViewMatrixFromYawPitchRoll(distance=2,
                                                                yaw=0,
                                                                pitch=-90,
                                                                roll=0,
                                                                cameraTargetPosition=[0, 0, 0],
                                                                upAxisIndex=2,
                                                                physicsClientId=self.env.CLIENT
                                                                )
            self.CAM_PRO = p.computeProjectionMatrixFOV(fov=60.0,
                                                        aspect=self.VID_WIDTH/self.VID_HEIGHT,
                                                        nearVal=0.1,
                                                        farVal=3.0
                                                        )
            # FIXME: share_observation_space is not used
            self.share_observation_space = [
                Box(0, 255, [4, self.VID_WIDTH, self.VID_HEIGHT], np.uint8)
            ] * self.n_agents
        else:
            self.share_observation_space = [
                Box(0, 255, [shp[-1] * self.n_agents, shp[1], shp[0]], np.uint8)
            ] * self.n_agents

    def initial_xyzs(self, is_initial_xyz, num_agents, map_name, action_type):
        """ initial position of agents 

        Args:
            is_initial_xyz (bool): if True,initial positions for agents, otherwise None
            num_agents (int): number of agents
            map_name (str): name of map
            action_type (str): type of action

        Returns:
            array: initial positions for agents
        """
        if is_initial_xyz:
            initial_xyz = np.random.uniform(0, 1, size=(num_agents, 3))
        else:
            initial_xyz = None  # use initial strategy in env when is_initial_xyz is False
        """
        if initial_xyzs is None:
            self.INIT_XYZS = np.vstack([np.array([x*4*self.L for x in range(self.NUM_DRONES)]), \
                                        np.array([y*4*self.L for y in range(self.NUM_DRONES)]), \
                                        np.ones(self.NUM_DRONES) * (self.COLLISION_H/2-self.COLLISION_Z_OFFSET+.1)]).transpose().reshape(self.NUM_DRONES, 3)
        """
        return initial_xyz

    def get_state(self):
        [w, h, rgb, dep, seg] = p.getCameraImage(width=self.VID_WIDTH,
                                                 height=self.VID_HEIGHT,
                                                 shadow=1,
                                                 viewMatrix=self.CAM_VIEW,
                                                 projectionMatrix=self.CAM_PRO,
                                                 renderer=p.ER_TINY_RENDERER,
                                                 flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                                 physicsClientId=self.env.CLIENT
                                                 )
        rgb = np.reshape(rgb, (h, w, 4))

        dep_im = (dep * 255).astype(np.uint8)
        # dep = np.reshape(dep, (h, w))
        # seg = np.reshape(seg, (h, w))
        rgb[:, :, 3] = dep_im
        # rgb = rgb.transpose(2, 0, 1)
        # a = rgb[0, :, :]
        # b = rgb[1, :, :]
        # c = rgb[2, :, :]
        # d = rgb[3, :, :]
        # frame_dir = Path.cwd() / "results"/"frames"
        # frame_dir.mkdir(parents=True, exist_ok=True)
        # (Image.fromarray(np.reshape(rgb, (h, w, 4)), 'RGBA')).save(frame_dir / f"test_{self.VID_WIDTH}_{self.VID_HEIGHT}.png")
        # self.FRAME_NUM += 1
        return rgb

    def reset(self):
        # TODO : reset the environment
        obs_dict = self.env.reset()
        obs_list = [np.transpose(obs_dict[i], (2, 1, 0)) for i in range(self.n_agents)]
        a = [np.concatenate(obs_list, axis=0)] * self.n_agents
        if self.use_camera_state:
            state = [np.transpose(self.get_state(), (2, 1, 0))] * self.n_agents
        else:
            state = [np.concatenate(obs_list, axis=0)] * self.n_agents
        return (
            obs_list,
            state,
            self.get_avail_actions(),
        )

    def step(self, actions):
        action_dcit = {i: actions[i] for i in range(self.n_agents)}
        obs_dict, reward_dict, done_dict, info_dict = self.env.step(action_dcit)
        obs_list = [np.transpose(obs_dict[i], (2, 1, 0)) for i in range(self.n_agents)]
        if self.use_camera_state:
            state = [np.transpose(self.get_state(), (2, 1, 0))] * self.n_agents
        else:
            state = [np.concatenate(obs_list, axis=0)] * self.n_agents
        return (
            obs_list,
            state,
            np.array([[sum(reward_dict.values())] * self.n_agents]).T,
            [done_dict[i] for i in range(self.n_agents)],
            [info_dict[i] for i in range(self.n_agents)],
            self.get_avail_actions(),
        )

    def get_avail_actions(self):  # all actions are always available
        return np.ones(shape=(self.n_agents, self.n_actions,))

    def seed(self, seed):
        pass

    def close(self):
        self.env.close()

    def render(self):
        self.env.render()

    def get_spaces(self):
        return self.observation_space, self.share_observation_space, self.action_space

    def get_num_agents(self):
        return self.n_agents
