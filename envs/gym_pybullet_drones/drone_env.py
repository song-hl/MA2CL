import cv2
from gym.spaces.box import Box
from gym_pybullet_drones.envs.multi_agent_rl.FlockAviary import FlockAviary
from gym_pybullet_drones.envs.multi_agent_rl.LeaderFollowerAviary import (
    LeaderFollowerAviary,
)
from gym_pybullet_drones.envs.multi_agent_rl.MeetupAviary import MeetupAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    ActionType,
    ObservationType,
)
import numpy as np
from .multiagentenv import MultiAgentEnv
from gym_pybullet_drones.utils.enums import DroneModel


class DroneEnv(MultiAgentEnv):
    def __init__(self, env_args):
        env_fn = {
            "flock": FlockAviary,
            "leader_follower": LeaderFollowerAviary,
            "meetup": MeetupAviary,
        }
        if env_args["is_initial_xyz"]:
            initial_xyz = np.random.uniform(0, 1, size=(env_args["num_agents"], 3))
        else:
            initial_xyz = None
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
        self.share_observation_space = [
            Box(0, 255, [shp[-1] * self.n_agents, shp[1], shp[0]], np.uint8)
        ] * self.n_agents

    def reset(self):
        obs_dict = self.env.reset()
        obs_list = [np.transpose(obs_dict[i], (2, 1, 0)) for i in range(self.n_agents)]
        return (
            obs_list,
            [np.concatenate(obs_list, axis=0)] * self.n_agents,
            self.get_avail_actions(),
        )

    def step(self, actions):
        action_dcit = {i: actions[i] for i in range(self.n_agents)}
        obs_dict, reward_dict, done_dict, info_dict = self.env.step(action_dcit)
        obs_list = [np.transpose(obs_dict[i], (2, 1, 0)) for i in range(self.n_agents)]
        return (
            obs_list,
            [np.concatenate(obs_list, axis=0)] * self.n_agents,
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
