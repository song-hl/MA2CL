import numpy as np
import supersuit as ss
from gym.spaces.box import Box
from pettingzoo.butterfly import (
    cooperative_pong_v5,
    knights_archers_zombies_v10,
    pistonball_v6,
    prospector_v4,
)
import cv2
from .multiagentenv import MultiAgentEnv


class ButterflyEnv(MultiAgentEnv):
    def __init__(self, all_args):
        map_name = all_args["map_name"]
        self.map_name = map_name
        episode_length = all_args["episode_length"]
        self.obs_size = all_args["pre_transform_image_size"]
        self.state_size = all_args["pre_transform_image_size"]
        self.stacked_frames = all_args["stacked_frames"]
        if all_args["use_stacked_frames"] is False:
            self.stacked_frames = 1
        if map_name == "prospector":
            env = prospector_v4.parallel_env(
                ind_reward=0.0,
                group_reward=0.5,
                other_group_reward=0.5,
                max_cycles=episode_length,
            )
        elif map_name == "knights_archers_zombies":
            env = knights_archers_zombies_v10.parallel_env(
                vector_state=False, max_cycles=episode_length,
            )
        elif map_name == "cooperative_pong":
            env = cooperative_pong_v5.parallel_env(max_cycles=episode_length,)
        elif map_name == "pistonball":
            env = pistonball_v6.parallel_env(max_cycles=episode_length)
        else:
            raise ValueError("map_name is not valid")
        env = ss.color_reduction_v0(env)
        env = ss.pad_action_space_v0(env)
        env = ss.pad_observations_v0(env)
        env = ss.resize_v1(
            env, x_size=self.obs_size, y_size=self.obs_size, linear_interp=True
        )
        if all_args["use_stacked_frames"]:
            env = ss.frame_stack_v2(env, stack_size=self.stacked_frames)
        if map_name == "knights_archers_zombies":
            env = ss.black_death_v3(env)

        self.env = env
        self._seed = 0
        self.n_agents = env.max_num_agents
        self.action_space = list(env.action_spaces.values())
        self.observation_space = [
            Box(0, 255, (self.stacked_frames, self.obs_size, self.obs_size), np.uint8)
        ] * self.n_agents
        self.share_observation_space = [
            Box(
                0,
                255,
                (self.stacked_frames*self.n_agents, self.state_size, self.state_size),
                np.uint8,
            )
        ] * self.n_agents
        self.stack_cent_obs = []

    def reset(self):
        self.stack_cent_obs.clear()
        obs_dict = self.env.reset(self._seed)
        cent_obs = cv2.cvtColor(self.env.state(), cv2.COLOR_RGB2GRAY)
        cent_obs = cv2.resize(cent_obs, (self.state_size, self.state_size))
        for _ in range(self.stacked_frames - 1):
            self.stack_cent_obs.append(np.zeros_like(cent_obs))
        self.stack_cent_obs.append(cent_obs)
        return (
            [np.transpose(obs, (2, 0, 1)) for obs in obs_dict.values()],
            self.get_cent_obs(),
            self.get_avail_actions(),
        )

    def step(self, actions):
        if self.action_space[0].__class__.__name__ == "Discrete":
            action_dict = {
                self.env.agents[i]: actions[i][0] for i in range(self.n_agents)
            }
        else:
            action_dict = {self.env.agents[i]: actions[i] for i in range(self.n_agents)}
        obs_dict, reward_dict, done_dict, info_dict = self.env.step(action_dict)
        cent_obs = cv2.cvtColor(self.env.state(), cv2.COLOR_RGB2GRAY)
        cent_obs = cv2.resize(cent_obs, (self.state_size, self.state_size))
        self.stack_cent_obs.append(cent_obs)
        return (
            [np.transpose(obs, (2, 0, 1)) for obs in obs_dict.values()],
            self.get_cent_obs(),
            np.array([[sum(reward_dict.values())] * self.n_agents]).T
            if self.map_name == "knights_archers_zombies"
            else np.array([list(reward_dict.values())]).T,
            list(done_dict.values()),
            list(info_dict.values()),
            self.get_avail_actions(),
        )

    def get_num_agents(self):
        return self.n_agents

    def get_spaces(self):
        return self.observation_space, self.share_observation_space, self.action_space

    def get_avail_actions(self):
        return None

    def seed(self, seed):
        self._seed = seed

    def get_cent_obs(self):
        cent_obs = np.stack(self.stack_cent_obs[-4:], axis=0)
        cent_obs = np.expand_dims(cent_obs, axis=0)
        return np.repeat(cent_obs, self.n_agents, axis=0)

    def close(self):
        pass

    def render(self):
        pass
