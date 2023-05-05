import gfootball.env as football_env
import gym
import numpy as np
from gym.spaces import Box

from .encode.obs_encode import FeatureEncoder
from .encode.rew_encode import Rewarder
from .multiagentenv import MultiAgentEnv


class FootballEnv(MultiAgentEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scenario = kwargs["env_args"]["scenario"]
        self.n_agents = kwargs["env_args"]["n_agent"]
        self.reward_type = kwargs["env_args"]["reward"]
        self.use_sight_range = kwargs["env_args"]["use_sight_range"]
        self.sight_range = kwargs["env_args"]["sight_range"]
        self.env = football_env.create_environment(
            env_name=self.scenario,
            number_of_left_players_agent_controls=self.n_agents,
            representation="raw",
            # representation="simple115v2",
            rewards=self.reward_type,
        )
        full_obs = self.env.unwrapped.observation()[0]
        self.feature_encoder = FeatureEncoder(self.use_sight_range, self.sight_range)
        self.reward_encoder = Rewarder()

        self.action_space = [
            gym.spaces.Discrete(self.env.action_space.nvec[1])
            for n in range(self.n_agents)
        ]
        tmp_obs_dicts = self.env.reset()
        tmp_obs = self._encode_obs(tmp_obs_dicts)[0]
        self.observation_space = [
            Box(
                low=float("-inf"),
                high=float("inf"),
                shape=tmp_obs[n].shape,
                dtype=np.float32,
            )
            for n in range(self.n_agents)
        ]
        self.share_observation_space = self.observation_space.copy()

        self.pre_obs = None
        # obs2 = self.get_simple_obs(0)

    def _encode_obs(self, raw_obs_dicts):
        obs = []
        ava = []
        state = []
        for obs_dict in raw_obs_dicts:
            obs_i = self.feature_encoder.encode(obs_dict.copy())
            state_i = self.feature_encoder.encode(obs_dict.copy(), False) if self.use_sight_range else obs_i

            obs_i_cat = np.hstack(
                [np.array(obs_i[k], dtype=np.float32).flatten() for k in sorted(obs_i)]
            )
            state_i_cat = np.hstack(
                [np.array(state_i[k], dtype=np.float32).flatten() for k in sorted(state_i)]
            )
            ava_i = state_i["avail"]
            obs.append(obs_i_cat)
            ava.append(ava_i)
            state.append(state_i_cat)
        return obs, state, ava

    def reset(self, **kwargs):
        """ Returns initial observations and states"""
        obs_dicts = self.env.reset()
        self.pre_obs = obs_dicts
        obs, state, ava = self._encode_obs(obs_dicts)
        return obs, state, ava

    def step(self, actions):
        actions_int = [int(a) for a in actions]
        o, r, d, i = self.env.step(actions_int)
        obs, state, ava = self._encode_obs(o)

        rewards = [
            [self.reward_encoder.calc_reward(_r, _prev_obs, _obs)]
            for _r, _prev_obs, _obs in zip(r, self.pre_obs, o)
        ]

        self.pre_obs = o

        dones = np.ones((self.n_agents), dtype=bool) * d
        infos = [i for n in range(self.n_agents)]
        return obs, state, rewards, dones, infos, ava

    def get_simple_obs(self, index=-1):
        full_obs = self.env.unwrapped.observation()[0]
        simple_obs = []

        if index == -1:
            # global state, absolute position
            simple_obs.append(full_obs['left_team']
                              [-self.n_agents:].reshape(-1))
            simple_obs.append(
                full_obs['left_team_direction'][-self.n_agents:].reshape(-1))

            simple_obs.append(full_obs['right_team'].reshape(-1))
            simple_obs.append(full_obs['right_team_direction'].reshape(-1))

            simple_obs.append(full_obs['ball'])
            simple_obs.append(full_obs['ball_direction'])

        else:
            # local state, relative position
            unsight_vec = np.array([0, 0])
            ego_position = full_obs['left_team'][-self.n_agents + index].reshape(-1)
            simple_obs.append(ego_position)

            allys_pos = np.delete(full_obs['left_team'][-self.n_agents:], index, axis=0) - ego_position

            for ally_pos in allys_pos:
                simple_obs.append(ally_pos if np.linalg.norm(ally_pos) < self.sight_range else unsight_vec)

            # player direction
            simple_obs.append(full_obs['left_team_direction'][-self.n_agents + index].reshape(-1))
            allys_direc = np.delete(full_obs['left_team_direction'][-self.n_agents:], index, axis=0)
            for ally_pos, ally_direc in zip(allys_pos, allys_direc):
                simple_obs.append(ally_direc if np.linalg.norm(ally_pos) < self.sight_range else unsight_vec)

            enemys_pos = (full_obs['right_team'] - ego_position)
            for enemy_pos in enemys_pos:
                simple_obs.append(enemy_pos if np.linalg.norm(enemy_pos) < self.sight_range else unsight_vec)

            enemys_direc = full_obs['right_team_direction']
            for enemy in enemys_direc:
                simple_obs.append(enemy if np.linalg.norm(enemy_pos) < self.sight_range else unsight_vec)

            ball_pos = full_obs['ball'][:2] - ego_position
            if np.linalg.norm(ball_pos) > self.sight_range:
                simple_obs.append(np.array([0, 0, 0, 0, 0, 0]))
            else:
                simple_obs.append(np.concatenate([ball_pos, full_obs['ball'][-1].reshape(-1), full_obs['ball_direction']]))

        simple_obs = np.concatenate(simple_obs)
        return simple_obs

    def render(self, **kwargs):
        # self.env.render(**kwargs)
        pass

    def close(self):
        pass

    def seed(self, args):
        pass

    def get_env_info(self):

        env_info = {
            "state_shape": self.observation_space[0].shape,
            "obs_shape": self.observation_space[0].shape,
            "n_actions": self.action_space[0].n,
            "n_agents": self.n_agents,
            "action_spaces": self.action_space,
        }
        return env_info
