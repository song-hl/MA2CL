from __future__ import absolute_import, division, print_function

import cv2
import gym
import numpy as np
from gfootball.env import config, football_env, observation_preprocessing, wrappers
from gym.spaces import Box

from .encode.obs_encode import FeatureEncoder
from .encode.rew_encode import Rewarder
from .multiagentenv import MultiAgentEnv


def _process_reward_wrappers(env, rewards):
    assert "scoring" in rewards.split(",")
    if "checkpoints" in rewards.split(","):
        env = wrappers.CheckpointRewardWrapper(env)
    return env


def _process_representation_wrappers(env, representation, channel_dimensions):
    """Wraps with necessary representation wrappers.

  Args:
    env: A GFootball gym environment.
    representation: See create_environment.representation comment.
    channel_dimensions: (width, height) tuple that represents the dimensions of
       SMM or pixels representation.
  Returns:
    Google Research Football environment.
  """
    if representation.startswith("pixels"):
        env = wrappers.PixelsStateWrapper(
            env, "gray" in representation, channel_dimensions
        )
    elif representation == "simple115":
        env = wrappers.Simple115StateWrapper(env)
    elif representation == "simple115v2":
        env = wrappers.Simple115StateWrapper(env, True)
    elif representation == "extracted":
        env = wrappers.SMMWrapper(env, channel_dimensions)
    elif representation == "raw":
        pass
    else:
        raise ValueError("Unsupported representation: {}".format(representation))
    return env


def _apply_output_wrappers(
    env,
    rewards,
    representation,
    channel_dimensions,
    apply_single_agent_wrappers,
    stacked,
):
    """Wraps with necessary wrappers modifying the output of the environment.

  Args:
    env: A GFootball gym environment.
    rewards: What rewards to apply.
    representation: See create_environment.representation comment.
    channel_dimensions: (width, height) tuple that represents the dimensions of
       SMM or pixels representation.
    apply_single_agent_wrappers: Whether to reduce output to single agent case.
    stacked: Should observations be stacked.
  Returns:
    Google Research Football environment.
  """
    env = _process_reward_wrappers(env, rewards)
    env = _process_representation_wrappers(env, representation, channel_dimensions)
    if apply_single_agent_wrappers:
        if representation != "raw":
            env = wrappers.SingleAgentObservationWrapper(env)
        env = wrappers.SingleAgentRewardWrapper(env)
    if stacked:
        env = wrappers.FrameStack(env, 4)
    env = wrappers.GetStateWrapper(env)
    return env


def create_environment(
    env_name="",
    stacked=False,
    representation="extracted",
    rewards="scoring",
    write_goal_dumps=False,
    write_full_episode_dumps=False,
    render=False,
    write_video=False,
    dump_frequency=1,
    logdir="",
    extra_players=None,
    number_of_left_players_agent_controls=1,
    number_of_right_players_agent_controls=0,
    channel_dimensions=(
        observation_preprocessing.SMM_WIDTH,
        observation_preprocessing.SMM_HEIGHT,
    ),
    other_config_options={},
):

    assert env_name

    scenario_config = config.Config({"level": env_name}).ScenarioConfig()
    players = [
        (
            "agent:left_players=%d,right_players=%d"
            % (
                number_of_left_players_agent_controls,
                number_of_right_players_agent_controls,
            )
        )
    ]

    # Enable MultiAgentToSingleAgent wrapper?
    multiagent_to_singleagent = False
    if scenario_config.control_all_players:
        if number_of_left_players_agent_controls in [
            0,
            1,
        ] and number_of_right_players_agent_controls in [0, 1]:
            multiagent_to_singleagent = True
            players = [
                (
                    "agent:left_players=%d,right_players=%d"
                    % (
                        scenario_config.controllable_left_players
                        if number_of_left_players_agent_controls
                        else 0,
                        scenario_config.controllable_right_players
                        if number_of_right_players_agent_controls
                        else 0,
                    )
                )
            ]

    if extra_players is not None:
        players.extend(extra_players)
    config_values = {
        "dump_full_episodes": write_full_episode_dumps,
        "dump_scores": write_goal_dumps,
        "players": players,
        "level": env_name,
        "tracesdir": logdir,
        "write_video": write_video,
        "render_resolution_x": channel_dimensions[0],  # ~
        "render_resolution_y": channel_dimensions[1],  # ~
    }
    config_values.update(other_config_options)
    c = config.Config(config_values)

    env = football_env.FootballEnv(c)
    if multiagent_to_singleagent:
        env = wrappers.MultiAgentToSingleAgent(
            env,
            number_of_left_players_agent_controls,
            number_of_right_players_agent_controls,
        )
    if dump_frequency > 1:
        env = wrappers.PeriodicDumpWriter(env, dump_frequency, render)
    elif render:
        env.render()
    env = _apply_output_wrappers(
        env,
        rewards,
        representation,
        channel_dimensions,
        (
            number_of_left_players_agent_controls
            + number_of_right_players_agent_controls
            == 1
        ),
        stacked,
    )
    return env


class PixelFootballEnv(MultiAgentEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scenario = kwargs["env_args"]["scenario"]
        self.n_agents = kwargs["env_args"]["n_agent"]
        self.reward_type = kwargs["env_args"]["reward"]
        # pre_transform_image_size = kwargs["env_args"]["pre_transform_image_size"]
        self.env = create_environment(
            env_name=self.scenario,
            number_of_left_players_agent_controls=self.n_agents,
            representation="extracted",
            # representation="simple115v2",
            rewards=self.reward_type,
            render=False,
            # channel_dimensions=[pre_transform_image_size, pre_transform_image_size],
        )
        self.feature_encoder = FeatureEncoder()
        self.reward_encoder = Rewarder()

        self.action_space = [
            gym.spaces.Discrete(self.env.action_space.nvec[1])
            for n in range(self.n_agents)
        ]

        # self.channel_dimensions = [pre_transform_image_size, pre_transform_image_size]
        # tmp_obs_dicts = self.env.reset()
        # tmp_obs = [self._encode_obs(obs_dict)[0] for obs_dict in tmp_obs_dicts]
        self.observation_space = [
            Box(
                low=float("-inf"),
                high=float("inf"),
                # shape=[3, self.channel_dimensions[1], self.channel_dimensions[0]],
                shape=[3, 96, 72],
                dtype=np.float32,
            )
            for n in range(self.n_agents)
        ]
        self.share_observation_space = self.observation_space.copy()

        self.pre_obs = None

    def _encode_obs(self, raw_obs):
        obs = self.feature_encoder.encode(raw_obs.copy())
        obs_cat = np.hstack(
            [np.array(obs[k], dtype=np.float32).flatten() for k in sorted(obs)]
        )
        ava = obs["avail"]
        return obs_cat, ava

    def reset(self, **kwargs):
        """ Returns initial observations and states"""
        obs_dicts = self.env.reset()  # ~ (num_agents, images_sizeH, images_sizeW,4)
        self.pre_obs = obs_dicts
        obs = []
        ava = []
        for obs_dict in obs_dicts:
            _, ava_i = self._encode_obs(obs_dict)
            obs_i = obs_dict["frame"]
            # obs_i = cv2.resize(
            #     frame,
            #     (self.channel_dimensions[0], self.channel_dimensions[1]),
            #     interpolation=cv2.INTER_AREA,
            # )
            obs.append(np.transpose(obs_i, (2, 0, 1)))
            ava.append(ava_i)
        state = obs.copy()
        return obs, state, ava

    def step(self, actions):
        actions_int = [int(a) for a in actions]
        o, r, d, i = self.env.step(actions_int)
        obs = []
        ava = []
        for obs_dict in o:
            _, ava_i = self._encode_obs(obs_dict)
            obs_i = obs_dict["frame"]
            # obs_i = cv2.resize(
            #     frame,
            #     (self.channel_dimensions[0], self.channel_dimensions[1]),
            #     interpolation=cv2.INTER_AREA,
            # )
            obs.append(np.transpose(obs_i, (2, 0, 1)))
            ava.append(ava_i)
        state = obs.copy()

        rewards = [
            [self.reward_encoder.calc_reward(_r, _prev_obs, _obs)]
            for _r, _prev_obs, _obs in zip(r, self.pre_obs, o)
        ]

        self.pre_obs = o

        dones = np.ones((self.n_agents), dtype=bool) * d
        infos = [i for n in range(self.n_agents)]
        return obs, state, rewards, dones, infos, ava

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
