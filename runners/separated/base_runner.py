import os

import numpy as np
import torch
import wandb
from gym.spaces import Box
from tensorboardX import SummaryWriter
from utils.separated_buffer import SeparatedReplayBuffer
from utils.util import random_crop, update_linear_schedule
from algorithms.utils.maskagent import MaskAgent
from algorithms.utils.util import check
from utils.util import (
    get_gard_norm,
    huber_loss,
    mse_loss,
    soft_update_params,
    update_linear_schedule,
)
import torch.nn as nn

def _t2n(x):
    return x.detach().cpu().numpy()


class Runner(object):
    def __init__(self, config):

        self.all_args = config["all_args"]
        self.envs = config["envs"]
        self.eval_envs = config["eval_envs"]
        self.device = config["device"]
        self.num_agents = config["num_agents"]

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N
        self.use_single_network = self.all_args.use_single_network
        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        elif self.use_render:
            pass

            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / "gifs")
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = f"{self.run_dir}/logs"
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = f"{self.run_dir}/models"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        if "happo" in self.all_args.algorithm_name:
            from algorithms.happo_policy import HAPPO_Policy as Policy
            from algorithms.happo_trainer import HAPPO as TrainAlgo
        elif "hatrpo" in self.all_args.algorithm_name:
            from algorithms.hatrpo_policy import HATRPO_Policy as Policy
            from algorithms.hatrpo_trainer import HATRPO as TrainAlgo
        else:
            raise NotImplementedError

        print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space)

        self.policy = []
        for agent_id in range(self.num_agents):
            pre_share_observation_space = (
                self.envs.share_observation_space[agent_id]
                if self.use_centralized_V
                else self.envs.observation_space[agent_id]
            )
            pre_observation_space = self.envs.observation_space[agent_id]
            if (
                isinstance(pre_share_observation_space, Box) and
                len(pre_share_observation_space.shape) == 3
                and self.all_args.pre_transform_image_size > self.all_args.image_size
            ):
                share_observation_space = Box(
                    0,
                    255,
                    (
                        pre_share_observation_space.shape[-3],
                        self.all_args.image_size,
                        self.all_args.image_size,
                    ),
                    np.uint8,
                )
            else:
                share_observation_space = pre_share_observation_space

            if (
                isinstance(pre_observation_space, Box) and
                len(pre_observation_space.shape) == 3
                and self.all_args.pre_transform_image_size > self.all_args.image_size
            ):
                observation_space = Box(
                    0,
                    255,
                    (
                        pre_observation_space.shape[-3],
                        self.all_args.image_size,
                        self.all_args.image_size,
                    ),
                    np.uint8,
                )
            else:
                observation_space = pre_observation_space

            # policy network
            po = Policy(
                self.all_args,
                observation_space,
                share_observation_space,
                self.envs.action_space[agent_id],
                device=self.device,
            )
            self.policy.append(po)

        if self.model_dir is not None:
            self.restore()

        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            # algorithm
            tr = TrainAlgo(self.all_args, self.policy[agent_id], device=self.device)
            # buffer
            pre_share_observation_space = (
                self.envs.share_observation_space[agent_id]
                if self.use_centralized_V
                else self.envs.observation_space[agent_id]
            )
            pre_observation_space = self.envs.observation_space[agent_id]
            bu = SeparatedReplayBuffer(
                self.all_args,
                pre_observation_space,
                pre_share_observation_space,
                self.envs.action_space[agent_id],
            )
            self.buffer.append(bu)
            self.trainer.append(tr)
        # self._use_jpr = self.all_args.use_jpr
        # if self._use_jpr:
        #     self.jpr = JPR(
        #         self.all_args,
        #         obs_encoder=[
        #             self.policy[agent_id].actor.obs_encoder
        #             for agent_id in range(self.num_agents)
        #         ],
        #         act_encoder=[
        #             self.policy[agent_id].actor.act_encoder
        #             for agent_id in range(self.num_agents)
        #         ],
        #         action_dim=self.policy[0].actor.act.action_dim,
        #         device=self.device,
        #         action_type=self.policy[0].actor.act.action_type,
        #     )
        #     self.jpr_optimizer = torch.optim.Adam(
        #         self.jpr.parameters(), lr=self.all_args.jpr_lr
        #     )
        #     self.jpr_bsz = self.all_args.jpr_bsz
        #     self.jpr_jumps = self.all_args.jpr_jumps
        #     self.jpr_update_freq = self.all_args.jpr_update_freq
        #     self.jpr_target_update_freq = self.all_args.jpr_target_update_freq
        #     self.jpr_momentum_tau = self.all_args.jpr_momentum_tau
        #     self.use_linear_jprlr_decay = self.all_args.use_linear_jprlr_decay
        #     self.jpr_epoch = self.all_args.ppo_epoch // self.all_args.jpr_update_freq
        #     self._use_max_grad_norm = self.all_args.use_max_grad_norm
        #     self.max_grad_norm = self.all_args.max_grad_norm
        #     self.tpdv = dict(dtype=torch.float32, device=self.device)
        #     self.num_updates = 0

        self._mask_agent = self.all_args.mask_agent
        if self._mask_agent:
            self.mask_agent = MaskAgent(
                self.all_args,
                obs_encoder=[
                    self.policy[agent_id].actor
                    for agent_id in range(self.num_agents)
                ],
                act_encoder=[
                    self.policy[agent_id].actor
                    for agent_id in range(self.num_agents)
                ],
                action_dim=self.policy[0].actor.act.action_dim,
                device=self.device,
                action_type=self.policy[0].actor.act.action_type,
            )
            self.maska_optimizer = torch.optim.Adam(self.mask_agent.parameters(), lr=self.all_args.maska_lr)
            # self.freeze_optimizer = torch.optim.Adam(self.mask_agent.freeze_params, lr=0)
            self.maska_update_freq = self.all_args.maska_update_freq
            self.maska_target_update_freq = self.all_args.maska_target_update_freq
            self.maska_momentum_tau = self.all_args.maska_momentum_tau
            self.use_linear_maskalr_decay = self.all_args.use_linear_maskalr_decay
            self.maska_bsz = self.all_args.maska_bsz
            self.maska_jumps = self.all_args.maska_jumps
            self.num_updates = 0
            self.tpdv = dict(dtype=torch.float32, device=self.device)
            self._use_max_grad_norm = self.all_args.use_max_grad_norm
            self.max_grad_norm = self.all_args.max_grad_norm

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(
                self.buffer[agent_id].share_obs[-1],
                self.buffer[agent_id].rnn_states_critic[-1],
                self.buffer[agent_id].masks[-1],
            )
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(
                next_value, self.trainer[agent_id].value_normalizer
            )

    def train(self):
        train_infos = []
        # random update order

        action_dim = self.buffer[0].actions.shape[-1]
        factor = np.ones(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32
        )

        for agent_id in torch.randperm(self.num_agents):
            self.trainer[agent_id].prep_training()
            self.buffer[agent_id].update_factor(factor)
            available_actions = (
                None
                if self.buffer[agent_id].available_actions is None
                else self.buffer[agent_id]
                .available_actions[:-1]
                .reshape(-1, *self.buffer[agent_id].available_actions.shape[2:])
            )

            if self.all_args.algorithm_name == "hatrpo":
                obs = (
                    self.buffer[agent_id]
                    .obs[:-1]
                    .reshape(-1, *self.buffer[agent_id].obs.shape[2:])
                )
                if len(obs.shape) > 2 and obs.shape[-1] > self.all_args["image_size"]:
                    obs = random_crop(obs, self.all_args["image_size"])
                old_actions_logprob, _, _, _, _ = self.trainer[
                    agent_id
                ].policy.actor.evaluate_actions(
                    obs,
                    self.buffer[agent_id]
                    .rnn_states[0:1]
                    .reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                    self.buffer[agent_id].actions.reshape(
                        -1, *self.buffer[agent_id].actions.shape[2:]
                    ),
                    self.buffer[agent_id]
                    .masks[:-1]
                    .reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                    available_actions,
                    self.buffer[agent_id]
                    .active_masks[:-1]
                    .reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]),
                )
            else:
                obs = (
                    self.buffer[agent_id]
                    .obs[:-1]
                    .reshape(-1, *self.buffer[agent_id].obs.shape[2:])
                )
                if len(obs.shape) > 2 and obs.shape[-1] > self.all_args["image_size"]:
                    obs = random_crop(obs, self.all_args["image_size"])
                old_actions_logprob, _ = self.trainer[
                    agent_id
                ].policy.actor.evaluate_actions(
                    obs,
                    self.buffer[agent_id]
                    .rnn_states[0:1]
                    .reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                    self.buffer[agent_id].actions.reshape(
                        -1, *self.buffer[agent_id].actions.shape[2:]
                    ),
                    self.buffer[agent_id]
                    .masks[:-1]
                    .reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                    available_actions,
                    self.buffer[agent_id]
                    .active_masks[:-1]
                    .reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]),
                )
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])

            if self.all_args.algorithm_name == "hatrpo":
                obs = (
                    self.buffer[agent_id]
                    .obs[:-1]
                    .reshape(-1, *self.buffer[agent_id].obs.shape[2:])
                )
                if len(obs.shape) > 2 and obs.shape[-1] > self.all_args["image_size"]:
                    obs = random_crop(obs, self.all_args["image_size"])
                new_actions_logprob, _, _, _, _ = self.trainer[
                    agent_id
                ].policy.actor.evaluate_actions(
                    obs,
                    self.buffer[agent_id]
                    .rnn_states[0:1]
                    .reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                    self.buffer[agent_id].actions.reshape(
                        -1, *self.buffer[agent_id].actions.shape[2:]
                    ),
                    self.buffer[agent_id]
                    .masks[:-1]
                    .reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                    available_actions,
                    self.buffer[agent_id]
                    .active_masks[:-1]
                    .reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]),
                )
            else:
                obs = (
                    self.buffer[agent_id]
                    .obs[:-1]
                    .reshape(-1, *self.buffer[agent_id].obs.shape[2:])
                )
                if len(obs.shape) > 2 and obs.shape[-1] > self.all_args["image_size"]:
                    obs = random_crop(obs, self.all_args["image_size"])
                new_actions_logprob, _ = self.trainer[
                    agent_id
                ].policy.actor.evaluate_actions(
                    obs,
                    self.buffer[agent_id]
                    .rnn_states[0:1]
                    .reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                    self.buffer[agent_id].actions.reshape(
                        -1, *self.buffer[agent_id].actions.shape[2:]
                    ),
                    self.buffer[agent_id]
                    .masks[:-1]
                    .reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                    available_actions,
                    self.buffer[agent_id]
                    .active_masks[:-1]
                    .reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]),
                )

            factor = factor * _t2n(
                torch.prod(
                    torch.exp(new_actions_logprob - old_actions_logprob), dim=-1
                ).reshape(self.episode_length, self.n_rollout_threads, 1)
            )
            train_infos.append(train_info)
            
        if self._mask_agent:
            train_info = {}
            train_info["maska_loss"] = 0
            train_info["maska_grad"] = 0
            for i in range(self.all_args.ppo_epoch):
                obs_batch = []
                obs_label_batch = []
                non_masked_batch = []
                actions_batch = []
                actions_label_batch = []
                for i in range(self.num_agents):
                    (obs,  actions) = self.buffer[i].sample_pre(self.maska_bsz, self.maska_jumps, self.mask_agent)
                    obs_batch.append(np.expand_dims(obs, axis=2))
                    actions_batch.append(np.expand_dims(actions, axis=2))
                obs_batch = np.concatenate(obs_batch, axis=2)
                actions_batch = np.concatenate(actions_batch, axis=2)
                non_masked = np.zeros((obs_batch.shape[0], self.num_agents), dtype=np.bool)
                obs_label = obs_batch[:, 1]
                actions_label = actions_batch[:, 1]
                obs, actions = self.mask_agent.mask_pre(obs_batch, actions_batch, non_masked)
                # train_info["maska_bsz"] += smaple_pre[0].shape[0]
                maska_loss, maska_grad = self.maska_update([obs, obs_label, non_masked, actions, actions_label])
                train_info["maska_loss"] += maska_loss.item()
                train_info["maska_grad"] += maska_grad.item()
            episode = self.num_updates // (self.all_args.ppo_epoch * self.all_args.num_mini_batch)
            episodes = (
                int(self.all_args.num_env_steps)
                // self.all_args.episode_length
                // self.all_args.n_rollout_threads
            )
            if self.use_linear_maskalr_decay:
                update_linear_schedule(
                    self.maska_optimizer, episode, episodes, self.all_args.maska_lr
                )

        # if self._use_jpr:
        #     train_info = {}
        #     train_info["jpr_loss"] = 0
        #     train_info["jpr_grad"] = 0
        #     for i in range(self.jpr_epoch):
        #         share_obs_batch = []
        #         obs_batch = []
        #         actions_batch = []
        #         for i in range(self.num_agents):
        #             (
        #                 share_obs_batch_single,
        #                 obs_batch_single,
        #                 actions_batch_single,
        #             ) = self.buffer[i].sample_jpr(self.jpr_bsz, self.jpr_jumps)
        #             share_obs_batch.append(
        #                 np.expand_dims(share_obs_batch_single, axis=2)
        #             )
        #             obs_batch.append(np.expand_dims(obs_batch_single, axis=2))
        #             actions_batch.append(np.expand_dims(actions_batch_single, axis=2))
        #         share_obs_batch = np.concatenate(share_obs_batch, axis=2)
        #         obs_batch = np.concatenate(obs_batch, axis=2)
        #         actions_batch = np.concatenate(actions_batch, axis=2)
        #         jpr_loss, jpr_grad = self.jpr_update(
        #             [share_obs_batch, obs_batch, actions_batch]
        #         )
        #         train_info["jpr_loss"] += jpr_loss.item()
        #         train_info["jpr_grad"] += jpr_grad.item()
        #     train_info["jpr_loss"] /= self.jpr_epoch
        #     train_info["jpr_grad"] /= self.jpr_epoch
        #     train_infos.append(train_info)
        #     self.num_updates += 1

        for i in range(self.num_agents):
            self.buffer[agent_id].after_update()

        return train_infos

    def save(self, episode):
        for agent_id in range(self.num_agents):
            if self.use_single_network:
                policy_model = self.trainer[agent_id].policy.model
                torch.save(
                    policy_model.state_dict(),
                    f"{self.save_dir}/model_agent{agent_id}_{episode}.pt",
                )
                torch.save(
                    policy_model.state_dict(),
                    f"{self.save_dir}/model_agent{agent_id}.pt",
                )
            else:
                policy_actor = self.trainer[agent_id].policy.actor
                torch.save(
                    policy_actor.state_dict(),
                    f"{self.save_dir}/actor_agent{agent_id}_{episode}.pt",
                )
                torch.save(
                    policy_actor.state_dict(),
                    f"{self.save_dir}/actor_agent{agent_id}.pt",
                )
                policy_critic = self.trainer[agent_id].policy.critic
                torch.save(
                    policy_critic.state_dict(),
                    f"{self.save_dir}/critic_agent{agent_id}_{episode}.pt",
                )
                torch.save(
                    policy_critic.state_dict(),
                    f"{self.save_dir}/critic_agent{agent_id}.pt",
                )

    def restore(self):
        for agent_id in range(self.num_agents):
            if self.use_single_network:
                policy_model_state_dict = torch.load(
                    f"{self.model_dir}/model_agent{agent_id}.pt"
                )
                self.policy[agent_id].model.load_state_dict(policy_model_state_dict)
            else:
                policy_actor_state_dict = torch.load(
                    f"{self.model_dir}/actor_agent{agent_id}.pt"
                )
                self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
                policy_critic_state_dict = torch.load(
                    f"{self.model_dir}/critic_agent{agent_id}.pt"
                )
                self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            v_scalar = np.mean(v) if len(v) > 0 else v
            if self.use_wandb:
                wandb.log({k: v_scalar}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v_scalar}, total_num_steps)

    def maska_update(self, sample):
        # TODO: add maska update
        (obs_batch, obs_label_batch, non_mask, action_batch, action_label) = sample
        obs_batch = check(obs_batch).to(**self.tpdv)
        obs_label_batch = check(obs_label_batch).to(**self.tpdv)
        action_batch = check(action_batch).to(**self.tpdv).flatten(0, 1)
        action_label = check(action_label).to(**self.tpdv)

        loss = self.mask_agent.get_loss(obs_batch, obs_label_batch, non_mask, action_batch, action_label)
        self.maska_optimizer.zero_grad()
        loss.backward()
        # self.freeze_optimizer.zero_grad()
        if self._use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(
                self.mask_agent.parameters(), self.max_grad_norm
            )
        else:
            grad_norm = 0
        self.maska_optimizer.step()
        if self.num_updates % self.maska_target_update_freq == 0:
            for i in range(self.num_agents):
                soft_update_params(
                    self.mask_agent.obs_encoder[i], self.mask_agent.target_obs_encoder[i], self.maska_momentum_tau
                )
                # soft_update_params(
                #     self.mask_agent.act_encoder[i], self.mask_agent.target_act_encoder[i], self.maska_momentum_tau
                # )
            soft_update_params(
                self.mask_agent.projector, self.mask_agent.target_projector, self.maska_momentum_tau
            )
        return loss, grad_norm
