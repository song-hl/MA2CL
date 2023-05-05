from itertools import cycle

import numpy as np
import torch
import torch.nn as nn

from algorithms.utils.jpr import JPR
from algorithms.utils.maskagent import MaskAgent
from algorithms.utils.popart import PopArt
from algorithms.utils.util import check
from algorithms.utils.valuenorm import ValueNorm
from utils.util import (
    get_gard_norm,
    huber_loss,
    mse_loss,
    soft_update_params,
    update_linear_schedule,
)
import datetime

class MAPPO:
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, policy, num_agents, device=torch.device("cpu")):

        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.num_agents = num_agents

        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.num_mini_batch = args["num_mini_batch"]
        self.data_chunk_length = args["data_chunk_length"]
        self.value_loss_coef = args["value_loss_coef"]
        self.entropy_coef = args["entropy_coef"]
        self.max_grad_norm = args["max_grad_norm"]
        self.huber_delta = args["huber_delta"]

        self._use_valuenorm = args["use_valuenorm"]
        self._use_recurrent_policy = args["use_recurrent_policy"]
        self._use_naive_recurrent = args["use_naive_recurrent_policy"]
        self._use_max_grad_norm = args["use_max_grad_norm"]
        self._use_clipped_value_loss = args["use_clipped_value_loss"]
        self._use_huber_loss = args["use_huber_loss"]
        self._use_popart = args["use_popart"]
        self._use_value_active_masks = args["use_value_active_masks"]
        self._use_policy_active_masks = args["use_policy_active_masks"]

        assert (
            self._use_popart and self._use_valuenorm
        ) == False, "self._use_popart and self._use_valuenorm can not be set True simultaneously"

        if self._use_popart:
            self.value_normalizer = PopArt(1, device=self.device)
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

        self._use_jpr = args.use_jpr
        self._mask_agent = args.mask_agent
        if args.use_jpr:
            self.jpr = JPR(
                args,
                act_encoder=policy.actor,
                obs_encoder=policy.critic,
                action_dim=policy.actor.act.action_dim,
                device=device,
                action_type=policy.actor.act.action_type,
            )
            self.jpr_optimizer = torch.optim.Adam(self.jpr.parameters(), lr=args.jpr_lr)
            self.jpr_bsz = args.jpr_bsz
            self.jpr_jumps = args.jpr_jumps
            self.jpr_update_freq = args.jpr_update_freq
            self.jpr_target_update_freq = args.jpr_target_update_freq
            self.jpr_momentum_tau = args.jpr_momentum_tau
            self.use_linear_jprlr_decay = args.use_linear_jprlr_decay
        if args.mask_agent:
            self.mask_agent = MaskAgent(
                args,
                act_encoder=policy.actor,
                obs_encoder=policy.actor,
                action_dim=policy.actor.act.action_dim,
                device=device,
                action_type=policy.actor.act.action_type,
            )
            self.maska_optimizer = torch.optim.Adam(self.mask_agent.parameters(), lr=args.maska_lr)
            # self.freeze_optimizer = torch.optim.Adam(self.mask_agent.freeze_params, lr=0)
            self.maska_update_freq = args.maska_update_freq
            self.maska_target_update_freq = args.maska_target_update_freq
            self.maska_momentum_tau = args.maska_momentum_tau
            self.use_linear_maskalr_decay = args.use_linear_maskalr_decay
            self.maska_bsz = args.maska_bsz
            self.maska_jumps = args.maska_jumps
        self.num_updates = 0

    def cal_value_loss(
        self, values, value_preds_batch, return_batch, active_masks_batch
    ):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.clip_param, self.clip_param
        )
        if self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = (
                self.value_normalizer.normalize(return_batch) - value_pred_clipped
            )
            error_original = self.value_normalizer.normalize(return_batch) - values

        if self._use_popart:
            error_clipped = self.value_normalizer(return_batch) - value_pred_clipped
            error_original = self.value_normalizer(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (
                value_loss * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        (
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            value_preds_batch,
            return_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
        ) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )
        # actor update
        imp_weights = torch.exp(
            (action_log_probs - old_action_log_probs_batch).sum(dim=-1, keepdim=True)
        )

        surr1 = imp_weights * adv_targ
        surr2 = (
            torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * adv_targ
        )

        if self._use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
                * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(
                torch.min(surr1, surr2), dim=-1, keepdim=True
            ).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(
            values, value_preds_batch, return_batch, active_masks_batch
        )

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.critic.parameters(), self.max_grad_norm
            )
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return (
            value_loss,
            critic_grad_norm,
            policy_loss,
            dist_entropy,
            actor_grad_norm,
            imp_weights,
        )

    def jpr_update(self, sample):
        (share_obs_batch, obs_batch, action_batch,) = sample
        share_obs_batch = check(share_obs_batch).to(**self.tpdv)
        obs_batch = check(obs_batch).to(**self.tpdv)
        action_batch = check(action_batch).to(**self.tpdv)

        loss = self.jpr.get_loss(share_obs_batch, obs_batch, action_batch,)
        self.jpr_optimizer.zero_grad()
        loss.backward()
        if self._use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(
                self.jpr.parameters(), self.max_grad_norm
            )
        else:
            grad_norm = 0
        self.jpr_optimizer.step()
        if self.num_updates % self.jpr_target_update_freq == 0:
            soft_update_params(
                self.jpr.obs_encoder, self.jpr.target_obs_encoder, self.jpr_momentum_tau
            )
            # soft_update_params(
            #     self.jpr.act_encoder, self.jpr.target_act_encoder, self.jpr_momentum_tau
            # )
            soft_update_params(
                self.jpr.projector, self.jpr.target_projector, self.jpr_momentum_tau
            )
        return loss, grad_norm

    def maska_update(self, sample):
        # TODO: add maska update
        (obs_batch, obs_label_batch, non_mask, action_batch, action_label) = sample
        obs_batch = check(obs_batch).to(**self.tpdv).flatten(0, 1)
        obs_label_batch = check(obs_label_batch).to(**self.tpdv).flatten(0, 1)
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
            soft_update_params(
                self.mask_agent.obs_encoder, self.mask_agent.target_obs_encoder, self.maska_momentum_tau
            )
            # soft_update_params(
            #     self.mask_agent.act_encoder, self.mask_agent.target_act_encoder, self.maska_momentum_tau
            # )
            soft_update_params(
                self.mask_agent.projector, self.mask_agent.target_projector, self.maska_momentum_tau
            )
        return loss, grad_norm

    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(
                buffer.value_preds[:-1]
            )
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}

        train_info["value_loss"] = 0
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["critic_grad_norm"] = 0
        train_info["ratio"] = 0

        if self._use_jpr:
            train_info["jpr_loss"] = 0
            train_info["jpr_grad"] = 0
        if self._mask_agent:
            train_info["maska_loss"] = 0
            train_info["maska_grad"] = 0
            train_info["maska_bsz"]  = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(
                    advantages, self.num_mini_batch, self.data_chunk_length
                )
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(
                    advantages, self.num_mini_batch
                )
            else:
                data_generator = buffer.feed_forward_generator(
                    advantages, self.num_mini_batch
                )

            for sample in data_generator:
                (
                    value_loss,
                    critic_grad_norm,
                    policy_loss,
                    dist_entropy,
                    actor_grad_norm,
                    imp_weights,
                ) = self.ppo_update(sample, update_actor)

                train_info["value_loss"] += value_loss.item()
                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["critic_grad_norm"] += critic_grad_norm
                train_info["ratio"] += imp_weights.mean()

                self.num_updates += 1
                if self._use_jpr and self.num_updates % self.jpr_update_freq == 0:
                    jpr_loss, jpr_grad = self.jpr_update(buffer.sample_jpr(self.jpr_bsz, self.jpr_jumps))
                    train_info["jpr_loss"] += jpr_loss.item()
                    train_info["jpr_grad"] += jpr_grad.item()
                if self._mask_agent and self.num_updates % self.maska_update_freq == 0:
                    smaple_pre = buffer.sample_pre(self.maska_bsz, self.maska_jumps, self.mask_agent)
                    train_info["maska_bsz"] += smaple_pre[0].shape[0]
                    # smaple_masked = buffer.sample_maska(self.maska_ratio, self.maska_bsz , self.maska_type)
                    maska_loss, maska_grad = self.maska_update(smaple_pre)
                    train_info["maska_loss"] += maska_loss.item()
                    train_info["maska_grad"] += maska_grad.item()
                    episode = self.num_updates // (self.ppo_epoch * self.num_mini_batch)
                    episodes = (
                        int(self.args.num_env_steps)
                        // self.args.episode_length
                        // self.args.n_rollout_threads
                    )
                    if self.use_linear_maskalr_decay:
                        update_linear_schedule(
                            self.maska_optimizer, episode, episodes, self.args.maska_lr
                        )
        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
