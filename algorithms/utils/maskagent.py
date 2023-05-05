import copy
from colorsys import hsv_to_rgb
from curses import noecho

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.augmentation import CenterCrop, RandomAffine, RandomCrop, RandomResizedCrop
from kornia.filters import GaussianBlur2d

from algorithms.ma_transformer import EncodeBlock, init_
from utils.util import get_model_params_volume, maybe_transform
from algorithms.utils.transformer_act import (
    discrete_parallel_act_logits,
    continuous_parallel_act_logits,
)

import wandb


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, length):
        return self.pe[:, :length]


class MLPHead(nn.Module):
    def __init__(
        self, in_channels, mlp_hidden_size, projection_size, use_bn=False, use_gelu=True
    ):
        super(MLPHead, self).__init__()
        self.fc1 = nn.Linear(in_channels, mlp_hidden_size)
        if use_bn:
            self.bn = nn.BatchNorm1d(mlp_hidden_size)
        self.act = nn.GELU() if use_gelu else nn.ReLU()
        self.fc2 = nn.Linear(mlp_hidden_size, projection_size)

    def forward(self, x):
        B, N, _ = x.shape
        x = self.fc1(x)
        if hasattr(self, "bn"):
            x = self.bn(x.flatten(0, 1)).view(B, N, -1)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Intensity(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise


class MaskAgent(nn.Module):
    def __init__(
        self,
        args,
        obs_encoder,
        act_encoder,
        action_dim,
        augmentation=["crop", "intensity"],
        aug_prob=1.0,
        device=torch.device("cpu"),
        action_type="Discrete",
    ):
        super(MaskAgent, self).__init__()
        self.n_head = args.maska_mawm_n_head
        self.n_layer = args.maska_mawm_n_layer
        self.algorithm_name = args.algorithm_name
        if "mat" in args.algorithm_name or "major" == args.algorithm_name:
            self.n_embd = obs_encoder.n_embd
            self.n_agent = obs_encoder.n_agent
            self.obs_encoder = obs_encoder
            self.target_obs_encoder = copy.deepcopy(self.obs_encoder)
            self.policy_type = 'mat'
        elif 'ppo' in args.algorithm_name:
            self.n_embd = args.hidden_size
            self.n_agent = args.num_agents
            self.obs_encoder = obs_encoder
            self.target_obs_encoder = copy.deepcopy(self.obs_encoder)
            self.policy_type = 'ppo'
        self.action_type = action_type
        if self.action_type == 'Discrete':
            self.use_one_hot_action = args.use_one_hot_action
            self.action_dim = action_dim if self.use_one_hot_action else 1
            # self.action_embedding = nn.Sequential(
            #     init_(nn.Linear(self.action_dim, self.n_embd, bias=False), activate=True),
            #     nn.GELU(),
            # )
        else:
            self.action_dim = action_dim
            # self.action_embedding = nn.Sequential(
            #     init_(nn.Linear(self.action_dim, self.n_embd), activate=True), nn.GELU()
            # )
        self.maska_ssl_tech = args.maska_ssl_tech               # 计算loss的方法
        self.maska_bsz = args.maska_bsz                         # batch size
        self.maska_latent_dim = args.maska_latent_dim           # latent dim 维度
        self.maska_projection_dim = args.maska_projection_dim   # projection dim 维度
        self.use_aug_in_maska = args.use_aug_in_maska           # 是否使用augmentation 数据增强
        self.use_bn_in_maska = args.use_bn_in_maska             # 是否使用BN 层
        self.use_gelu_in_maska = args.use_gelu_in_maska         # 是否使用gelu 激活函数
        self.use_action_info = args.use_action_info

        self.maska_ratio = args.maska_ratio
        self.maska_type = args.maska_type
        self.mask_agent_num = args.mask_agent_num
        if self.use_action_info:
            self.Pre_layer = nn.Sequential(
                *[EncodeBlock(self.n_embd+self.action_dim, self.n_head, self.n_agent) for _ in range(self.n_layer)]
            )
        else:
            self.Pre_layer = nn.Sequential(
                *[EncodeBlock(self.n_embd, self.n_head, self.n_agent) for _ in range(self.n_layer)]
            )

        self.W = nn.Parameter(torch.rand(self.n_embd, self.n_embd))
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.use_project = args.use_project
        self.projector = MLPHead(
            self.n_embd,
            self.maska_latent_dim,
            self.maska_projection_dim,
            use_bn=self.use_bn_in_maska,
            use_gelu=self.use_gelu_in_maska,
        )
        self.target_projector = MLPHead(
            self.n_embd,
            self.maska_latent_dim,
            self.maska_projection_dim,
            use_bn=self.use_bn_in_maska,
            use_gelu=self.use_gelu_in_maska,
        )

        self.ln = nn.LayerNorm(self.n_embd)
        self.position = PositionalEmbedding(self.n_embd)
        self.use_pos = args.pos_emb

        # count the volume of parameters of model
        self.get_params_volume()
        # self.get_update_params()
        self.rl_rep_bsz(args)

        # 数据增强部分
        # self.transforms = []
        # self.eval_transforms = []
        # self.uses_augmentation = True
        # self.aug_prob = aug_prob
        # for aug in augmentation:
        #     if aug == "affine":
        #         transformation = RandomAffine(5, (0.14, 0.14), (0.9, 1.1), (-5, 5))
        #         eval_transformation = nn.Identity()
        #         self.uses_augmentation = True
        #     elif aug == "crop":
        #         transformation = RandomCrop((args.image_size, args.image_size))
        #         # Crashes if aug-prob not 1: use CenterCrop((args.image_size, args.image_size)) or Resize((args.image_size, args.image_size)) in that case.
        #         eval_transformation = CenterCrop((args.image_size, args.image_size))
        #         self.uses_augmentation = True
        #     elif aug == "rrc":
        #         transformation = RandomResizedCrop((100, 100), (0.8, 1))
        #         eval_transformation = nn.Identity()
        #         self.uses_augmentation = True
        #     elif aug == "blur":
        #         transformation = GaussianBlur2d((5, 5), (1.5, 1.5))
        #         eval_transformation = nn.Identity()
        #         self.uses_augmentation = True
        #     elif aug == "shift":
        #         transformation = nn.Sequential(
        #             nn.ReplicationPad2d(4),
        #             RandomCrop((args.image_size, args.image_size)),
        #         )
        #         eval_transformation = nn.Identity()
        #     elif aug == "intensity":
        #         transformation = Intensity(scale=0.05)
        #         eval_transformation = nn.Identity()
        #     elif aug == "none":
        #         transformation = eval_transformation = nn.Identity()
        #     else:
        #         raise NotImplementedError()
        #     self.transforms.append(transformation)
        #     self.eval_transforms.append(eval_transformation)

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.to(device)

    def get_loss(self, obs, obs_label, non_masked, action, action_label):
        state = None
        # DATA AUGMENTATION
        # if obs.dim() > 4 and obs.shape[-1] == obs.shape[-2]:
        #     obs = self.transform(obs, augment=True)
        #     obs_label = self.transform(obs_label, augment=True)
        # 得到 obs 的表征
        if "happo" in self.algorithm_name:
            obs_rep = []
            for i in range(self.n_agent):
                obs_rep_single = self.obs_encoder[i].get_rep(state, obs[:, i])
                obs_rep.append(obs_rep_single)
            obs_rep = torch.stack(obs_rep, dim=1).flatten(0, 1)
        else:
            obs_rep = self.obs_encoder.get_rep(state, obs)

        if self.policy_type == 'ppo':
            obs_rep = obs_rep.reshape(-1, self.n_agent, self.n_embd)
        # projector 层 映射变换
        if self.use_project:
            obs_rep = self.projector(obs_rep)
        # 加入位置编码
        if self.use_pos:
            position = self.position(self.n_agent).to(self.device)
            obs_rep = obs_rep + position
        # 加入action信息
        if self.use_action_info:
            if self.action_type == 'Discrete':
                if self.use_one_hot_action:
                    action = F.one_hot(action.long(), self.action_dim).float()
                    action = action.reshape(-1, self.n_agent, self.action_dim)
                else:
                    action = action.reshape(-1, self.n_agent, 1)
            else:
                action = action.reshape(-1, self.n_agent, self.action_dim)

            obs_act = torch.cat([obs_rep, action], dim=-1)
            # Dynamic Model Prediction 预测mask之后的obs
            obs_act_rep = self.Pre_layer(obs_act)
            obs_pred = obs_act_rep[:, :, :-self.action_dim]
        else:
            obs_pred = self.Pre_layer(obs_rep)

        # 得到 label 的表征
        with torch.no_grad():
            if "happo" in self.algorithm_name:
                target_rep = []
                for i in range(self.n_agent):
                    target_rep_single = self.target_obs_encoder[i].get_rep(state, obs_label[:, i])
                    target_rep.append(target_rep_single)
                target_rep = torch.stack(target_rep, dim=1).flatten(0, 1)
            else:
                target_rep = self.target_obs_encoder.get_rep(state, obs_label)
            if self.policy_type == 'ppo':
                target_rep = target_rep.reshape(-1, self.n_agent, self.n_embd)
            if self.use_project:
                target_rep = self.target_projector(target_rep)
        # resize to (bsz * n_agent, n_embd)

        obs_pred = obs_pred.flatten(0, 1)
        target_rep = target_rep.flatten(0, 1)
        non_masked = non_masked.reshape(-1)

        # 被mask的 obs
        mask_idx = torch.arange(obs_pred.shape[0]).long().to(target_rep.device)[non_masked]
        obs_pred_m = obs_pred[non_masked]

        if self.maska_ssl_tech == "cpc":
            obs_loss = self.cpc_loss(obs_pred_m, target_rep, mask_idx)
        elif self.maska_ssl_tech == "byol":
            obs_loss = self.byol_loss(obs_pred_m, target_rep[non_masked])
        else:
            raise NotImplementedError()

        # if self.use_action_loss:
        #     if self.policy_type == 'ppo':
        #         action_logits_pre = self.act_encoder(obs_mgroup)
        #         with torch.no_grad():
        #             action_logits_pre_target = self.target_act_encoder(target_mgroup)
        #         action_loss = torch.distributions.kl.kl_divergence(action_logits_pre, action_logits_pre_target).mean()
        #         loss = action_loss + obs_loss
        #     elif self.policy_type == 'mat':
        #         # FIXME: this is not correct
        #         actions = action_label[maska_group]
        #         action_logits_pre = self.transformer_action_logits(obs_mgroup, actions, self.act_encoder)
        #         with torch.no_grad():
        #             action_logits_pre_target = self.transformer_action_logits(target_mgroup, actions, self.target_act_encoder)
        #         action_loss = torch.distributions.kl.kl_divergence(action_logits_pre, action_logits_pre_target).mean()
        #         # action_loss = self.gaussian_kl(pre_action_means, pre_action_stds, target_action_means, target_action_stds)
        #         loss = action_loss + obs_loss
        # else:
            # loss = obs_loss

        loss = obs_loss
        return loss

    def byol_loss(self, rep, target_rep):
        # B, T, N, D = rep.shape
        # proj_rep = self.projector(rep.flatten(0, 1))
        # pred_rep = self.predictor(proj_rep)
        # with torch.no_grad():
        #     target_proj_rep = self.target_projector(target_rep.flatten(0, 1))
        # pred_rep = pred_rep.view(B, T, N, D)
        # target_proj_rep = target_proj_rep.view(B, T, N, D)
        loss = self.norm_mse_loss(rep, target_rep)
        return loss

    def infonce_loss(self, rep, target):
        # TODO not implemented yet
        raise NotImplementedError

    def cpc_loss(self, rep, target, true_idx):
        logits = self.compute_logits(rep, target)
        loss = self.cross_entropy_loss(logits, true_idx)
        return loss

    def mse_loss(self, rep, target_rep):
        loss = torch.mean(
            F.mse_loss(rep, target_rep.clone().detach(), reduction="none"),
            dim=[1, 2, 3],
        )
        return loss

    def norm_mse_loss(self, rep, target_rep):
        loss = (
            F.mse_loss(
                F.normalize(rep, dim=-1, p=2.0, eps=1e-3),
                F.normalize(target_rep, dim=-1, p=2.0, eps=1e-3),
                reduction="none",
            )
            .sum(dim=-1)
            # .mean(2)
            # .mean(1)
            .mean(0)
        )
        return loss

    def apply_transforms(self, transforms, eval_transforms, image):
        if eval_transforms is None:
            for transform in transforms:
                image = transform(image)
        else:
            for transform, eval_transform in zip(transforms, eval_transforms):
                image = maybe_transform(
                    image, transform, eval_transform, p=self.aug_prob
                )
        return image

    def compute_logits(self, z_a, z_pos):  # TAG 计算距离
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1, keepdim=True)[0]  # 每行元素减去每行最大值 训练稳定性
        return logits

    @torch.no_grad()
    def transform(self, images, augment=False):
        images = images.float() / 255.0
        flat_images = images.reshape(-1, *images.shape[-3:])
        if augment:
            processed_images = self.apply_transforms(
                self.transforms, self.eval_transforms, flat_images
            )
        else:
            processed_images = self.apply_transforms(
                self.eval_transforms, None, flat_images
            )
        processed_images = processed_images.view(
            *images.shape[:-3], *processed_images.shape[1:]
        )
        return processed_images * 255.0

    def get_params_volume(self):
        total_params = get_model_params_volume(self)[0]
        online_encoder_params = 0
        trainable_params = 0
        ema_params = 0
        for name, model in self.named_children():
            params = get_model_params_volume(model)[0]
            if "target" in name:
                ema_params += params  # 更新ema的参数
            elif 'encoder' in name:
                online_encoder_params += params
            elif 'Pre' in name:
                Predicted_params = params
                trainable_params += params
            else:
                trainable_params += params
        print("=" * 60)
        print(f"Total params of MaskAgent: {total_params}")
        print(f"Trainable params of MaskAgent: {trainable_params}+{online_encoder_params}")
        print(f"Predicted params of MaskAgent: {Predicted_params}")
        print(f"EMA params of MaskAgent: {ema_params}")
        print("=" * 60)

    def rl_rep_bsz(self, args):
        # 计算ppo_update 的样本数，和，rep的样本数
        ppo_update_bsz = args.n_rollout_threads * args.episode_length // args.num_mini_batch
        if args.maska_bsz is None:
            self.maska_bsz = ppo_update_bsz
            args.maska_bsz = ppo_update_bsz
        else:
            self.maska_bsz = args.maska_bsz
        print(f"ppo_update_bsz: {ppo_update_bsz}, maska_bsz: {self.maska_bsz}")
        if args.use_wandb:
            wandb.config.update({"rep_bsz": self.maska_bsz, "ppo_update_bsz": ppo_update_bsz, "bsz_ratio": self.maska_bsz/ppo_update_bsz}, allow_val_change=True)

    def mask_pre(self, obs, actions, non_masked):

        if self.maska_type == "next":
            masked_obses = np.array(obs[:, 0], copy=True)
            masked_actions = np.array(actions[:, 0], copy=True)
        else:
            masked_obses = np.array(obs[:, 1], copy=True)
            masked_actions = np.array(actions[:, 1], copy=True)
        
        mask_ratio = 1

        for row in range(masked_obses.shape[0]):
            cols = np.random.choice(masked_obses.shape[1], self.mask_agent_num, replace=False)
            prob = np.random.rand()

            if self.maska_type == "last":
                if prob < mask_ratio:
                    masked_obses[row, cols] = obs[row, 0, cols]
                    masked_actions[row, cols] = actions[row, 0, cols]
                    non_masked[row, cols] = True
            elif self.maska_type == "zero":
                if prob < mask_ratio:
                    masked_obses[row, cols] = 0
                    masked_actions[row, cols] = actions[row, 0, cols]
                    non_masked[row, cols] = True
            elif self.maska_type == "random":
                if prob < mask_ratio:
                    random_state = np.random.randn(*masked_obses[row, cols].shape)
                    masked_obses[row, cols] = random_state + obs[row, 0, cols]
                    if self.action_type == 'Discrete':
                        masked_actions[row, cols] = np.random.randint(self.action_dim, size=random_state.shape)
                    else:
                        masked_actions[row, cols] = actions[row, 0, cols] + np.clip(np.random.randn(*masked_actions[row, cols].shape), -0.1, 0.1)
                    non_masked[row, cols] = True
            elif self.maska_type == "mix_zero":
                if prob < mask_ratio * self.maska_ratio:
                    masked_obses[row, cols] = obs[row, 0, cols]
                    masked_actions[row, cols] = actions[row, 0, cols]
                    non_masked[row, cols] = True
                elif prob < mask_ratio:
                    masked_obses[row, cols] = 0
                    masked_actions[row, cols] = actions[row, 0, cols]
                    non_masked[row, cols] = True
            elif self.maska_type == "mix_random":
                if prob < mask_ratio * self.maska_ratio:
                    masked_obses[row, cols] = obs[row, 0, cols]
                    masked_actions[row, cols] = actions[row, 0, cols]
                    non_masked[row, cols] = True
                elif prob < mask_ratio:
                    random_state = np.random.randn(*masked_obses[row, cols].shape)
                    masked_obses[row, cols] = random_state + obs[row, 0, cols]
                    if self.action_type == 'Discrete':
                        masked_actions[row, cols] = np.random.randint(self.action_dim, size=random_state.shape)
                    else:
                        masked_actions[row, cols] = actions[row, 1, cols] + np.clip(np.random.randn(*masked_actions[row, cols].shape), -0.1, 0.1)
                    non_masked[row, cols] = True
            elif self.maska_type == "next":
                if prob < mask_ratio:
                    masked_obses[row, cols] = 0
                    non_masked[row, cols] = True
            else:
                raise NotImplementedError

        # print(f"total :{len(masked_obses)},masked:{np.sum(non_masked==True)}")
        return masked_obses, masked_actions

    def get_update_params(self):
        self.freeze_params = []
        # if "happo" in self.algorithm_name:
        #     pass
        # else:
        #     for pname, p in self.act_encoder.named_parameters():
        #         self.freeze_params.append(p)

    def transformer_action_logits(self, obs_rep, action, encoder):
        batch_size = np.shape(obs_rep)[0]
        obs = None
        if self.action_type == "Discrete":
            output_action_logists = discrete_parallel_act_logits(
                encoder,
                obs_rep,
                obs,
                action,
                batch_size,
                self.n_agent,
                self.action_dim,
                self.tpdv,
            )
        else:
            output_action_logists = continuous_parallel_act_logits(
                encoder,
                obs_rep,
                obs,
                action,
                batch_size,
                self.n_agent,
                self.action_dim,
                self.tpdv,
            )
        return output_action_logists

    # KL divergence between two gaussian distributions
    def gaussian_kl(self, mu1, std1, mu2, std2):
        var1 = std1 ** 2
        var2 = std2 ** 2
        kl = torch.log(std2 / std1) + (var1 + (mu1 - mu2) ** 2) / (2 * var2) - 0.5
        return kl.mean()
