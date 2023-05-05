import copy
from colorsys import hsv_to_rgb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.augmentation import CenterCrop, RandomAffine, RandomCrop, RandomResizedCrop
from kornia.filters import GaussianBlur2d

from algorithms.ma_transformer import EncodeBlock, init_
from utils.util import get_model_params_volume, maybe_transform


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


class JPR(nn.Module):
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
        super(JPR, self).__init__()
        self.n_head = args.jpr_mawm_n_head
        self.n_layer = args.jpr_mawm_n_layer
        self.algorithm_name = args.algorithm_name
        if "ppo" in args.algorithm_name:
            self.n_embd = args.hidden_size
            self.n_agent = args.num_agents
        else:
            self.n_embd = obs_encoder.n_embd
            self.n_agent = obs_encoder.n_agent
        self.action_type = action_type
        self.action_dim = action_dim
        self.jpr_ssl_tech = args.jpr_ssl_tech
        self.jpr_bsz = args.jpr_bsz
        self.jpr_latent_dim = args.jpr_latent_dim
        self.jpr_projection_dim = args.jpr_projection_dim
        self.use_aug_in_jpr = args.use_aug_in_jpr
        self.use_bn_in_jpr = args.use_bn_in_jpr
        self.use_gelu_in_jpr = args.use_gelu_in_jpr

        self.obs_encoder = obs_encoder
        self.act_encoder = act_encoder
        self.target_obs_encoder = copy.deepcopy(obs_encoder)
        # self.target_act_encoder = copy.deepcopy(act_encoder)

        self.projector = MLPHead(
            self.n_embd,
            self.jpr_latent_dim,
            self.jpr_projection_dim,
            use_bn=self.use_bn_in_jpr,
            use_gelu=self.use_gelu_in_jpr,
        )
        self.target_projector = MLPHead(
            self.n_embd,
            self.jpr_latent_dim,
            self.jpr_projection_dim,
            use_bn=self.use_bn_in_jpr,
            use_gelu=self.use_gelu_in_jpr,
        )
        self.predictor = MLPHead(
            self.jpr_projection_dim,
            self.jpr_latent_dim,
            self.jpr_projection_dim,
            use_bn=self.use_bn_in_jpr,
            use_gelu=self.use_gelu_in_jpr,
        )

        self.W = nn.Parameter(torch.rand(self.n_embd, self.n_embd))

        self.ln = nn.LayerNorm(self.n_embd)
        self.position = PositionalEmbedding(self.n_embd)
        self.ma_transition_model = nn.Sequential(
            *[
                EncodeBlock(self.n_embd, self.n_head, self.n_agent * 2)
                for _ in range(self.n_layer)
            ]
        )

        # count the volume of parameters of model
        self.get_params_volume()

        self.transforms = []
        self.eval_transforms = []
        self.uses_augmentation = True
        self.aug_prob = aug_prob
        for aug in augmentation:
            if aug == "affine":
                transformation = RandomAffine(5, (0.14, 0.14), (0.9, 1.1), (-5, 5))
                eval_transformation = nn.Identity()
                self.uses_augmentation = True
            elif aug == "crop":
                transformation = RandomCrop((args.image_size, args.image_size))
                # Crashes if aug-prob not 1: use CenterCrop((args.image_size, args.image_size)) or Resize((args.image_size, args.image_size)) in that case.
                eval_transformation = CenterCrop((args.image_size, args.image_size))
                self.uses_augmentation = True
            elif aug == "rrc":
                transformation = RandomResizedCrop((100, 100), (0.8, 1))
                eval_transformation = nn.Identity()
                self.uses_augmentation = True
            elif aug == "blur":
                transformation = GaussianBlur2d((5, 5), (1.5, 1.5))
                eval_transformation = nn.Identity()
                self.uses_augmentation = True
            elif aug == "shift":
                transformation = nn.Sequential(
                    nn.ReplicationPad2d(4),
                    RandomCrop((args.image_size, args.image_size)),
                )
                eval_transformation = nn.Identity()
            elif aug == "intensity":
                transformation = Intensity(scale=0.05)
                eval_transformation = nn.Identity()
            elif aug == "none":
                transformation = eval_transformation = nn.Identity()
            else:
                raise NotImplementedError()
            self.transforms.append(transformation)
            self.eval_transforms.append(eval_transformation)

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.to(device)

    def forward(self, state_seq, obs_seq, action_seq):
        B, T, N, *_ = state_seq.shape

        obs_rep = self.obs_encoder.get_rep(state_seq[:, 0], obs_seq[:, 0])
        # pos_emb = self.position(N).repeat_interleave(B, dim=0)
        online_next_obs_rep = [obs_rep]
        target_next_obs_rep = []

        for t in range(T - 1):
            obs = obs_seq[:, t]
            if self.action_type == "Discrete":
                action = action_seq[:, t].long()
                one_hot_action = F.one_hot(
                    action.squeeze(-1), num_classes=self.action_dim
                )
                shifted_action = torch.zeros((B, N, self.action_dim + 1)).to(
                    **self.tpdv
                )
                shifted_action[:, 0, 0] = 1
                shifted_action[:, 1:, 1:] = one_hot_action[:, :-1, :]
            else:
                action = action_seq[:, t]
                shifted_action = torch.zeros((B, N, self.action_dim)).to(**self.tpdv)
                shifted_action[:, 1:, :] = action[:, :-1, :]
            act_rep = self.act_encoder.get_rep(
                shifted_action, online_next_obs_rep[-1], obs
            )
            x = torch.zeros(B, N * 2, self.n_embd).to(self.device)
            # x[:, ::2, :] = online_next_obs_rep[-1] + pos_emb
            # x[:, 1::2, :] = act_rep + pos_emb
            x[:, ::2, :] = online_next_obs_rep[-1]
            x[:, 1::2, :] = act_rep
            online_next_obs_rep.append(self.ma_transition_model(self.ln(x))[:, ::2, :])
            if self.jpr_ssl_tech in ["byol", "infonce"]:
                with torch.no_grad():
                    target_next_obs_rep.append(
                        self.target_obs_encoder.get_rep(state_seq[:, t+1], obs_seq[:, t+1])
                    )
            elif self.jpr_ssl_tech in ["simsiam", "mse"]:
                target_next_obs_rep.append(
                    self.obs_encoder.get_rep(state_seq[:, t+1], obs_seq[:, t+1])
                )

        online_next_obs_rep = torch.stack(online_next_obs_rep[1:], dim=1)
        target_next_obs_rep = torch.stack(target_next_obs_rep, dim=1)
        return online_next_obs_rep, target_next_obs_rep

    def get_loss(self, state, obs, action):
        # if state.dim() > 4 and state.shape[-1] == state.shape[-2]:
        #     state = self.transform(state, augment=True)
        # if obs.dim() > 4 and state.shape[-1] == state.shape[-2]:
        #     obs = self.transform(obs, augment=True)
        rep, target_rep = self.forward(state, obs, action)
        return getattr(self, f"{self.jpr_ssl_tech}_loss")(rep, target_rep)

    def byol_loss(self, rep, target_rep):
        B, T, N, D = rep.shape
        proj_rep = self.projector(rep.flatten(0, 1))
        pred_rep = self.predictor(proj_rep)
        with torch.no_grad():
            target_proj_rep = self.target_projector(target_rep.flatten(0, 1))
        pred_rep = pred_rep.view(B, T, N, D)
        target_proj_rep = target_proj_rep.view(B, T, N, D)
        loss = self.norm_mse_loss(pred_rep, target_proj_rep)
        return loss

    def infonce_loss(self, rep, target):
        raise NotImplementedError

    def simsiam_loss(self, rep, target_rep):
        return 0.5 * (
            self._simsiam_loss(rep, target_rep) + self._simsiam_loss(target_rep, rep)
        )

    def _simsiam_loss(self, rep, target_rep):
        B, T, N, D = rep.shape
        proj_rep = self.projector(rep).flatten(0, 1)
        pred_rep = self.predictor(proj_rep)
        pred_rep = pred_rep.view(B, T, N, D)
        with torch.no_grad():
            target_proj_rep = self.projector(target_rep).flatten(0, 1)
            target_proj_rep = target_proj_rep.view(B, T, N, D)
        loss = self.norm_mse_loss(pred_rep, target_proj_rep)
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
            .mean(2)
            .mean(1)
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
                ema_params += params
            elif "encoder" not in name:
                trainable_params += params
            else:
                online_encoder_params += params
        print("=" * 60)
        print(f"Total params of JPR: {total_params}")
        print(f"Trainable params of JPR: {trainable_params}+{online_encoder_params}")
        print(f"EMA params of JPR: {ema_params}")
        print("=" * 60)
