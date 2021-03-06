"""Actor critic."""

import numpy as np
import os
import torch
import torch.nn as nn

from torch.distributions import MultivariateNormal

import policies.mvp.vit as vit


###############################################################################
# Pixels
###############################################################################

_HOI_MODELS = {
    "maevit-s16": "mae_pretrain_hoi_vit_small.pth",
}

_IN_MODELS = {
    "vit-s16": "sup_pretrain_imagenet_vit_small.pth",
    "maevit-s16": "mae_pretrain_imagenet_vit_small.pth",
}


class Encoder(nn.Module):

    def __init__(self, model_type, pretrain_dir, pretrain_type, freeze, emb_dim):
        super(Encoder, self).__init__()
        assert pretrain_type in ["imagenet", "hoi", "none"]
        if pretrain_type == "imagenet":
            assert model_type in _IN_MODELS
            pretrain_fname = _IN_MODELS[model_type]
            pretrain_path = os.path.join(pretrain_dir, pretrain_fname)
        elif pretrain_type == "hoi":
            assert model_type in _HOI_MODELS
            pretrain_fname = _HOI_MODELS[model_type]
            pretrain_path = os.path.join(pretrain_dir, pretrain_fname)
        else:
            pretrain_path = "none"
        assert pretrain_type == "none" or os.path.exists(pretrain_path)
        self.backbone, gap_dim = vit.vit_s16(pretrain_path)
        self.gap_dim = gap_dim
        if freeze:
            self.backbone.freeze()
        self.freeze = freeze
        self.projector = nn.Linear(gap_dim, emb_dim)

    @torch.no_grad()
    def forward(self, x):
        feat = self.backbone.extract_feat(x)
        return self.projector(self.backbone.forward_norm(feat)), feat

    def forward_feat(self, feat):
        return self.projector(self.backbone.forward_norm(feat))


class PixelActorCritic(nn.Module):

    def __init__(
        self,
        image_shape,
        states_shape,
        actions_shape,
        initial_std,
        encoder_cfg,
        policy_cfg,
    ):
        super(PixelActorCritic, self).__init__()
        assert encoder_cfg is not None

        self.is_recurrent = False
        self.recurrent_hidden_state_size = 1

        # Encoder params
        model_type = encoder_cfg["model_type"]
        pretrain_dir = encoder_cfg["pretrain_dir"]
        pretrain_type = encoder_cfg["pretrain_type"]
        freeze = encoder_cfg["freeze"]
        self.emb_dim = emb_dim = encoder_cfg["emb_dim"]

        # Policy params
        actor_hidden_dim = policy_cfg["pi_hid_sizes"]
        critic_hidden_dim = policy_cfg["vf_hid_sizes"]
        activation = nn.SELU()

        # Obs and state encoders
        self.obs_enc = Encoder(
            model_type=model_type,
            pretrain_dir=pretrain_dir,
            pretrain_type=pretrain_type,
            freeze=freeze,
            emb_dim=emb_dim
        )
        self.image_shape = image_shape
        self.state_dim = states_shape[0]
        self.state_enc = nn.Linear(states_shape[0], emb_dim)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(emb_dim * 2, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for li in range(len(actor_hidden_dim)):
            if li == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[li], *actions_shape))
            else:
                actor_layers.append(
                    nn.Linear(actor_hidden_dim[li], actor_hidden_dim[li + 1])
                )
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(emb_dim * 2, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for li in range(len(critic_hidden_dim)):
            if li == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[li], 1))
            else:
                critic_layers.append(
                    nn.Linear(critic_hidden_dim[li], critic_hidden_dim[li + 1])
                )
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(self.obs_enc)
        print(self.state_enc)
        print(self.actor)
        print(self.critic)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        self.init_weights(self.actor, actor_weights)
        self.init_weights(self.critic, critic_weights)

    @property
    def obs_feature_size(self):
        return self.obs_enc.gap_dim + self.state_dim

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(
                mod for mod in sequential if isinstance(mod, nn.Linear)
            )
        ]

    @torch.no_grad()
    def act(self, feature_obs, rnn_hxs=None, rnn_masks=None, deterministic=False):
        image_feat = torch.narrow(feature_obs, dim=1, start=0, length=self.obs_enc.gap_dim)
        state_obs = torch.narrow(feature_obs, dim=1, start=self.obs_enc.gap_dim, length=self.state_dim)
        image_emb = self.obs_enc.forward_feat(image_feat)
        state_emb = self.state_enc(state_obs)
        joint_emb = torch.cat([image_emb, state_emb], dim=1)

        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        if deterministic:
            actions = actions_mean
        else:
            actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions).unsqueeze(dim=-1)

        value = self.critic(joint_emb)

        return value, actions, actions_log_prob, rnn_hxs

    @torch.no_grad()
    def encode_obs(self, observations):
        assert observations.shape[1] == int(np.prod(self.image_shape)) + self.state_dim
        image_obs = torch.narrow(observations, dim=1, start=0, length=int(observations.shape[1] - self.state_dim))
        image_obs = image_obs.reshape((-1, *self.image_shape))
        state_obs = torch.narrow(observations, dim=1, start=int(np.prod(self.image_shape)), length=self.state_dim)
        obs_emb, obs_feat = self.obs_enc(image_obs)
        return torch.cat([obs_feat, state_obs], dim=-1).detach()

    def get_value(self, feature_obs, rnn_hxs=None, rnn_masks=None):
        image_feat = torch.narrow(feature_obs, dim=1, start=0, length=self.obs_enc.gap_dim)
        state_obs = torch.narrow(feature_obs, dim=1, start=self.obs_enc.gap_dim, length=self.state_dim)
        image_emb = self.obs_enc.forward_feat(image_feat)
        state_emb = self.state_enc(state_obs)
        joint_emb = torch.cat([image_emb, state_emb], dim=1)
        value = self.critic(joint_emb)
        return value

    def evaluate_actions(self, feature_obs, rnn_hxs=None, rnn_masks=None, actions=None):
        image_feat = torch.narrow(feature_obs, dim=1, start=0, length=self.obs_enc.gap_dim)
        state_obs = torch.narrow(feature_obs, dim=1, start=self.obs_enc.gap_dim, length=self.state_dim)
        image_emb = self.obs_enc.forward_feat(image_feat)
        state_emb = self.state_enc(state_obs)
        joint_emb = torch.cat([image_emb, state_emb], dim=1)

        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions).unsqueeze(dim=-1)
        entropy = distribution.entropy()

        return actions_log_prob, entropy, rnn_hxs
