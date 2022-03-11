import numpy as np

from policies.base import ActorCriticPolicy
import torch
import torch.nn as nn


class MultiModalPolicy(ActorCriticPolicy):
    def __init__(self, image_shape, state_shape, action_dim):
        super(MultiModalPolicy, self).__init__()
        self._is_recurrent = False
        self._recurrent_hidden_state_size = 1
        self.image_shape = image_shape  # (n_frame, h, w)
        self.state_shape = state_shape  # int
        self.image_encoder = nn.Sequential(
            nn.Conv2d(self.image_shape[0], 2 * self.image_shape[0], 8, 4, 0),
            nn.ReLU(),
            nn.Conv2d(2 * self.image_shape[0], 4 * self.image_shape[0], 4, 2, 0),
            nn.ReLU(),
            nn.Conv2d(4 * self.image_shape[0], 4 * self.image_shape[0], 3, 1, 0),
            nn.ReLU(), nn.Flatten(),
        )
        with torch.no_grad():
            image_feature_dim = self.image_encoder(torch.zeros(1, *self.image_shape)).shape[-1]
        self.pi_state_encoder = nn.Sequential(
            nn.Linear(self.state_shape, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.pi_mean = nn.Sequential(
            nn.Linear(image_feature_dim + 128, 64), nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        self.pi_logstd = nn.Parameter(torch.zeros(action_dim), requires_grad=True)
        self.vf_state_encoder = nn.Sequential(
            nn.Linear(self.state_shape, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.vf_linears = nn.Sequential(
            nn.Linear(image_feature_dim + 128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, obs, rnn_hxs=None, rnn_masks=None, forward_vf=True):
        image_obs = torch.narrow(obs, dim=-1, start=0, length=int(np.prod(self.image_shape)))
        image_obs = torch.reshape(image_obs, [-1] + list(self.image_shape))
        state_obs = torch.narrow(obs, dim=-1, start=int(np.prod(self.image_shape)), length=self.state_shape)
        image_feature = self.image_encoder(image_obs)
        pi_state_feature = self.pi_state_encoder(state_obs)
        pi_joint_feature = torch.cat([image_feature, pi_state_feature], dim=-1)
        action_mean = self.pi_mean(pi_joint_feature)
        action_dist = torch.distributions.Normal(action_mean, torch.exp(self.pi_logstd))
        if forward_vf:
            vf_state_feature = self.vf_state_encoder(state_obs)
            vf_joint_feature = torch.cat([image_feature, vf_state_feature], dim=-1)
            value = self.vf_linears(vf_joint_feature)
        else:
            value = None
        return value, action_dist, rnn_hxs

    def act(self, obs, rnn_hxs=None, rnn_masks=None, deterministic=False):
        value, action_dist, rnn_hxs = self.forward(obs, rnn_hxs, rnn_masks)
        if deterministic:
            action = action_dist.mean
        else:
            action = action_dist.sample()
        log_probs = torch.sum(action_dist.log_prob(action), dim=-1, keepdim=True)
        return value, action, log_probs, rnn_hxs

    def evaluate_actions(self, obs, rnn_hxs, rnn_masks, actions):
        _, action_dist, rnn_hxs = self.forward(obs, rnn_hxs, rnn_masks, forward_vf=False)
        log_probs = torch.sum(action_dist.log_prob(actions), dim=-1, keepdim=True)
        entropy = action_dist.entropy()
        return log_probs, entropy, rnn_hxs
