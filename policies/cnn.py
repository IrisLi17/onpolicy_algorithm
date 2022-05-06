from turtle import forward, hideturtle
from policies.base import ActorCriticPolicy
import torch
import torch.nn as nn
import numpy as np


class CNNStatePolicy(ActorCriticPolicy):
    def __init__(self, image_shape, state_dim, action_dim, hidden_size) -> None:
        super().__init__()
        if len(image_shape) == 4:
            pass
        else:
            assert len(image_shape) == 3
            image_shape = (1, *image_shape)
        self.image_shape = image_shape  # (L, C, H, W)
        assert image_shape[-1] == image_shape[-2] == 84
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        
        self.image_encoder = nn.Sequential(
            nn.Conv2d(self.image_shape[1], 2 * self.image_shape[1], 8, 4, 0),
            nn.ReLU(),
            nn.Conv2d(2 * self.image_shape[1], 4 * self.image_shape[1], 4, 2, 0),
            nn.ReLU(),
            nn.Conv2d(4 * self.image_shape[1], 4 * self.image_shape[1], 3, 1, 0),
            nn.ReLU(), nn.Flatten(),
        )
        with torch.no_grad():
            image_feature_dim = self.image_encoder(torch.zeros(1, *self.image_shape[1:])).shape[-1]
        self.image_projector = nn.Linear(image_feature_dim * self.image_shape[0], hidden_size)
        self.pi_state_encoder = nn.Linear(self.state_dim, self.hidden_size)
        self.vf_state_encoder = nn.Linear(self.state_dim, self.hidden_size)
        self.pi_layer_norm = nn.LayerNorm(hidden_size + hidden_size)
        self.vf_layer_norm = nn.LayerNorm(hidden_size + hidden_size)
        self.pi_mean_layers = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.action_dim)
        )
        self.pi_logstd = nn.Parameter(torch.zeros(self.action_dim), requires_grad=True)
        self.vf_layers = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.is_recurrent = False
        self.recurrent_hidden_state_size = 1
        self._initialize()

    def _initialize(self):
        for net in self.vf_layers:
            if isinstance(net, nn.Linear):
                nn.init.orthogonal_(net.weight, gain=np.sqrt(2))
                nn.init.constant_(net.bias, 0.)
        nn.init.orthogonal_(self.pi_mean_layers[-1].weight, gain=0.01)
        nn.init.constant_(self.pi_mean_layers[-1].bias, 0.)
    
    @torch.jit.ignore
    def forward(self, obs, rnn_hxs=None, rnn_masks=None):
        assert obs.shape[-1] == torch.prod(torch.tensor(self.image_shape)) + self.state_dim
        batch_size = obs.shape[0]
        image_obs = torch.narrow(obs, dim=1, start=0, length=torch.prod(torch.tensor(self.image_shape)))
        image_obs = image_obs.reshape((-1, *self.image_shape[1:]))  # (batch_size * L, C, H, W)
        state_obs = torch.narrow(obs, dim=1, start=torch.prod(torch.tensor(self.image_shape)), length=self.state_dim)
        image_feature = self.image_projector(self.image_encoder(image_obs).reshape((batch_size, -1)))
        pi_state_feature = self.pi_state_encoder(state_obs)
        vf_state_feature = self.vf_state_encoder(state_obs)
        pi_feature = self.pi_layer_norm(torch.cat([image_feature, pi_state_feature], dim=-1))
        vf_feature = self.vf_layer_norm(torch.cat([image_feature, vf_state_feature], dim=-1))
        action_mean = self.pi_mean_layers(pi_feature)
        dist = torch.distributions.Normal(loc=action_mean, scale=torch.exp(self.pi_logstd))
        value = self.vf_layers(vf_feature)
        return value, dist, rnn_hxs
    
    def act(self, obs, rnn_hxs=None, rnn_masks=None, deterministic=False):
        value, action_dist, rnn_hxs = self.forward(obs, rnn_hxs, rnn_masks)
        if deterministic:
            action = action_dist.mean
        else:
            action = action_dist.sample()
        log_probs = torch.sum(action_dist.log_prob(action), dim=-1, keepdim=True)
        return value, action, log_probs, rnn_hxs
    
    def evaluate_actions(self, obs, rnn_hxs=None, rnn_masks=None, actions=None):
        _, action_dist, rnn_hxs = self.forward(obs, rnn_hxs, rnn_masks)
        log_probs = torch.sum(action_dist.log_prob(actions), dim=-1, keepdim=True)
        entropy = action_dist.entropy()
        return log_probs, entropy, rnn_hxs