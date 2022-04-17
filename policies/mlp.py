from policies.base import ActorCriticPolicy
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import torch.nn as nn
import torch
import numpy as np


class MlpPolicy(ActorCriticPolicy):
    def __init__(self, obs_dim, action_dim, hidden_size, num_bin):
        super(MlpPolicy, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_bin = num_bin
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size, hidden_size]
        assert isinstance(hidden_size, list) or isinstance(hidden_size, tuple)
        assert len(hidden_size) >= 2
        self.feature_extractor = nn.ModuleList()
        self.feature_extractor.append(nn.Linear(self.obs_dim, hidden_size[0]))
        for i in range(len(hidden_size) - 1):
            self.feature_extractor.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
        self.critic_linears = nn.Sequential(
            nn.Linear(hidden_size[-1], 1)
        )
        if num_bin ** self.action_dim > 100:
            self.arch = "independent"
            self.actor_linears = nn.ModuleList()
            for i in range(self.action_dim):
                self.actor_linears.append(
                    nn.Sequential(
                        nn.Linear(hidden_size[-1], num_bin)
                    )
                )
        else:
            self.arch = "joint"
            self.actor_linears = nn.Linear(hidden_size[-1], num_bin ** self.action_dim)
        self.is_recurrent = False
        self.recurrent_hidden_state_size = 1
    
    def forward(self, obs, rnn_hxs=None, rnn_masks=None):
        features = obs
        for m in self.feature_extractor:
            features = nn.functional.relu(m(features))
        values = self.critic_linears(features)
        if self.arch == "independent":
            action_dist = []
            for i in range(self.action_dim):
                logits = self.actor_linears[i](features)
                action_dist.append(Categorical(logits=logits))
        else:
            logits = self.actor_linears(features)
            action_dist = Categorical(logits=logits)
        return values, action_dist, rnn_hxs
    
    def act(self, obs, rnn_hxs=None, rnn_masks=None, deterministic=False):
        values, action_dist, rnn_hxs = self.forward(obs, rnn_hxs, rnn_masks)
        if self.arch == "independent":
            actions = []
            log_probs = []
            for i in range(self.action_dim):
                if deterministic:
                    actions.append(action_dist[i].probs.argmax(dim=-1, keepdim=True))
                else:
                    actions.append(action_dist[i].sample().unsqueeze(dim=-1))
                log_probs.append(action_dist[i].log_prob(actions[-1].squeeze(dim=-1)))
            actions = torch.cat(actions, dim=-1)
            log_probs = torch.sum(torch.stack(log_probs, dim=-1), dim=-1, keepdim=True)
            # Rescale to [-1, 1]
            actions = actions / (self.num_bin - 1.0) * 2 - 1
        else:
            if deterministic:
                actions = action_dist.probs.argmax(dim=-1, keepdim=True)
            else:
                actions = action_dist.sample().unsqueeze(dim=-1)
            log_probs = action_dist.log_prob(actions.squeeze(dim=-1)).unsqueeze(dim=-1)
            c_actions = []
            for _ in reversed(range(self.action_dim)):
                c_actions.insert(0, actions % self.num_bin)
                actions = actions // self.num_bin
            actions = torch.cat(c_actions, dim=-1) / (self.num_bin - 1.0) * 2 - 1
        return values, actions, log_probs, rnn_hxs
    
    def evaluate_actions(self, obs, rnn_hxs=None, rnn_masks=None,  actions=None):
        _, action_dist, rnn_hxs = self.forward(obs, rnn_hxs, rnn_masks)
        # Scale up actions
        actions = torch.round((actions + 1.0) / 2 * (self.num_bin - 1.0)).int().detach()
        if self.arch == "independent":
            log_probs = []
            for i in range(self.action_dim):
                log_probs.append(action_dist[i].log_prob(actions[:, i]))
            log_probs = torch.sum(torch.stack(log_probs, dim=-1), dim=-1, keepdim=True)
            entropy = torch.sum(torch.stack([dist.entropy() for dist in action_dist], dim=-1), dim=-1)
        else:
            digit = actions[:, 0]
            for i in range(1, actions.shape[1]):
                digit = (self.num_bin * digit + actions[:, i]).detach()
            log_probs = action_dist.log_prob(digit).unsqueeze(dim=-1)
            entropy = action_dist.entropy()
        entropy = entropy.mean()
        return log_probs, entropy, rnn_hxs


class MlpGaussianPolicy(ActorCriticPolicy):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super(MlpGaussianPolicy, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.policy_feature = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.actor_mean = nn.Linear(hidden_size, self.act_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(self.act_dim), requires_grad=True)
        self.value_feature = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden_size, 1)
        self.is_recurrent = False
        self.recurrent_hidden_state_size = 1
        self._initialize()

    def _initialize(self):
        nn.init.orthogonal_(self.value_head.weight, gain=np.sqrt(2))
        nn.init.constant_(self.value_head.bias, 0.)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.)

    @torch.jit.ignore
    def forward(self, obs, rnn_hxs=None, rnn_masks=None):
        policy_features = self.policy_feature(obs)
        policy_mean = self.actor_mean(policy_features)
        # policy_logstd = torch.clamp(self.actor_logstd, -5, 0)
        policy_logstd = self.actor_logstd
        try:
            action_dist = Normal(loc=policy_mean, scale=torch.exp(policy_logstd))
        except:
            print("policy mean", torch.isnan(policy_mean).any())
            print("actor mean weight", torch.isnan(self.actor_mean.weight).any(), "actor mean bias", torch.isnan(self.actor_mean.bias).any())
            print("policy features", torch.isnan(policy_features).any())
            print("obs", torch.isnan(obs).any())
            raise RuntimeError
        value_features = self.value_feature(obs)
        values = self.value_head(value_features)
        return values, action_dist, rnn_hxs

    @torch.jit.ignore
    def act(self, obs, rnn_hxs=None, rnn_masks=None, deterministic=False):
        values, action_dist, rnn_hxs = self.forward(obs, rnn_hxs, rnn_masks)
        if deterministic:
            actions = action_dist.loc
        else:
            actions = action_dist.sample()
        # todo: check shape
        log_prob = action_dist.log_prob(actions).sum(dim=-1, keepdim=True)
        return values, actions, log_prob, rnn_hxs

    @torch.jit.ignore
    def evaluate_actions(self, obs, rnn_hxs, rnn_masks, actions):
        _, action_dist, rnn_hxs = self.forward(obs, rnn_hxs, rnn_masks)
        log_prob = action_dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = action_dist.entropy()
        return log_prob, entropy, rnn_hxs

    @torch.jit.export
    def take_action(self, obs):
        policy_features = self.policy_feature(obs)
        policy_mean = self.actor_mean(policy_features)
        policy_mean = torch.clip(policy_mean, -1, 1)
        return policy_mean


class MlpDynamics(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super(MlpDynamics, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.state_feature = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.sa_feature = nn.Sequential(
            nn.Linear(self.obs_dim + self.act_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.value_head = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())

    def forward(self, obs, actions, rnn_hxs=None, rnn_masks=None):
        next_features = self.sa_feature.forward(torch.cat([obs, actions], dim=-1))
        values = self.value_head.forward(next_features)
        return values, next_features

    def get_state_feature(self, obs, rnn_hxs=None, rnn_masks=None):
        state_features = self.state_feature.forward(obs)
        return state_features

    def predict_without_action(self, obs, rnn_hxs=None, rnn_masks=None):
        state_feature = self.get_state_feature(obs, rnn_hxs, rnn_masks)
        values = self.value_head.forward(state_feature)
        return values
