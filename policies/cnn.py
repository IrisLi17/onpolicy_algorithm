from policies.base import ActorCriticPolicy
import torch
import torch.nn as nn
import numpy as np
from utils.distributions import Normal
from torch.distributions import Categorical
import torchvision


class CNNStatePolicy(ActorCriticPolicy):
    def __init__(self, image_shape, state_dim, action_dim, hidden_size, previ_dim=0, state_only_critic=False) -> None:
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
        self.previ_dim = previ_dim
        self.state_only_critic = state_only_critic
        
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
        if not self.state_only_critic:
            self.critic_image_encoder = nn.Sequential(
                nn.Conv2d(self.image_shape[1], 2 * self.image_shape[1], 8, 4, 0),
                nn.ReLU(),
                nn.Conv2d(2 * self.image_shape[1], 4 * self.image_shape[1], 4, 2, 0),
                nn.ReLU(),
                nn.Conv2d(4 * self.image_shape[1], 4 * self.image_shape[1], 3, 1, 0),
                nn.ReLU(), nn.Flatten(),
            )
            self.critic_image_projector = nn.Linear(image_feature_dim * self.image_shape[0], hidden_size)
        
        self.pi_state_encoder = nn.Linear(self.state_dim, self.hidden_size)
        self.vf_state_encoder = nn.Linear(self.state_dim + self.previ_dim, self.hidden_size)
        self.pi_layer_norm = nn.LayerNorm(hidden_size + hidden_size)
        self.vf_layer_norm = nn.LayerNorm(hidden_size + hidden_size)
        self.pi_feature_layer = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.pi_mean_layer = nn.Linear(hidden_size, self.action_dim - 1)
        self.gripper_layer = nn.Linear(hidden_size, 3)
        # Modify architecture as in MLP!
        self.pi_logstd = nn.Parameter(torch.zeros(self.action_dim - 1), requires_grad=True)
        vf_input_dim = hidden_size if self.state_only_critic else 2 * hidden_size
        self.vf_layers = nn.Sequential(
            nn.Linear(vf_input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.aux_layer = nn.Sequential(
            nn.Linear(2 * hidden_size, 3)
        )
        self.aux_metric = nn.L1Loss()
        self.is_recurrent = False
        self.recurrent_hidden_state_size = 1
        self._initialize()

    def _initialize(self):
        for net in self.vf_layers:
            if isinstance(net, nn.Linear):
                nn.init.orthogonal_(net.weight, gain=np.sqrt(2))
                nn.init.constant_(net.bias, 0.)
        nn.init.orthogonal_(self.pi_mean_layer.weight, gain=0.01)
        nn.init.constant_(self.pi_mean_layer.bias, 0.)
    
    @torch.jit.ignore
    def forward(self, obs, rnn_hxs=None, rnn_masks=None, previ_obs=None, forward_value=True):
        assert obs.shape[-1] == torch.prod(torch.tensor(self.image_shape)) + self.state_dim
        batch_size = obs.shape[0]
        image_obs = torch.narrow(obs, dim=1, start=0, length=torch.prod(torch.tensor(self.image_shape)))
        image_obs = image_obs.reshape((-1, *self.image_shape[1:]))  # (batch_size * L, C, H, W)
        state_obs = torch.narrow(obs, dim=1, start=torch.prod(torch.tensor(self.image_shape)), length=self.state_dim)
        image_feature = self.image_projector(self.image_encoder(image_obs).reshape((batch_size, -1)))
        pi_state_feature = self.pi_state_encoder(state_obs)
        pi_feature = self.pi_layer_norm(torch.cat([image_feature, pi_state_feature], dim=-1))
        pi_feature = self.pi_feature_layer(pi_feature)
        arm_mean = self.pi_mean_layer(pi_feature)
        gripper_logits = self.gripper_layer(pi_feature)
        dist = [Normal(loc=arm_mean, scale=torch.exp(self.pi_logstd)), Categorical(logits=gripper_logits)]
        if forward_value:
            if not self.state_only_critic:
                critic_image_feature = self.critic_image_projector(self.critic_image_encoder(image_obs).reshape((batch_size, -1)))
            if self.previ_dim > 0:
                assert previ_obs.shape[1] == self.previ_dim
                critic_state_obs = torch.cat([state_obs, previ_obs], dim=-1)
            else:
                critic_state_obs = torch.clone(state_obs)
            vf_state_feature = self.vf_state_encoder(critic_state_obs)
            if not self.state_only_critic:
                vf_feature = self.vf_layer_norm(torch.cat([critic_image_feature, vf_state_feature], dim=-1))
            else:
                vf_feature = vf_state_feature
            value = self.vf_layers(vf_feature)
        else:
            value = None
        return value, dist, rnn_hxs
    
    def act(self, obs, rnn_hxs=None, rnn_masks=None, deterministic=False, previ_obs=None, forward_value=True):
        value, action_dist, rnn_hxs = self.forward(obs, rnn_hxs, rnn_masks, previ_obs, forward_value)
        if deterministic:
            arm_action = action_dist[0].mean
            gripper_action = torch.argmax(action_dist[1], dim=-1, keepdim=True)
        else:
            arm_action = action_dist[0].sample([])
            gripper_action = action_dist[1].sample().unsqueeze(dim=-1)
        normed_gripper_action = gripper_action / 2.0 * 2 - 1
        action = torch.cat([arm_action, normed_gripper_action], dim=-1)
        log_probs = action_dist[0].log_prob(arm_action).sum(dim=-1, keepdim=True) + action_dist[1].log_prob(gripper_action.squeeze(dim=-1)).unsqueeze(dim=-1)
        return value, action, log_probs, rnn_hxs
    
    def evaluate_actions(self, obs, rnn_hxs=None, rnn_masks=None, actions=None):
        _, action_dist, rnn_hxs = self.forward(obs, rnn_hxs, rnn_masks, forward_value=False)
        arm_action = torch.narrow(actions, dim=1, start=0, length=3)
        gripper_action = ((torch.narrow(actions, dim=1, start=3, length=1) + 1) / 2 * 2).to(torch.int)
        log_probs = action_dist[0].log_prob(arm_action).sum(dim=-1, keepdim=True) + action_dist[1].log_prob(gripper_action.squeeze(dim=-1)).unsqueeze(dim=-1)
        entropy = action_dist[0].entropy()
        return log_probs, entropy, rnn_hxs
    
    def compute_aux_loss(self, obs, rnn_hxs=None, rnn_masks=None, aux_input=None):
        batch_size = obs.shape[0]
        image_obs = torch.narrow(obs, dim=1, start=0, length=torch.prod(torch.tensor(self.image_shape)))
        image_obs = image_obs.reshape((-1, *self.image_shape[1:]))  # (batch_size * L, C, H, W)
        state_obs = torch.narrow(obs, dim=1, start=torch.prod(torch.tensor(self.image_shape)), length=self.state_dim)
        image_feature = self.image_projector(self.image_encoder(image_obs).reshape((batch_size, -1)))
        pi_state_feature = self.pi_state_encoder(state_obs)
        pi_feature = self.pi_layer_norm(torch.cat([image_feature, pi_state_feature], dim=-1))
        pi_feature = self.pi_feature_layer(pi_feature)
        predict = self.aux_layer(pi_feature)
        aux_ground_truth = torch.narrow(aux_input, dim=1, start=0, length=3)
        loss = self.aux_metric(predict, aux_ground_truth.detach())
        return loss

    @torch.jit.export
    def take_action(self, obs):
        assert obs.shape[-1] == torch.prod(torch.tensor(self.image_shape)) + self.state_dim
        batch_size = obs.shape[0]
        image_obs = torch.narrow(obs, dim=1, start=0, length=torch.prod(torch.tensor(self.image_shape)))
        image_obs = image_obs.reshape((-1, *self.image_shape[1:]))  # (batch_size * L, C, H, W)
        state_obs = torch.narrow(obs, dim=1, start=torch.prod(torch.tensor(self.image_shape)), length=self.state_dim)
        image_feature = self.image_projector(self.image_encoder(image_obs).reshape((batch_size, -1)))
        pi_state_feature = self.pi_state_encoder(state_obs)
        pi_feature = self.pi_layer_norm(torch.cat([image_feature, pi_state_feature], dim=-1))
        pi_feature = self.pi_feature_layer(pi_feature)
        arm_action = self.pi_mean_layer(pi_feature)
        gripper_action = torch.argmax(self.gripper_layer(pi_feature), dim=-1, keepdim=True)
        normed_gripper_action = gripper_action / 2.0 * 2 - 1
        action_mean = torch.cat([arm_action, normed_gripper_action], dim=-1)
        return action_mean


class CNNStateHistoryPolicy(ActorCriticPolicy):
    def __init__(self, image_shape, state_dim, action_dim, hidden_size, lstm_hidden_size, previ_dim=0, state_only_critic=False) -> None:
        super().__init__()
        assert len(image_shape) == 3
        assert image_shape[1] == image_shape[2]
        self.image_shape = image_shape  # (C, H, W)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.previ_dim = previ_dim
        self.state_only_critic = state_only_critic
        self.image_transformer = torchvision.transforms.ColorJitter(hue=0.1)

        # Recurrent perception part
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
        self.image_projector = nn.Linear(image_feature_dim, hidden_size)
        if not state_only_critic:
            self.critic_image_encoder = nn.Sequential(
                nn.Conv2d(self.image_shape[0], 2 * self.image_shape[0], 8, 4, 0),
                nn.ReLU(),
                nn.Conv2d(2 * self.image_shape[0], 4 * self.image_shape[0], 4, 2, 0),
                nn.ReLU(),
                nn.Conv2d(4 * self.image_shape[0], 4 * self.image_shape[0], 3, 1, 0),
                nn.ReLU(), nn.Flatten(),
            )
            self.critic_image_projector = nn.Linear(image_feature_dim, hidden_size)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.critic_state_encoder = nn.Sequential(
            nn.Linear(self.state_dim + self.previ_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.memory_cell = nn.LSTMCell(2 * hidden_size, lstm_hidden_size)
        # self.critic_memory_cell = nn.LSTMCell(2 * hidden_size, lstm_hidden_size)
        self.pi_mean_layers = nn.Sequential(
            nn.Linear(lstm_hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, self.action_dim)
        )
        self.pi_logstd = nn.Parameter(torch.zeros(self.action_dim), requires_grad=True)
        vf_input_dim = hidden_size if self.state_only_critic else 2 * hidden_size
        self.vf_layers = nn.Sequential(
            nn.Linear(vf_input_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.aux_layer = nn.Sequential(
            nn.Linear(lstm_hidden_size, 3)
        )
        self.aux_metric = nn.L1Loss()
        self.is_recurrent = True
        self.recurrent_hidden_state_size = 2 * lstm_hidden_size
        self._initialize()
    
    def _initialize(self):
        for net in self.vf_layers:
            if isinstance(net, nn.Linear):
                nn.init.orthogonal_(net.weight, gain=np.sqrt(2))
                nn.init.constant_(net.bias, 0.)
        nn.init.orthogonal_(self.pi_mean_layers[-1].weight, gain=0.01)
        nn.init.constant_(self.pi_mean_layers[-1].bias, 0.)
    
    def _forward_feature(self, obs, rnn_hxs: torch.Tensor, rnn_masks: torch.Tensor, previ_obs: torch.Tensor, forward_value: bool=True):
        output_feature, new_rnn_hxs = self._forward_policy_feature(obs, rnn_hxs, rnn_masks)
        if forward_value:
            critic_output_feature = self._forward_critic_feature(obs, previ_obs)
        else:
            critic_output_feature = None
        return output_feature, critic_output_feature, new_rnn_hxs

    def _forward_policy_feature(self, obs: torch.Tensor, rnn_hxs: torch.Tensor, rnn_masks: torch.Tensor):
        assert obs.shape[-1] == torch.prod(torch.tensor(self.image_shape)) + self.state_dim
        image_obs = torch.narrow(obs, dim=1, start=0, length=torch.prod(torch.tensor(self.image_shape))).reshape((-1, *self.image_shape))
        image_obs = self.image_transformer(image_obs)
        state_obs = torch.narrow(obs, dim=1, start=torch.prod(torch.tensor(self.image_shape)), length=self.state_dim)
        image_feature = self.image_projector(self.image_encoder(image_obs))
        state_feature = self.state_encoder(state_obs)
        input_feature = torch.cat([image_feature, state_feature], dim=-1)
        N = rnn_hxs.shape[0]
        T = obs.shape[0] // N
        input_feature = input_feature.view(T, N, -1)
        rnn_masks = rnn_masks.view(T, N, 1)
        lstm_hxs = torch.narrow(rnn_hxs, dim=1, start=0, length=self.recurrent_hidden_state_size // 2)
        lstm_cell = torch.narrow(rnn_hxs, dim=1, start=self.recurrent_hidden_state_size // 2, length=self.recurrent_hidden_state_size // 2)
        output_feature = []
        for t in range(T):
            lstm_hxs, lstm_cell = self.memory_cell(input_feature[t], (lstm_hxs * rnn_masks[t], lstm_cell))
            output_feature.append(lstm_hxs)
        output_feature = torch.stack(output_feature, dim=0).reshape((T * N, -1))
        new_rnn_hxs = torch.cat([lstm_hxs, lstm_cell], dim=-1)
        return output_feature, new_rnn_hxs

    def _forward_critic_feature(self, obs: torch.Tensor, previ_obs: torch.Tensor):
        assert obs.shape[-1] == torch.prod(torch.tensor(self.image_shape)) + self.state_dim
        image_obs = torch.narrow(obs, dim=1, start=0, length=torch.prod(torch.tensor(self.image_shape))).reshape((-1, *self.image_shape))
        image_obs = self.image_transformer(image_obs)
        state_obs = torch.narrow(obs, dim=1, start=torch.prod(torch.tensor(self.image_shape)), length=self.state_dim)
        if not self.state_only_critic:
            critic_image_feature = self.critic_image_projector(self.critic_image_encoder(image_obs))
        if self.previ_dim > 0:
            assert previ_obs.shape[1] == self.previ_dim
            critic_state_obs = torch.cat([state_obs, previ_obs], dim=-1)
        else:
            critic_state_obs = torch.clone(state_obs)
        critic_state_feature = self.critic_state_encoder(critic_state_obs)
        if not self.state_only_critic:
            critic_output_feature = torch.cat([critic_image_feature, critic_state_feature], dim=-1)
        else:
            critic_output_feature = critic_state_feature
        return critic_output_feature

    @torch.jit.ignore
    def forward(self, obs, rnn_hxs: torch.Tensor, rnn_masks: torch.Tensor, previ_obs: torch.Tensor=None, forward_value: bool=True):
        output_feature, critic_output_feature, rnn_hxs = self._forward_feature(obs, rnn_hxs, rnn_masks, previ_obs, forward_value)
        action_mean = self.pi_mean_layers(output_feature)
        action_dist = Normal(action_mean, torch.exp(self.pi_logstd))
        if forward_value:
            value = self.vf_layers(critic_output_feature)
        else:
            value = None
        return value, action_dist, rnn_hxs
    
    def act(self, obs, rnn_hxs=None, rnn_masks=None, deterministic=False, previ_obs=None, forward_value=True):
        value, action_dist, rnn_hxs = self.forward(obs, rnn_hxs, rnn_masks, previ_obs, forward_value)
        if deterministic:
            action = action_dist.mean
        else:
            action = action_dist.sample([])
        log_probs = torch.sum(action_dist.log_prob(action), dim=-1, keepdim=True)
        return value, action, log_probs, rnn_hxs
    
    def evaluate_actions(self, obs, rnn_hxs=None, rnn_masks=None, actions=None):
        _, action_dist, rnn_hxs = self.forward(obs, rnn_hxs, rnn_masks, forward_value=False)
        log_probs = torch.sum(action_dist.log_prob(actions), dim=-1, keepdim=True)
        entropy = action_dist.entropy()
        return log_probs, entropy, rnn_hxs
    
    def compute_aux_loss(self, obs, rnn_hxs, rnn_masks, aux_input):
        output_feature, _, rnn_hxs = self._forward_feature(obs, rnn_hxs, rnn_masks, None, forward_value=False)
        predict = self.aux_layer(output_feature)
        aux_ground_truth = torch.narrow(aux_input, dim=1, start=0, length=3)
        loss = self.aux_metric(predict, aux_ground_truth.detach())
        return loss
    
    def get_value(self, obs, rnn_hxs=None, rnn_masks=None, previ_obs=None):
        value, dist, rnn_hxs = self.forward(obs, rnn_hxs, rnn_masks, previ_obs)
        return value
    
    @torch.jit.export
    def take_action(self, obs, rnn_hxs, rnn_masks):
        output_feature, rnn_hxs = self._forward_policy_feature(obs, rnn_hxs, rnn_masks)
        action = self.pi_mean_layers(output_feature)
        # _, action_dist, rnn_hxs = self.forward(obs, rnn_hxs, rnn_masks, previ_obs, forward_value=False)
        # action = action_dist.mean
        return action, rnn_hxs
