from policies.base import ActorCriticPolicy
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class HybridMlpPolicy(ActorCriticPolicy):
    def __init__(
        self, mvp_feat_dim, state_obs_dim, n_primitive, act_dim, num_bin,
        hidden_dim, proj_img_dim, proj_state_dim, use_param_mask=False,
        privilege_dim=0
    ) -> None:
        super().__init__()
        self.mvp_feat_dim = mvp_feat_dim
        self.state_obs_dim = state_obs_dim
        self.privilege_dim = privilege_dim
        self.n_primitive = n_primitive
        self.act_dim = act_dim
        self.num_bin = num_bin
        self.mvp_projector = nn.Linear(mvp_feat_dim, proj_img_dim)
        self.state_linear = nn.Linear(state_obs_dim, proj_state_dim)
        if privilege_dim > 0:
            self.privilege_state_linear = nn.Linear(state_obs_dim + privilege_dim, proj_state_dim)
        else:
            self.privilege_state_linear = None
        self.act_type = nn.Sequential(
            nn.Linear(2 * proj_img_dim + proj_state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_primitive),
        )
        self.act_param = nn.ModuleList([nn.Sequential(
            nn.Linear(2 * proj_img_dim + proj_state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_bin)
        ) for _ in range(act_dim)])
        self.value_layers = nn.Sequential(
            nn.Linear(2 * proj_img_dim + proj_state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.is_recurrent = False
        self.recurrent_hidden_state_size = 1
        self.use_param_mask = use_param_mask
        # self.init_weights(self.act_type, [np.sqrt(2), np.sqrt(2), 0.01])
        # for i in range(act_dim):
        #     self.init_weights(self.act_param[i], [np.sqrt(2), np.sqrt(2), 0.01])
        # self.init_weights(self.value_layers, [np.sqrt(2), np.sqrt(2), 1.0])
    
    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def _obs_parser(self, obs):
        cur_img_feat = torch.narrow(obs, dim=-1, start=0, length=self.mvp_feat_dim)
        cur_state = torch.narrow(obs, dim=-1, start=self.mvp_feat_dim, length=self.state_obs_dim)
        goal_img_feat = torch.narrow(obs, dim=-1, start=self.mvp_feat_dim + self.state_obs_dim, length=self.mvp_feat_dim)
        privilege_info = torch.narrow(obs, dim=-1, start=2 * self.mvp_feat_dim + self.state_obs_dim, length=self.privilege_dim)
        return cur_img_feat.detach(), cur_state.detach(), goal_img_feat.detach(), privilege_info.detach()

    def forward(self, obs, rnn_hxs=None, rnn_masks=None):
        cur_img_feat, cur_state, goal_img_feat, privilege_info = self._obs_parser(obs)
        proj_cur_feat = self.mvp_projector(cur_img_feat)
        proj_goal_feat = self.mvp_projector(goal_img_feat)
        proj_state_feat = self.state_linear(cur_state)
        proj_input = torch.cat([proj_cur_feat, proj_state_feat, proj_goal_feat], dim=-1)
        if self.privilege_state_linear is not None:
            proj_priv_state_feat = self.privilege_state_linear(torch.cat([cur_state, privilege_info], dim=-1))
            proj_value_input = torch.cat([proj_cur_feat, proj_priv_state_feat, proj_goal_feat], dim=-1)
        else:
            proj_value_input = proj_input
        act_type_logits = self.act_type(proj_input)
        act_type_dist = Categorical(logits=act_type_logits)
        act_param_logits = [self.act_param[i](proj_input) for i in range(len(self.act_param))]
        act_param_dist = [Categorical(logits=act_param_logits[i]) for i in range(self.act_dim)]
        value_pred = self.value_layers(proj_value_input)
        return value_pred, (act_type_dist, act_param_dist), rnn_hxs
    
    def act(self, obs, rnn_hxs=None, rnn_masks=None, deterministic=False):
        value_pred, (act_type_dist, act_param_dist), rnn_hxs = self.forward(obs, rnn_hxs, rnn_masks)
        if deterministic:
            act_type = act_type_dist.probs.argmax(dim=-1, keepdim=True)
            act_params = [act_param_dist[i].probs.argmax(dim=-1, keepdim=True) for i in range(len(act_param_dist))]
        else:
            act_type = act_type_dist.sample().unsqueeze(dim=-1)
            act_params = [act_param_dist[i].sample().unsqueeze(dim=-1) for i in range(len(act_param_dist))]
        act_type_logprob = act_type_dist.log_prob(act_type.squeeze(dim=-1)).unsqueeze(dim=-1)
        if self.use_param_mask:
            use_param_mask = (act_type == 0).float().detach()
        else:
            use_param_mask = 1
        act_param_logprob = [
            act_param_dist[i].log_prob(act_params[i].squeeze(dim=-1)).unsqueeze(dim=-1) * use_param_mask
            for i in range(self.act_dim)
        ]
        act_params = [2 * (act_param / (self.num_bin - 1.0)) - 1 for act_param in act_params]
        actions = torch.cat([act_type] + act_params, dim=-1)
        log_prob = torch.sum(torch.stack([act_type_logprob] + act_param_logprob, dim=-1), dim=-1)
        return value_pred, actions, log_prob, rnn_hxs
    
    def evaluate_actions(self, obs, rnn_hxs, rnn_masks, actions):
        _, (act_type_dist, act_param_dist), _ = self.forward(obs, rnn_hxs, rnn_masks)
        act_type = actions[:, 0].int()
        act_params = torch.round((actions[:, 1:] + 1) / 2 * (self.num_bin - 1.0)).int()
        if self.use_param_mask:
            use_param_mask = (act_type == 0).float().detach()
        else:
            use_param_mask = 1
        act_type_logprob = act_type_dist.log_prob(act_type)
        act_param_logprob = [
            act_param_dist[i].log_prob(act_params[:, i]) * use_param_mask
            for i in range(self.act_dim)
        ]
        log_prob = torch.sum(torch.stack([act_type_logprob] + act_param_logprob, dim=-1), dim=-1, keepdim=True)
        act_type_ent = act_type_dist.entropy()
        act_param_ent = [act_param_dist[i].entropy() * use_param_mask for i in range(self.act_dim)]
        entropy = torch.sum(torch.stack([act_type_ent] + act_param_ent, dim=-1), dim=-1).mean()
        return log_prob, entropy, rnn_hxs
