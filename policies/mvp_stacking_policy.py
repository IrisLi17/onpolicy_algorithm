from policies.base import ActorCriticPolicy
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical


class MvpStackingPolicy(ActorCriticPolicy):
    def __init__(
        self, mvp_feat_dim, n_primitive, act_dim, num_bin,
        proj_img_dim, privilege_dim=0, state_only_value=False,
    ) -> None:
        super().__init__()
        self.mvp_feat_dim = mvp_feat_dim
        self.robot_state_dim = 7
        self.privilege_dim = privilege_dim
        self.n_primitive = n_primitive
        self.act_dim = act_dim
        self.num_bin = num_bin
        self.mvp_projector = nn.Linear(mvp_feat_dim, proj_img_dim)
        if not state_only_value:
            self.value_mvp_projector = nn.Linear(mvp_feat_dim, proj_img_dim)
        else:
            assert privilege_dim > 0
            self.value_mvp_projector = None
        self.act_feature = nn.Sequential(
            nn.Linear(2 * proj_img_dim, 256), nn.SELU(),
            nn.Linear(256, 256), nn.SELU(),
            nn.Linear(256, 256), nn.SELU()
        )
        self.act_type = nn.Sequential(
            nn.Linear(256, n_primitive)
        )
        self.act_param = nn.ModuleList([nn.Sequential(
            nn.Linear(256, num_bin)
        ) for _ in range(act_dim)])
        if not state_only_value:
            self.value_layers = nn.Sequential(
                nn.Linear(2 * proj_img_dim + privilege_dim, 256), nn.SELU(),
                nn.Linear(256, 128), nn.SELU(),
                nn.Linear(128, 64), nn.SELU(),
                nn.Linear(64, 1)
            )
        else:
            # TODO: use a transformer may be more appropriate
            self.value_layers = nn.Sequential(
                nn.Linear(privilege_dim, 256), nn.SELU(),
                nn.Linear(256, 128), nn.SELU(),
                nn.Linear(128, 64), nn.SELU(),
                nn.Linear(64, 1)
            )
        self.is_recurrent = False
        self.recurrent_hidden_state_size = 1
        self.use_param_mask = False
        self.init_weights(self.act_type, [0.01])
        for i in range(act_dim):
            self.init_weights(self.act_param[i], [0.01])
        self.init_weights(self.value_layers, [np.sqrt(2), np.sqrt(2), np.sqrt(2), 1.0])
    
    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def _obs_parser(self, obs):
        cur_img_feat = torch.narrow(obs, dim=-1, start=0, length=self.mvp_feat_dim)
        goal_img_feat = torch.narrow(obs, dim=-1, start=self.mvp_feat_dim + self.robot_state_dim, length=self.mvp_feat_dim)
        privilege_info = torch.narrow(obs, dim=-1, start=2 * self.mvp_feat_dim + self.robot_state_dim, length=self.privilege_dim)
        return cur_img_feat.detach(), goal_img_feat.detach(), privilege_info.detach()

    def forward(self, obs, rnn_hxs=None, rnn_masks=None):
        cur_img_feat, goal_img_feat, privilege_info = self._obs_parser(obs)
        proj_cur_feat = self.mvp_projector(cur_img_feat)
        proj_goal_feat = self.mvp_projector(goal_img_feat)
        proj_input = torch.cat([proj_cur_feat, proj_goal_feat], dim=-1)
        proj_input = self.act_feature(proj_input)
        # print("input std", torch.std(proj_input, dim=0).mean(), "input mean", torch.mean(proj_input))
        if self.value_mvp_projector is not None:
            value_proj_cur_feat = self.value_mvp_projector(cur_img_feat)
            value_proj_goal_feat = self.value_mvp_projector(goal_img_feat)
            proj_value_input = torch.cat([value_proj_cur_feat, value_proj_goal_feat], dim=-1)
            if self.privilege_dim > 0:
                proj_value_input = torch.cat([proj_value_input, privilege_info], dim=-1)
        else:
            proj_value_input = privilege_info
        act_type_logits = self.act_type(proj_input)
        # print("act type logits", act_type_logits[:5])
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
        act_type_logprob = act_type_dist.log_prob(act_type)
        act_param_logprob = [
            act_param_dist[i].log_prob(act_params[:, i])
            for i in range(self.act_dim)
        ]
        log_prob = torch.sum(torch.stack([act_type_logprob] + act_param_logprob, dim=-1), dim=-1, keepdim=True)
        act_type_ent = act_type_dist.entropy()
        act_param_ent = [act_param_dist[i].entropy() for i in range(self.act_dim)]
        entropy = torch.sum(torch.stack([act_type_ent] + act_param_ent, dim=-1), dim=-1).mean()
        return log_prob, entropy, rnn_hxs
    
    def get_bc_loss(self, obs, rnn_hxs, rnn_masks, actions):
        actions[:, 1:] = torch.clamp(actions[:, 1:], -1., 1.)
        log_prob, _, _ = self.evaluate_actions(obs, rnn_hxs, rnn_masks, actions)
        # TODO: do we need clip
        loss = -log_prob.mean()
        return loss
