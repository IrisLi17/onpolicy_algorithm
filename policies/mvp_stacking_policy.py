from policies.base import ActorCriticPolicy
from policies.attention_discrete import SelfAttentionBase
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical


class MvpStackingPolicy(ActorCriticPolicy):
    def __init__(
        self, mvp_feat_dim, n_primitive, act_dim, num_bin,
        proj_img_dim, privilege_dim=0, state_only_value=False,
        attn_value=False,
    ) -> None:
        super().__init__()
        self.mvp_feat_dim = mvp_feat_dim
        self.robot_state_dim = 7
        self.privilege_dim = privilege_dim
        self.n_primitive = n_primitive
        self.act_dim = act_dim
        self.num_bin = num_bin
        self.attn_value = attn_value
        self.mvp_projector = nn.Sequential(
            nn.LayerNorm(mvp_feat_dim, eps=1e-6),
            nn.Linear(mvp_feat_dim, proj_img_dim)
        )
        self.mvp_diff_projector = nn.Sequential(
            nn.LayerNorm(mvp_feat_dim, eps=1e-6),
            nn.Linear(mvp_feat_dim, proj_img_dim)
        )
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
            if attn_value:
                self.value_object_encode_linear = nn.Linear(8, 64)
                _encoder_layer = nn.TransformerEncoderLayer(
                    d_model=64, nhead=1, dim_feedforward=64, dropout=0.0,
                    ) # seq, batch, feature
                self.value_attn_encoder = nn.TransformerEncoder(_encoder_layer, num_layers=3)
                self.value_agg = nn.Linear(64, 1)
            else:
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
        if not attn_value:
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
        proj_goal_feat = self.mvp_diff_projector(goal_img_feat - cur_img_feat)
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
        if not self.attn_value:
            value_pred = self.value_layers(proj_value_input)
        else:
            obj_and_goals = proj_value_input.reshape((proj_value_input.shape[0], 2, -1, 7))
            bsz = obj_and_goals.shape[0]
            dtype = obj_and_goals.dtype
            device = obj_and_goals.device
            token_embed = torch.cat([
                torch.zeros((bsz, 1, obj_and_goals.shape[2], 7), dtype=dtype, device=device),
                torch.ones((bsz, 1, obj_and_goals.shape[2], 7), dtype=dtype, device=device)
            ], dim=1)
            obj_and_goals = torch.cat([obj_and_goals, token_embed], dim=-1).reshape((bsz, -1, 8)).transpose(0, 1)
            obj_goal_embed = self.value_object_encode_linear(obj_and_goals)
            value_feature = self.value_attn_encoder(obj_goal_embed)
            value_pred = self.value_agg(torch.mean(value_feature, dim=0))
        
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
        loss = -torch.clamp(log_prob, max=-2).mean()
        # loss = -log_prob.mean()
        return loss


# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class SelfAttentionExtractor(nn.Module):
    def __init__(self, robot_dim, object_dim, hidden_size, n_attention_blocks, n_heads, is_recurrent=False):
        super(SelfAttentionExtractor, self).__init__()
        self.hidden_size = hidden_size
        #print("model input size, robot dim:", robot_dim, "object_dim:", object_dim)
        self.embed = nn.Sequential(
            nn.Linear(robot_dim + object_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size),
        )
        self.embed_goal = nn.Sequential(
            nn.Linear(20, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size),
        )
        self.embed_obj = nn.Sequential(
            nn.Linear(30, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size),
        )
        self.n_attention_blocks = n_attention_blocks
        self.attention_blocks = nn.ModuleList(
            [SelfAttentionBase(hidden_size, hidden_size, n_heads) for _ in range(n_attention_blocks)]
        )
        self.layer_norm1 = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(n_attention_blocks)])
        self.feed_forward_network = nn.ModuleList(
            [nn.ModuleList([nn.Linear(hidden_size, hidden_size),
                            nn.Linear(hidden_size, hidden_size)]) for _ in range(n_attention_blocks)])
        self.feed_forward_network = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size)
            ) for _ in range(n_attention_blocks)
        )
        self.layer_norm2 = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(n_attention_blocks)])
        if is_recurrent:
            self.recurrent_layer = nn.LSTMCell(hidden_size, hidden_size)
        else:
            self.recurrent_layer = None
    
    def forward(self, robot_obs, objects_obs, masks=None, tokenize=False, recurrent_hxs=None, recurrent_masks=None, agg=True):
        # recurrent_hxs: batch_size, 2 * hidden_size. Contains both hidden state and cell state
        # print("selfAttention/robot_obs:", robot_obs.shape)
        # print("selfAttention/objects_obs:", objects_obs.shape)
        if tokenize:
            assert (robot_obs.shape[1] - 11) % 12 == 0
            n_max_goal = int((robot_obs.shape[1] - 11) / 12)
            n_object = objects_obs.shape[1]
            goal_obs = torch.zeros((robot_obs.shape[0], n_max_goal * n_object, 6)).cuda()
            goal_masks = torch.zeros((robot_obs.shape[0], n_max_goal * n_object)).cuda()
            for i in range(n_max_goal):
                goal_masks[:, n_object*i: n_object*(i+1)] = (torch.norm(robot_obs[:, 14 + 6 * i: 17 + 6 * i] + 1, dim=-1) < 1e-3).unsqueeze(dim=-1).repeat(1, n_object)
                for row in range(n_object*i, n_object*(i+1)):
                    goal_obs[:, row, :3] = robot_obs[:, 14 + 6 * i: 17 + 6 * i]
                    goal_obs[:, row, 3:] = robot_obs[:, 14 + 6 * i + 6 * n_max_goal: 17 + 6 * i + 6 * n_max_goal]
            objects_obs = torch.cat([robot_obs[:, :11].unsqueeze(dim=1).repeat(1, n_object*n_max_goal, 1), goal_obs.cuda(), objects_obs[:, :, 3:].repeat(1, n_max_goal, 1)], dim=-1)
            features = self.embed(objects_obs)
            action_masks = masks.clone().repeat(1, n_max_goal)
            masks = torch.logical_or(masks.repeat(1, n_max_goal), goal_masks)
            """  
            goal_obs = torch.zeros((robot_obs.shape[0], n_max_goal, 9)).cuda()
            for i in range(n_max_goal):
                goal_obs[:, i, :6] = robot_obs[:, 11 + 6 * i: 17 + 6 * i]
                goal_obs[:, i, 6:] = robot_obs[:, 14 + 6 * i + 6 * n_max_goal: 17 + 6 * i + 6 * n_max_goal]
            goal_obs = torch.cat([robot_obs[:, :11].unsqueeze(dim=1).repeat(1, n_max_goal, 1), goal_obs], dim=-1)
            objects_obs = torch.cat([robot_obs[:, :11].unsqueeze(dim=1).repeat(1, n_object, 1), objects_obs], dim=-1)
            goal_masks = torch.norm(goal_obs[:, :, 11:] + 1, dim=-1) < 1e-3
            goal_feature = self.embed_goal(goal_obs)
            obj_feature = self.embed_obj(objects_obs)
            features = torch.cat([goal_feature, obj_feature], dim=1)
            masks = torch.cat([goal_masks, masks], dim=-1)
            all_masks = masks.clone()
            all_masks[:, :n_max_goal] = True
            """
        else:
            assert (robot_obs.shape[1]-11) % 6 == 0
            n_max_goal = int((robot_obs.shape[1]-11) / 6)
            n_object = objects_obs.shape[1]
            goal_obs = torch.zeros((robot_obs.shape[0], n_max_goal * n_object, 6))
            for i in range(n_max_goal):
                for row in range(n_object*i, n_object*(i+1)):
                    goal_obs[:, row, :3] = robot_obs[:, 11 + 3 * i: 14 + 3 * i]
                    goal_obs[:, row, 3:] = robot_obs[:, 11 + 3 * i + 3 * n_max_goal: 14 + 3 * i + 3 * n_max_goal]
            objects_obs = torch.cat([robot_obs[:, :11].unsqueeze(dim=1).repeat(1, n_object*n_max_goal, 1), goal_obs.cuda(), objects_obs.repeat(1, n_max_goal, 1)], dim=-1)  # TODO
            features = self.embed(objects_obs)
            masks = masks.repeat(1, n_max_goal)

        for i in range(self.n_attention_blocks):
            attn_output = self.attention_blocks[i](features, features, features, masks)
            out1 = self.layer_norm1[i](features + attn_output)
            ffn_out = self.feed_forward_network[i](out1)
            features = self.layer_norm2[i](ffn_out)
        # Aggregate all the objects?
        if agg:
            features = torch.mean(features, dim=1)
            # Optional recurrent layer
            if self.recurrent_layer is not None:
                recurrent_hxs = recurrent_hxs * recurrent_masks
                hx, cx = torch.split(recurrent_hxs, self.hidden_size, dim=-1)
                hx_new, cx_new = self.recurrent_layer(features, (hx, cx))
                recurrent_hxs = torch.cat([hx_new, cx_new], dim=-1)
        if tokenize:
            return features, recurrent_hxs, masks, action_masks
        return features, recurrent_hxs


class AttentionDiscretePolicy(ActorCriticPolicy):
    def __init__(self, obs_parser, action_dim, hidden_size, num_bin, feature_extractor="cross_attention",
                 shared=True, n_critic_layers=3, n_actor_layers=3, n_object=None, tokenize=False, is_recurrent=False, kwargs={}):
        super(AttentionDiscretePolicy, self).__init__()
        self.obs_parser = obs_parser
        # robot_obs, objects_obs, masks = self.obs_parser(torch.zeros(1, obs_dim))
        # self.obj_dim = objects_obs.shape[-1]
        # self.robot_dim = robot_obs.shape[-1]
        self.obj_dim = self.obs_parser.obj_dim
        self.robot_dim = self.obs_parser.robot_dim
        self.action_dim = action_dim
        self.lff = kwargs.get("lff", False)
        self.ff_dim = kwargs.get("ff_dim", 0) if self.lff else 0
        self.num_bin = num_bin
        self.n_object = n_object
        self.tokenize = tokenize
        if feature_extractor == "self_attention":
            self.robot_B = None
            self.obj_B = None
            self.feature_extractor = SelfAttentionExtractor(
                self.robot_dim + self.ff_dim, self.obj_dim + self.ff_dim, hidden_size, kwargs["n_attention_blocks"],
                kwargs["n_heads"], is_recurrent
            )
            self.critic_feature_extractor = None if shared else SelfAttentionExtractor(
                self.robot_dim + self.ff_dim, self.obj_dim + self.ff_dim, hidden_size, kwargs["n_attention_blocks"],
                kwargs["n_heads"], is_recurrent
            )
        else:
            raise NotImplementedError
        self.is_recurrent = is_recurrent
        self.recurrent_hidden_state_size = 2 * hidden_size if is_recurrent else 1
        self.critic_linears = nn.Sequential(
            *([nn.Linear(hidden_size, hidden_size), nn.ReLU()] * (n_critic_layers - 1) +
              [nn.Linear(hidden_size, 1)]),
        )
        self.actor_linears = nn.ModuleList()
        if tokenize:
            n_actor_linears = self.action_dim - 1
        else:
            n_actor_linears = self.action_dim
        for i in range(n_actor_linears):
            self.actor_linears.append(
                nn.Sequential(
                    *([nn.Linear(hidden_size, hidden_size), nn.ReLU()] * (n_actor_layers - 1) +
                      [nn.Linear(hidden_size, num_bin)]),
                )
            )
        if tokenize:
            self.obj_id_linear = nn.Linear(hidden_size, 1)
        # for sanity check, predict success prob using policy feature to check its quality
        self.debug_head = nn.Sequential(
            *([nn.Linear(hidden_size, hidden_size), nn.ReLU()] * (n_critic_layers - 1) +
              [nn.Linear(hidden_size, 1), nn.Sigmoid()])
        )
        self._initialize()

    def _initialize(self):
        for net in self.critic_linears:
            if isinstance(net, nn.Linear):
                nn.init.orthogonal_(net.weight, gain=np.sqrt(2))
                nn.init.constant_(net.bias, 0.)
        for net in self.debug_head:
            if isinstance(net, nn.Linear):
                nn.init.orthogonal_(net.weight, gain=np.sqrt(2))
                nn.init.constant_(net.bias, 0.)
        # for f in self.actor_linears:
        #     net = f[-1]
        #     if isinstance(net, nn.Linear):
        #         nn.init.orthogonal_(net.weight, gain=0.01)
        #         nn.init.constant_(net.bias, 0.)

    def _parse_obs(self, obs):
        robot_obs, objects_obs, masks = self.obs_parser.forward(obs)
        if self.robot_B is not None:
            robot_obs = torch.cat(
                [torch.sin(2 * np.pi * torch.matmul(robot_obs, self.robot_B)),
                 torch.cos(2 * np.pi * torch.matmul(robot_obs, self.robot_B)),
                 robot_obs], dim=-1
            )
        if self.obj_B is not None:
            objects_obs = torch.cat(
                [torch.sin(2 * np.pi * torch.matmul(objects_obs, self.obj_B)),
                 torch.cos(2 * np.pi * torch.matmul(objects_obs, self.obj_B)),
                 objects_obs], dim=-1
            )
        return robot_obs, objects_obs, masks

    def forward(self, obs, rnn_hxs=None, rnn_masks=None, return_logits=False):
        robot_obs, objects_obs, masks = self._parse_obs(obs)
        if self.tokenize:
            n_max_goal = int((robot_obs.shape[1] - 11) / 12)
            n_object = objects_obs.shape[1]
            features, new_rnn_hxs, mask, action_mask = self.feature_extractor(robot_obs, objects_obs, masks, self.tokenize,
                                                                 rnn_hxs, rnn_masks, agg=False)  # features[128,18,64]
            # todo: add mask
            critic_features = features if self.critic_feature_extractor is None else \
                self.critic_feature_extractor(robot_obs, objects_obs, masks, self.tokenize, rnn_hxs, rnn_masks, agg=False)[0]
            critic_out = torch.sum(critic_features * (1. - torch.unsqueeze(mask.float(), dim=-1)), dim=1) / torch.sum(
                1. - torch.unsqueeze(mask.float(), dim=-1), dim=1)
            values = self.critic_linears(critic_out)
        else:
            features, new_rnn_hxs = self.feature_extractor(robot_obs, objects_obs, masks, self.tokenize, rnn_hxs, rnn_masks)
            critic_features = features if self.critic_feature_extractor is None else \
                self.critic_feature_extractor(robot_obs, objects_obs, masks, self.tokenize, rnn_hxs, rnn_masks)[0]
            values = self.critic_linears(critic_features)
        action_dist, action_logits = [], []
        if self.tokenize:
            obj_id_logit = self.obj_id_linear(features)  # [128, 18, 1]
            obj_id_logit -= torch.unsqueeze(mask, dim=-1) * 1e9
            obj_id_logit = obj_id_logit.squeeze(dim=-1)
            total_id_logit = torch.zeros((obj_id_logit.shape[0], n_object)).to("cuda")
            for i in range(n_object):
                idx = np.linspace(i, (n_max_goal - 1) * n_object + i, n_max_goal)
                total_id_logit[:, i] = torch.max(obj_id_logit[:, idx], dim=-1)[0]
            action_dist.append(Categorical(logits=total_id_logit))

            object_states_logit = [self.actor_linears[i](features) - torch.unsqueeze(mask, dim=-1).repeat(1, 1, self.num_bin) * 1e9
                                   for i in range(self.action_dim - 1)]  # # [batch_size, seq_len, n_bin]
            total_state_logit = []  # torch.zeros((object_states_logit.shape[0], n_object, object_states_logit.shape[2]))
            for j in range(len(object_states_logit)):
                tmp_logit = torch.zeros((object_states_logit[0].shape[0], n_object, object_states_logit[0].shape[-1])).to("cuda")
                for i in range(n_object):
                    idx = np.linspace(i, (n_max_goal - 1) * n_object + i, n_max_goal)
                    tmp_logit[:, i, :] = torch.max(object_states_logit[j][:, idx, :], dim=1)[0]
                total_state_logit.append(tmp_logit)
            dist = tuple([total_id_logit] + total_state_logit)
            return values, dist, new_rnn_hxs
        else:
            for i in range(self.action_dim):
                logits = self.actor_linears[i](features)
                action_dist.append(Categorical(logits=logits))
                action_logits.append(logits)
        if return_logits:
            return values, action_logits, new_rnn_hxs
        else:
            return values, action_dist, new_rnn_hxs

    def act(self, obs, rnn_hxs=None, rnn_masks=None, deterministic=False):
        values, action_dist, rnn_hxs = self.forward(obs, rnn_hxs, rnn_masks)
        actions = []
        log_probs = []
        if self.tokenize:
            dist_obj_id = action_dist[0]
            dist_obj_id = FixedCategorical(logits=dist_obj_id)
            dist_obj_states = action_dist[1:]

            if deterministic:
                object_id = dist_obj_id.mode()
            else:
                object_id = dist_obj_id.sample()

            onehot = torch.zeros(dist_obj_states[0].size(0), dist_obj_states[0].size(1)).to(object_id.device)
            onehot.scatter_(1, object_id, 1)
            # onehot = onehot[:, 1:]
            reconstr_logits = [(dist * onehot.unsqueeze(dim=-1).detach()).sum(dim=1) for dist in dist_obj_states]
            dist_object_states = [FixedCategorical(logits=logit) for logit in reconstr_logits]

            if deterministic:
                object_states = [dist.mode() for dist in dist_object_states]
            else:
                object_states = [dist.sample() for dist in dist_object_states]

            action = torch.cat([object_id] + object_states, dim=-1).float()
            action[:, 1:] = action[:, 1:] / (self.num_bin - 1) * 2 - 1

            object_id_log_prob = dist_obj_id.log_probs(object_id)
            # print(len(object_states), object_states[0].shape, object_id.shape)
            log_probs = [object_id_log_prob] + [dist_object_states[i].log_probs(object_states[i])
                                                for i in range(self.action_dim - 1)]
            log_probs = torch.sum(torch.stack(log_probs, dim=1), dim=1)  # , keepdim=True
            return values, action, log_probs, rnn_hxs

        for i in range(self.action_dim):
            if deterministic:
                actions.append(action_dist[i].probs.argmax(dim=-1, keepdim=True))
            else:
                try:
                    actions.append(action_dist[i].sample().unsqueeze(dim=-1))
                except:
                    import IPython
                    IPython.embed()
                    exit()
            log_probs.append(action_dist[i].log_prob(actions[-1].squeeze(dim=-1)))
        actions = torch.cat(actions, dim=-1)
        log_probs = torch.sum(torch.stack(log_probs, dim=-1), dim=-1)  # , keepdim=True
        actions = actions / (self.num_bin - 1.0) * 2 - 1
        return values, actions, log_probs, rnn_hxs

    def predict_success_prob(self, obs, rnn_hxs=None, rnn_masks=None):
        robot_obs, objects_obs, masks = self._parse_obs(obs)
        features, new_rnn_hxs = self.feature_extractor(robot_obs, objects_obs, masks, rnn_hxs, rnn_masks)
        # Only want to check the quality of learned policy representation, do not want to train it
        success_prob = self.debug_head.forward(features.detach())
        return success_prob
    
    def evaluate_actions(self, obs, rnn_hxs=None, rnn_masks=None,  actions=None):
        _, action_dist, rnn_hxs = self.forward(obs, rnn_hxs, rnn_masks)
        # Scale up actions
        if self.tokenize:
            dist_object_id = action_dist[0]
            dist_object_id = FixedCategorical(logits=dist_object_id)
            dist_object_states = action_dist[1:]
            actions_obj = torch.narrow(actions, dim=-1, start=0, length=1).long().to("cuda")
            actions_other = torch.narrow(actions, dim=-1, start=1, length=actions.shape[1] - 1)
            actions_other = torch.round((actions_other + 1.0) / 2 * (self.num_bin - 1.0)).int().detach()
            # print(actions_obj[0])
            # print(actions_other[0])
            # actions = torch.cat([actions_obj.int().detach().unsqueeze(1), actions_other], dim=-1)

            onehot = torch.from_numpy(np.zeros((dist_object_states[0].size(0), dist_object_states[0].size(1)))).to(
                actions_obj.device)
            onehot.scatter_(1, actions_obj, 1)
            # onehot = onehot[:, 1:]
            # print(onehot[0])
            reconstr_logits = [(dist * onehot.unsqueeze(dim=-1).detach()).sum(dim=1) for dist in dist_object_states]
            dist_object_states = [FixedCategorical(logits=logit) for logit in reconstr_logits]
            object_id_log_prob = dist_object_id.log_probs(actions_obj)
            log_probs = [object_id_log_prob] + [
                dist_object_states[j].log_probs(actions_other[:, j].detach())
                for j in range(self.action_dim - 1)
            ]
            log_probs = torch.sum(torch.stack(log_probs, dim=1), dim=1)  # , keepdim=True

            dist_entropy = dist_object_id.entropy()
            for j in range(self.action_dim - 1):
                dist_entropy += dist_object_states[j].entropy()
            dist_entropy = dist_entropy.mean()
            return log_probs, dist_entropy, rnn_hxs
        else:
            actions = torch.round((actions + 1.0) / 2 * (self.num_bin - 1.0)).int().detach()
        log_probs = []
        for i in range(self.action_dim):
            log_probs.append(action_dist[i].log_prob(actions[:, i]))
        log_probs = torch.sum(torch.stack(log_probs, dim=-1), dim=-1, keepdim=True)
        entropy = torch.sum(torch.stack([dist.entropy() for dist in action_dist], dim=-1), dim=-1)
        # entropy = entropy.mean()
        return log_probs, entropy, rnn_hxs

    def get_logits(self, obs, actions):
        print("get logits!")
        exit(0)
        _, action_logits, _ = self.forward(obs, return_logits=True)
        if tokenize:
            actions_obj = actions[:, 0]
            actions_other = actions[:, 1:]
            actions_other = torch.round((actions_other + 1.0) / 2 * (self.num_bin - 1.0)).long().detach()
            actions = torch.cat([actions_obj.long().detach().unsqueeze(1), actions_other], dim=-1)
            # actions[:, 1:] = torch.round((actions[:, 1:] + 1.0) / 2 * (self.num_bin - 1.0)).long().detach()
        else:
            actions = torch.round((actions + 1.0) / 2 * (self.num_bin - 1.0)).long().detach()
        # expand as one hot
        logits = []
        for i in range(self.action_dim):
            if tokenize and i == 0:
                action_onehot = torch.zeros((actions.shape[0], self.n_object), device=actions.device)
                action_onehot.scatter_(1, actions[:, i].unsqueeze(dim=-1), 1)
                logits.append(torch.sum(action_logits[i] * action_onehot.detach(), dim=-1))
                continue
            action_onehot = torch.zeros((actions.shape[0], self.num_bin), device=actions.device)
            action_onehot.scatter_(1, actions[:, i].unsqueeze(dim=-1), 1)
            logits.append(torch.sum(action_logits[i] * action_onehot.detach(), dim=-1))
        logits = torch.sum(torch.stack(logits, dim=-1), dim=-1, keepdim=True)
        return logits

    def get_weighted_log_prob(self, obs, actions):
        print("get weighted log prob!")
        exit(0)
        _, action_dist, _ = self.forward(obs)
        if tokenize:
            actions_obj = actions[:, 0]
            actions_other = actions[:, 1:]
            actions_other = torch.round((actions_other + 1.0) / 2 * (self.num_bin - 1.0)).long().detach()
            actions = torch.cat([actions_obj.long().detach().unsqueeze(1), actions_other], dim=-1)
            # actions[:, 1:] = torch.round((actions[:, 1:] + 1.0) / 2 * (self.num_bin - 1.0)).long().detach()
        else:
            actions = torch.round((actions + 1.0) / 2 * (self.num_bin - 1.0)).long().detach()
        weighted_log_probs = []
        for i in range(self.action_dim):
            if tokenize and i == 0:
                weight = torch.exp(
                    -(torch.arange(self.n_object).unsqueeze(dim=0).repeat(actions.shape[0], 1).to(obs.device)
                      - actions[:, i: i + 1]) ** 2).detach()
                normalizer = torch.sum(weight, dim=-1).detach()
                weighted_log_probs.append(torch.log(torch.sum(action_dist[i].probs * weight, dim=-1) / normalizer))
                continue
            weight = torch.exp(-(torch.arange(self.num_bin).unsqueeze(dim=0).repeat(actions.shape[0], 1).to(obs.device)
                                 - actions[:, i: i + 1]) ** 2).detach()
            normalizer = torch.sum(weight, dim=-1).detach()
            weighted_log_probs.append(torch.log(torch.sum(action_dist[i].probs * weight, dim=-1) / normalizer))
        weighted_log_probs = torch.sum(torch.stack(weighted_log_probs, dim=-1), dim=-1, keepdim=True)
        return weighted_log_probs

    def get_feature(self, obs, rnn_hxs=None, rnn_masks=None, net="policy"):
        robot_obs, objects_obs, masks = self._parse_obs(obs)
        if net == "policy":
            features, new_rnn_hxs = self.feature_extractor(robot_obs, objects_obs, masks, rnn_hxs, rnn_masks)
        elif net == "value":
            assert self.critic_feature_extractor is not None
            features = self.critic_feature_extractor(robot_obs, objects_obs, masks, rnn_hxs, rnn_masks)[0]
        else:
            raise NotImplementedError
        return features