import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical, Normal
from policies.base import ActorCriticPolicy


class CrossAttentionExtractor(nn.Module):
    def __init__(self, robot_dim, object_dim, hidden_size):
        super(CrossAttentionExtractor, self).__init__()
        self.robot_embed = nn.Sequential(
            nn.Linear(robot_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
        )
        self.object_embed = nn.Sequential(
            nn.Linear(object_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU()
        )
        self.ln = nn.LayerNorm(hidden_size // 2)
    
    def forward(self, robot_obs, objects_obs, masks=None):
        # assume robot_obs [batch_size, robot_dim]
        # objects_obs [batch_size, n_obj, object_dim]
        # masks [batch_size, n_obj]
        robot_embedding = self.robot_embed(robot_obs)  # (batch_size, hidden_size/2)
        objects_embedding = self.object_embed(objects_obs)  # (batch_size, n_obj, hidden_size/2)
        weights = torch.matmul(robot_embedding.unsqueeze(dim=1), objects_embedding.transpose(1, 2)) / np.sqrt(objects_embedding.size()[2])  # (batch_size, 1, n_obj)
        if masks is not None:
            weights -= masks.unsqueeze(dim=1) * 1e9
        weights = nn.functional.softmax(weights, dim=-1)  # (batch_size, 1, n_obj)
        weighted_feature = torch.matmul(weights, objects_embedding).squeeze(dim=1)  # (batch_size, hidden_size/2)
        weighted_feature = nn.functional.relu(self.ln(weighted_feature))
        return torch.cat([robot_embedding, weighted_feature], dim=-1)  # (batch_size, hidden_size)


def scaled_dot_product_attention(q, k, v, mask=None):
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # (..., seq_len_q, seq_len_k)
    dk = k.shape[-1]
    scaled_qk = matmul_qk / np.sqrt(dk)
    if mask is not None:
        scaled_qk += (mask * -1e9)  # 1: we don't want it, 0: we want it
    attention_weights = nn.functional.softmax(scaled_qk, dim=-1)  # (..., seq_len_q, seq_len_k)
    output = torch.matmul(attention_weights, v)  # (..., seq_len_q, feature_dim)
    return output, attention_weights


class SelfAttentionBase(nn.Module):
    def __init__(self, input_dim, feature_dim, n_heads=1):
        super(SelfAttentionBase, self).__init__()
        self.n_heads = n_heads
        self.q_linear = nn.Linear(input_dim, feature_dim)
        self.k_linear = nn.Linear(input_dim, feature_dim)
        self.v_linear = nn.Linear(input_dim, feature_dim)
        self.dense = nn.Linear(feature_dim, feature_dim)

    def split_head(self, x):
        x_size = x.size()
        assert isinstance(x_size[2] // self.n_heads, int)
        x = torch.reshape(x, [-1, x_size[1], self.n_heads, x_size[2] // self.n_heads])
        x = torch.transpose(x, 1, 2)  # (batch_size, n_heads, seq_len, depth)
        return x

    def forward(self, q, k, v, mask):
        assert len(q.size()) == 3
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)
        q_heads = self.split_head(q)
        k_heads = self.split_head(k)
        v_heads = self.split_head((v))
        mask = torch.unsqueeze(mask, dim=1).unsqueeze(dim=2)  # (batch_size, 1, 1, seq_len)
        attention_out, weights = scaled_dot_product_attention(q_heads, k_heads, v_heads, mask)
        attention_out = torch.transpose(attention_out, 1, 2)  # (batch_size, seq_len_q, n_heads, depth)
        out_size = attention_out.size()
        attention_out = torch.reshape(attention_out, [-1, out_size[1], out_size[2] * out_size[3]])
        attention_out = self.dense(attention_out)
        return attention_out


class SelfAttentionExtractor(nn.Module):
    def __init__(self, robot_dim, object_dim, hidden_size, n_attention_blocks, n_heads, is_recurrent=False):
        super(SelfAttentionExtractor, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Sequential(
            nn.Linear(robot_dim + object_dim, hidden_size), nn.ReLU(),
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
    
    def forward(self, robot_obs, objects_obs, masks=None, recurrent_hxs=None, recurrent_masks=None, agg=True):
        # recurrent_hxs: batch_size, 2 * hidden_size. Contains both hidden state and cell state
        n_object = objects_obs.shape[1]
        objects_obs = torch.cat([robot_obs.unsqueeze(dim=1).repeat(1, n_object, 1), objects_obs], dim=-1)
        features = self.embed(objects_obs)
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
        return features, recurrent_hxs


class AttentionDiscretePolicy(ActorCriticPolicy):
    def __init__(self, obs_parser, action_dim, hidden_size, num_bin, feature_extractor="cross_attention",
                 shared=True, n_critic_layers=3, n_actor_layers=3, is_recurrent=False, kwargs={}):
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
        if feature_extractor == "cross_attention":
            self.feature_extractor = CrossAttentionExtractor(self.robot_dim, self.obj_dim, hidden_size)
            self.critic_feature_extractor = None if shared else\
                CrossAttentionExtractor(self.robot_dim, self.obj_dim, hidden_size)
        elif feature_extractor == "self_attention":
            self.robot_B = nn.Parameter(
                torch.normal(0., 0.01, size=(self.robot_dim, self.ff_dim // 2)), requires_grad=True
            ) if self.lff else None
            self.obj_B = nn.Parameter(
                torch.normal(0., 0.01, size=(self.obj_dim, self.ff_dim // 2)), requires_grad=True
            ) if self.lff else None
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
        for i in range(self.action_dim):
            self.actor_linears.append(
                nn.Sequential(
                    *([nn.Linear(hidden_size, hidden_size), nn.ReLU()] * (n_actor_layers - 1) +
                      [nn.Linear(hidden_size, num_bin)]),
                )
            )
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
        features, new_rnn_hxs = self.feature_extractor(robot_obs, objects_obs, masks, rnn_hxs, rnn_masks)
        critic_features = features if self.critic_feature_extractor is None else \
            self.critic_feature_extractor(robot_obs, objects_obs, masks, rnn_hxs, rnn_masks)[0]
        values = self.critic_linears(critic_features)
        action_dist, action_logits = [], []
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
        log_probs = torch.sum(torch.stack(log_probs, dim=-1), dim=-1, keepdim=True)
        # Rescale to [-1, 1]
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
        actions = torch.round((actions + 1.0) / 2 * (self.num_bin - 1.0)).int().detach()
        log_probs = []
        for i in range(self.action_dim):
            log_probs.append(action_dist[i].log_prob(actions[:, i]))
        log_probs = torch.sum(torch.stack(log_probs, dim=-1), dim=-1, keepdim=True)
        entropy = torch.sum(torch.stack([dist.entropy() for dist in action_dist], dim=-1), dim=-1)
        # entropy = entropy.mean()
        return log_probs, entropy, rnn_hxs

    def get_logits(self, obs, actions):
        _, action_logits, _ = self.forward(obs, return_logits=True)
        actions = torch.round((actions + 1.0) / 2 * (self.num_bin - 1.0)).long().detach()
        # expand as one hot
        logits = []
        for i in range(self.action_dim):
            action_onehot = torch.zeros((actions.shape[0], self.num_bin), device=actions.device)
            action_onehot.scatter_(1, actions[:, i].unsqueeze(dim=-1), 1)
            logits.append(torch.sum(action_logits[i] * action_onehot.detach(), dim=-1))
        logits = torch.sum(torch.stack(logits, dim=-1), dim=-1, keepdim=True)
        return logits

    def get_weighted_log_prob(self, obs, actions):
        _, action_dist, _ = self.forward(obs)
        actions = torch.round((actions + 1.0) / 2 * (self.num_bin - 1.0)).long().detach()
        weighted_log_probs = []
        for i in range(self.action_dim):
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


class AttentionValue(nn.Module):
    def __init__(self, obs_parser, action_dim, hidden_size, with_action=False, n_critic_layers=3,
                 is_recurrent=False, action_repr="self", kwargs={}):
        super(AttentionValue, self).__init__()
        self.obs_parser = obs_parser
        self.obj_dim = self.obs_parser.obj_dim
        self.robot_dim = self.obs_parser.robot_dim
        self.lff = kwargs.get('lff', False)
        self.ff_dim = kwargs.get('ff_dim', 0) if self.lff else 0
        self.with_action = with_action
        self.action_repr = action_repr
        self.action_dim = action_dim if with_action and action_repr == "self" else 0
        self.robot_B = nn.Parameter(
            torch.normal(0., 0.01, size=(self.robot_dim + self.action_dim, self.ff_dim // 2)), requires_grad=True
        ) if self.lff else None
        self.obj_B = nn.Parameter(
            torch.normal(0., 0.01, size=(self.obj_dim, self.ff_dim // 2)), requires_grad=True
        ) if self.lff else None
        self.feature_extractor = SelfAttentionExtractor(
            self.robot_dim + self.action_dim + self.ff_dim, self.obj_dim + self.ff_dim, hidden_size,
            kwargs["n_attention_blocks"], kwargs["n_heads"], is_recurrent
        )
        if with_action and action_repr == "cross":
            self.action_embed = nn.Sequential(
                nn.Linear(action_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size)
            )
        else:
            self.action_embed = None
        self.is_recurrent = is_recurrent
        self.recurrent_hidden_state_size = 2 * hidden_size if is_recurrent else 1
        self.critic_linears = nn.Sequential(
            *([nn.Linear(hidden_size, hidden_size), nn.ReLU()] * (n_critic_layers - 1) +
              [nn.Linear(hidden_size, 1), nn.Sigmoid()]),
        )
        self._initialize()

    def _initialize(self):
        for net in self.critic_linears:
            if isinstance(net, nn.Linear):
                nn.init.orthogonal_(net.weight, gain=np.sqrt(2))
                nn.init.constant_(net.bias, 0.)

    def _parse_obs(self, obs, actions):
        robot_obs, objects_obs, masks = self.obs_parser.forward(obs)
        if self.with_action and self.action_repr == "self":
            assert actions is not None
            robot_obs = torch.cat([robot_obs, actions], dim=-1)
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

    def forward(self, obs, actions=None, rnn_hxs=None, rnn_masks=None):
        robot_obs, objects_obs, masks = self._parse_obs(obs, actions)
        features, rnn_hxs = self.feature_extractor(robot_obs, objects_obs, masks, rnn_hxs, rnn_masks, agg=False)
        # add cross attention between action and objects features
        if self.with_action and self.action_repr == "cross":
            action_features = self.action_embed(actions).unsqueeze(dim=1)
            features, _ = scaled_dot_product_attention(action_features, features, features)
            features = torch.squeeze(features, dim=1)
        else:
            features = torch.mean(features, dim=1)
        values = self.critic_linears.forward(features)
        return values


class AttentionDynamics(nn.Module):
    def __init__(self, obs_parser, action_dim, hidden_size, n_critic_layers=3,
                 is_recurrent=False, kwargs={}):
        '''
        Two branches. The first one is a state-based feature extractor. The second one is an action-conditioned transition model plus a value prediction head.
        '''
        super(AttentionDynamics, self).__init__()
        self.obs_parser = obs_parser
        self.obj_dim = self.obs_parser.obj_dim
        self.robot_dim = self.obs_parser.robot_dim
        self.action_dim = action_dim
        self.lff = kwargs.get('lff', False)
        self.ff_dim = kwargs.get('ff_dim', 0) if self.lff else 0
        self.robot_B = nn.Parameter(
            torch.normal(0., 0.01, size=(self.robot_dim, self.ff_dim // 2)), requires_grad=True
        ) if self.lff else None
        self.obj_B = nn.Parameter(
            torch.normal(0., 0.01, size=(self.obj_dim, self.ff_dim // 2)), requires_grad=True
        ) if self.lff else None
        self.act_B = nn.Parameter(
            torch.normal(0., 0.01, size=(self.action_dim, self.ff_dim // 2)), requires_grad=True
        )
        self.is_recurrent = is_recurrent
        self.recurrent_hidden_state_size = 2 * hidden_size if is_recurrent else 1
        # state only
        self.feature_extractor = SelfAttentionExtractor(
            self.robot_dim + self.ff_dim, self.obj_dim + self.ff_dim, hidden_size, kwargs["n_attention_blocks"],
            kwargs["n_heads"], is_recurrent
        )
        # self.action_embed = nn.Sequential(
        #     nn.Linear(action_dim, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size)
        # )

        # dynamics
        self.transition_model = SelfAttentionExtractor(
            self.robot_dim + self.action_dim + 2 * self.ff_dim, self.obj_dim + self.ff_dim, hidden_size,
            kwargs["n_attention_blocks"], kwargs["n_heads"], is_recurrent
        )
        # prediction head
        self.critic_linears = nn.Sequential(
            *([nn.Linear(hidden_size, hidden_size), nn.ReLU()] * (n_critic_layers - 1) +
              [nn.Linear(hidden_size, 1), nn.Sigmoid()]),
        )
        self._initialize()

    def _initialize(self):
        for net in self.critic_linears:
            if isinstance(net, nn.Linear):
                nn.init.orthogonal_(net.weight, gain=np.sqrt(2))
                nn.init.constant_(net.bias, 0.)

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

    def forward(self, obs, actions, rnn_hxs=None, rnn_masks=None):
        robot_obs, objects_obs, masks = self._parse_obs(obs)
        # state_features, rnn_hxs = self.feature_extractor(robot_obs, objects_obs, masks, rnn_hxs, rnn_masks, agg=False)
        # action_embed = self.action_embed.forward(actions)
        if self.act_B is not None:
            actions = torch.cat(
                [torch.sin(2 * np.pi * torch.matmul(actions, self.act_B)),
                 torch.cos(2 * np.pi * torch.matmul(actions, self.act_B)),
                 actions], dim=-1
            )
        action_embed = torch.cat([robot_obs, actions], dim=-1)
        next_features, _ = self.transition_model(action_embed, objects_obs, masks, rnn_hxs, rnn_masks, agg=False)
        agg_features = torch.mean(next_features, dim=1)
        values = self.critic_linears.forward(agg_features)
        return values, next_features

    def get_state_feature(self, obs, rnn_hxs=None, rnn_masks=None):
        robot_obs, objects_obs, masks = self._parse_obs(obs)
        state_features, rnn_hxs = self.feature_extractor(robot_obs, objects_obs, masks, rnn_hxs, rnn_masks, agg=False)
        return state_features

    def predict_without_action(self, obs, rnn_hxs=None, rnn_masks=None):
        state_features = self.get_state_feature(obs, rnn_hxs, rnn_masks)
        agg_features = torch.mean(state_features, dim=1)
        values = self.critic_linears.forward(agg_features)
        return values


class AttentionDiscreteQ(nn.Module):
    def __init__(self, obs_parser, action_dim, hidden_size, num_bin, n_critic_layers=3,
                 is_recurrent=False, activation=torch.sigmoid, kwargs={}):
        super(AttentionDiscreteQ, self).__init__()
        self.obs_parser = obs_parser
        self.obj_dim = self.obs_parser.obj_dim
        self.robot_dim = self.obs_parser.robot_dim
        self.action_dim = action_dim
        self.feature_extractor = SelfAttentionExtractor(
            self.robot_dim, self.obj_dim, hidden_size, kwargs["n_attention_blocks"],
            kwargs["n_heads"], is_recurrent
        )
        self.num_bin = num_bin
        self.is_recurrent = is_recurrent
        self.recurrent_hidden_state_size = 2 * hidden_size if is_recurrent else 1
        self.critic_linears = nn.ModuleList([nn.Sequential(
            *([nn.Linear(hidden_size, hidden_size), nn.ReLU()] * (n_critic_layers - 1) +
              [nn.Linear(hidden_size, num_bin)]),
        ) for _ in range(self.action_dim)])
        self.activation = activation
        self._initialize()

    def _initialize(self):
        for model in self.critic_linears:
            for net in model:
                if isinstance(net, nn.Linear):
                    nn.init.orthogonal_(net.weight, gain=np.sqrt(2))
                    nn.init.constant_(net.bias, 0.)

    def forward(self, obs, action, rnn_hxs=None, rnn_masks=None):
        robot_obs, objects_obs, masks = self.obs_parser.forward(obs)
        features, rnn_hxs = self.feature_extractor(robot_obs, objects_obs, masks, rnn_hxs, rnn_masks)
        predictions = [self.critic_linears[i](features) for i in range(len(self.critic_linears))]
        action = torch.round((action + 1) / 2 * (self.num_bin - 1)).long()
        logits = []
        for a_idx in range(self.action_dim):
            action_head = action[:, a_idx].unsqueeze(dim=-1)
            selected_logit = torch.gather(predictions[a_idx], 1, action_head)
            logits.append(selected_logit)
        logits = torch.cat(logits, dim=-1).mean(dim=-1, keepdim=True)
        return self.activation(logits)


class AttentionGaussianPolicy(ActorCriticPolicy):
    def __init__(self, obs_parser, action_dim, hidden_size, feature_extractor="cross_attention",
                 shared=True, n_critic_layers=3, n_actor_layers=3, kwargs={}):
        super(AttentionGaussianPolicy, self).__init__()
        self.obs_parser = obs_parser
        # robot_obs, objects_obs, masks = self.obs_parser(torch.zeros(1, obs_dim))
        # self.obj_dim = objects_obs.shape[-1]
        # self.robot_dim = robot_obs.shape[-1]
        self.obj_dim = self.obs_parser.obj_dim
        self.robot_dim = self.obs_parser.robot_dim
        self.action_dim = action_dim
        if feature_extractor == "cross_attention":
            self.feature_extractor = CrossAttentionExtractor(self.robot_dim, self.obj_dim, hidden_size)
            self.critic_feature_extractor = None if shared else\
                CrossAttentionExtractor(self.robot_dim, self.obj_dim, hidden_size)
        elif feature_extractor == "self_attention":
            self.feature_extractor = SelfAttentionExtractor(
                self.robot_dim, self.obj_dim, hidden_size, kwargs["n_attention_blocks"], kwargs["n_heads"]
            )
            self.critic_feature_extractor = None if shared else SelfAttentionExtractor(
                self.robot_dim, self.obj_dim, hidden_size, kwargs["n_attention_blocks"], kwargs["n_heads"]
            )
        else:
            raise NotImplementedError
        self.critic_linears = nn.Sequential(
            *([nn.Linear(hidden_size, hidden_size), nn.ReLU()] * (n_critic_layers - 1) +
              [nn.Linear(hidden_size, 1)]),
        )
        self.actor_linears = nn.Sequential(
            *([nn.Linear(hidden_size, hidden_size), nn.ReLU()] * (n_actor_layers - 1) +
              [nn.Linear(hidden_size, self.action_dim)])
        )
        # self.actor_logstd = nn.Sequential(
        #     *([nn.Linear(hidden_size, hidden_size), nn.ReLU()] * (n_actor_layers - 1) +
        #       [nn.Linear(hidden_size, self.action_dim)])
        # )
        self.actor_logstd = nn.Parameter(-torch.ones(self.action_dim))
        self._initialize()

    def _initialize(self):
        for net in self.critic_linears:
            if isinstance(net, nn.Linear):
                nn.init.orthogonal_(net.weight, gain=np.sqrt(2))
                nn.init.constant_(net.bias, 0.)
        for net in self.actor_linears:
            if isinstance(net, nn.Linear):
                nn.init.orthogonal_(net.weight, gain=0.01)
                nn.init.constant_(net.bias, 0.)

    def forward(self, obs, rnn_hxs=None, rnn_masks=None):
        # robot_obs, objects_obs, masks = self.parse_obs(obs)
        robot_obs, objects_obs, masks = self.obs_parser.forward(obs)
        features = self.feature_extractor(robot_obs, objects_obs, masks)
        critic_features = features if self.critic_feature_extractor is None else \
            self.critic_feature_extractor(robot_obs, objects_obs, masks)
        values = self.critic_linears(critic_features)
        action_mean = self.actor_linears(features)
        # action_logstd = self.actor_logstd(features)
        # TODO: speed up
        # action_dist = MultivariateNormal(action_mean, torch.diag_embed(torch.exp(action_logstd)))
        action_dist = Normal(action_mean, torch.exp(self.actor_logstd))
        return values, action_dist, rnn_hxs

    def act(self, obs, rnn_hxs=None, rnn_masks=None, deterministic=False):
        values, action_dist, rnn_hxs = self.forward(obs, rnn_hxs, rnn_masks)
        if deterministic:
            actions = action_dist.mean
        else:
            actions = action_dist.sample()
        log_probs = action_dist.log_prob(actions).sum(dim=-1, keepdim=True)
        return values, actions, log_probs, rnn_hxs

    def evaluate_actions(self, obs, rnn_hxs=None, rnn_masks=None,  actions=None):
        _, action_dist, rnn_hxs = self.forward(obs, rnn_hxs, rnn_masks)
        log_probs = action_dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = action_dist.entropy().sum(dim=-1).mean()
        return log_probs, entropy, rnn_hxs
