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
    def __init__(self, obs_dim, act_dim, hidden_size, n_layers=2):
        super(MlpGaussianPolicy, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        # self.policy_feature = nn.Sequential(
        #     nn.Linear(self.obs_dim, hidden_size), nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        # )
        self.policy_feature = [nn.Linear(self.obs_dim, hidden_size), nn.ReLU()]
        for _ in range(n_layers - 1):
            self.policy_feature.append(nn.Linear(hidden_size, hidden_size))
            self.policy_feature.append(nn.ReLU())
        self.policy_feature = nn.Sequential(*self.policy_feature)
        self.actor_mean = nn.Linear(hidden_size, self.act_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(self.act_dim), requires_grad=True)
        # self.value_feature = nn.Sequential(
        #     nn.Linear(self.obs_dim, hidden_size), nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        # )
        self.value_feature = [nn.Linear(self.obs_dim, hidden_size), nn.ReLU()]
        for _ in range(n_layers - 1):
            self.value_feature.append(nn.Linear(hidden_size, hidden_size))
            self.value_feature.append(nn.ReLU())
        self.value_feature = nn.Sequential(*self.value_feature)
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
    def forward(self, obs, rnn_hxs=None, rnn_masks=None, previ_obs=None):
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
    def act(self, obs, rnn_hxs=None, rnn_masks=None, deterministic=False, previ_obs=None, forward_value=True):
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


class PandaSepPolicy(ActorCriticPolicy):
    def __init__(self, obs_dim, hidden_size, n_layers=2) -> None:
        super(PandaSepPolicy, self).__init__()
        self.obs_dim = obs_dim
        self.hidden_size = hidden_size
        self.policy_feature = [nn.Linear(self.obs_dim, hidden_size), nn.ReLU()]
        for _ in range(n_layers - 1):
            self.policy_feature.append(nn.Linear(hidden_size, hidden_size))
            self.policy_feature.append(nn.ReLU())
        self.policy_feature = nn.Sequential(*self.policy_feature)
        self.arm_action_mean = nn.Linear(hidden_size, 3)
        self.arm_action_logstd = nn.Parameter(torch.zeros(3), requires_grad=True)
        self.gripper_action = nn.Linear(hidden_size, 3)
        self.value_feature = [nn.Linear(self.obs_dim, hidden_size), nn.ReLU()]
        for _ in range(n_layers - 1):
            self.value_feature.append(nn.Linear(hidden_size, hidden_size))
            self.value_feature.append(nn.ReLU())
        self.value_feature = nn.Sequential(*self.value_feature)
        self.value_head = nn.Linear(hidden_size, 1)
        self.is_recurrent = False
        self.recurrent_hidden_state_size = 1
        self._initialize()

    def _initialize(self):
        nn.init.orthogonal_(self.value_head.weight, gain=np.sqrt(2))
        nn.init.constant_(self.value_head.bias, 0.)
        nn.init.orthogonal_(self.arm_action_mean.weight, gain=0.01)
        nn.init.constant_(self.arm_action_mean.bias, 0.)
    
    def forward(self, obs, rnn_hxs=None, rnn_masks=None, previ_obs=None):
        policy_features = self.policy_feature(obs)
        arm_action_mean = self.arm_action_mean(policy_features)
        gripper_action_logit = self.gripper_action(policy_features)
        value_features = self.value_feature(obs)
        values = self.value_head(value_features)
        action_dist = (Normal(arm_action_mean, torch.exp(self.arm_action_logstd)), Categorical(logits=gripper_action_logit))
        return values, action_dist, rnn_hxs
    
    def act(self, obs, rnn_hxs=None, rnn_masks=None, deterministic=False, previ_obs=None, forward_value=True):
        values, action_dist, rnn_hxs = self.forward(obs, rnn_hxs, rnn_masks, previ_obs)
        if deterministic:
            arm_action = action_dist[0].loc
            gripper_action = torch.argmax(action_dist[1].probs, dim=-1, keepdim=True)
        else:
            arm_action = action_dist[0].sample()
            gripper_action = action_dist[1].sample().unsqueeze(dim=-1)
        normed_gripper_action = gripper_action / 2.0 * 2 - 1
        actions = torch.cat([arm_action, normed_gripper_action], dim=-1)
        log_prob = action_dist[0].log_prob(arm_action).sum(dim=-1, keepdim=True) + action_dist[1].log_prob(gripper_action.squeeze(dim=-1)).unsqueeze(dim=-1)
        return values, actions, log_prob, rnn_hxs
    
    def evaluate_actions(self, obs, rnn_hxs, rnn_masks, actions):
        _, action_dist, rnn_hxs = self.forward(obs, rnn_hxs, rnn_masks)
        arm_action = torch.narrow(actions, dim=1, start=0, length=3)
        gripper_action = ((torch.narrow(actions, dim=1, start=3, length=1) + 1) / 2 * 2).to(torch.int)
        log_prob = action_dist[0].log_prob(arm_action).sum(dim=-1, keepdim=True) + action_dist[1].log_prob(gripper_action.squeeze(dim=-1)).unsqueeze(dim=-1)
        entropy = action_dist[0].entropy()
        return log_prob, entropy, rnn_hxs
    
    def take_action(self, obs):
        policy_features = self.policy_feature(obs)
        arm_action_mean = self.arm_action_mean(policy_features)
        gripper_action_logit = self.gripper_action(policy_features)
        gripper_action = torch.argmax(gripper_action_logit, dim=1, keepdim=True) / 2.0 * 2 - 1
        action = torch.clamp(torch.cat([arm_action_mean, gripper_action], dim=-1), min=-1, max=1)
        return action


class PandaExpertPolicy(ActorCriticPolicy):
    def __init__(self, device):
        super(PandaExpertPolicy, self).__init__()
        self.device = device
        self.phase = None
    
    def act(self, obs, deterministic):
        assert obs.shape[-1] == 18
        if self.phase is None:
            self.phase = torch.zeros(obs.shape[0], dtype=torch.int, device=self.device)
        box_pos = obs[:, :3]
        approach_pos = box_pos + torch.tensor([[0., 0., 0.1]], device=self.device, dtype=torch.float)
        eef_pos = obs[:, 3:6]
        gripper_width = torch.sum(obs[:, 10:12], dim=-1)
        goal_pos = obs[:, 15:18]
        goal_approach_pos = goal_pos + torch.tensor([[0., 0., 0.02]], device=self.device, dtype=torch.float)
        # before_reach_before_open = torch.logical_and(torch.norm(box_pos - eef_pos, dim=-1) > 0.005, gripper_width < 0.075)
        # before_reach_after_open = torch.logical_and(torch.norm(box_pos - eef_pos, dim=-1) > 0.005, gripper_width >= 0.075)
        # after_reach_before_close = torch.logical_and(torch.norm(box_pos - eef_pos, dim=-1) <= 0.01, gripper_width >= 0.051)
        # after_reach_after_close = torch.logical_and(torch.norm(box_pos - eef_pos, dim=-1) <= 0.01, gripper_width < 0.051)
        inc_phase = torch.zeros(obs.shape[0], dtype=torch.bool, device=self.device)
        inc_phase[torch.logical_and(self.phase == 0, gripper_width >= 0.075)] = True
        inc_phase[torch.logical_and(self.phase == 1, torch.norm(approach_pos - eef_pos, dim=-1) <= 0.01)] = True
        inc_phase[torch.logical_and(self.phase == 2, torch.norm(box_pos - eef_pos, dim=-1) <= 0.01)] = True
        inc_phase[torch.logical_and(self.phase == 3, gripper_width < 0.051)] = True
        inc_phase[torch.logical_and(self.phase == 4, torch.norm(goal_approach_pos - eef_pos, dim=-1) <= 0.01)] = True
        self.phase += inc_phase.to(torch.int)
        actions = torch.zeros(obs.shape[0], 4, device=self.device)
        # actions[torch.where(before_reach_before_open), 3] = 1.0
        actions[:, 3] = 1.0 * (self.phase == 0).to(torch.float)
        to_approach = approach_pos - eef_pos
        actions[:, :3] += 20 * to_approach * (self.phase == 1).unsqueeze(dim=-1)
        to_box = box_pos - eef_pos
        # actions[torch.where(before_reach_after_open), :3] = torch.clamp(20 * to_box[before_reach_after_open], -1, 1)
        actions[:, :3] += 20 * to_box * (self.phase == 2).unsqueeze(dim=-1)
        # actions[torch.where(after_reach_before_close), 3] = -1.0
        actions[:, 3] += -1.0 * (self.phase == 3).to(torch.float)
        
        actions[:, :3] += 20 * (goal_approach_pos - eef_pos) * (self.phase == 4).unsqueeze(dim=-1)
        to_goal = goal_pos - eef_pos
        # actions[torch.where(after_reach_after_close), :3] = torch.clamp(20 * to_goal[after_reach_after_close], -1, 1)
        actions[:, :3] += 20 * to_goal * (self.phase == 5).unsqueeze(dim=-1)
        
        # print(actions[0])
        return None, actions, None, None
    
    def reset(self, idx):
        if self.phase is not None:
            self.phase[idx] *= 0
     
class PandaHybridPolicy(ActorCriticPolicy):
    def __init__(self, obs_dim, hidden_size):
        super(PandaHybridPolicy, self).__init__()
        self.obs_dim = obs_dim
        self.policy_feature = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU()
        )
        self.action_type_predictor = nn.Linear(hidden_size, 2)
        self.action_mean = nn.Linear(hidden_size, 4)
        self.action_logstd = nn.Parameter(torch.zeros(4), requires_grad=True)
        self.value_feature = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU()
        )
        self.value_head = nn.Linear(hidden_size, 1)
        self.is_recurrent = False
        self.recurrent_hidden_state_size = 1
        self._initialize()
    
    def _initialize(self):
        nn.init.orthogonal_(self.value_head.weight, gain=np.sqrt(2))
        nn.init.constant_(self.value_head.bias, 0.)
        nn.init.orthogonal_(self.action_mean.weight, gain=0.01)
        nn.init.constant_(self.action_mean.bias, 0.)
    
    def forward(self, obs, rnn_hxs=None, rnn_masks=None):
        policy_feature = self.policy_feature(obs)
        action_type = torch.softmax(self.action_type_predictor(policy_feature),dim=-1)
        action_mean = self.action_mean(policy_feature)
        action_type_dist = Categorical(probs=action_type)
        pos_action_dist = Normal(action_mean[:, :3], torch.exp(self.action_logstd[:3]))
        gripper_action_dist = Normal(action_mean[:, 3:], torch.exp(self.action_logstd[3:]))
        value_features = self.value_feature(obs)
        values = self.value_head(value_features)
        return values, (action_type_dist, pos_action_dist, gripper_action_dist),rnn_hxs
    
    def act(self, obs, rnn_hxs=None, rnn_masks=None, deterministic=False):
        values, (action_type_dist, pos_action_dist, gripper_action_dist), rnn_hxs = self.forward(obs, rnn_hxs, rnn_masks)
        if deterministic:
            action_type = torch.argmax(action_type_dist.probs, dim=1, keepdim=True)
            pos_actions = pos_action_dist.mean
            gripper_actions = gripper_action_dist.mean
        else:
            action_type = action_type_dist.sample().unsqueeze(dim=-1)
            pos_actions = pos_action_dist.sample()
            gripper_actions = gripper_action_dist.sample()
        actions = torch.cat([action_type, pos_actions, gripper_actions], dim=-1)
        log_prob = action_type_dist.log_prob(action_type.squeeze(dim=-1)).unsqueeze(dim=-1) \
                   + pos_action_dist.log_prob(pos_actions).sum(dim=-1, keepdim=True) \
                   + gripper_action_dist.log_prob(gripper_actions).sum(dim=-1, keepdim=True)
        return values, actions, log_prob, rnn_hxs
    
    def evaluate_actions(self, obs, rnn_hxs, rnn_masks, actions):
        values, (action_type_dist, pos_action_dist, gripper_action_dist), rnn_hxs = self.forward(obs, rnn_hxs, rnn_masks)
        action_type = torch.narrow(actions, dim=1, start=0, length=1)
        pos_actions = torch.narrow(actions, dim=1, start=1, length=3)
        gripper_actions = torch.narrow(actions, dim=1, start=4, length=1)
        log_prob = action_type_dist.log_prob(action_type.squeeze(dim=-1)).unsqueeze(dim=-1) \
                   + pos_action_dist.log_prob(pos_actions).sum(dim=-1, keepdim=True) \
                   + gripper_action_dist.log_prob(gripper_actions).sum(dim=-1, keepdim=True)
        entropy = (action_type_dist.entropy() + pos_action_dist.entropy().sum(dim=-1) + gripper_action_dist.entropy().sum(dim=-1)) / 5
        return log_prob, entropy, rnn_hxs
