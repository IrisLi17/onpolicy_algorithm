# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Slot Attention model for object discovery and set prediction."""
import numpy as np
import torch
import torch.nn as nn
from policies.base import ActorCriticPolicy
from policies.mvp_stacking_policy import MvpStackingPolicy
from policies.vit_policy.segm.model.factory import load_model
from policies.vit_policy.segm.vit_feature import fetch_feature


STATS = {
    "vit": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
    "deit": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
}

class ViTPoicy(ActorCriticPolicy):
    def __init__(self, resolution=(128, 128), act_dim=6, num_bin=21,
                 privilege_dim=0, encoder_path="/data/zkxu/segmenter/seg_tiny_mask/checkpoint.pth", enc=True, cls=True, type='mean', layer_id=0) -> None:
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.resolution = resolution
        self.act_dim = act_dim
        if isinstance(num_bin, int):
            num_bin = [num_bin] * self.act_dim
        self.num_bin = np.array(num_bin)
        self.privilege_dim = privilege_dim
        self.robot_state_dim = 7
        self.load_encoder(encoder_path, device)
        self.patch_size = self.oc_encoder.patch_size
        self.image_size = self.variant["dataset_kwargs"]["image_size"]
        self.n_heads = self.variant["net_kwargs"]["n_heads"]
        assert self.image_size == self.resolution[0]
        self.type = type
        self.enc = enc
        self.cls = cls
        self.layer_id = layer_id
        self.feat_dim = (self.image_size // self.patch_size)**2 if self.type=='mean' else self.n_heads * (self.image_size[0] // self.patch_size)**2
        _actor_attn_layer = nn.TransformerEncoderLayer(
            d_model=2 * self.feat_dim, nhead=1, dim_feedforward=2 * self.feat_dim, dropout=0.0, batch_first=True
        )  # batch, seq, feature
        self.actor_attn_encoder = nn.TransformerEncoder(_actor_attn_layer, num_layers=2)
        self.act_type = nn.Sequential(
            nn.Linear(2 * self.feat_dim, 1)
        )
        self.act_param = nn.ModuleList([nn.Sequential(
            nn.Linear(2 * self.feat_dim, num_bin[i])
        ) for i in range(act_dim)])

        self.value_object_encode_layer = nn.Sequential(
            nn.Linear(14, 64), nn.ReLU(),
            nn.Linear(64, 64)
        )
        _encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=1, dim_feedforward=64, dropout=0.0,
        )  # seq, batch, feature
        self.value_attn_encoder = nn.TransformerEncoder(_encoder_layer, num_layers=3)
        self.value_agg = nn.Linear(64, 1)
        self.is_recurrent = False
        self.recurrent_hidden_state_size = 1
        self.use_param_mask = False
        self.init_weights(self.act_type, [0.01])
        for i in range(act_dim):
            self.init_weights(self.act_param[i], [0.01])
        self.to(device)

    def load_encoder(self, encoder_path, device):
        self.oc_encoder, self.variant = load_model(encoder_path)
        # freeze parameters
        for p in self.oc_encoder.parameters():
            p.requires_grad = False
        normalization = self.variant["dataset_kwargs"]["normalization"]
        stats = STATS[normalization]
        self.im_mean, self.im_std = torch.FloatTensor(stats["mean"])[None, :, None, None].to(device), torch.FloatTensor(stats["std"])[None, :,
                                                                           None, None].to(device)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def _obs_parser(self, obs):
        assert obs.shape[-1] == 2 * 3 * self.resolution[0] * self.resolution[
            1] + self.robot_state_dim + self.privilege_dim
        cur_image = torch.narrow(obs, dim=-1, start=0, length=3 * self.resolution[0] * self.resolution[1]).reshape(
            obs.shape[0], 3, *self.resolution)
        goal_image = torch.narrow(obs, dim=-1, start=3 * self.resolution[0] * self.resolution[1],
                                  length=3 * self.resolution[0] * self.resolution[1]).reshape(obs.shape[0], 3,
                                                                                              *self.resolution)
        privilege_info = torch.narrow(
            obs, dim=-1, start=2 * 3 * self.resolution[0] * self.resolution[1] + self.robot_state_dim,
            length=self.privilege_dim)
        return cur_image.detach(), goal_image.detach(), privilege_info.detach()

    def forward(self, obs, rnn_hxs=None, rnn_masks=None, forward_policy=True):
        cur_image, goal_image, privilege_info = self._obs_parser(obs)
        if forward_policy:
            cur_image = (cur_image / 255.0 - self.im_mean) / self.im_std
            cur_obj_feature = fetch_feature(cur_image, self.oc_encoder, self.variant, layer_id=self.layer_id, enc=self.enc, cls=self.cls, type=self.type, device=self.device)
            goal_image = (goal_image / 255.0 - self.im_mean) / self.im_std
            goal_obj_feature = fetch_feature(goal_image, self.oc_encoder, self.variant, layer_id=self.layer_id,
                                            enc=self.enc, cls=self.cls, type=self.type, device=self.device)
            obj_feature = torch.cat([cur_obj_feature, goal_obj_feature], dim=-1)
            act_slot_feature = self.actor_attn_encoder(obj_feature)
            act_type_logits = self.act_type(act_slot_feature).squeeze(dim=-1)  # bsz, n_obj
            act_param_logits = [self.act_param[i](act_slot_feature) for i in range(len(self.act_param))]
        else:
            act_type_logits, act_param_logits = None, None
        # value
        bsz = privilege_info.shape[0]
        obj_and_goals = privilege_info.reshape(
            (bsz, 2, -1, 7)
        ).permute((2, 0, 1, 3)).reshape((-1, bsz, 14))
        obj_goal_embed = self.value_object_encode_layer(obj_and_goals)
        value_feature = self.value_attn_encoder(obj_goal_embed)
        value_pred = self.value_agg(torch.mean(value_feature, dim=0))
        return value_pred, (act_type_logits, act_param_logits), rnn_hxs

    def act(self, obs, rnn_hxs=None, rnn_masks=None, deterministic=False):
        return MvpStackingPolicy.act(self, obs, rnn_hxs, rnn_masks, deterministic)

    def evaluate_actions(self, obs, rnn_hxs, rnn_masks, actions):
        return MvpStackingPolicy.evaluate_actions(self, obs, rnn_hxs, rnn_masks, actions)

    def get_value(self, obs, rnn_hxs=None, rnn_masks=None):
        return self.forward(obs, rnn_hxs, rnn_masks, forward_policy=False)[0]

    def get_bc_loss(self, obs, rnn_hxs, rnn_masks, actions):
        return MvpStackingPolicy.get_bc_loss(self, obs, rnn_hxs, rnn_masks, actions)

    def get_aux_loss(self, obs):
        return torch.tensor([np.nan], device=obs.device), torch.tensor([np.nan], device=obs.device)