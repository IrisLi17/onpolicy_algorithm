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


from policies.seg_dataset import transform
from policies.seg_model.segmentation_module import make_model
from PIL import Image
import torch.nn.functional as functional


def fetch_feature(feature, mask, num_classes=6):
    out_feature = []
    mask = torch.cat([mask.unsqueeze(1)]*feature.size()[1], dim=1)

    for i in range(1, num_classes+1):
        mask_ = (mask==i).float()
        feature_ = (feature * mask_).sum(-1).sum(-1) / torch.clamp(mask_.sum(-1).sum(-1), min=1e-5)
        out_feature.append(feature_.unsqueeze(1))
    return torch.cat(out_feature, dim=1)

def fetch_mask(img, colors = np.array([[1.0, 0, 0], [1, 1, 0], [0.2, 0.8, 0.8], [0.8, 0.2, 0.8], [0.2, 0.8, 0.2], [0.0, 0.0, 1.0]]), thr=5500.):
    ### img : np.ndarray(uint8) (B, H, W, 3)
    ### return masks np.ndarray(uint8) (B, H, W, 3), masks_colors np.ndarray(float) (B, H, W, 3)
    masks = np.zeros_like(img)
    masks_color = np.zeros_like(img)
    for i, color in enumerate(colors):
        cs = np.zeros_like(img)
        cs[..., :] = color
        if (color == [0.2, 0.8, 0.2]).all():
            thr *= 0.5
        if (color == [0.8, 0.2, 0.8]).all():
            thr *= 0.9
        if (color == [1, 1, 0]).all():
            thr *= 1.3
        mse = np.mean((img-cs*255.)**2, -1)
        # print(np.where(mse<=thr))
        masks[np.where(mse<=thr)] = (i+1)
        masks_color[np.where(mse <= thr)] = color
        if (color == [0.2, 0.8, 0.2]).all():
            thr /= 0.5
        if (color == [0.8, 0.2, 0.8]).all():
            thr /= 0.9
        if (color == [1, 1, 0]).all():
            thr /= 1.3
    return masks, masks_color


class SegAttentionPoicy(ActorCriticPolicy):
    def __init__(self, resolution=(128, 128), encoder_path=None, act_dim=6, num_bin=21,
                 privilege_dim=0) -> None:
        super().__init__()
        self.resolution = resolution

        self.act_dim = act_dim
        if isinstance(num_bin, int):
            num_bin = [num_bin] * act_dim
        self.num_bin = np.array(num_bin)
        self.privilege_dim = privilege_dim
        self.robot_state_dim = 7
        self.oc_encoder = make_model()
        _actor_attn_layer = nn.TransformerEncoderLayer(
            d_model=2 * 256, nhead=1, dim_feedforward=2 * 256, dropout=0.0, batch_first=True
        )  # batch, seq, feature
        self.actor_attn_encoder = nn.TransformerEncoder(_actor_attn_layer, num_layers=2)
        self.act_type = nn.Sequential(
            nn.Linear(2 * 256, 1)
        )
        self.act_param = nn.ModuleList([nn.Sequential(
            nn.Linear(2 * 256, num_bin[i])
        ) for i in range(act_dim)])

        self.value_object_encode_layer = nn.Sequential(
            nn.Linear(14, 64), nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.transform = transform.Compose([
            transform.RandomResizedCrop(128, (0.5, 2.0)),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
        self.im_mean = nn.Parameter(
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).reshape((1, 3, 1, 1)), 
            requires_grad=False)
        self.im_std = nn.Parameter(
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).reshape((1, 3, 1, 1)), 
            requires_grad=False)
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.load_encoder(encoder_path, device)

    def load_encoder(self, path, device):
        pre_dict, new_dict = \
        torch.load(path, map_location=device)[
            "model_state"], {}
        for k, v in pre_dict.items():
            new_dict[k[7:]] = v
        self.oc_encoder.load_state_dict(new_dict)
        del pre_dict
        del new_dict

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
        return cur_image.detach().cpu().numpy(), goal_image.detach().cpu().numpy(), privilege_info.detach()

    def forward(self, obs, rnn_hxs=None, rnn_masks=None, forward_policy=True):
        cur_image, goal_image, privilege_info = self._obs_parser(obs)
        if forward_policy:
            # prepare masks
            cur_image_ = cur_image.transpose((0,2,3,1)).copy()
            cur_masks, _ = fetch_mask(cur_image_)
            cur_masks = torch.from_numpy(cur_masks).to(obs.device).float().mean(-1).long()
            cur_image = (torch.from_numpy(cur_image).to(obs.device) / 255.0 - self.im_mean) / self.im_std
            # cur_image = Image.fromarray(cur_image)
            # cur_image = self.transforms(cur_image)
            # cur_image = cur_image.permute((0, 2, 3, 1))
            with torch.no_grad():
                _, out = self.oc_encoder(cur_image, ret_intermediate=True)
                out["pre_logits"] = functional.interpolate(out["pre_logits"], size=self.resolution, mode="bilinear",
                                                           align_corners=False)
                cur_feat = fetch_feature(out["pre_logits"], cur_masks) # B * 6 * 256
            goal_image_ = goal_image.transpose((0, 2, 3, 1)).copy()
            goal_masks, _ = fetch_mask(goal_image_)
            goal_masks = torch.from_numpy(goal_masks).to(obs.device).float().mean(-1).long()
            goal_image = (torch.from_numpy(goal_image).to(obs.device) / 255.0 - self.im_mean) / self.im_std
            # goal_image = self.transforms(goal_image)
            # goal_image = goal_image.permute((0, 2, 3, 1))
            with torch.no_grad():
                _, out = self.oc_encoder(goal_image, ret_intermediate=True)
                out["pre_logits"] = functional.interpolate(out["pre_logits"], size=self.resolution, mode="bilinear",
                                                           align_corners=False)
                goal_feat = fetch_feature(out["pre_logits"], goal_masks)  # B * 6 * 256
                # goal_combined_recon, goal_recons, goal_masks, goal_slot_feature = self.oc_encoder.forward(goal_image)
            # import matplotlib.pyplot as plt
            # debug_cur = ((cur_combined_recon[0:1].permute(0, 3, 1, 2) * self.im_std + self.im_mean) * 255.0)[0].permute(1, 2, 0)
            # plt.imsave("tmp/tmp0.png", debug_cur.detach().cpu().numpy().astype(np.uint8))
            # plt.imsave("tmp/tmp0_gt.png", ((cur_image * self.im_std.squeeze() + self.im_mean.squeeze()) * 255)[0].detach().cpu().numpy().astype(np.uint8))
            # debug_goal = ((goal_combined_recon[0:1].permute(0, 3, 1, 2) * self.im_std + self.im_mean) * 255.0)[0].permute(1, 2, 0)
            # plt.imsave("tmp/tmp1.png", debug_goal.detach().cpu().numpy().astype(np.uint8))
            # plt.imsave("tmp/tmp1_gt.png", ((goal_image * self.im_std.squeeze() + self.im_mean.squeeze()) * 255)[0].detach().cpu().numpy().astype(np.uint8))
            # cur_obj_feature, cur_assignment = assign_object(
            #     cur_image * self.im_std.squeeze() + self.im_mean.squeeze(), cur_masks, self.color, cur_slot_feature)
            # goal_obj_feature, goal_assignment = assign_object(
            #     goal_image * self.im_std.squeeze() + self.im_mean.squeeze(), goal_masks, self.color, goal_slot_feature)
            # # print("cur assignment", cur_assignment[0], "goal_assignment", goal_assignment[0])
            # # exit()
            assert not torch.isnan(cur_feat).any(), cur_feat
            assert not torch.isnan(goal_feat).any(), goal_feat
            obj_feature = torch.cat([cur_feat, goal_feat], dim=-1)
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


def spatial_broadcast(slots, resolution):
    """Broadcast slot features to a 2D grid and collapse slot dimension."""
    # `slots` has shape: [batch_size, num_slots, slot_size].
    # slots = tf.reshape(slots, [-1, slots.shape[-1]])[:, None, None, :]
    # grid = tf.tile(slots, [1, resolution[0], resolution[1], 1])
    slots = slots.reshape((-1, slots.shape[-1]))[:, None, None, :]
    grid = torch.tile(slots, (1, resolution[0], resolution[1], 1))
    # `grid` has shape: [batch_size*num_slots, width, height, slot_size].
    return grid


def spatial_flatten(x):
    # return tf.reshape(x, [-1, x.shape[1] * x.shape[2], x.shape[-1]])
    return x.reshape((-1, x.shape[1] * x.shape[2], x.shape[-1]))


def unstack_and_split(x, batch_size, num_channels=3):
    """Unstack batch dimension and split into channels and alpha mask."""
    # unstacked = tf.reshape(x, [batch_size, -1] + x.shape.as_list()[1:])
    # channels, masks = tf.split(unstacked, [num_channels, 1], axis=-1)
    unstacked = x.reshape((batch_size, -1, *(x.shape[1:])))
    channels = unstacked[..., :num_channels]
    masks = unstacked[..., -1:]
    return channels, masks

def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return np.concatenate([grid, 1.0 - grid], axis=-1)


class SoftPositionEmbed(nn.Module):
    """Adds soft positional embedding with learnable projection."""

    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.

        Args:
          hidden_size: Size of input feature dimension.
          resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.dense = nn.Linear(4, hidden_size)
        self.grid = nn.Parameter(torch.from_numpy(build_grid(resolution)), requires_grad=False)

    def forward(self, inputs):
        return inputs + self.dense(self.grid)


def detect_background(masks):
    # TODO: also filter out empty slots if there is redundency
    # get key mask for later transformer network
    area = masks.sum(dim=3).sum(dim=2)  # batch_size, num_slots, 1
    is_background = (area == area.max(dim=1, keepdim=True)[0]).squeeze(dim=-1)  # batch_size, num_slots
    return is_background


# def correct_object_id(slot_id, recons, masks, COLORS: torch.Tensor):
#   # binarize mask
#   masks = masks * (masks > 0.8)
#   # avg_colors shape: batch_size, num_slots, num_channels
#   avg_colors = (recons * masks).sum(dim=3).sum(dim=2) / torch.maximum(
#     masks.sum(dim=3).sum(dim=2), 1e-5 * torch.ones(masks.shape[0], masks.shape[1], 1, device=masks.device))
#   slot_id = slot_id.reshape(slot_id.shape[0])
#   # selected_colors: batch_size, num_channels
#   selected_colors = avg_colors[torch.arange(slot_id.shape[0], device=slot_id.device), slot_id]
#   print("selected colors", selected_colors)
#   # compare with used colors
#   num_objects = COLORS.shape[0]
#   color_distances = torch.stack([color_distance(selected_colors, COLORS[i]) for i in range(num_objects)], dim=-1)
#   # batch_size, num colors
#   object_id = torch.argmin(color_distances, dim=-1) # batch_size
#   return object_id


def color_distance(color1: np.ndarray, color2: np.ndarray):
    assert color1.shape[-1] == 3
    assert color2.shape[-1] == 3
    import matplotlib
    hsv1 = matplotlib.colors.rgb_to_hsv(color1)
    h1 = hsv1[..., 0]
    hsv2 = matplotlib.colors.rgb_to_hsv(color2)
    h2 = hsv2[..., 0]
    return np.abs(h1 - h2)


def assign_object(obs, masks, COLORS, slot_features):
    is_background = detect_background(masks)  # batch_size, num_slots
    # print("is background", is_background.shape, is_background)
    # binarize mask
    masks = masks * (masks > 0.9)
    avg_colors = (obs.unsqueeze(dim=1) * masks).sum(dim=3).sum(dim=2) / torch.maximum(
        masks.sum(dim=3).sum(dim=2),
        1e-5 * torch.ones(masks.shape[0], masks.shape[1], 1, device=masks.device)
    )  # batch_size, num_slots, num_channels
    distances = np.stack(
        [color_distance(avg_colors.detach().cpu().numpy(), COLORS[i]) for i in range(COLORS.shape[0])], axis=-1
    )  # batch_size, num_slots, num_objects
    distances = torch.from_numpy(distances).to(obs.device)

    # print("distances", distances, distances.shape)
    distances = distances + is_background.unsqueeze(dim=-1).float() * 1e5  # Background slots should not be assigned
    # print("with bg distances", distances) # bug
    # Each object will be assigned a best slot. There may be slots that do not belong to any object, or belong to multiple objects
    assignment = (distances == distances.min(dim=1, keepdim=True)[0]).float()  # batch_size, num_slots, num_objects
    object_features = torch.matmul(assignment.transpose(1, 2), slot_features)  # batch_size, num_objects, feature_dim
    return object_features, assignment


def build_model(resolution, batch_size, num_slots, num_iterations,
                num_channels=3, model_type="object_discovery"):
    """Build keras model."""
    if model_type == "object_discovery":
        model_def = SlotAttentionAutoEncoder
    # elif model_type == "set_prediction":
    #   model_def = SlotAttentionClassifier
    else:
        raise ValueError("Invalid name for model type.")

    image = torch.randn((5, *list(resolution), num_channels))
    model = model_def(resolution, num_slots, num_iterations)
    output = model.forward(image)
    for item in output:
        print(item.shape)
    # image = tf.keras.Input(list(resolution) + [num_channels], batch_size)
    # outputs = model_def(resolution, num_slots, num_iterations)(image)
    # model = tf.keras.Model(inputs=image, outputs=outputs)
    # return model


if __name__ == "__main__":
    build_model((128, 128), None, 7, 3)


