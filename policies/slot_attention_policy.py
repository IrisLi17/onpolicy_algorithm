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
import sys
sys.path.append("../stacking_env")
from bullet_envs.env.primitive_stacking import COLOR
sys.path.remove("../stacking_env")


class SlotAttentionPoicy(ActorCriticPolicy):
  def __init__(self, resolution=(128, 128), num_slots=7, encoder_path=None, act_dim=6, num_bin=21, privilege_dim=0) -> None:
    super().__init__()
    self.resolution = resolution
    self.act_dim = act_dim
    self.num_bin = num_bin
    self.privilege_dim = privilege_dim
    self.robot_state_dim = 7
    self.oc_encoder = SlotAttentionAutoEncoder(resolution, num_slots, num_iterations=3)
    _actor_attn_layer = nn.TransformerEncoderLayer(
      d_model=2 * 64, nhead=1, dim_feedforward=2 * 64, dropout=0.0, batch_first=True
    ) # batch, seq, feature
    self.actor_attn_encoder = nn.TransformerEncoder(_actor_attn_layer, num_layers=2)
    self.act_type = nn.Sequential(
      nn.Linear(2 * 64, 1)
    )
    self.act_param = nn.ModuleList([nn.Sequential(
      nn.Linear(2 * 64, num_bin)
    ) for _ in range(act_dim)])
    
    self.value_object_encode_layer = nn.Sequential(
      nn.Linear(14, 64), nn.ReLU(), 
      nn.Linear(64, 64)
    )
    _encoder_layer = nn.TransformerEncoderLayer(
      d_model=64, nhead=1, dim_feedforward=64, dropout=0.0,
    ) # seq, batch, feature
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
    checkpoint = torch.load(path, map_location=device)
    self.oc_encoder.load_state_dict(checkpoint["param"])
    # freeze parameters
    for p in self.oc_encoder.parameters():
      p.requires_grad = False
    if "im_mean" in checkpoint:
      self.im_mean = checkpoint["im_mean"].to(device)
    else:
      self.im_mean = torch.Tensor([0.5, 0.5, 0.5]).reshape((1, 3, 1, 1)).to(device)
    if "im_std" in checkpoint:
      self.im_std = checkpoint["im_std"].to(device)
    else:
      self.im_std = torch.Tensor([0.5, 0.5, 0.5]).reshape((1, 3, 1, 1)).to(device)
  
  @staticmethod
  def init_weights(sequential, scales):
      [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
        enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]
  
  def _obs_parser(self, obs):
    assert obs.shape[-1] == 2 * 3 * self.resolution[0] * self.resolution[1] + self.robot_state_dim + self.privilege_dim
    cur_image = torch.narrow(obs, dim=-1, start=0, length=3 * self.resolution[0] * self.resolution[1]).reshape(obs.shape[0], 3, *self.resolution)
    goal_image = torch.narrow(obs, dim=-1, start=3 * self.resolution[0] * self.resolution[1], 
                              length=3 * self.resolution[0] * self.resolution[1]).reshape(obs.shape[0], 3, *self.resolution)
    privilege_info = torch.narrow(
      obs, dim=-1, start=2 * 3 * self.resolution[0] * self.resolution[1] + self.robot_state_dim, 
      length=self.privilege_dim)
    return cur_image.detach(), goal_image.detach(), privilege_info.detach()

  def forward(self, obs, rnn_hxs=None, rnn_masks=None, forward_policy=True):
    cur_image, goal_image, privilege_info = self._obs_parser(obs)
    if forward_policy:
      cur_image = (cur_image / 255.0 - self.im_mean) / self.im_std
      cur_image = cur_image.permute((0, 2, 3, 1))
      with torch.no_grad():
        cur_combined_recon, cur_recons, cur_masks, cur_slot_feature = self.oc_encoder.forward(cur_image)
      goal_image = (goal_image / 255.0 - self.im_mean) / self.im_std
      goal_image = goal_image.permute((0, 2, 3, 1))
      with torch.no_grad():
        goal_combined_recon, goal_recons, goal_masks, goal_slot_feature = self.oc_encoder.forward(goal_image)
      # import matplotlib.pyplot as plt
      # debug_cur = ((cur_combined_recon[0:1].permute(0, 3, 1, 2) * self.im_std + self.im_mean) * 255.0)[0].permute(1, 2, 0)
      # plt.imsave("tmp/tmp0.png", debug_cur.detach().cpu().numpy().astype(np.uint8))
      # plt.imsave("tmp/tmp0_gt.png", ((cur_image * self.im_std.squeeze() + self.im_mean.squeeze()) * 255)[0].detach().cpu().numpy().astype(np.uint8))
      # debug_goal = ((goal_combined_recon[0:1].permute(0, 3, 1, 2) * self.im_std + self.im_mean) * 255.0)[0].permute(1, 2, 0)
      # plt.imsave("tmp/tmp1.png", debug_goal.detach().cpu().numpy().astype(np.uint8))
      # plt.imsave("tmp/tmp1_gt.png", ((goal_image * self.im_std.squeeze() + self.im_mean.squeeze()) * 255)[0].detach().cpu().numpy().astype(np.uint8))
      if not hasattr(self, "color"):
        self.color = np.array(COLOR[:6])
      cur_obj_feature, cur_assignment = assign_object(
        cur_image * self.im_std.squeeze() + self.im_mean.squeeze(), cur_masks, self.color, cur_slot_feature)
      goal_obj_feature, goal_assignment = assign_object(
        goal_image * self.im_std.squeeze() + self.im_mean.squeeze(), goal_masks, self.color, goal_slot_feature)
      # print("cur assignment", cur_assignment[0], "goal_assignment", goal_assignment[0])
      # exit()
      obj_feature = torch.cat([cur_obj_feature, goal_obj_feature], dim=-1)
      act_slot_feature = self.actor_attn_encoder(obj_feature)
      act_type_logits = self.act_type(act_slot_feature).squeeze(dim=-1) # bsz, n_obj
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


class SlotAttention(nn.Module):
  """Slot Attention module."""

  def __init__(self, num_iterations, num_slots, slot_size, mlp_hidden_size,
               epsilon=1e-8):
    """Builds the Slot Attention module.

    Args:
      num_iterations: Number of iterations.
      num_slots: Number of slots.
      slot_size: Dimensionality of slot feature vectors.
      mlp_hidden_size: Hidden layer size of MLP.
      epsilon: Offset for attention coefficients before normalization.
    """
    super().__init__()
    self.num_iterations = num_iterations
    self.num_slots = num_slots
    self.slot_size = slot_size
    self.mlp_hidden_size = mlp_hidden_size
    self.epsilon = epsilon

    self.norm_inputs = nn.LayerNorm(64)
    self.norm_slots = nn.LayerNorm(self.slot_size)
    self.norm_mlp = nn.LayerNorm(self.slot_size)

    # Parameters for Gaussian init (shared by all slots).
    self.slots_mu = nn.Parameter(torch.zeros((1, 1, self.slot_size), dtype=torch.float32))
    torch.nn.init.xavier_uniform_(self.slots_mu.data)
    self.slots_log_sigma = nn.Parameter(torch.zeros((1, 1, self.slot_size), dtype=torch.float32))
    torch.nn.init.xavier_uniform_(self.slots_log_sigma.data)

    # Linear maps for the attention module.
    self.project_q = nn.Linear(self.slot_size, self.slot_size, bias=False)
    self.project_k = nn.Linear(64, self.slot_size, bias=False)
    self.project_v = nn.Linear(64, self.slot_size, bias=False)

    # Slot update functions.
    # self.gru = layers.GRUCell(self.slot_size)
    self.gru = nn.GRUCell(self.slot_size, self.slot_size)
    # self.mlp = tf.keras.Sequential([
    #     layers.Dense(self.mlp_hidden_size, activation="relu"),
    #     layers.Dense(self.slot_size)
    # ], name="mlp")
    self.mlp = nn.Sequential(
        nn.Linear(self.slot_size, self.mlp_hidden_size), nn.ReLU(),
        nn.Linear(self.mlp_hidden_size, self.slot_size)
    )

  def forward(self, inputs):
    # `inputs` has shape [batch_size, num_inputs, inputs_size].
    inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
    k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
    v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].

    # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
    # slots = self.slots_mu + tf.exp(self.slots_log_sigma) * tf.random.normal(
    #     [tf.shape(inputs)[0], self.num_slots, self.slot_size])
    slots = self.slots_mu + torch.exp(self.slots_log_sigma) * torch.randn(
        (inputs.shape[0], self.num_slots, self.slot_size), device=inputs.device)

    # Multiple rounds of attention.
    for _ in range(self.num_iterations):
      slots_prev = slots
      slots = self.norm_slots(slots)

      # Attention.
      q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
      q *= self.slot_size ** -0.5  # Normalization.
      # `attn` has shape: [batch_size, num_inputs, num_slots].
      attn_logits = torch.matmul(k, q.transpose(-2, -1))
      attn = torch.softmax(attn_logits, dim=-1)

      # Weigted mean.
      attn = attn + self.epsilon
      # attn /= tf.reduce_sum(attn, axis=-2, keepdims=True)
      attn = attn / attn.sum(dim=-2, keepdim=True)
      # updates = tf.keras.backend.batch_dot(attn, v, axes=-2)
      updates = torch.matmul(attn.transpose(-2, -1), v)
      # `updates` has shape: [batch_size, num_slots, slot_size].

      # Slot update.
      # slots, _ = self.gru(updates, [slots_prev])
      slots = self.gru(
        updates.view(-1, self.slot_size), slots_prev.view(-1, self.slot_size)
      ).view(-1, self.num_slots, self.slot_size)
      slots = slots + self.mlp(self.norm_mlp(slots))

    return slots


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


class SlotAttentionAutoEncoder(nn.Module):
  """Slot Attention-based auto-encoder for object discovery."""

  def __init__(self, resolution, num_slots, num_iterations):
    """Builds the Slot Attention-based auto-encoder.

    Args:
      resolution: Tuple of integers specifying width and height of input image.
      num_slots: Number of slots in Slot Attention.
      num_iterations: Number of iterations in Slot Attention.
    """
    super().__init__()
    self.resolution = resolution
    self.num_slots = num_slots
    self.num_iterations = num_iterations

    # self.encoder_cnn = tf.keras.Sequential([
    #     layers.Conv2D(64, kernel_size=5, padding="SAME", activation="relu"),
    #     layers.Conv2D(64, kernel_size=5, padding="SAME", activation="relu"),
    #     layers.Conv2D(64, kernel_size=5, padding="SAME", activation="relu"),
    #     layers.Conv2D(64, kernel_size=5, padding="SAME", activation="relu")
    # ], name="encoder_cnn")
    self.encoder_cnn = nn.Sequential(
      nn.Conv2d(3, 64, 5, padding="same"), nn.ReLU(),
      nn.Conv2d(64, 64, 5, padding="same"), nn.ReLU(),
      nn.Conv2d(64, 64, 5, padding="same"), nn.ReLU(),
      nn.Conv2d(64, 64, 5, padding="same"), nn.ReLU(),
    )

    self.decoder_initial_size = (8, 8)
    # self.decoder_cnn = tf.keras.Sequential([
    #     layers.Conv2DTranspose(
    #         64, 5, strides=(2, 2), padding="SAME", activation="relu"),
    #     layers.Conv2DTranspose(
    #         64, 5, strides=(2, 2), padding="SAME", activation="relu"),
    #     layers.Conv2DTranspose(
    #         64, 5, strides=(2, 2), padding="SAME", activation="relu"),
    #     layers.Conv2DTranspose(
    #         64, 5, strides=(2, 2), padding="SAME", activation="relu"),
    #     layers.Conv2DTranspose(
    #         64, 5, strides=(1, 1), padding="SAME", activation="relu"),
    #     layers.Conv2DTranspose(
    #         4, 3, strides=(1, 1), padding="SAME", activation=None)
    # ], name="decoder_cnn")
    self.decoder_cnn = nn.Sequential(
      nn.ConvTranspose2d(64, 64, 5, stride=2, padding=2, output_padding=1), nn.ReLU(),
      nn.ConvTranspose2d(64, 64, 5, stride=2, padding=2, output_padding=1), nn.ReLU(),
      nn.ConvTranspose2d(64, 64, 5, stride=2, padding=2, output_padding=1), nn.ReLU(),
      nn.ConvTranspose2d(64, 64, 5, stride=2, padding=2, output_padding=1), nn.ReLU(),
      nn.ConvTranspose2d(64, 64, 5, stride=1, padding=2), nn.ReLU(),
      nn.ConvTranspose2d(64, 4, 3, stride=1, padding=1)
    )

    self.encoder_pos = SoftPositionEmbed(64, self.resolution)
    self.decoder_pos = SoftPositionEmbed(64, self.decoder_initial_size)

    self.layer_norm = nn.LayerNorm(64)
    # self.mlp = tf.keras.Sequential([
    #     layers.Dense(64, activation="relu"),
    #     layers.Dense(64)
    # ], name="feedforward")
    self.mlp = nn.Sequential(
      nn.Linear(64, 64), nn.ReLU(),
      nn.Linear(64, 64)
    )

    self.slot_attention = SlotAttention(
        num_iterations=self.num_iterations,
        num_slots=self.num_slots,
        slot_size=64,
        mlp_hidden_size=128)

  def batch_forward(self, image: torch.Tensor):
    # `image` has shape: [batch_size, width, height, num_channels].

    # Convolutional encoder with position embedding.
    x = self.encoder_cnn.forward(image.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # CNN Backbone.
    x = self.encoder_pos(x)  # Position embedding.
    x = spatial_flatten(x)  # Flatten spatial dimensions (treat image as set).
    x = self.mlp(self.layer_norm(x))  # Feedforward network on set.
    # `x` has shape: [batch_size, width*height, input_size].

    # Slot Attention module.
    slots = self.slot_attention(x)
    # `slots` has shape: [batch_size, num_slots, slot_size].

    # Spatial broadcast decoder.
    x = spatial_broadcast(slots, self.decoder_initial_size)
    # `x` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
    x = self.decoder_pos(x)
    x = self.decoder_cnn(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    # `x` has shape: [batch_size*num_slots, width, height, num_channels+1].

    # Undo combination of slot and batch dimension; split alpha masks.
    recons, masks = unstack_and_split(x, batch_size=image.shape[0])
    # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
    # `masks` has shape: [batch_size, num_slots, width, height, 1].

    # Normalize alpha masks over slots.
    masks = torch.softmax(masks, dim=1)
    recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image.
    # `recon_combined` has shape: [batch_size, width, height, num_channels].

    return recon_combined, recons, masks, slots
  
  def forward(self, image: torch.Tensor):
    if image.shape[0] <= 64:
      return self.batch_forward(image)
    n_rounds = image.shape[0] // 64 if image.shape[0] % 64 == 0 else image.shape[0] // 64 + 1
    recon_combined, recons, masks, slots = [], [], [], []
    for i in range(n_rounds):
      image_batch = image[64 * i: 64 * (i + 1)]
      res_batch = self.batch_forward(image_batch)
      recon_combined.append(res_batch[0])
      recons.append(res_batch[1])
      masks.append(res_batch[2])
      slots.append(res_batch[3])
    recon_combined, recons, masks, slots = map(
      lambda x: torch.cat(x, dim=0), [recon_combined, recons, masks, slots]
    )
    return recon_combined, recons, masks, slots


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
  ) # batch_size, num_slots, num_channels
  distances = np.stack(
    [color_distance(avg_colors.detach().cpu().numpy(), COLORS[i]) for i in range(COLORS.shape[0])], axis=-1
  ) # batch_size, num_slots, num_objects
  distances = torch.from_numpy(distances).to(obs.device)
   
  # print("distances", distances, distances.shape)
  distances = distances + is_background.unsqueeze(dim=-1).float() * 1e5 # Background slots should not be assigned
  # print("with bg distances", distances) # bug
  # Each object will be assigned a best slot. There may be slots that do not belong to any object, or belong to multiple objects
  assignment = (distances == distances.min(dim=1, keepdim=True)[0]).float() # batch_size, num_slots, num_objects
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


