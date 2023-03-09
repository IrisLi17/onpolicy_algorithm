import einops
import torch
import torchvision

import matplotlib.pyplot as plt
# import segm.utils.torch as ptu
import torch.nn.functional as F

from pathlib import Path
from PIL import Image
# from segm import config
# from segm.data.utils import STATS
from policies.vit_policy.segm.model.decoder import MaskTransformer
from policies.vit_policy.segm.model.factory import load_model
from torchvision import transforms
from einops import rearrange
import numpy as np
# import cv2

def fetch_feature(
    input,
    model,
    variant,
    layer_id=0,
    x_patch=0,
    y_patch=0,
    enc=False,
    cls=True,
    type='mean',
    device='cuda',
):
    model.eval()
    model.to(device)

    # Get model config
    patch_size = model.patch_size
    normalization = variant["dataset_kwargs"]["normalization"]
    image_size = variant["dataset_kwargs"]["image_size"]
    n_cls = variant["net_kwargs"]["n_cls"]
    # stats = STATS[normalization]
    # mean, std = torch.FloatTensor(stats["mean"])[None, :, None, None], torch.FloatTensor(stats["std"])[None, :, None, None]

    assert input.dim() == 4

    # normalize
    # input = (input-mean) / std
    # input = input.to(device)

    # Make the image divisible by the patch size
    w, h = (
        image_size - image_size % patch_size,
        image_size - image_size % patch_size,
    )

    # Crop to image size
    input = input[: ,:, :w, :h]

    w_featmap = input.shape[-2] // patch_size
    h_featmap = input.shape[-1] // patch_size

    # Sanity checks
    if not enc and not isinstance(model.decoder, MaskTransformer):
        raise ValueError(
            f"Attention maps for decoder are only availabe for MaskTransformer. Provided model with decoder type: {model.decoder}."
        )

    if not cls:
        if x_patch > w_featmap or y_patch > h_featmap:
            raise ValueError(
                f"Provided patch x: {x_patch} y: {y_patch} is not valid. Patch should be in the range x: [0, {w_featmap}), y: [0, {h_featmap})"
            )
        num_patch = w_featmap * y_patch + x_patch

    if layer_id < 0:
        raise ValueError("Provided layer_id should be positive.")

    if enc and model.encoder.n_layers <= layer_id:
        raise ValueError(
            f"Provided layer_id: {layer_id} is not valid for encoder with {model.encoder.n_layers}."
        )

    if not enc and model.decoder.n_layers <= layer_id:
        raise ValueError(
            f"Provided layer_id: {layer_id} is not valid for decoder with {model.decoder.n_layers}."
        )

    _, cls_seg_feat = model(input.to(device), True)

    # Process input and extract attention maps
    if enc:
        # print(f"Generating Feature Mapping for Encoder Layer Id {layer_id}")
        fmap = model.get_feature_map_enc(input.to(device), layer_id)
        # attentions = model.get_attention_map_enc(input.to(device), layer_id)
        # print(attentions.size())
        num_extra_tokens = 1 + model.encoder.distilled
        if cls:
            fmap = fmap[:, num_extra_tokens:]
            # patches = fmap @ model.decoder.proj_patch
            # patches = patches / patches.norm(dim=-1, keepdim=True)
            masks = fmap @ cls_seg_feat.transpose(1, 2)
            # masks = model.decoder.mask_norm(masks)
            masks = rearrange(masks, "b (h w) n -> b n h w", h=8).reshape(masks.shape[0], n_cls, -1)
            return masks
        else:
            fmap = fmap[
                         :, :, num_patch + num_extra_tokens, num_extra_tokens:
                         ]
            print(fmap.size())
    else:
        print(f"Generating Attention Mapping for Decoder Layer Id {layer_id}")
        attentions = model.get_attention_map_dec(input.to(device), layer_id)
        fmap = model.get_feature_map_dec(input.to(device), layer_id)
        patches = fmap[:, :-n_cls]
        patches = patches @ model.decoder.proj_patch
        patches = patches / patches.norm(dim=-1, keepdim=True)
        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = model.decoder.mask_norm(masks)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=8).reshape(masks.shape[0], n_cls, -1)
        if cls:
            # attentions = attentions[:, :, -n_cls:, :-n_cls]
            # print(attentions.size())
            return masks
        else:
            attentions = attentions[:, :, num_patch, :-n_cls]

    # Reshape into image shape
    # bs, nh = attentions.shape[0], attentions.shape[1]  # Number of heads
    # attentions = attentions.reshape(bs, nh, n_cls, -1).permute(0, 2, 1, 3).mean(-2) if type == 'mean' else attentions.reshape(bs, nh, n_cls, -1).permute(0, 2, 1, 3).reshape(bs, n_cls, -1)
    #
    return attentions