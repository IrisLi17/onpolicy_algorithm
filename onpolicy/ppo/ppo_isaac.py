import time
import torch
import torch.nn as nn
import torch.optim as optim
from utils import logger
from onpolicy.storage import RolloutStorage
from collections import deque
import numpy as np
from typing import List, Dict
import pickle
import wandb


class PPO(object):
    def __init__(self, env, policy: nn.Module, device="cpu", n_steps=1024, nminibatches=32, noptepochs=10, gamma=0.99,
                 lam=0.95, learning_rate=2.5e-4, cliprange=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, eps=1e-5,
                 use_gae=True, use_clipped_value_loss=True, use_linear_lr_decay=False, feature_only=False, use_wandb=False):
        self.env = env
        self.policy = policy
        self.device = device
        self.n_steps = n_steps
        self.nminibatches = nminibatches
        self.noptepochs = noptepochs
        self.gamma = gamma
        self.lam = lam
        self.learning_rate = learning_rate
        self.cliprange = cliprange
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.use_gae = use_gae
        self.use_clipped_value_loss = use_clipped_value_loss
        self.use_linear_lr_decay = use_linear_lr_decay
        self.feature_only = feature_only
        self.use_wandb = use_wandb

        self.n_envs = self.env.num_envs

        self.rollouts = RolloutStorage(self.n_steps, self.n_envs,
                                       (self.env.num_obs,) if not self.feature_only else (self.policy.obs_feature_size,),
                                       self.env.num_actions,
                                       self.policy.recurrent_hidden_state_size)

        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate, eps=eps)

    def learn(self, total_timesteps, callback=None):
        ep_infos = dict()
        obs = self.env.reset()
        if self.feature_only:
            self.rollouts.obs[0].copy_(self.policy.encode_obs(obs))
        else:
            self.rollouts.obs[0].copy_(obs)
        self.rollouts.to(self.device)
        self.num_timesteps = 0
        # loss_names = ["value_loss", "policy_loss", "entropy", "grad_norm", "param_norm",
        #               "il_loss", "predictor_loss", "sa_predictor_loss"]

        start = time.time()
        num_updates = int(total_timesteps) // self.n_steps // self.n_envs

        for j in range(num_updates):
            step_time = 0
            if not isinstance(callback, list):
                callback = [callback]
            for cb in callback:
                if callable(cb):
                    cb(locals(), globals())
            if self.use_linear_lr_decay:
                # decrease learning rate linearly
                update_linear_schedule(self.optimizer, j, num_updates, self.learning_rate)

            for step in range(self.n_steps):
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = self.policy.act(
                        self.rollouts.obs[step], self.rollouts.recurrent_hidden_states[step],
                        self.rollouts.masks[step])

                # Obser reward and next obs
                step_start = time.time()
                obs, reward, dones, info = self.env.step(action)
                step_time += time.time() - step_start
                reward = reward.float().unsqueeze(dim=-1)
                if self.feature_only:
                    store_obs = self.policy.encode_obs(obs)
                else:
                    store_obs = obs
                with torch.no_grad():
                    _next_value = self.policy.get_value(
                        store_obs, None, None).detach()
                    # If terminate due to time limit, mimic infinite horizon
                    reward += self.gamma * _next_value * info["time_out"].float().unsqueeze(dim=-1)
                self.num_timesteps += self.n_envs

                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    for k in maybe_ep_info:
                        if not (k in ep_infos):
                            ep_infos[k] = deque(maxlen=1000)
                        ep_infos[k].extend(maybe_ep_info[k])

                # If done then clean the history of observations.
                masks = (1 - dones.float()).reshape((self.n_envs, 1))
                bad_masks = torch.ones((self.n_envs, 1), dtype=torch.float)
                self.rollouts.insert(store_obs, recurrent_hidden_states, action,
                                     action_log_prob, value, reward, masks, bad_masks)

            with torch.no_grad():
                next_value = self.policy.get_value(
                    self.rollouts.obs[-1], self.rollouts.recurrent_hidden_states[-1],
                    self.rollouts.masks[-1]).detach()

            self.rollouts.compute_returns(next_value, self.use_gae, self.gamma, self.lam)

            losses = self.update()

            self.rollouts.after_update()

            fps = int(self.num_timesteps / (time.time() - start))
            log_dict = {}
            log_dict["serial_timesteps"] = j * self.n_steps
            log_dict["n_updates"] = j
            log_dict["total_timesteps"] = self.num_timesteps
            log_dict["fps"] = fps
            for key in ep_infos:
                if key == "r":
                    log_dict["ep_reward_mean"] = torch.mean(torch.tensor(ep_infos["r"]).float()).item()
                elif key == "l":
                    log_dict["ep_len_mean"] = torch.mean(torch.tensor(ep_infos["l"]).float()).item()
                elif key == "t":
                    pass
                else:
                    log_dict[key] = torch.mean(torch.tensor(ep_infos[key]).float()).item()
            log_dict["time_elapsed"] = time.time() - start
            log_dict["time_env"] = step_time
            for loss_name in losses.keys():
                log_dict[loss_name] = losses[loss_name]
            for k, v in log_dict.items():
                logger.logkv(k, v)
            logger.dumpkvs()
            if self.use_wandb:
                wandb.log(log_dict, step=self.num_timesteps)

    def update(self):
        advantages = self.rollouts.returns[:-1] - self.rollouts.value_preds[:-1]
        adv_mean, adv_std = advantages.mean(), advantages.std()
        advantages = (advantages - adv_mean) / (adv_std + 1e-5)

        losses = dict(value_loss=[], policy_loss=[], entropy=[], grad_norm=[], param_norm=[],
                      clip_ratio=[])

        for e in range(self.noptepochs):
            if self.policy.is_recurrent:
                data_generator = self.rollouts.recurrent_generator(
                    advantages, self.nminibatches)
            else:
                data_generator = self.rollouts.feed_forward_generator(
                    advantages, self.nminibatches)

            for mb_idx, sample in enumerate(data_generator):
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                episode_success_batch, adv_targ, next_obs_batch, next_masks_batch, *_ = sample
                # todo: bug with recurrent generator
                # print("hxs shape", recurrent_hidden_states_batch.shape, "mask shape", masks_batch.shape)  # (2, 128), (8192, 1)
                # Reshape to do in a single forward pass for all steps
                action_log_probs, dist_entropy, _ = self.policy.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)
                dist_entropy = dist_entropy.mean()

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.cliprange,
                                    1.0 + self.cliprange) * adv_targ
                clipped_ratio = (torch.abs(ratio - 1) > self.cliprange).sum().item() / ratio.shape[0]
                action_loss = -torch.min(surr1, surr2).mean()

                values = self.policy.get_value(obs_batch, recurrent_hidden_states_batch, masks_batch)
                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                                         (values - value_preds_batch).clamp(-self.cliprange, self.cliprange)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                            value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.vf_coef + action_loss -
                 dist_entropy * self.ent_coef).backward()
                total_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                # print('total norm', total_norm)
                self.optimizer.step()

                params = list(filter(lambda p: p[1].grad is not None, self.policy.named_parameters()))
                param_norm = torch.norm(
                    torch.stack([torch.norm(p[1].detach().to(self.device)) for p in params]))

                isnan = torch.tensor([torch.isnan(p[1].detach()).any() for p in self.policy.named_parameters()]).any()
                if isnan:
                    logger.log("value loss", value_loss, "action loss", action_loss, "dist entropy", dist_entropy,
                               "grad norm", total_norm)
                    logger.log("ratio", torch.isnan(ratio).any(), "advantage", torch.isnan(advantages).any(),
                               "action log probs", torch.isnan(action_log_probs).any(),
                               "old action log probs", torch.isnan(old_action_log_probs_batch).any(),
                               "obs batch", torch.isnan(obs_batch).any(), "action batch",
                               torch.isnan(actions_batch).any())
                    with open("debug_data.pkl", "wb") as f:
                        pickle.dump(dict(action_log_probs=action_log_probs.detach().cpu().numpy(),
                                         old_action_log_probs_batch=old_action_log_probs_batch.detach().cpu().numpy(),
                                         ratio=ratio.detach().cpu().numpy(),
                                         advantages=advantages.detach().cpu().numpy(),
                                         obs_batch=obs_batch.detach().cpu().numpy(),
                                         actions_batch=actions_batch.detach().cpu().numpy()), f)
                    raise RuntimeError

                losses["value_loss"].append(value_loss.item())
                losses["policy_loss"].append(action_loss.item())
                losses["entropy"].append(dist_entropy.item())
                losses["grad_norm"].append(total_norm.item())
                losses["param_norm"].append(param_norm.item())
                losses["clip_ratio"].append(clipped_ratio)

        for key in losses:
            losses[key] = safe_mean(losses[key])

        return losses

    def save(self, save_path):
        save_dict = {'optimizer': self.optimizer.state_dict()}
        if hasattr(self.policy, "get_save_dict"):
            save_dict['policy'] = self.policy.get_save_dict()
        else:
            save_dict['policy'] = self.policy.state_dict()
        torch.save(save_dict, save_path)

    def load(self, load_pth, eval=True):
        checkpoint = torch.load(load_pth, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'], strict=False)
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            pass
        if eval:
            self.policy.eval()
        else:
            self.policy.train()
    
    def pretrain(self, obs_buffer, action_buffer):
        optimizer = optim.Adam(self.policy.parameters(), lr=2.5e-4)
        n_epoch = 30
        batch_size = 64
        n_data = obs_buffer.shape[0]
        assert action_buffer.shape[0] == n_data
        inds = np.arange(n_data)
        if self.feature_only:
            with torch.no_grad():
                # TODO: divide into minibatch to avoid too much memory usage
                _n_batch = n_data // 256 if n_data % 256 == 0 else n_data // 256 + 1
                _obs_buffer = []
                for i in range(_n_batch):
                    _obs_buffer.append(self.policy.encode_obs(torch.from_numpy(obs_buffer[i * 256: (i + 1) * 256]).to(self.device)))
                obs_buffer = torch.cat(_obs_buffer, dim=0)
        else:
            obs_buffer = torch.from_numpy(obs_buffer).to(self.device)
        action_buffer = torch.from_numpy(action_buffer).to(self.device)
        losses = deque(maxlen=100)
        for e in range(n_epoch):
            np.random.shuffle(inds)
            for m in range(n_data // batch_size):
                mb_ids = inds[m * batch_size: (m + 1) * batch_size]
                obs_batch = obs_buffer[mb_ids]
                actions_batch = action_buffer[mb_ids]
                action_log_probs, dist_entropy, _ = self.policy.evaluate_actions(
                    obs_batch, None, None,
                    actions_batch)
                loss = -action_log_probs.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            print(e, np.mean(losses))

def safe_mean(arr):
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return nan. It is used for logging only.

    :param arr: (np.ndarray)
    :return: (float)
    """
    return np.nan if len(arr) == 0 else np.mean(arr)


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def update_linear_clip(epoch, total_num_epochs, initial_clip):
    cur_clip = initial_clip - (initial_clip * (epoch / float(total_num_epochs)))
    return cur_clip
