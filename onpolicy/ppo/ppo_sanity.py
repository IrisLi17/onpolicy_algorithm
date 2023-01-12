import time
import torch
import torch.nn as nn
import torch.optim as optim
from utils import logger
from vec_env.base_vec_env import VecEnv
from onpolicy.storage import RolloutStorage
from collections import deque
import numpy as np
from typing import List, Dict
import pickle
import wandb
import os
from onpolicy.ppo.expansion import OfflineExpansion


class PPO(object):
    def __init__(self, env, policy: nn.Module, device="cpu", n_steps=1024, nminibatches=32, noptepochs=10, gamma=0.99,
                 lam=0.95, learning_rate=2.5e-4, cliprange=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, eps=1e-5,
                 use_gae=True, use_clipped_value_loss=True, use_linear_lr_decay=False, use_wandb=False, warmup_dataset:dict=None):
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
        self.use_wandb = use_wandb
        self.warmup_dataset = warmup_dataset

        # if isinstance(self.env, VecEnv):
        #     self.n_envs = self.env.num_envs
        # else:
        #     self.n_envs = 1
        if hasattr(self.env, "num_envs"):
            self.n_envs = self.env.num_envs
        else:
            self.n_envs = 1
        self.rollouts = RolloutStorage(self.n_steps, self.n_envs,
                                       self.env.observation_space.shape, self.env.action_space,
                                       self.policy.recurrent_hidden_state_size)

        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate, eps=eps)
        self.env_id = self.env.get_attr("spec")[0].id
        self.expansion = OfflineExpansion(self.rollouts, self.env, self.policy)

    def learn(self, total_timesteps, callback=None):
        if self.warmup_dataset is not None:
            self.il_warmup(self.warmup_dataset)
        episode_rewards = deque(maxlen=1000)
        ep_infos = deque(maxlen=1000)
        obs = self.env.reset()
        # initial_states = self.env.env_method("get_state")
        # goal_dim = self.env.get_attr("goal")[0].shape[0]
        # if self.reduction_strategy == "simultaneous":

        self.rollouts.obs[0].copy_(obs)
        self.rollouts.to(self.device)
        self.num_timesteps = 0
        # loss_names = ["value_loss", "policy_loss", "entropy", "grad_norm", "param_norm",
        #               "il_loss", "predictor_loss", "sa_predictor_loss"]

        start = time.time()
        num_updates = int(total_timesteps) // self.n_steps // self.n_envs

        for j in range(num_updates):
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
                obs, reward, done, infos = self.env.step(action)
                self.num_timesteps += self.n_envs

                for e_idx, info in enumerate(infos):
                    maybe_ep_info = info.get('episode')
                    if maybe_ep_info is not None:
                        ep_infos.append(maybe_ep_info)
                        episode_rewards.append(maybe_ep_info['r'])

                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                     for info in infos])
                self.rollouts.insert(obs, recurrent_hidden_states, action,
                                     action_log_prob, value, reward, masks, bad_masks)

            with torch.no_grad():
                next_value = self.policy.get_value(
                    self.rollouts.obs[-1], self.rollouts.recurrent_hidden_states[-1],
                    self.rollouts.masks[-1]).detach()

            self.rollouts.compute_returns(next_value, self.use_gae, self.gamma, self.lam)

            losses = self.update()

            self.rollouts.after_update()

            fps = int(self.num_timesteps / (time.time() - start))
            logger.logkv("serial_timesteps", j * self.n_steps)
            logger.logkv("n_updates", j)
            logger.logkv("total_timesteps", self.num_timesteps)
            logger.logkv("fps", fps)
            if len(ep_infos) > 0 and len(ep_infos[0]) > 0:
                logger.logkv('ep_reward_mean', safe_mean([ep_info['r'] for ep_info in ep_infos]))
                logger.logkv('ep_len_mean', safe_mean([ep_info['l'] for ep_info in ep_infos]))
                for key in ep_infos[0]:
                    if key not in ["r", "l", "t"]:
                        logger.logkv(key, safe_mean([ep_info[key] for ep_info in ep_infos]))
                # if "is_success" in ep_infos[0]:
                #     logger.logkv('success_rate', safe_mean([ep_info['is_success'] for ep_info in ep_infos]))
            logger.logkv('time_elapsed', time.time() - start)
            for loss_name in losses.keys():
                logger.logkv(loss_name, losses[loss_name])
            logger.dumpkvs()
            if self.use_wandb:
                log_data = dict()
                log_data["n_updates"] = j
                log_data["fps"] = fps
                if len(ep_infos) > 0 and len(ep_infos[0]) > 0:
                    log_data["ep_reward_mean"] = safe_mean([ep_info['r'] for ep_info in ep_infos])
                    log_data["ep_len_mean"] = safe_mean([ep_info['l'] for ep_info in ep_infos])
                    for key in ep_infos[0]:
                        if key not in ["r", "l", "t"]:
                            log_data[key] = safe_mean([ep_info[key] for ep_info in ep_infos])
                log_data["time_elapsed"] = time.time() - start
                for loss_name in losses.keys():
                    log_data[loss_name] = losses[loss_name]
                wandb.log(log_data, step=self.num_timesteps)
            if j % 20 == 0:
                im_dataset, new_tasks = self.expansion.expand()
                with open("im_dataset_%d.pkl" % (j // 20), "wb") as f:
                    pickle.dump(im_dataset, f)
                with open("generated_tasks_%d.pkl" % (j // 20), "wb") as f:
                    pickle.dump(new_tasks, f)
                task_per_env = new_tasks.shape[0] // self.n_envs if (
                    new_tasks.shape[0] % self.n_envs) == 0 else new_tasks.shape[0] // self.n_envs + 1
                for i in range(self.n_envs):
                    self.env.env_method("add_tasks", new_tasks[task_per_env * i: task_per_env * (i + 1)])
                self.il_warmup(im_dataset)

    def il_warmup(self, dataset, train_value=False):
        dataset["obs"] = torch.from_numpy(dataset["obs"]).float().to(self.device)
        dataset["action"] = torch.from_numpy(dataset["action"]).float().to(self.device)
        num_sample = dataset["obs"].shape[0]
        print("Num of samples", num_sample)
        indices = np.arange(num_sample)
        n_value_epoch = 5
        n_epoch = 15
        batch_size = 128
        eval_interval = 10
        losses = dict(policy_loss=deque(maxlen=50), grad_norm=deque(maxlen=50), 
                      param_norm=deque(maxlen=50), value_loss=deque(maxlen=50),
                      action_param_error=deque(maxlen=50), action_type_error=deque(maxlen=50))
        optimizer = optim.Adam([p[1] for p in self.policy.named_parameters() if not "log_std" in p[0]], lr=1e-3, weight_decay=0.0)
        self.save(os.path.join(logger.get_dir(), "model_init.pt"))
        
        if train_value:
            target_returns = torch.from_numpy(self.compute_discounted_reward(dataset["boundary"] + [dataset["obs"].shape[0]])).unsqueeze(dim=-1).float().to(self.device)
            # TODO: get negative samples? cross sample the goals
            original_goal_img = dataset["terminate_obs"][:, 768 + 14: 768 * 2 + 14]
            original_goal_info = torch.cat([dataset["terminate_obs"][:, -7: -6], dataset["terminate"][:, -3:]], dim=-1)
            # shuffled_idx = 
            
            for i in range(n_value_epoch):
                np.random.shuffle(indices)
                for j in range(num_sample // batch_size):
                    mb_idx = indices[batch_size * j: batch_size * (j + 1)]
                    obs_batch = dataset["obs"][mb_idx]
                    return_batch = target_returns[mb_idx]
                    recurrent_hidden_states_batch = torch.zeros((obs_batch.shape[0], 1))
                    masks_batch = torch.ones((obs_batch.shape[0], 1))
                    pred_value = self.policy.get_value(obs_batch, recurrent_hidden_states_batch, masks_batch)
                    loss = torch.mean((pred_value - return_batch) ** 2)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses["value_loss"].append(loss.item())
                print("Value epoch", i, "Value loss", np.mean(losses["value_loss"]))
                
        from utils.evaluation import evaluate_fixed_states
        success_episodes, total_episodes = evaluate_fixed_states(self.env, self.policy, self.device, None, None, 200, deterministic=True)
        print("Initial success episode: drawer %d / %d, object %d / %d" % (success_episodes[0], total_episodes[0], success_episodes[1], total_episodes[1]))
        for i in range(n_epoch):
            np.random.shuffle(indices)
            for j in range(num_sample // batch_size):
                mb_idx = indices[batch_size * j: batch_size * (j + 1)]
                obs_batch = dataset["obs"][mb_idx]
                actions_batch = dataset["action"][mb_idx]
                recurrent_hidden_states_batch = torch.zeros((obs_batch.shape[0], 1))
                masks_batch = torch.ones((obs_batch.shape[0], 1))
                # action_log_probs, dist_entropy, _ = self.policy.evaluate_actions(
                #     obs_batch, recurrent_hidden_states_batch, masks_batch,
                #     actions_batch)
                # loss = -action_log_probs.mean()
                loss = self.policy.get_bc_loss(obs_batch, recurrent_hidden_states_batch, masks_batch, actions_batch)
                optimizer.zero_grad()
                loss.backward()
                grads = [p.grad for p in self.policy.parameters() if p.grad is not None]
                device = grads[0].device
                total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2).to(device) for g in grads]), 2)
                optimizer.step()
                with torch.no_grad():
                    _, pred_actions, pred_logprob, _ = self.policy.act(obs_batch, deterministic=True)
                    action_param_error = torch.norm(pred_actions[:, 1:] - actions_batch[:, 1:], 1, dim=-1).mean()
                    action_type_error = torch.norm(pred_actions[:, 0: 1] - actions_batch[:, 0: 1], 0, dim=-1).mean()
                    if j == 0:
                        print("pred action", pred_actions[0], "gt action", actions_batch[0])
                losses["action_param_error"].append(action_param_error.item())
                losses["action_type_error"].append(action_type_error.item())
                losses["param_norm"].append(
                    torch.norm(torch.stack([torch.norm(p.detach()) for p in self.policy.parameters()])).item()
                )
                # for name, p in self.policy.named_parameters():
                #     if not name in losses:
                #         losses[name] = deque(maxlen=50)
                #     losses[name].append(torch.norm(p.detach()).item())
                losses["policy_loss"].append(loss.item())
                losses["grad_norm"].append(total_norm.item())
            if i % eval_interval == 0 or i == n_epoch - 1:
                success_episodes, total_episodes = evaluate_fixed_states(self.env, self.policy, self.device, None, None, 200, deterministic=True)
                print("Success episode: drawer %d / %d, object %d / %d" % (success_episodes[0], total_episodes[0], success_episodes[1], total_episodes[1]))
                self.save(os.path.join(logger.get_dir(), "model_init%d.pt" % i))
            logger.logkv("epoch", i)
            for k in losses:
                # print(k, np.mean(losses[k]))
                logger.logkv(k, np.mean(losses[k]))
            logger.dump_tabular()

    def compute_discounted_reward(self, boundary: list):
        returns = []
        for i in range(1, len(boundary)):
            ep_return = np.power(self.gamma, np.arange(boundary[i] - boundary[i - 1] - 1, -1, -1))
            returns.append(ep_return)
        returns = np.concatenate(returns)
        assert len(returns) == boundary[-1]
        return returns
    
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
        save_dict = {'policy': self.policy.state_dict(),
                     'optimizer': self.optimizer.state_dict()}
        if hasattr(self.env, "obs_rms"):
            save_dict["obs_rms"] = self.env.obs_rms
        if hasattr(self.env, "ret_rms"):
            save_dict["ret_rms"] = self.env.ret_rms
        torch.save(save_dict, save_path)

    def load(self, load_pth, eval=True):
        checkpoint = torch.load(load_pth, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'], strict=False)
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            pass
        if hasattr(self.env, "obs_rms"):
            self.env.venv.obs_rms.mean = checkpoint["obs_rms"].mean
            self.env.venv.obs_rms.var = checkpoint["obs_rms"].var
        # if hasattr(self.env, "ret_rms"):
        #     self.env.venv.set_attr("ret_rms", checkpoint["ret_rms"])
        #     print(self.env.ret_rms)
        if eval:
            self.policy.eval()
        else:
            self.policy.train()


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
