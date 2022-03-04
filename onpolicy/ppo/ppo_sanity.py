import time
import torch
import torch.nn as nn
import torch.optim as optim
from utils import logger
from vec_env.base_vec_env import VecEnv
from onpolicy.storage import RolloutStorage, compute_gae_return_multi
from collections import deque
import numpy as np
from typing import List, Dict
import pickle


class PPO(object):
    def __init__(self, env, policy: nn.Module, device="cpu", n_steps=1024, nminibatches=32,
                 noptepochs=10, gamma=0.99, lam=0.95, learning_rate=2.5e-4, cliprange=0.2, ent_coef=0.01, vf_coef=0.5,
                 max_grad_norm=0.5, eps=1e-5, use_gae=True, use_clipped_value_loss=True, use_linear_lr_decay=False,
                 il_coef=1, dataset=None, task_reduction=False, reduction_strategy="incremental", bc_ent_coef=0,
                 keep_success_ratio=0.1, pred_rew_coef=0.5,
                 il_weighted=False, sil=False, relabel=False, data_interval=10, use_rl=True, eval_env=None,
                 off_value_coef=0):
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
        self.il_coef = il_coef
        self.dataset = dataset
        self.bc_ent_coef = bc_ent_coef
        self.off_value_coef = off_value_coef

        if isinstance(self.env, VecEnv):
            self.n_envs = self.env.num_envs
        else:
            self.n_envs = 1

        self.rollouts = RolloutStorage(self.n_steps, self.n_envs,
                                       self.env.observation_space.shape, self.env.action_space,
                                       self.policy.recurrent_hidden_state_size)

        # self.success_predictor = success_predictor
        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate, eps=eps)
        # if success_predictor is not None:
        #     self.predictor_optimizer = optim.Adam(success_predictor.parameters(), lr=learning_rate, eps=eps)
        # else:
        #     self.predictor_optimizer = None

        self.env_id = env_id = self.env.get_attr("spec")[0].id
        self.data_strategy = reduction_strategy
        self.task_reduction = task_reduction
        self.keep_success_ratio = keep_success_ratio
        self.data_interval = data_interval
        self.sil = sil
        self.relabel = relabel

        if self.data_strategy == None:
            self.auxiliary_rollouts = None
            self.offline_datasets = None
        elif self.data_strategy == "simultaneous":
            iters_to_cache = 1
            if env_id == "BulletStack-v1":
                iters_to_cache = 10
            elif env_id.startswith("AntMaze"):
                iters_to_cache = 1
            elif env_id.startswith("PointMaze"):
                iters_to_cache = 1
            self.auxiliary_rollouts = {
                key: dict(obs=deque(maxlen=iters_to_cache), actions=deque(maxlen=iters_to_cache),
                          last_step_masks=deque(maxlen=iters_to_cache), reward=deque(maxlen=iters_to_cache))
                for key in ["self", "reduction"]
            }
            self.offline_datasets = None
        else:
            self.auxiliary_rollouts = None
            if self.data_strategy == "fixed_interval":
                self.offline_datasets = {key: deque(maxlen=1) for key in ["self", "reduction"]}
            else:
                self.offline_datasets = {key: deque(maxlen=self.data_interval) for key in ["self", "reduction"]}

        self.eval_env = eval_env
        self.log_restart_states = deque(maxlen=100)

        self.pred_rew_coef = pred_rew_coef
        self.il_weighted = il_weighted

        self.use_rl = use_rl
        if not use_rl:
            assert (not il_weighted) and (not task_reduction) and self.data_strategy != "simultaneous"

        self.log_prob_clip_min = -20
        self.log_prob_clip_max = 0
        if self.env_id == "BulletStack-v1":
            self.log_prob_clip_min = -self.env.action_space.shape[0] * 3.5
        if self.env_id.startswith("AntMaze"):
            self.log_prob_clip_min = -10
            self.log_prob_clip_max = np.inf

    def load_dataset(self, dataset):
        self.dataset = dataset

    def learn(self, total_timesteps, callback=None):
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
            '''
            if j == 0:
                load_demo_train()
            if j == 2:
                exit()
            '''
            # '''
            # curriculum on current_n_to_stack
            proceed_cl = False
            success_rate = safe_mean([ep_info["r"] for ep_info in ep_infos])

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

            if self.use_rl:
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
            if self.use_rl:
                for loss_name in losses.keys():
                    logger.logkv(loss_name, losses[loss_name])
            logger.dumpkvs()

    def update(self):
        advantages = self.rollouts.returns[:-1] - self.rollouts.value_preds[:-1]
        adv_mean, adv_std = advantages.mean(), advantages.std()
        advantages = (advantages - adv_mean) / (adv_std + 1e-5)

        losses = dict(value_loss=[], policy_loss=[], entropy=[], grad_norm=[], param_norm=[],
                      sa_predictor_loss=[], neg_il_loss=[])

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
                # if self.success_predictor is not None:
                #     success_prediction = self.success_predictor(obs_batch, actions_batch, recurrent_hidden_states_batch, masks_batch)
                #     prediction_loss = nn.functional.binary_cross_entropy(success_prediction, episode_success_batch)
                #     # add some loss to ensure temporal consistency?
                #     # current prediction is lagged behind. only when failure happens, the success rate goes down
                #     # prediction_loss += 0.5 * nn.functional.mse_loss(success_prediction[:-1], success_prediction[1:].detach())
                #     self.predictor_optimizer.zero_grad()
                #     (prediction_loss * self.vf_coef).backward()
                #     self.predictor_optimizer.step()
                # else:
                #     prediction_loss = torch.FloatTensor([np.nan]).to(self.device)

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

        for key in losses:
            losses[key] = safe_mean(losses[key])

        return losses

    def train_from_dataset(self, dataset: List[Dict], failed_dataset: List[Dict] = [], batch_size=64, n_epoch=30,
                           positive_ratio=1.):
        # positive_ratio: the ratio of successful reduction steps among all reduction steps
        goal_dim = self.env.get_attr("goal")[0].shape[0]
        reduction_obs, reduction_action = [], []
        origin_obs, origin_action = [], []
        last_step_masks = []
        train_initial_states = []
        train_goals = []
        train_rewards = []
        for traj in dataset:
            if traj["tag"] == "reduction":
                reduction_obs.append(traj["obs"])
                reduction_action.append(traj["actions"])
            else:
                origin_obs.append(traj["obs"])
                origin_action.append(traj["actions"])
            last_step_mask = np.ones(len(traj["obs"]))
            last_step_mask[-1] = 0
            last_step_masks.append(last_step_mask)
            train_reward = np.zeros(len(traj["obs"]))
            train_reward[-1] = 1
            train_rewards.append(train_reward)
            # train_traj_start_idx.append(train_traj_start_idx[-1] + len(traj["obs"]))
            if "states" in traj:
                train_initial_states.append(traj["states"][0])
                train_goals.append(traj["obs"][0][-goal_dim:])
        # reduction_obs = np.concatenate(reduction_obs, axis=0)
        # reduction_action = np.concatenate(reduction_action, axis=0)
        # origin_obs = np.concatenate(origin_obs, axis=0)
        # origin_action = np.concatenate(origin_action, axis=0)
        train_obs = np.concatenate(reduction_obs + origin_obs, axis=0)
        train_action = np.concatenate(reduction_action + origin_action, axis=0)
        last_step_masks = np.concatenate(last_step_masks, axis=0)
        train_rewards = np.concatenate(train_rewards, axis=0)
        if len(reduction_obs):
            n_reduction = np.sum([len(traj) for traj in reduction_obs])
        else:
            n_reduction = 0
        if len(origin_obs):
            n_original = np.sum([len(traj) for traj in origin_obs])
        else:
            n_original = 0
        logger.log("n reduction:", n_reduction, "n original:", n_original)
        # negative data, for fitting success predictor only
        # try to train sa predictor
        # logger.log("n train obs: ", train_obs.shape)
        il_weights = compute_il_weight(train_obs, train_action, last_step_masks, self.device, None,
                                       train_rewards, self.policy, self.gamma, self.lam, self.il_weighted)

        if self.use_rl:
            cur_values = compute_value(train_obs, self.device, self.policy)
            gae_return = compute_gae_return_multi(
                train_rewards, cur_values, np.concatenate([[0.], last_step_masks], axis=0), self.gamma, self.lam)
        losses = deque(maxlen=500)
        gradient_norms = deque(maxlen=500)

        negative_obs, negative_action, negative_pred_diff, negative_states = [], [], [], []
        # negative_obs, negative_action, negative_pred_diff, negative_states = load_negative_data()
        # logger.log("N negative data:", negative_obs.shape[0])

        for epoch in range(n_epoch):
            indices = np.arange(train_obs.shape[0])
            np.random.shuffle(indices)
            for m in range(len(indices) // batch_size):
                mb_indices = indices[m * batch_size: (m + 1) * batch_size]
                mb_obs = torch.from_numpy(train_obs[mb_indices]).float().to(self.device)
                mb_actions = torch.from_numpy(train_action[mb_indices]).to(self.device)
                mb_tags = mb_indices < n_reduction
                mb_weights = torch.from_numpy(il_weights[mb_indices]).to(self.device)

                # add ctrs data
                # if self.success_predictor is not None and len(contrastive_obs):
                #     ctrs_idx = np.random.randint(0, len(contrastive_obs), size=n_ctrs_traj_per_batch)
                #     mb_ctrs_obs = torch.from_numpy(np.concatenate([contrastive_obs[_idx] for _idx in ctrs_idx], axis=0)).float().to(self.device)
                #     mb_ctrs_action = torch.from_numpy(np.concatenate([contrastive_action[_idx] for _idx in ctrs_idx], axis=0)).to(self.device)
                #     mb_ctrs_weights = torch.from_numpy(np.concatenate([contrastive_weight[_idx] for _idx in ctrs_idx])).to(self.device)
                #     if epoch == 0 and m == 0:
                #         logger.log("ctrs size", mb_ctrs_obs.shape, mb_ctrs_action.shape, mb_ctrs_weights.shape)
                #         logger.log("original size", mb_obs.shape, mb_actions.shape, mb_weights.shape)
                #     mb_obs = torch.cat([mb_obs, mb_ctrs_obs], dim=0)
                #     mb_actions = torch.cat([mb_actions, mb_ctrs_action], dim=0)
                #     mb_weights = torch.cat([mb_weights, mb_ctrs_weights], dim=0)
                action_log_probs, dist_entropy, _ = self.policy.evaluate_actions(
                    mb_obs, None, None, mb_actions)
                action_log_probs = torch.clamp(action_log_probs, min=self.log_prob_clip_min, max=self.log_prob_clip_max)
                if self.use_rl:
                    mb_gae_returns = torch.from_numpy(gae_return[mb_indices]).float().to(self.device)
                    values = self.policy.get_value(mb_obs)
                    value_loss = 0.5 * torch.mean((values - mb_gae_returns.unsqueeze(dim=-1)) ** 2)
                else:
                    value_loss = torch.FloatTensor([0.0])
                # add negative data
                loss = -(mb_weights.unsqueeze(dim=-1) * action_log_probs).mean() \
                       - self.bc_ent_coef * dist_entropy.mean() + self.off_value_coef * value_loss
                self.optimizer.zero_grad()
                loss.backward()
                total_norm = get_grad_norm(self.policy.parameters())
                self.optimizer.step()
                losses.append(loss.item())
                gradient_norms.append(total_norm.item())
                # if self.use_contrastive_offline:
                #     ctrs_pos_losses.append(-(mb_ctrs_weights.unsqueeze(dim=-1) * action_log_probs[mb_indices.shape[0]:])[torch.where(mb_ctrs_weights > 0)[0]].mean().item())
                #     ctrs_neg_losses.append(-(mb_ctrs_weights.unsqueeze(dim=-1) * action_log_probs[mb_indices.shape[0]:])[torch.where(mb_ctrs_weights < 0)[0]].mean().item())

                if (epoch * (len(indices) // batch_size) + m) % 1000 == 0:
                    # if self.success_predictor is not None and len(contrastive_obs):
                    #     with torch.no_grad():
                    #         debug_ctrs_obs = torch.from_numpy(contrastive_obs[0]).float().to(self.device)
                    #         debug_ctrs_actions = torch.from_numpy(contrastive_action[0]).to(self.device)
                    #         debug_ctrs_weights = contrastive_weight[0]
                    #         _, debug_actions, _, _ = self.policy.act(debug_ctrs_obs)
                    #         debug_log_probs, _, _ = self.policy.evaluate_actions(
                    #             debug_ctrs_obs, None, None, debug_ctrs_actions
                    #         )
                    #     logger.log("debug action", debug_actions, "debug log prob", debug_log_probs,
                    #                "ctrs action", debug_ctrs_actions, "ctrs weight", debug_ctrs_weights)
                    logger.log("epoch %d" % epoch, "m %d" % m, "loss", np.mean(losses),
                               "gradient norm", np.mean(gradient_norms),
                               )

    def save(self, save_path):
        save_dict = {'policy': self.policy.state_dict(),
                     'optimizer': self.optimizer.state_dict()}
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


def get_grad_norm(parameters, norm_type: float = 2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    from torch._six import inf
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                                norm_type)
    return total_norm


def get_last_step_mask(l):
    arr = np.ones(l)
    arr[-1] = 0
    return arr


def compute_il_weight(train_obs, train_action, last_step_masks, device, sa_predictor=None,
                      train_rewards=None, policy=None, gamma=0.99, gae_lambda=0.95, weighted=False):
    il_weights = np.ones(train_obs.shape[0])
    if sa_predictor is not None:
        assert isinstance(sa_predictor, list) and isinstance(sa_predictor[0], nn.Module)
        predictions = []
        chunk_size = 1024
        n_chunk = train_obs.shape[0] // chunk_size if train_obs.shape[0] % chunk_size == 0 else train_obs.shape[
                                                                                                    0] // chunk_size + 1
        with torch.no_grad():
            for chunk_idx in range(n_chunk):
                obs_batch = train_obs[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size]
                if isinstance(obs_batch, np.ndarray):
                    obs_batch = torch.from_numpy(obs_batch).float().to(device)
                action_batch = train_action[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size]
                if isinstance(action_batch, np.ndarray):
                    action_batch = torch.from_numpy(action_batch).float().to(device)
                prediction = get_success_prediction(
                    sa_predictor, obs_batch, action_batch, None, None).cpu().numpy().squeeze(axis=-1)
                predictions.append(prediction)
        predictions = np.concatenate(predictions, axis=0)
        diffs = np.concatenate([predictions[1:] - predictions[:-1], [0.]])
        diffs = diffs * last_step_masks
        assert len(diffs) == len(train_obs)
        il_weights = np.exp(diffs)
    elif weighted and policy is not None:
        values = compute_value(train_obs, device, policy)
        masks = np.concatenate([[0.], last_step_masks], axis=0)
        gae_returns = compute_gae_return_multi(train_rewards, values, masks, gamma, gae_lambda)
        il_weights = np.exp(gae_returns - values)
        # il_weights = np.maximum(gae_returns - values, 0)
        assert len(il_weights.shape) == 1
    return il_weights


def compute_value(train_obs, device, policy):
    values = []
    chunk_size = 1024
    n_chunk = train_obs.shape[0] // chunk_size if train_obs.shape[0] % chunk_size == 0 else train_obs.shape[
                                                                                                0] // chunk_size + 1
    with torch.no_grad():
        for chunk_idx in range(n_chunk):
            obs_batch = train_obs[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size]
            if isinstance(obs_batch, np.ndarray):
                obs_batch = torch.from_numpy(obs_batch).float().to(device)
            value = policy.get_value(obs_batch).cpu().numpy().squeeze(axis=-1)
            values.append(value)
    values = np.concatenate(values, axis=0)
    return values


def load_negative_data():
    negative_obs, negative_action, negative_pred_diff, negative_states = [], [], [], []
    with open("debug_negative_data.pkl", "rb") as f:
        for _ in range(10000):
            try:
                data = pickle.load(f)
            except EOFError:
                break
            preds = np.array(data["predictions"]).squeeze(axis=-1)
            pred_diff = preds[1:] - preds[:-1]
            bad_idx = np.where(pred_diff < 0)[0]
            bad_obs, bad_action = np.array(data["obs"])[bad_idx], np.array(data["actions"])[bad_idx]
            negative_obs.append(bad_obs)
            negative_action.append(bad_action)
            negative_pred_diff.append(pred_diff[bad_idx])
            negative_states.extend([data["states"][i] for i in bad_idx])
    if len(negative_obs):
        negative_obs = np.concatenate(negative_obs, axis=0)
        negative_action = np.concatenate(negative_action, axis=0)
        negative_pred_diff = np.concatenate(negative_pred_diff, axis=0)
    return negative_obs, negative_action, negative_pred_diff, negative_states


def get_positive_data(negative_states, negative_obs, policy, env, sa_predictor, device):
    positive_actions = []
    goal_dim = None
    _count = 0
    for idx, state in enumerate(negative_states):
        env.reset()
        if goal_dim is None:
            goal_dim = env.goal.shape[0]
        env.set_state(state)
        env.set_goals([negative_obs[idx][-goal_dim:].cpu().numpy()])
        env.sync_attr()
        obs = env.get_obs()
        obs = torch.from_numpy(
            np.concatenate([obs[key] for key in ['observation', 'achieved_goal', 'desired_goal']])).float().to(device)
        with torch.no_grad():
            _, action, _, _ = policy.act(obs.unsqueeze(dim=0), deterministic=True)
        next_obs, _, _, _ = env.step(action[0].cpu().numpy())
        next_obs = torch.from_numpy(next_obs).float().to(device)
        total_obs = torch.stack([obs, next_obs], dim=0)
        predictions = get_success_prediction(sa_predictor, total_obs, None, with_action=False)
        if predictions[1] > predictions[0]:
            positive_actions.append(action[0].cpu().numpy())
            _count += 1
        else:
            positive_actions.append(None)
    logger.log("match count", _count, "negative count", negative_obs.shape[0])
    return positive_actions


def load_contrastive_data(n_total_ctrs_traj, device, policy, success_predictor=None):
    ctrs_neg_seg_len = 3
    adapt_seg_idx = True
    start_traj = 30001
    end_traj = start_traj + n_total_ctrs_traj
    contrastive_obs, contrastive_action, contrastive_weight = [], [], []
    with open("contrastive_buffer.pkl", "rb") as f:
        try:
            for _ in range(start_traj):
                pickle.load(f)
            for tj_idx in range(end_traj - start_traj):
                pair = pickle.load(f)
                pos = 0 if pair[0]["is_success"] else 1
                neg = 0 if pair[1]["is_success"] else 1
                if adapt_seg_idx:
                    with torch.no_grad():
                        _neg_predictions = success_predictor(
                            torch.from_numpy(pair[neg]["obs"]).float().to(device)).cpu().numpy()
                    _prediction_drop = (_neg_predictions[:-1] - _neg_predictions[1:]).squeeze()
                    if np.max(_prediction_drop) > 0.4:
                        # drop_idx = np.where(_prediction_drop > 0.2)[0][0]
                        drop_idx = np.argmax(_prediction_drop)
                    else:
                        # drop_idx = np.argmax(_prediction_drop)
                        continue
                    with torch.no_grad():
                        features = policy.get_feature(torch.from_numpy(
                            np.concatenate(
                                [pair[pos]["obs"], pair[neg]["obs"][drop_idx: drop_idx + 1]], axis=0)
                        ).float().to(device), net="value").cpu().numpy()
                    feature_dist = np.linalg.norm(features[:-1] - features[-1:], axis=-1)
                    pos_idx = np.argmin(feature_dist)
                else:
                    pos_idx, drop_idx = 0, 0
                contrastive_obs.append(
                    np.concatenate(
                        [pair[pos]["obs"][pos_idx:],
                         pair[neg]["obs"][drop_idx: drop_idx + ctrs_neg_seg_len]], axis=0)
                )
                # train_traj_start_idx.append(train_traj_start_idx[-1] + len(contrastive_obs[-1]))
                with torch.no_grad():
                    _pos_predictions = success_predictor(
                        torch.from_numpy(pair[pos]["obs"][pos_idx:]).float().to(device)).cpu().numpy()
                pos_weight = np.exp(np.concatenate([_pos_predictions[1:] - _pos_predictions[:-1], [0.]]))
                contrastive_weight.append(
                    np.concatenate(
                        [pos_weight, -np.ones(
                            pair[neg]["obs"][drop_idx: drop_idx + ctrs_neg_seg_len].shape[0]
                        )], axis=0)
                )
                # contrastive_obs.append(pair[neg]["obs"][drop_idx: drop_idx + 10])
                # train_traj_start_idx.append(train_traj_start_idx[-1] + len(contrastive_obs[-1]))
                # contrastive_weight.append(-np.ones(contrastive_obs[-1].shape[0]))
                contrastive_action.append(
                    np.concatenate(
                        [pair[pos]["actions"][pos_idx:],
                         pair[neg]["actions"][drop_idx: drop_idx + ctrs_neg_seg_len]], axis=0)
                )
                # contrastive_action.append(pair[neg]["actions"][drop_idx: drop_idx + 10])
        except EOFError:
            pass
    return contrastive_obs, contrastive_action, contrastive_weight


def get_success_prediction(sa_predictor: list, obs: torch.Tensor, action,
                           recurrent_hidden_states=None, masks=None, with_action=True):
    success_predictions = []
    with torch.no_grad():
        for m_idx in range(len(sa_predictor)):
            if with_action:
                success_predictions.append(sa_predictor[m_idx](
                    obs, action, recurrent_hidden_states, masks)[0])
            else:
                success_predictions.append(sa_predictor[m_idx].predict_without_action(
                    obs, recurrent_hidden_states, masks))
        success_prediction = torch.min(torch.stack(success_predictions, dim=0), dim=0)[0]
    return success_prediction
