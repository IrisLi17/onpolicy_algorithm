import copy
import time, os
import torch
import torch.nn as nn
import torch.optim as optim
from utils import logger
from vec_env.base_vec_env import VecEnv
from onpolicy.storage import RolloutStorage, compute_gae_return_multi
from policies.base import ActorCriticPolicy
from collections import deque
import numpy as np
from typing import List, Dict
from utils.evaluation import evaluate_fixed_states
import pickle


class PAIR(object):
    def __init__(self, env, policy: ActorCriticPolicy, reduction_env=None, device="cpu", n_steps=1024, nminibatches=32,
                 noptepochs=10, gamma=0.99, lam=0.95, learning_rate=2.5e-4, cliprange=0.2, ent_coef=0.01, vf_coef=0.5,
                 max_grad_norm=0.5, eps=1e-5, use_gae=True, use_clipped_value_loss=True, use_linear_lr_decay=False,
                 il_coef=1, task_reduction=False, reduction_strategy="fixed_interval", bc_ent_coef=0,
                 keep_success_ratio=0.1, pred_rew_coef=0.5, il_weighted=True, sil=False, relabel=False,
                 data_interval=10, eval_env=None, off_value_coef=0, tr_kwargs={}, log_prob_clip_min=-np.inf,
                 log_prob_clip_max=np.inf):
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

        self.env_id = self.env.get_attr("spec")[0].id
        self.data_strategy = reduction_strategy
        self.task_reduction = task_reduction
        _reduction_env = reduction_env if reduction_env is not None else env
        self.keep_success_ratio = keep_success_ratio
        if self.task_reduction:
            from onpolicy.ppo.task_reduction import TaskReduction
            self.task_reducer = TaskReduction(
                _reduction_env, self.policy, keep_success_ratio=keep_success_ratio,
                env_kwargs=tr_kwargs
            )
        self.data_interval = data_interval
        self.sil = sil
        self.relabel = relabel

        if self.data_strategy is None:
            self.offline_datasets = None
        else:
            if self.data_strategy == "fixed_interval":
                self.offline_datasets = {key: deque(maxlen=1) for key in ["self", "reduction"]}
            else:
                self.offline_datasets = {key: deque(maxlen=self.data_interval) for key in ["self", "reduction"]}

        self.eval_env = eval_env

        self.pred_rew_coef = pred_rew_coef
        self.il_weighted = il_weighted

        self.log_prob_clip_min = log_prob_clip_min
        self.log_prob_clip_max = log_prob_clip_max
        self._check_env()

    def _check_env(self):
        if self.relabel or self.task_reduction:
            goal = self.env.get_attr("goal")
            if self.task_reduction:
                # todo: incomplete check list of required api
                state = self.env.env_method("get_state")
                self.env.env_method("set_state", state[0])
                self.env.env_method("get_obs")
            self.env.reset()

    def learn(self, total_timesteps, callback=None):
        episode_rewards = deque(maxlen=1000)
        ep_infos = deque(maxlen=1000)
        if self.env_id == "BulletStack-v1":
            detailed_stats = [deque(maxlen=500) for _ in range(self.env.get_attr("n_object")[0])]
            current_n_to_stack = 1
            if len(self.env.get_attr("n_to_stack_choices")[0]) > 1:
                self.env.env_method("set_choice_prob", [current_n_to_stack], [0.7])
        obs = self.env.reset()
        goal_dim = self.env.get_attr("goal")[0].shape[0]
        # if self.reduction_strategy == "simultaneous":
        buffer = [dict(obs=[], states=[], actions=[], values=[], rewards=[], log_probs=[]) for _ in
                  range(self.env.num_envs)]
        reduction_initial_states, reduction_initial_obs, reduction_goal_seqs = [], [], []
        success_data_per_update = []

        self.rollouts.obs[0].copy_(obs)
        self.rollouts.to(self.device)
        last_reduce_time = 0
        if self.env_id == "SawyerPush-v1":
            last_reduce_time = 0
        if self.env_id == "AntMaze-v1":
            last_reduce_time = 0
        if self.env_id == "BulletStack-v1":
            last_reduce_time = np.inf
        start_reduction = False

        def start_reduction_condition():
            if self.env_id == "BulletStack-v1":
                return current_n_to_stack >= 2
            return True

        self.num_timesteps = 0

        start = time.time()
        num_updates = int(total_timesteps) // self.n_steps // self.n_envs

        for j in range(num_updates):
            # curriculum on current_n_to_stack
            # curriculum_callback(locals(), globals())
            if self.env_id == "BulletStack-v1":
                success_rate = safe_mean([ep_info["r"] for ep_info in ep_infos])
                threshold = min(0.8, 1 - 1 / (self.env.get_attr("n_object")[0] + 1 - current_n_to_stack))
                if success_rate > 0.6 \
                        and np.mean(detailed_stats[current_n_to_stack - 1]) > threshold \
                        and ((current_n_to_stack == 1 and self.env.get_attr("cl_ratio")[0] == 0) or
                             1 < current_n_to_stack < self.env.get_attr("n_object")[0]):
                    current_n_to_stack += 1
                    self.env.env_method("set_choice_prob", [current_n_to_stack], [0.7])
            # in some condition, try task reduction, and imitation
            if self.task_reduction and self.data_strategy == "fixed_interval":
                if (not start_reduction) and start_reduction_condition():
                    start_reduction = True
                    last_reduce_time = j
                if start_reduction and j - last_reduce_time >= self.data_interval:
                    last_reduce_time = j
            # '''

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

                if self.task_reduction:
                    states = self.env.env_method("get_state")

                # Obser reward and next obs
                obs, reward, done, infos = self.env.step(action)
                self.num_timesteps += self.n_envs

                for e_idx in range(self.env.num_envs):
                    if self.task_reduction:
                        buffer[e_idx]["states"].append(states[e_idx])
                    buffer[e_idx]["obs"].append(self.rollouts.obs[step][e_idx].cpu().numpy())
                    buffer[e_idx]["actions"].append(action[e_idx].cpu().numpy())
                    buffer[e_idx]["log_probs"].append(action_log_prob[e_idx].cpu().numpy())
                    buffer[e_idx]["values"].append(value[e_idx].cpu().numpy())
                    buffer[e_idx]["rewards"].append(reward[e_idx].cpu().numpy())
                    if len(buffer[e_idx]["values"]) > 1:
                        self.rollouts.rewards[step - 1][e_idx] += self.pred_rew_coef * torch.from_numpy(
                            buffer[e_idx]["values"][-1] - buffer[e_idx]["values"][-2]).to(self.device)

                for e_idx, info in enumerate(infos):
                    maybe_ep_info = info.get('episode')
                    if maybe_ep_info is not None:
                        ep_infos.append(maybe_ep_info)
                        episode_rewards.append(maybe_ep_info['r'])
                        if self.env_id == "BulletStack-v1":
                            n_to_stack = info["n_to_stack"]
                            detailed_stats[n_to_stack - 1].append(info["is_success"])
                        if info["is_success"]:
                            # store successful data for SIL and partially for task reduction
                            if self.sil or self.task_reduction:
                                traj = dict(obs=np.stack(buffer[e_idx]["obs"], axis=0),
                                            actions=np.stack(buffer[e_idx]["actions"], axis=0),
                                            last_step_mask=get_last_step_mask(len(buffer[e_idx]["obs"])),
                                            reward=np.concatenate(buffer[e_idx]["rewards"]),
                                            tag="origin")
                                success_data_per_update.append(traj)
                        else:
                            # store reduction state, goal for task reduction, and optionally do goal relabeling
                            if self.task_reduction and start_reduction:
                                reduction_idx = 0
                                reduction_initial_states.append(buffer[e_idx]["states"][reduction_idx])
                                reduction_initial_obs.append(buffer[e_idx]["obs"][reduction_idx].copy())
                            # goal relabeling
                            if self.relabel:
                                relabel_obs = np.stack(buffer[e_idx]["obs"], axis=0)
                                relabel_idx = np.random.randint(0, relabel_obs.shape[0])
                                relabel_obs[:, -goal_dim:] = np.tile(relabel_obs[relabel_idx: relabel_idx + 1, -2 * goal_dim: -goal_dim], (relabel_obs.shape[0], 1))
                                artificial_reward = self.env.env_method(
                                    "compute_reward", relabel_obs[:, -2 * goal_dim: -goal_dim], relabel_obs[:, -goal_dim:], None, indices=0)[0]
                                seg_idx = np.where(artificial_reward >= artificial_reward[relabel_idx])[0][0]
                                if seg_idx > 0:
                                    traj = dict(obs=relabel_obs[:seg_idx + 1],
                                                actions=np.stack(buffer[e_idx]["actions"][:seg_idx + 1], axis=0),
                                                last_step_mask=get_last_step_mask(seg_idx + 1),
                                                reward=artificial_reward[:seg_idx + 1],
                                                tag="origin")
                                    success_data_per_update.append(traj)

                        buffer[e_idx] = dict(obs=[], states=[], actions=[], values=[], rewards=[], log_probs=[])

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

            # collect reduction task if any
            if len(reduction_initial_states) and \
                    (self.data_strategy != "fixed_interval" or
                     (self.data_strategy == "fixed_interval" and j - last_reduce_time == self.data_interval - 1)):
                assert self.task_reducer is not None
                _to_delete = []
                for r_idx in range(len(reduction_initial_states)):
                    subgoal = self.task_reducer.generate_subgoal(
                        reduction_initial_obs[r_idx], reduction_initial_obs[r_idx][-goal_dim:]
                    )
                    if subgoal is None:
                        _to_delete.append(r_idx)
                    else:
                        reduction_goal_seqs.append([subgoal, reduction_initial_obs[r_idx][-goal_dim:]])
                reduction_initial_states = [reduction_initial_states[idx] for idx in
                                            range(len(reduction_initial_states)) if idx not in _to_delete]
                logger.log("number of reduction trials", len(reduction_initial_states))
                reduction_dataset, n_interactions = self.task_reducer.rollout(
                    reduction_initial_states, reduction_goal_seqs)
                self.num_timesteps += n_interactions
                n_reduction_traj = len(reduction_dataset)
                reduction_initial_states, reduction_initial_obs, reduction_goal_seqs = [], [], []
                if n_reduction_traj > 0:
                    self.offline_datasets["reduction"].append(reduction_dataset)
            else:
                n_reduction_traj = None

            if len(success_data_per_update) and \
                    (self.data_strategy != "fixed_interval" or
                     (self.data_strategy == "fixed_interval" and j - last_reduce_time == self.data_interval - 1)):
                # Maximum number of trajectories
                _horizon = self.env.get_attr("spec")[0].max_episode_steps
                if _horizon is None:
                    _horizon = 100  # workaround for flexible timelimit stacking
                max_n_traj = int(500000 / _horizon / self.offline_datasets["self"].maxlen)

                if len(success_data_per_update) < max_n_traj:
                    self.offline_datasets["self"].append(success_data_per_update)
                else:
                    traj_idx = np.random.choice(np.arange(len(success_data_per_update)), max_n_traj)
                    self.offline_datasets["self"].append([success_data_per_update[_i] for _i in traj_idx])
                success_data_per_update = []

            if self.offline_datasets is not None and (self.data_strategy != "fixed_interval" or (self.data_strategy == "fixed_interval" and j - last_reduce_time == self.data_interval - 1)):
                dataset = []
                for d in self.offline_datasets["reduction"]:
                    dataset += d
                reduction_traj_count = len(dataset)
                if reduction_traj_count == 0:
                    _ratio = 1.0
                else:
                    _ratio = min(1.0, self.keep_success_ratio * reduction_traj_count / sum([len(d) for d in self.offline_datasets["self"]]))
                for d in self.offline_datasets["self"]:
                    dataset += [d[_i] for _i in np.random.choice(np.arange(len(d)),
                                                                 np.floor(len(d) * _ratio).astype(int))]
                self.train_from_dataset(dataset, n_epoch=self.noptepochs)

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
            if self.env_id == "BulletStack-v1":
                for i in range(len(detailed_stats)):
                    logger.logkv("eval_stack_%d" % (i + 1), safe_mean(detailed_stats[i]))
                logger.logkv("current_n_to_stack", current_n_to_stack)
            logger.logkv('time_elapsed', time.time() - start)
            for loss_name in losses.keys():
                logger.logkv(loss_name, losses[loss_name])
            logger.dumpkvs()

    def update(self):
        advantages = self.rollouts.returns[:-1] - self.rollouts.value_preds[:-1]
        adv_mean, adv_std = advantages.mean(), advantages.std()
        advantages = (advantages - adv_mean) / (adv_std + 1e-5)

        losses = dict(value_loss=[], policy_loss=[], entropy=[], grad_norm=[], param_norm=[],
                      clipped_ratio=[], il_loss=[], )

        il_indices, il_batch_size, il_total_obs, il_total_actions, il_weights = None, None, None, None, None

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

                if il_batch_size is not None:
                    if e == 0 and mb_idx == 0:
                        print("il batch size", il_batch_size, "original batch size", obs_batch.shape[0])
                    il_mb_indices = il_indices[il_batch_size * mb_idx: il_batch_size * (mb_idx + 1)]
                    il_obs_batch = il_total_obs[il_mb_indices]
                    il_actions_batch = il_total_actions[il_mb_indices]
                    il_weight_batch = il_weights[il_mb_indices]
                    il_log_probs, _, _ = self.policy.evaluate_actions(
                        il_obs_batch, None, None, il_actions_batch
                    )
                    il_log_probs = torch.clamp(il_log_probs, min=self.log_prob_clip_min, max=self.log_prob_clip_max)
                    il_loss = (-il_weight_batch * il_log_probs).mean()
                else:
                    il_loss = torch.FloatTensor([np.nan]).to(self.device)

                self.optimizer.zero_grad()
                (value_loss * self.vf_coef + action_loss -
                 dist_entropy * self.ent_coef + il_loss * self.il_coef).backward()
                total_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                # print('total norm', total_norm)
                self.optimizer.step()

                params = list(filter(lambda p: p[1].grad is not None, self.policy.named_parameters()))
                param_norm = torch.norm(
                    torch.stack([torch.norm(p[1].detach().to(self.device)) for p in params]))

                isnan = torch.tensor([torch.isnan(p[1].detach()).any() for p in self.policy.named_parameters()]).any()
                if isnan:
                    logger.log("value loss", value_loss, "action loss", action_loss, "dist entropy", dist_entropy,
                               "il loss", il_loss, "grad norm", total_norm)
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
                losses["il_loss"].append(il_loss.item())
                losses["clipped_ratio"].append(clipped_ratio)

        for key in losses:
            losses[key] = safe_mean(losses[key])

        return losses

    def train_from_dataset(self, dataset: List[Dict], batch_size=64, n_epoch=30):
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
            train_reward = traj["reward"]
            train_rewards.append(train_reward)
            # train_traj_start_idx.append(train_traj_start_idx[-1] + len(traj["obs"]))
            if "states" in traj:
                train_initial_states.append(traj["states"][0])
                train_goals.append(traj["obs"][0][-goal_dim:])
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
        il_weights = compute_il_weight(train_obs, last_step_masks, self.device, train_rewards, self.policy, self.gamma,
                                       self.lam, self.il_weighted)

        cur_values = compute_value(train_obs, self.device, self.policy)
        gae_return = compute_gae_return_multi(
            train_rewards, cur_values, np.concatenate([[0.], last_step_masks], axis=0), self.gamma, self.lam)
        losses = deque(maxlen=500)
        gradient_norms = deque(maxlen=500)

        for epoch in range(n_epoch):
            if epoch % 30 == 0:
                detail_eval_success, detail_eval_episode = evaluate_fixed_states(
                    self.eval_env, self.policy, self.device, None, None, 500, deterministic=False)
                logger.log("epoch %d " % epoch, detail_eval_success, "/", detail_eval_episode)
                if len(train_initial_states):
                    detail_train_success, detail_train_episode = evaluate_fixed_states(
                        self.env, self.policy, self.device, train_initial_states[:200], np.array(train_goals[:200]))
                    logger.log("train: epoch %d" % epoch, detail_train_success, "/", detail_train_episode)
            indices = np.arange(train_obs.shape[0])
            np.random.shuffle(indices)
            for m in range(len(indices) // batch_size):
                mb_indices = indices[m * batch_size: (m + 1) * batch_size]
                mb_obs = torch.from_numpy(train_obs[mb_indices]).float().to(self.device)
                mb_actions = torch.from_numpy(train_action[mb_indices]).to(self.device)
                mb_weights = torch.from_numpy(il_weights[mb_indices]).to(self.device)

                action_log_probs, dist_entropy, _ = self.policy.evaluate_actions(
                    mb_obs, None, None, mb_actions)
                action_log_probs = torch.clamp(action_log_probs, min=self.log_prob_clip_min, max=self.log_prob_clip_max)
                mb_gae_returns = torch.from_numpy(gae_return[mb_indices]).float().to(self.device)
                values = self.policy.get_value(mb_obs)
                value_loss = 0.5 * torch.mean((values - mb_gae_returns.unsqueeze(dim=-1)) ** 2)
                # add negative data
                loss = -(mb_weights.unsqueeze(dim=-1) * action_log_probs).mean() \
                       - self.bc_ent_coef * dist_entropy.mean() + self.off_value_coef * value_loss
                self.optimizer.zero_grad()
                loss.backward()
                total_norm = get_grad_norm(self.policy.parameters())
                self.optimizer.step()
                losses.append(loss.item())
                gradient_norms.append(total_norm.item())

                if (epoch * (len(indices) // batch_size) + m) % 1000 == 0:
                    logger.log("epoch %d" % epoch, "m %d" % m, "loss", np.mean(losses),
                               "gradient norm", np.mean(gradient_norms),
                               )
        detail_eval_success, detail_eval_episode = evaluate_fixed_states(
            self.eval_env, self.policy, self.device, None, None, 500, deterministic=False)
        logger.log("final", detail_eval_success, "/", detail_eval_episode)

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


def compute_il_weight(train_obs, last_step_masks, device, train_rewards=None, policy=None, gamma=0.99, gae_lambda=0.95,
                      weighted=False):
    il_weights = np.ones(train_obs.shape[0])
    if weighted and policy is not None:
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
