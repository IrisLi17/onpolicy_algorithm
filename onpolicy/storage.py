import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 recurrent_hidden_state_size, aux_shape=None):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.states = np.empty([num_steps + 1, num_processes], dtype=object)
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if isinstance(action_space, int):
            action_shape = action_space
        elif action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.episode_success = torch.zeros(num_steps + 1, num_processes, 1)

        self.success = torch.zeros(num_steps + 1, num_processes, 1)

        # Storage for auxiliary task
        self.aux = torch.zeros(num_steps, num_processes, *aux_shape) if aux_shape is not None else None

        # Step to go
        self.step_to_go = torch.zeros(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        self.success = self.success.to(device)
        self.episode_success = self.episode_success.to(device)
        self.step_to_go = self.step_to_go.to(device)
        if self.aux is not None:
            self.aux = self.aux.to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks, aux_info=None,
               state_dict=None, success_masks=None):
        self.obs[self.step + 1].copy_(obs)
        if state_dict is not None:
            if not isinstance(state_dict, np.ndarray):
                state_dict = np.asarray(state_dict)
            assert isinstance(state_dict[0], dict)
            self.states[self.step + 1] = state_dict
        self.recurrent_hidden_states[self.step +
                                     1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        if success_masks is not None:
            self.success[self.step + 1].copy_(success_masks)
        if self.aux is not None:
            self.aux[self.step].copy_(aux_info)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        # self.states[0] = self.states[-1]
        _shape = self.states.shape
        last_state = self.states[-1]
        del self.states
        self.states = np.empty(_shape, dtype=object)
        self.states[0] = last_state
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])
        self.success[0].copy_(self.success[-1])

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True):
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                        + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]
        for step in reversed(range(self.rewards.size(0))):
            self.episode_success[step] = \
                self.masks[step + 1] * self.episode_success[step + 1] + (1 - self.masks[step + 1]) * self.rewards[step]

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None,
                               with_aux=False):
        if with_aux:
            assert self.aux is not None
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            episode_success_batch = self.episode_success[:-1].view(-1, 1)[indices]
            next_indices = np.array([(ind + num_processes) % batch_size for ind in indices])
            next_masks = self.masks[:-1].view(-1, 1)[next_indices]
            next_masks[np.where(next_indices < num_processes)] = 0.
            next_obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[next_indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                  value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                  episode_success_batch, adv_targ, next_obs_batch, next_masks

    # TODO: implement recurrent version with aux info and value ensemble if needed
    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind])
                # recurrent_hidden_states_batch.append(
                #     self.recurrent_hidden_states[:-1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            # recurrent_hidden_states_batch = torch.stack(recurrent_hidden_states_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            # recurrent_hidden_states_batch = _flatten_helper(T, N, recurrent_hidden_states_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ


class NpRolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 recurrent_hidden_state_size, n_values=1, aux_shape=None, n_action_level=1):
        self.obs = np.zeros((num_steps + 1, num_processes, *obs_shape))
        self.states = np.empty([num_steps + 1, num_processes], dtype=object)
        self.recurrent_hidden_states = np.zeros(
            (num_steps + 1, num_processes, recurrent_hidden_state_size))
        self.rewards = np.zeros((num_steps, num_processes, 1))
        # TODO: value ensemble
        self.n_values = n_values
        self.value_preds = np.zeros((num_steps + 1, num_processes, n_values, 1))
        self.returns = np.zeros((num_steps + 1, num_processes, n_values, 1))
        self.n_action_level = n_action_level
        self.action_log_probs = np.zeros((num_steps, num_processes, 1, n_action_level))
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = np.zeros((num_steps, num_processes, action_shape))
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = np.ones((num_steps + 1, num_processes, 1))

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = np.ones((num_steps + 1, num_processes, 1))

        self.success = np.zeros((num_steps + 1, num_processes, 1))

        # Storage for auxiliary task
        self.aux = np.zeros((num_steps, num_processes, *aux_shape)) if aux_shape is not None else None

        # Is out_of_reach or not
        self.reset_actions = np.zeros((num_steps, num_processes, 1))

        # Step to go
        self.step_to_go = np.zeros((num_steps + 1, num_processes, 1))

        self.num_steps = num_steps
        self.step = 0
    
    def insert(self, obs: np.ndarray, recurrent_hidden_states: np.ndarray, actions: np.ndarray, action_log_probs: np.ndarray,
               value_preds, rewards, masks, bad_masks, aux_info=None,
               state_dict=None, success_masks=None, reset_action=None):
        self.obs[self.step + 1] = obs.copy()
        if state_dict is not None:
            if not isinstance(state_dict, np.ndarray):
                state_dict = np.asarray(state_dict)
            assert isinstance(state_dict[0], dict)
            self.states[self.step + 1] = state_dict
        self.recurrent_hidden_states[self.step + 1] = recurrent_hidden_states.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = np.transpose(action_log_probs, [1, 2, 0]).copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        self.bad_masks[self.step + 1] = bad_masks.copy()
        self.success[self.step + 1] = success_masks.copy()
        if self.aux is not None:
            self.aux[self.step] = aux_info.copy()
        self.reset_actions[self.step] = reset_action.copy()

        self.step = (self.step + 1) % self.num_steps
    
    def after_update(self):
        self.obs[0] = self.obs[-1].copy()
        self.states[0] = self.states[-1]
        # _shape = self.states.shape
        # last_state = self.states[-1]
        # del self.states
        # self.states = np.empty(_shape, dtype=object)
        # self.states[0] = last_state
        self.recurrent_hidden_states[0] = self.recurrent_hidden_states[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.success[0] = self.success[-1].copy()

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True):
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    delta = np.expand_dims(self.rewards[step], axis=1) + gamma * self.value_preds[
                        step + 1] * np.expand_dims(self.masks[step +
                                               1], axis=1) - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * np.expand_dims(self.masks[step +
                                                                  1], axis=1) * gae
                    gae = gae * np.expand_dims(self.bad_masks[step + 1], axis=1)
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = (self.returns[step + 1] * \
                        gamma * np.expand_dims(self.masks[step + 1], axis=1) + np.expand_dims(self.rewards[step], axis=1)) * np.expand_dims(self.bad_masks[step + 1], axis=1) \
                        + (1 - np.expand_dims(self.bad_masks[step + 1], axis=1)) * self.value_preds[step]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    delta = np.expand_dims(self.rewards[step], axis=1) + gamma * self.value_preds[
                        step + 1] * np.expand_dims(self.masks[step +
                                               1], axis=1) - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * np.expand_dims(self.masks[step +
                                                                  1], axis=1) * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * \
                        gamma * np.expand_dims(self.masks[step + 1], dim=1) + self.rewards[step]

    def compute_step_to_go(self):
        self.step_to_go[-1] = 30
        self.step_to_go[-1][self.success[-1] == 1] = 0
        for step in reversed(range(self.step_to_go.shape[0] - 1)):
            self.step_to_go[step] = np.clip(self.step_to_go[step + 1] + 1, 0, 30)
            self.step_to_go[step][self.success[step] == 1] = 0

    def feed_forward_generator(self,
                               advantages: np.ndarray,
                               num_mini_batch=None,
                               mini_batch_size=None,
                               with_aux=False):
        if with_aux:
            assert self.aux is not None
        num_steps, num_processes = self.rewards.shape[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = self.obs[:-1].reshape((-1, *self.obs.shape[2:]))[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].reshape(
                (-1, self.recurrent_hidden_states.shape[-1]))[indices]
            actions_batch = self.actions.reshape((-1,
                                              self.actions.shape[-1]))[indices]
            # TODO: shape changed!
            value_preds_batch = self.value_preds[:-1].reshape((-1, self.n_values, 1))[indices]
            return_batch = self.returns[:-1].reshape((-1, self.n_values, 1))[indices]
            masks_batch = self.masks[:-1].reshape((-1, 1))[indices]
            old_action_log_probs_batch = self.action_log_probs.reshape((-1, 1, self.n_action_level))[indices].transpose([2, 0, 1])
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.reshape((-1, 1))[indices]

            # if with_aux:
            #     aux_batch = self.aux.view(-1, *self.aux.size()[2:])[indices]
            # else:
            #     aux_batch = None

            # debug_supervised_action_batch = self.debug_supervised_actions.view(-1, self.actions.size(-1))[indices]
            # debug_supervised_action_idx = torch.where(torch.norm(debug_supervised_action_batch + torch.ones(self.actions.size(-1)).to(debug_supervised_action_batch.device), dim=1) != 0)[0]
            # yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
            #     value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ, aux_batch, \
            #     debug_supervised_action_batch, debug_supervised_action_idx
            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                  value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    # TODO: implement recurrent version with aux info and value ensemble if needed
    def recurrent_generator(self, advantages, num_mini_batch):
        raise NotImplementedError
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind])
                # recurrent_hidden_states_batch.append(
                #     self.recurrent_hidden_states[:-1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            # recurrent_hidden_states_batch = torch.stack(recurrent_hidden_states_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            # recurrent_hidden_states_batch = _flatten_helper(T, N, recurrent_hidden_states_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ


def compute_gae_return(rewards: np.ndarray, values: np.ndarray, gamma, gae_lambda):
    assert rewards.shape[0] == values.shape[0]
    values = np.concatenate([values, values[0: 1]], axis=0)
    values[-1] = 0.
    mask = np.concatenate([np.ones_like(rewards), np.zeros_like(rewards[0: 1])], axis=0)
    returns = np.zeros_like(rewards)
    gae = 0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * mask[step + 1] - values[step]
        gae = delta + gamma * gae_lambda * mask[step + 1] * gae
        returns[step] = gae + values[step]
    return returns


def compute_gae_return_multi(rewards: np.ndarray, values: np.ndarray, masks: np.ndarray, gamma, gae_lambda):
    assert rewards.shape[0] == values.shape[0]
    assert rewards.shape[0] == masks.shape[0] - 1
    values = np.concatenate([values, np.zeros_like(values[0: 1])], axis=0)
    returns = np.zeros_like(rewards)
    gae = 0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step + 1] - values[step]
        gae = delta + gamma * gae_lambda * masks[step + 1] * gae
        returns[step] = gae + values[step]
    return returns
