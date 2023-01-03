from onpolicy.storage import RolloutStorage
import torch
import numpy as np
import pickle


class OfflineExpansion(object):
    def __init__(self, rollout: RolloutStorage, env, value_op) -> None:
        self.rollout = rollout
        self.value_op = value_op
        self.env = env
    
    def expand(self):
        initial_obs, end_obs, all_initial_idx, all_end_idx = self.parse_dataset()
        n_original = end_obs.shape[0]
        # Create more. 
        multiplier = 50
        _idx = np.random.choice(end_obs.shape[0], multiplier * end_obs.shape[0])
        sampled_end_obs = end_obs[_idx]
        initial_obs = initial_obs.repeat(multiplier, 1)
        end_obs = end_obs.repeat(multiplier, 1)
        sampled_achieved = self.obs_to_achieved(sampled_end_obs)
        relabel_initial_obs = self.relabel(initial_obs, sampled_achieved)
        relabel_interm_obs = self.relabel(end_obs, sampled_achieved)
        initial_value = self.safe_get_value(relabel_initial_obs)
        interm_value = self.safe_get_value(relabel_interm_obs)
        # initial_value_clip = torch.clamp(initial_value, 1e-6, 1.).squeeze(dim=-1)
        # interm_value_clip = torch.clamp(interm_value, 1e-6, 1.).squeeze(dim=-1)
        # TODO: value may not be good. try state-only value function or value ensemble
        metric = interm_value - initial_value
        # Get the top 2/multiplier goals
        goal_idx = torch.where(metric > torch.quantile(metric, 1.0 - 1.0 / multiplier))[0]
        # Get imitation dataset
        im_dataset = dict(obs=[], action=[], boundary=[], original_obs=[], initial_value=[], interm_value=[])
        im_count = 0
        for g_idx in goal_idx:
            # in original idx
            original_idx = g_idx % n_original
            traj_initial_idx = all_initial_idx[original_idx]
            traj_end_idx = all_end_idx[original_idx]
            assert traj_initial_idx[1] == traj_end_idx[1]
            assert len(traj_initial_idx) == 2 and len(traj_end_idx) == 2
            new_goal = (item[g_idx] for item in sampled_achieved)
            # inclusive
            traj_obs = self.rollout.obs[traj_initial_idx[0]: traj_end_idx[0] + 1, traj_initial_idx[1]]
            im_traj_obs = self.relabel(traj_obs, new_goal)
            im_traj_action = self.rollout.actions[traj_initial_idx[0]: traj_end_idx[0] + 1, traj_initial_idx[1]]
            im_dataset["obs"].append(im_traj_obs.detach().cpu().numpy())
            im_dataset["action"].append(im_traj_action.detach().cpu().numpy())
            im_dataset["boundary"].append(im_count)
            im_dataset["original_obs"].append(traj_obs.detach().cpu().numpy())
            im_dataset["initial_value"].append(initial_value[g_idx, 0].item())
            im_dataset["interm_value"].append(interm_value[g_idx, 0].item())
            im_count += im_traj_obs.shape[0]
        im_dataset["obs"] = np.concatenate(im_dataset["obs"], axis=0)
        im_dataset["action"] = np.concatenate(im_dataset["action"], axis=0)
        im_dataset["boundary"] = np.array(im_dataset["boundary"])
        im_dataset["original_obs"] = np.concatenate(im_dataset["original_obs"], axis=0)
        im_dataset["initial_value"] = np.array(im_dataset["initial_value"])
        im_dataset["interm_value"] = np.array(im_dataset["interm_value"])
        with open("im_dataset.pkl", "wb") as f:
            pickle.dump(im_dataset, f)

        # Prepare for the next round of environment interaction
        generated_obs = relabel_initial_obs[goal_idx]
        # Convert to states that can pass into the environment
        reset_env_states = self.obs_to_env_states(generated_obs)            
        return reset_env_states
        
    def parse_dataset(self):
        num_processes = self.rollout.obs.shape[1]
        # inclusive
        episode_start = [[] for _ in range(num_processes)]
        for i in range(self.rollout.obs.shape[0] - 1):
            start_worker_idx = torch.where(self.rollout.masks[i, :, 0] == 0)[0]
            for j in start_worker_idx:
                episode_start[j].append(i)
        all_initial_obs = []
        all_end_obs = []
        all_initial_idx = []
        all_end_idx = []
        for i in range(num_processes):
            start_steps = np.array(episode_start[i][:-1])
            end_steps = np.array(episode_start[i][1:]) - 1
            initial_obs = self.rollout.obs[torch.from_numpy(start_steps).long(), i]
            end_obs = self.rollout.obs[torch.from_numpy(end_steps).long(), i]
            all_initial_obs.append(initial_obs)
            all_end_obs.append(end_obs)
            all_initial_idx.append(np.stack([start_steps, i * np.ones_like(start_steps)], axis=-1))
            all_end_idx.append(np.stack([end_steps, i * np.ones_like(end_steps)], axis=-1))
            # obs to state
        all_initial_obs = torch.cat(all_initial_obs, dim=0)
        all_end_obs = torch.cat(all_end_obs, dim=0)
        all_initial_idx = np.concatenate(all_initial_idx, axis=0)
        all_end_idx = np.concatenate(all_end_idx, axis=0)
        return all_initial_obs, all_end_obs, all_initial_idx, all_end_idx
        # get (initial state, end state)
    
    def obs_to_achieved(self, obs):
        # the achieved image feature
        achieved_img = obs[..., :768]
        achieved_drawer = obs[..., 768 * 2 + 7: 768 * 2 + 7 + 1]
        achieved_obj = obs[..., 768 * 2 + 7 + 2: 768 * 2 + 7 + 5]
        # low level states
        return (achieved_img, achieved_drawer, achieved_obj)
    
    def obs_to_env_states(self, obs):
        # reset state and goal state
        reset_robot_state = obs[..., 768: 768 + 7]
        reset_drawer_state = obs[..., 768 * 2 + 7: 768 * 2 + 7 + 1]
        reset_obj_state = obs[..., 768 * 2 + 7 + 2: 768 * 2 + 7 + 5]
        goal_drawer_state = obs[..., 768 * 2 + 7 + 1: 768 * 2 + 7 + 2]
        goal_obj_state = obs[..., 768 * 2 + 7 + 5: 768 * 2 + 7 + 8]
        return {
            "reset_robot_state": reset_robot_state, "reset_drawer_state": reset_drawer_state,
            "reset_obj_state": reset_obj_state, "goal_drawer_state": goal_drawer_state, 
            "goal_obj_state": goal_obj_state,
        }
    
    def relabel(self, obs, goal: tuple):
        goal_img, goal_drawer, goal_obj = goal
        new_obs = torch.clone(obs)
        new_obs[..., 768 + 7: 768 * 2 + 7] = goal_img
        new_obs[..., 768 * 2 + 7 + 1: 768 * 2 + 7 + 2] = goal_drawer
        new_obs[..., 768 * 2 + 7 + 5:] = goal_obj
        return new_obs

    def safe_get_value(self, obs):
        if obs.shape[0] % 1024 == 0:
            n_chunk = obs.shape[0] // 1024
        else:
            n_chunk = obs.shape[0] // 1024 + 1
        values = []
        with torch.no_grad():
            for i in range(n_chunk):
                values.append(self.value_op(obs[i * 1024: (i + 1) * 1024]))
        values = torch.cat(values, dim=0)
        return values
