from onpolicy.storage import RolloutStorage
import torch
import numpy as np
import pickle


class OfflineExpansion(object):
    def __init__(self, rollout: RolloutStorage, env, policy) -> None:
        self.rollout = rollout
        self.policy = policy
        self.value_op = policy.get_value
        self.env = env
    
    def expand(self):
        initial_obs, end_obs, all_initial_idx, all_end_idx = self.parse_dataset()
        n_original = end_obs.shape[0]
        # TODO: new goals should not only come from achieved end obs, they can from any achieved obs. 
        # In this way the agent can potentially discover something new.
        # Create more. 
        multiplier = 20
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
        print("Sampled goal", metric.shape[0])
        # Get the top 2/multiplier goals
        goal_idx = torch.where(metric > torch.quantile(metric, 1.0 - 1.0 / multiplier))[0]
        # TODO: For debugging only, oracle goal selection
        goal_idx = torch.where(
            torch.logical_or(
                (relabel_interm_obs[:, -8] - relabel_interm_obs[:, -7]).abs() < 0.02,
                torch.norm(relabel_interm_obs[:, -6:-3] - relabel_interm_obs[:, -3:]) < 0.04
            )
        )[0]
        # Verify the second half in environment
        second_half = self.roll_out(relabel_interm_obs[goal_idx], interm_value[goal_idx])
        valid_goal_idx = torch.tensor([goal_idx[item["task_idx"]] for item in second_half])
        print("proposed and valid goal", goal_idx.shape, valid_goal_idx.shape)
        # Get full imitation traj
        # TODO: keep some original successful data
        # Get imitation dataset
        im_dataset = dict(obs=[], action=[], boundary=[], original_obs=[], 
                          initial_value=[], interm_value=[], terminate_obs=[])
        im_count = 0
        for i in range(len(second_half)):
            second_traj = second_half[i]
            g_idx = valid_goal_idx[i]
            # in original idx
            original_idx = g_idx % n_original
            traj_initial_idx = all_initial_idx[original_idx]
            traj_end_idx = all_end_idx[original_idx]
            assert traj_initial_idx[1] == traj_end_idx[1]
            assert len(traj_initial_idx) == 2 and len(traj_end_idx) == 2
            new_goal = (item[g_idx] for item in sampled_achieved)
            # inclusive
            traj_obs = self.rollout.obs[traj_initial_idx[0]: traj_end_idx[0] + 1, traj_initial_idx[1]]
            im_traj1_obs = self.relabel(traj_obs, new_goal)
            im_traj1_action = self.rollout.actions[traj_initial_idx[0]: traj_end_idx[0] + 1, traj_initial_idx[1]]
            im_traj2_obs = second_traj["obs"]
            im_traj_obs = np.concatenate(
                [im_traj1_obs.detach().cpu().numpy(), im_traj2_obs[:-1]], axis=0
            )
            im_traj_action = np.concatenate(
                [im_traj1_action.detach().cpu().numpy(), second_traj["action"]], axis=0
            )
            im_dataset["obs"].append(im_traj_obs)
            im_dataset["action"].append(im_traj_action)
            im_dataset["terminate_obs"].append(im_traj2_obs[-1])
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

        # Prepare for the next round of environment interaction
        generated_obs = relabel_initial_obs[goal_idx]
        # Convert to states that can pass into the environment
        reset_env_states = self.obs_to_env_states(generated_obs)            
        return im_dataset, reset_env_states
        
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
        # TODO: end obs is incorrect. need to record terminate obs
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
        # Let the environment parse init Robot states + init and goal privileged states
        # parse goal image features as well, and modify env wrapper to accept both raw image and features 
        return self.env.get_state_from_obs(obs).detach().cpu().numpy()
    
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

    def roll_out(self, start_obs, second_value):
        # environment should implement set_task, get_obs method.
        # wrapper should wrap get_obs properly.
        n_used_goals = 0
        dones = [False for _ in range(self.env.num_envs)]
        is_running = np.array([False] * self.env.num_envs)
        tracked_task_idx = np.array([-1] * self.env.num_envs)
        self.env.reset()
        # set tasks and get obs
        for i in range(min(start_obs.shape[0], self.env.num_envs)):
            self.env.env_method("set_task", self.obs_to_env_states(start_obs[i]), indices=i)
            is_running[i] = True
            tracked_task_idx[i] = n_used_goals
            n_used_goals += 1
        obs = self.env.get_obs()
        buffer = [{"obs": [obs[i].detach().cpu().numpy()], "action": []} for i in range(self.env.num_envs)]
        dataset = []
        while np.any(is_running):
            with torch.no_grad():
                actions = self.policy.act(obs)[1]
            obs, reward, dones, infos = self.env.step(actions)
            for i in range(self.env.num_envs):
                buffer[i]["obs"].append(obs[i].detach().cpu().numpy())
                buffer[i]["action"].append(actions[i].detach().cpu().numpy())
                if dones[i]:
                    # if is_running[i] and not infos[i]["is_success"]:
                    #     print("start obs", start_obs[tracked_task_idx[i], -8:])
                    #     print("value", second_value[tracked_task_idx[i]])
                    if is_running[i] and infos[i]["is_success"]:
                        # append last observation and move from buffer to dataset
                        buffer[i]["obs"][-1] = infos[i]["terminal_observation"].detach().cpu().numpy()
                        buffer[i]["obs"] = np.array(buffer[i]["obs"])
                        buffer[i]["action"] = np.array(buffer[i]["action"])
                        dataset.append({**buffer[i], "task_idx": tracked_task_idx[i]})
                    if n_used_goals >= start_obs.shape[0]:
                        is_running[i] = False
                    else:
                        # The environment is already reset by subproc vec wrapper, so we have to override here
                        self.env.env_method("set_task", self.obs_to_env_states(start_obs[n_used_goals]), indices=i)
                        obs[i] = self.env.get_obs(indices=i)
                        tracked_task_idx[i] = n_used_goals
                        n_used_goals += 1 
                    buffer[i] = {"obs": [obs[i].detach().cpu().numpy()], "action": []}                   
        return dataset
