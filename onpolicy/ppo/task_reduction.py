from policies.base import ActorCriticPolicy
import torch
import numpy as np
from utils import logger


class TaskReduction(object):
    def __init__(self, env, policy: ActorCriticPolicy, keep_success_ratio=0.1, env_kwargs={}):
        self.env = env
        self.policy = policy
        for p in self.policy.parameters():
            self.device = p.data.device
            break
        self.env_id = self.env.get_attr("spec")[0].id
        if self.env_id == "BulletStack-v1":
            self.x_workspace = self.env.get_attr("robot")[0].x_workspace
            self.y_workspace = self.env.get_attr("robot")[0].y_workspace
            self.base_pos = self.env.get_attr("robot")[0].base_pos
        elif self.env_id.startswith("AntMaze") or self.env_id.startswith("PointMaze"):
            self.x_workspace = (self.env.get_attr("min_x")[0], self.env.get_attr("max_x")[0])
            self.y_workspace = (self.env.get_attr("min_y")[0], self.env.get_attr("max_y")[0])
        elif self.env_id.startswith("SawyerPush"):
            # search space is more than 2 dimensional
            self.workspace = (self.env.get_attr("goal_space")[0].low,
                              self.env.get_attr("goal_space")[0].high)
        else:
            raise NotImplementedError
        self.goal_dim = self.env.get_attr("goal")[0].shape[0]

        self.keep_success_ratio = keep_success_ratio
        self.env_kwargs = env_kwargs

    def _act(self, obs, deterministic):
        values, actions, log_probs, _ = self.policy.act(obs, deterministic)
        return values, actions, log_probs

    def _evaluate_actions(self, obs, actions):
        action_log_probs, _, _ = self.policy.evaluate_actions(obs, None, None, actions)
        return action_log_probs

    def _get_value(self, obs):
        values = self.policy.get_value(obs, None, None)
        return values

    def reduction(self, n_desired_traj, manual_subgoal=False, return_terminal_obs=False):
        obs = self.env.reset()
        # initial_states = self.env.env_method("get_state")
        is_reduction = [False] * self.env.num_envs
        buffer = [dict(obs=[], actions=[], states=[], values=[], tag="") for _ in range(self.env.num_envs)]
        dataset = []
        failed_dataset = []
        n_total_traj = 0
        n_interactions = 0
        _n_original_success = 0
        _n_need_reduction = 0
        _n_reduction_success = 0
        _n_reduction_fail = 0
        # for computing importance weight
        _step_reduction_success = 0
        _step_reduction_fail = 0
        _last_logged_len = 0
        if self.env_id == "BulletStack-v1":
            _detail_reduction_success = [0] * self.env.get_attr("n_object")[0]
        else:
            _detail_reduction_success = None
        # first execute learned policy, then select subgoal for those failed trajectories
        # perform multi-goal rollout in single thread
        while len(dataset) < n_desired_traj and n_total_traj < 10 * n_desired_traj:
            if len(dataset) % 100 == 0 and len(dataset) > _last_logged_len:
                logger.log("N reduction trajectory", len(dataset), "n total traj", n_total_traj,
                           _detail_reduction_success)
                _last_logged_len = len(dataset)
            states = self.env.env_method("get_state")
            with torch.no_grad():
                values, actions, log_probs = self._act(obs, deterministic=False)
            new_obs, rewards, dones, infos = self.env.step(actions)
            n_interactions += self.env.num_envs
            for e_idx in range(self.env.num_envs):
                buffer[e_idx]["obs"].append(obs[e_idx])
                buffer[e_idx]["actions"].append(actions[e_idx])
                buffer[e_idx]["states"].append(states[e_idx])
                buffer[e_idx]["values"].append(values[e_idx].cpu().numpy())
                if dones[e_idx]:
                    n_total_traj += 1
                    ultimate_goal = infos[e_idx]["terminal_observation"][-self.goal_dim:]
                    if not is_reduction[e_idx] and not infos[e_idx]["is_success"]:
                        _n_need_reduction += 1
                        # need reduction
                        is_reduction[e_idx] = True
                        # task reduction from critical states
                        # reduction_start_idx = select_reduction_idx(buffer[e_idx]["values"])
                        reduction_start_idx = 0
                        subgoal = self.generate_subgoal(buffer[e_idx]["obs"][reduction_start_idx].cpu().numpy(),
                                                        ultimate_goal)
                        if subgoal is None:
                            is_reduction[e_idx] = False
                            buffer[e_idx] = dict(obs=[], actions=[], states=[], values=[], tag="")
                            continue
                        # for debugging only
                        if manual_subgoal:
                            subgoal[:2] = ultimate_goal[:2]
                        goal_seq = [subgoal, ultimate_goal]
                        # print("goal seq", goal_seq)
                        self.env.env_method("set_state", buffer[e_idx]["states"][reduction_start_idx], indices=e_idx)
                        self.env.env_method("set_goals", goal_seq, indices=e_idx)
                        if self.env_id == "BulletStack-v1":
                            self.env.env_method("sync_attr", indices=e_idx)
                        new_obs[e_idx] = dict2tensor(self.env.env_method("get_obs", indices=e_idx), self.device)[0]
                    elif is_reduction[e_idx] and infos[e_idx]["is_success"]:
                        if return_terminal_obs:
                            buffer[e_idx]["obs"].append(
                                torch.from_numpy(infos[e_idx]["terminal_observation"]).float().to(self.device)
                            )
                        _n_reduction_success += 1
                        _step_reduction_success += len(buffer[e_idx]["obs"])
                        if self.env_id == "BulletStack-v1":
                            _detail_reduction_success[infos[e_idx]["n_to_stack"] - 1] += 1
                        # print("n_reduction_success", _n_reduction_success, "n_need_reduction", _n_need_reduction,
                        #       "n_original_success", _n_original_success, "dataset length", len(dataset),
                        #       "n_traj", n_total_traj, "n_interactions", n_interactions)
                        relabel_obs(self.env, buffer[e_idx]["obs"], torch.from_numpy(ultimate_goal).to(self.device))
                        # try to use less gpu mem
                        buffer[e_idx]["obs"] = torch.stack(buffer[e_idx]["obs"], dim=0).cpu().numpy()
                        buffer[e_idx]["actions"] = torch.stack(buffer[e_idx]["actions"], dim=0).cpu().numpy()
                        buffer[e_idx]["tag"] = "reduction"
                        dataset.append(buffer[e_idx])
                        is_reduction[e_idx] = False
                    elif not is_reduction[e_idx] and infos[e_idx]["is_success"]:
                        if _n_original_success < self.keep_success_ratio * len(dataset):
                            if return_terminal_obs:
                                buffer[e_idx]["obs"].append(
                                    torch.from_numpy(infos[e_idx]["terminal_observation"]).float().to(self.device)
                                )
                            buffer[e_idx]["obs"] = torch.stack(buffer[e_idx]["obs"], dim=0).cpu().numpy()
                            buffer[e_idx]["actions"] = torch.stack(buffer[e_idx]["actions"], dim=0).cpu().numpy()
                            buffer[e_idx]["tag"] = "origin"
                            dataset.append(buffer[e_idx])
                            _n_original_success += 1
                    elif is_reduction[e_idx] and not infos[e_idx]["is_success"]:
                        _n_reduction_fail += 1
                        _step_reduction_fail += len(buffer[e_idx]["obs"])
                        if len(failed_dataset) < len(dataset):
                            if return_terminal_obs:
                                buffer[e_idx]["obs"].append(
                                    torch.from_numpy(infos[e_idx]["terminal_observation"]).float().to(self.device)
                                )
                            relabel_obs(self.env, buffer[e_idx]["obs"], torch.from_numpy(ultimate_goal).to(self.device))
                            buffer[e_idx]["obs"] = torch.stack(buffer[e_idx]["obs"], dim=0).cpu().numpy()
                            buffer[e_idx]["actions"] = torch.stack(buffer[e_idx]["actions"], dim=0).cpu().numpy()
                        if len(failed_dataset) < len(dataset):
                            failed_dataset.append(buffer[e_idx])
                        is_reduction[e_idx] = False
                    buffer[e_idx] = dict(obs=[], actions=[], states=[], values=[], tag="")
                    # initial_states[e_idx] = self.env.env_method("get_state", indices=e_idx)[0]
            obs = new_obs
        return dataset, n_interactions, failed_dataset, _step_reduction_success / (_step_reduction_success + _step_reduction_fail)

    def generate_subgoal(self, obs: np.ndarray, goal: np.ndarray):
        # numpy version
        assert isinstance(obs, np.ndarray) and isinstance(goal, np.ndarray)
        assert len(obs.shape) == 1 and len(goal.shape) == 1
        # given start obs and ultimate goal, generate appropriate subgoal
        if self.env_id == "BulletStack-v1":
            robot_obs, objects_obs, masks = self.policy.obs_parser.forward(
                torch.from_numpy(obs).float().to(self.device).unsqueeze(dim=0))
            info = self.env.env_method(
                "get_info_from_objects", objects_obs[0], goal, indices=0)[0]
            # 1, n_obj, obj_dim
            # n_base,
            # goal_xy = goal[:2]
            # objects_xy = objects_obs[0, :, :2]
            # n_base = int(torch.sum(torch.norm(objects_xy - goal_xy.unsqueeze(dim=0), dim=-1) < 0.01).item())
            n_base = info["n_base"]
            n_active_objects = info["n_active"]
            # goal_idx = np.argmax(goal[3:])
            subgoal_idx_choice = list(range(n_base, n_active_objects))
            # try:
            #     subgoal_idx_choice.remove(goal_idx)
            # except ValueError:
            #     pass
            if len(subgoal_idx_choice) == 0 or info["n_to_stack"] <= 1:
                return None
            subgoal_idx = np.random.choice(subgoal_idx_choice, 1024, replace=True)
            subgoals = np.zeros((1024, goal.shape[0]))
            subgoals[np.arange(1024), 3 + subgoal_idx] = 1
            # generate goal position
            if self.env_kwargs["multi_height"]:
                # loosen the constraint on sub-goal height
                goal_positions = np.stack([
                    np.random.uniform(*self.x_workspace, size=1024),
                    np.random.uniform(*self.y_workspace, size=1024),
                    # self.base_pos[2] + 0.025 + (np.random.randint(low=0, high=info["n_to_stack"] - 1, size=(1024,)) + n_base) * 0.05
                    (self.base_pos[2] + 0.025) * np.ones(1024)
                ], axis=-1)
                above_base_mask = np.where(np.linalg.norm(goal_positions[:, :2] - goal[:2], axis=-1) < 0.03)[0]
                if len(above_base_mask):
                    goal_positions[above_base_mask, 2] = self.base_pos[2] + 0.025 + (np.random.randint(
                        low=0, high=info["n_to_stack"] - 1, size=(len(above_base_mask),)) + n_base) * 0.05
            else:
                goal_positions = np.stack([
                    np.random.uniform(*self.x_workspace, size=1024),
                    np.random.uniform(*self.y_workspace, size=1024),
                    np.ones(1024) * (self.base_pos[2] + 0.025 + 0.05 * n_base)
                ], axis=-1)
            subgoals[:, :3] = goal_positions
        elif self.env_id.startswith("AntMaze") or self.env_id.startswith("PointMaze"):
            subgoals = np.stack([
                np.random.uniform(*self.x_workspace, size=1024),
                np.random.uniform(*self.y_workspace, size=1024),
            ], axis=-1)
            info = None
        elif self.env_id.startswith("SawyerPush"):
            subgoals = np.random.uniform(self.workspace[0], self.workspace[1], size=(1024, len(self.workspace[0])))
            info = None
        else:
            raise NotImplementedError
        # subgoals = subgoals.to(self.device)
        # print("In generating subgoal, ultimate goal", goal, "subgoal height", subgoals[0, 2])
        # print("obs", obs)
        # imagined_obs_old = self.env.env_method(
        #     "imagine_obs", torch.tile(obs.unsqueeze(dim=0), (1024, 1)), subgoals, indices=0)[0]
        imagined_obs = self.dispatch_imagine_obs(
            np.tile(np.expand_dims(obs, axis=0), (1024, 1)), subgoals, [info] * 1024
        )
        # print("subgoal", subgoals[0], "imagined obs", imagined_obs[0])
        switched_obs = np.tile(np.expand_dims(obs, axis=0), (1024, 1))
        # switched_obs_old = self.env.env_method("relabel_obs", switched_obs, subgoals, indices=0)[0]
        switched_obs = self.dispatch_relabel_obs(switched_obs, subgoals)
        # print("switched obs", switched_obs[0])
        with torch.no_grad():
            v_total = self._get_value(
                torch.cat([torch.from_numpy(switched_obs),
                           torch.from_numpy(imagined_obs)], dim=0).to(self.device).float())
            v_init2sub = v_total[:switched_obs.shape[0]]
            v_sub2final = v_total[switched_obs.shape[0]:]
            v = torch.min(v_init2sub, v_sub2final)
        # todo: visualize with 3D scatter
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # goal_positions = goal_positions.cpu().numpy()
        # v_init2sub = v_init2sub.cpu().numpy()
        # v_sub2final = v_sub2final.cpu().numpy()
        # print("subgoals", subgoals[:10])
        # print("v init2sub", v_init2sub[:10])
        # print("v sub2final", v_sub2final[:10])
        # for j in range(v.shape[0]):
        #     ax.scatter(goal_positions[j, 0], goal_positions[j, 1], v_init2sub[j], c='blue')
        #     ax.scatter(goal_positions[j, 0], goal_positions[j, 1], v_sub2final[j], c='green')
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        # plt.savefig("value_landscape.png")
        best_idx = torch.argmax(v)
        best_subgoal = subgoals[best_idx.item()]
        return best_subgoal

    def rollout(self, initial_states, goal_seqs, max_size=300_000):
        self.env.reset()
        n_interactions = 0
        n_used_traj = 0
        n_valid_transition = 0
        is_running = [False] * self.env.num_envs
        dataset = []
        storage = [dict(obs=[], actions=[], reward=np.empty(0), tag="") for _ in range(self.env.num_envs)]
        if len(goal_seqs) == 0:
            return dataset, n_interactions
        self.env.dispatch_env_method("set_goals", *(goal_seqs[:min(self.env.num_envs, len(goal_seqs))]),
                                     indices=range(min(self.env.num_envs, len(goal_seqs))))
        self.env.dispatch_env_method("set_state", *(initial_states[:min(self.env.num_envs, len(initial_states))]),
                                     indices=range(min(self.env.num_envs, len(initial_states))))
        if self.env_id == "BulletStack-v1":
            self.env.env_method("sync_attr", indices=list(range(min(self.env.num_envs, len(initial_states)))))
        # todo: get obs
        obs = self.env.get_obs()
        n_used_traj += min(self.env.num_envs, len(goal_seqs))
        is_running[:min(self.env.num_envs, len(goal_seqs))] = [True] * min(self.env.num_envs, len(goal_seqs))
        while np.any(is_running):
            with torch.no_grad():
                values, actions, log_probs = self._act(obs, deterministic=False)
            new_obs, rewards, dones, infos = self.env.step(actions)
            n_interactions += sum(is_running)
            for e_idx in range(len(dones)):
                storage[e_idx]["obs"].append(obs[e_idx])
                storage[e_idx]["actions"].append(actions[e_idx])
                if dones[e_idx]:
                    if is_running[e_idx] and infos[e_idx]["is_success"]:
                        relabel_obs(
                            self.env, storage[e_idx]["obs"],
                            torch.from_numpy(infos[e_idx]["terminal_observation"][-self.goal_dim:]).to(self.device)
                        )
                        storage[e_idx]["obs"] = torch.stack(storage[e_idx]["obs"], dim=0).cpu().numpy()
                        storage[e_idx]["actions"] = torch.stack(storage[e_idx]["actions"], dim=0).cpu().numpy()
                        _rewards = np.zeros(len(storage[e_idx]["actions"]))
                        _rewards[-1] = 1.0
                        storage[e_idx]["reward"] = _rewards
                        storage[e_idx]["tag"] = "reduction"
                        dataset.append(storage[e_idx])
                        n_valid_transition += len(storage[e_idx]["obs"])
                        if n_valid_transition >= max_size:
                            return dataset, n_interactions
                    storage[e_idx] = dict(obs=[], actions=[], reward=np.empty(0), tag="")
                    if n_used_traj < len(initial_states):
                        self.env.env_method("set_goals", goal_seqs[n_used_traj], indices=e_idx)
                        self.env.env_method("set_state", initial_states[n_used_traj], indices=e_idx)
                        if self.env_id == "BulletStack-v1":
                            self.env.env_method("sync_attr", indices=e_idx)
                        new_obs[e_idx] = self.env.get_obs()[e_idx]
                        n_used_traj += 1
                    else:
                        is_running[e_idx] = False
            obs = new_obs
        return dataset, n_interactions

    def execute_self_demo(self, n_desired_traj, return_terminal_obs=False):
        obs = self.env.reset()
        buffer = [dict(obs=[], actions=[], tag="") for _ in range(self.env.num_envs)]
        dataset = []
        n_total_traj = 0
        n_interactions = 0
        while len(dataset) < n_desired_traj and n_total_traj < 10 * n_desired_traj:
            with torch.no_grad():
                values, actions, log_probs = self._act(obs, deterministic=False)
            new_obs, rewards, dones, infos = self.env.step(actions)
            n_interactions += self.env.num_envs
            for e_idx in range(self.env.num_envs):
                buffer[e_idx]["obs"].append(obs[e_idx].cpu().numpy())
                buffer[e_idx]["actions"].append(actions[e_idx].cpu().numpy())
                if dones[e_idx]:
                    n_total_traj += 1
                    if infos[e_idx]["is_success"]:
                        if return_terminal_obs:
                            buffer[e_idx]["obs"].append(infos[e_idx]["terminal_obs"])
                        buffer[e_idx]["obs"] = np.stack(buffer[e_idx]["obs"], axis=0)
                        buffer[e_idx]["actions"] = np.stack(buffer[e_idx]["actions"], axis=0)
                        buffer[e_idx]["tag"] = "origin"
                        dataset.append(buffer[e_idx])
                    buffer[e_idx] = dict(obs=[], actions=[], tag="")
            obs = new_obs
        return dataset, n_interactions

    def dispatch_imagine_obs(self, obs: np.ndarray, goal: np.ndarray, infos):
        imagined_obs = []
        for i in range(obs.shape[0] // self.env.num_envs):
            result = self.env.dispatch_env_method(
                "imagine_obs", obs[i * self.env.num_envs: (i + 1) * self.env.num_envs],
                goal[i * self.env.num_envs: (i + 1) * self.env.num_envs],
                infos[i * self.env.num_envs: (i + 1) * self.env.num_envs],
                n_args=3
            )
            imagined_obs.append(np.stack(result, axis=0).squeeze(axis=1))
        n = (obs.shape[0] // self.env.num_envs) * self.env.num_envs
        if n < obs.shape[0]:
            result = self.env.dispatch_env_method(
                "imagine_obs", obs[n:], goal[n:], infos[n:],
                n_args=3, indices=list(range(obs.shape[0] - n))
            )
            imagined_obs.append(np.stack(result, axis=0).squeeze(axis=1))
        return np.concatenate(imagined_obs, axis=0)

    def dispatch_relabel_obs(self, obs: np.ndarray, goal: np.ndarray):
        switched_obs = []
        for i in range(obs.shape[0] // self.env.num_envs):
            result = self.env.dispatch_env_method(
                "relabel_obs", obs[i * self.env.num_envs: (i + 1) * self.env.num_envs],
                goal[i * self.env.num_envs: (i + 1) * self.env.num_envs], n_args=2
            )
            switched_obs.append(np.stack(result, axis=0).squeeze(axis=1))
        n = (obs.shape[0] // self.env.num_envs) * self.env.num_envs
        if n < obs.shape[0]:
            result = self.env.dispatch_env_method(
                "relabel_obs", obs[n:], goal[n:], n_args=2, indices=list(range(obs.shape[0] - n))
            )
            switched_obs.append(np.stack(result, axis=0).squeeze(axis=1))
        return np.concatenate(switched_obs, axis=0)


def dict2tensor(obs_dicts, device):
    tensors = [torch.from_numpy(np.concatenate(
        [d[key] for key in ["observation", "achieved_goal", "desired_goal"]]
    )).to(device) for d in obs_dicts]
    return torch.stack(tensors, dim=0)


def select_reduction_idx(values: list):
    traj_value = np.array(values).squeeze()
    value_diff = traj_value[:-1] - traj_value[1:]
    if np.max(value_diff) < 0.1:
        reduction_start_idx = 0
    else:
        reduction_start_idx = np.random.randint(
            max(np.argmax(value_diff) - 4, 0), min(np.argmax(value_diff) + 5, len(traj_value))
        )
    return reduction_start_idx


def relabel_obs(env, obs_seq, goal):
    assert isinstance(obs_seq, list)
    env_id = env.get_attr("spec")[0].id
    if env_id == "BulletStack-v1":
        robot_dim = env.get_attr("robot_dim")[0]
        object_dim = env.get_attr("object_dim")[0]
        goal_dim = env.get_attr("goal")[0].shape[0]
        goal_idx = torch.argmax(goal[3:])
        for obs in obs_seq:
            goal_indicator_idx = torch.from_numpy(np.arange(robot_dim + object_dim - 1, obs.shape[0] - 2 * goal_dim, object_dim, dtype=np.long))
            obs[goal_indicator_idx] = 0.
            obs[robot_dim + goal_idx * object_dim + object_dim - 1] = 1.
            obs[-2 * goal_dim: -2 * goal_dim + 3] = \
                obs[robot_dim + goal_idx * object_dim: robot_dim + goal_idx * object_dim + 3]
            obs[-2 * goal_dim + 3: -goal_dim] = goal[3:]
            obs[-goal_dim:] = goal[:]
    else:
        goal_dim = env.get_attr("goal")[0].shape[0]
        for obs in obs_seq:
            obs[-goal_dim:] = goal
