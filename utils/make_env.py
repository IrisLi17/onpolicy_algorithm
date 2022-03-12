import os
import gym
from utils.wrapper import DoneOnSuccessWrapper, VecPyTorch, ScaleRewardWrapper, FlattenDictWrapper
from utils.monitor import Monitor
from vec_env.subproc_vec_env import SubprocVecEnv
import torch
try:
    import panda_gym
except:
    import sys
    sys.path.append("../panda-gym")
    import panda_gym
import sys
sys.path.append("../motion_imitation")
import envs
sys.path.remove("../motion_imitation")


def make_env(env_id, rank, log_dir=None, obs_keys=None, done_when_success=False, reward_scale=1.0,
             info_keywords=("is_success",), kwargs={}):
    env = gym.make(env_id, **kwargs)
    if obs_keys is not None and isinstance(obs_keys, list):
        env = FlattenDictWrapper(env, obs_keys)
    if done_when_success:
        env = DoneOnSuccessWrapper(env, reward_offset=1.0)
    env = ScaleRewardWrapper(env, reward_scale=reward_scale)
    if log_dir is not None:
        env = Monitor(env, os.path.join(log_dir, "%d.monitor.csv" % rank), info_keywords=info_keywords)
    return env


def make_vec_env(env_id, num_workers, device, **kwargs):
    def make_env_thunk(i):
        return lambda: make_env(env_id, i, **kwargs)
    env = SubprocVecEnv([make_env_thunk(i) for i in range(num_workers)])
    return VecPyTorch(env, device)


class ObsParser(object):
    def __init__(self, robot_dim, obj_dim, goal_dim):
        self.robot_dim = robot_dim
        self.obj_nogoal_dim = obj_dim  # per object
        self.goal_dim = goal_dim  # per object
        self.obj_dim = self.obj_nogoal_dim + self.goal_dim

    def forward(self, obs: torch.Tensor):
        assert len(obs.shape) == 2
        robot_obs = torch.narrow(obs, dim=1, start=0, length=self.robot_dim)
        assert (obs.shape[-1] - self.robot_dim) % (self.obj_nogoal_dim + 2 * self.goal_dim) == 0
        num_object = int((obs.shape[-1] - self.robot_dim) / (self.obj_nogoal_dim + 2 * self.goal_dim))
        objects_obs = torch.narrow(obs, dim=1, start=self.robot_dim,
                                   length=self.obj_nogoal_dim * num_object)
        objects_obs = torch.reshape(objects_obs, (obs.shape[0], num_object, self.obj_nogoal_dim))
        desired_goals = torch.narrow(obs, dim=1, start=obs.shape[-1]-self.goal_dim * num_object,
                                     length=self.goal_dim * num_object)
        desired_goals = torch.reshape(desired_goals, (obs.shape[0], num_object, self.goal_dim))
        objects_obs = torch.cat([objects_obs, desired_goals], dim=-1)
        masks = torch.norm(objects_obs, dim=-1) < 1e-3
        return robot_obs, objects_obs, masks
