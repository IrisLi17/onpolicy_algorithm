import os
import gym
from utils.wrapper import DoneOnSuccessWrapper, VecPyTorch, ScaleRewardWrapper, FlattenDictWrapper, ScaleActionWrapper
from utils.monitor import Monitor
from vec_env.subproc_vec_env import SubprocVecEnv
import torch
import sys
'''
try:
    import panda_gym
except ImportError:
    import sys
    sys.path.append("../panda-gym")
    import panda_gym
try:
    sys.path.append("../motion_imitation")
    import envs
    sys.path.remove("../motion_imitation")
except:
    pass
'''
try:
    sys.path.append("../stacking_env")
    import env
    sys.path.remove("../stacking_env")
except:
    pass


def make_env(env_id, rank, log_dir=None, obs_keys=None, done_when_success=False, reward_offset=1.0, reward_scale=1.0,
             flexible_time_limit=False, allow_switch_goal=False, scale_action=False, info_keywords=("is_success",),
             kwargs={}):
    env = gym.make(env_id, **kwargs)
    if flexible_time_limit:
        from utils.wrapper import FlexibleTimeLimitWrapper
        env = FlexibleTimeLimitWrapper(env)
    if obs_keys is not None and isinstance(obs_keys, list):
        env = FlattenDictWrapper(env, obs_keys)
    if done_when_success:
        env = DoneOnSuccessWrapper(env, reward_offset=reward_offset)
    if allow_switch_goal:
        from utils.wrapper import SwitchGoalWrapper
        env = SwitchGoalWrapper(env)
    if scale_action:
        env = ScaleActionWrapper(env)
    env = ScaleRewardWrapper(env, reward_scale=reward_scale)
    if log_dir is not None:
        env = Monitor(env, os.path.join(log_dir, "%d.monitor.csv" % rank), info_keywords=info_keywords)
    return env


def make_vec_env(env_id, num_workers, device, normalize=False, training=True, **kwargs):
    def make_env_thunk(i):
        return lambda: make_env(env_id, i, **kwargs)
    env = SubprocVecEnv([make_env_thunk(i) for i in range(num_workers)])
    if normalize:
        from vec_env.vec_normalize import VecNormalize
        env = VecNormalize(env, training=training, norm_obs=("obs" in normalize), norm_reward=("ret" in normalize))
    return VecPyTorch(env, device)
