import sys
sys.path.append("../isaac_projects/panda-isaac")
from panda_isaac.panda_push import PandaPushEnv
from panda_isaac.base_config import BaseConfig
sys.path.remove("../isaac_projects/panda-isaac")


class PushConfig(BaseConfig):
    class env(BaseConfig.env):
        seed = 42
        num_envs = 1024
        # num_observations = 3 * 224 * 224 + 12
        num_observations = (3 + 15) * 2
        num_actions = 4
        max_episode_length = 100
    
    class obs(BaseConfig.obs):
        type = "state"
        history_length = 2
    
    class control(BaseConfig.control):
        decimal = 6
        controller = "ik"
        # controller = "osc"
        # common_speed = 2.17 * 1
        # common_speed = 1.5 * decimal
    
    class reward(BaseConfig.reward):
        type = "dense"


def goal_in_air_cl(_locals, _globals):
    if _locals["j"] > 1:
        import torch
        ep_infos = _locals["ep_infos"]
        success_rate = torch.mean(torch.tensor(ep_infos["is_success"]).float()).item()
        cur_goal_in_air_ratio = _locals["self"].env.goal_in_air
        if success_rate > 0.6:
            _locals["self"].env.set_goal_in_air_ratio(min(cur_goal_in_air_ratio + 0.1, 1.0))
        print("Goal in air ratio =", _locals["self"].env.goal_in_air)


config = dict(
    env_id="IsaacPandaPushState-v0",
    algo="ppo",
    name="test_ik_dense_filter_obs15_2frame",
    # name="test_joint_decimal6_1024w_step64_dense",
    total_timesteps=int(1e8),
    entry_point=PandaPushEnv,
    env_config=PushConfig(),
    policy_type="mlp",
    policy=dict(
        hidden_size=64,
        # num_bin=21,
    ),
    train=dict(
      # n_steps=1024,
      n_steps=64,
      nminibatches=32,
      # learning_rate=1e-3,
      learning_rate=2.5e-4,
      # cliprange=0.1,
      use_wandb=False
    ),
    # callback=[goal_in_air_cl],
)