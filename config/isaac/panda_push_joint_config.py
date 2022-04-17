import sys
sys.path.append("../isaac_projects/panda-isaac")
from panda_isaac.panda_push_joint import PandaPushEnv
from panda_isaac.base_config import BaseConfig
sys.path.remove("../isaac_projects/panda-isaac")


class PushConfig(BaseConfig):
    class env(BaseConfig.env):
        seed = 42
        num_envs = 1024
        # num_observations = 3 * 224 * 224 + 12
        num_observations = 7 + 27
        num_actions = 8
        max_episode_length = 100
    
    class obs(BaseConfig.obs):
        type = "state"
    
    class control(BaseConfig.control):
        decimal = 6
        controller = "joint"
        # common_speed = 2.17 * 1
        # common_speed = 1.5 * decimal
    
    class reward(BaseConfig.reward):
        type = "sparse"


config = dict(
    env_id="IsaacPandaPushState-v0",
    algo="ppo",
    name="test_joint_decimal6_1024w_step64_penori_saturate_horizon100_debug",
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
)