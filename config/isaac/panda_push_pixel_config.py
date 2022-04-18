import sys
sys.path.append("../isaac_projects/panda-isaac")
from panda_isaac.panda_push_joint import PandaPushEnv
from panda_isaac.base_config import BaseConfig
sys.path.remove("../isaac_projects/panda-isaac")


class PushConfig(BaseConfig):
    class env(BaseConfig.env):
        seed = 42
        num_envs = 225
        num_observations = 3 * 224 * 224 + 30
        # num_observations = 7 + 27
        num_actions = 8
        max_episode_length = 100
    
    class obs(BaseConfig.obs):
        type = "pixel"
        im_size = 224
    
    class control(BaseConfig.control):
        decimal = 6
        controller = "joint"
        # common_speed = 2.17 * 1
        # common_speed = 1.5 * decimal
    
    class reward(BaseConfig.reward):
        type = "sparse"


config = dict(
    env_id="IsaacPandaPushPixel-v0",
    algo="ppo",
    name="test",
    total_timesteps=int(1e9),
    entry_point=PandaPushEnv,
    env_config=PushConfig(),
    policy_type=("policies.mvp.mvp_policy", "PixelActorCritic"),
    policy=dict(
        image_shape=(3, 224, 224),
        states_shape=(30,),
        actions_shape=(8,),
        initial_std=1.0,
        encoder_cfg=dict(
            model_type="maevit-s16",
            pretrain_dir="policies/mvp/pretrained",
            pretrain_type="hoi",
            freeze=True,
            emb_dim=128),
        policy_cfg=dict(pi_hid_sizes=[256, 128, 64], vf_hid_sizes=[256, 128, 64])
    ),
    train=dict(
      feature_only=True,
      # n_steps=1024,
      n_steps=64,
      nminibatches=8,
      # learning_rate=1e-3,
      learning_rate=2.5e-4,
      # cliprange=0.1,
      use_wandb=False
    ),
)