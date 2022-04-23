import sys
sys.path.append("../isaac_projects/panda-isaac")
from panda_isaac.panda_push import PandaPushEnv
from panda_isaac.base_config import BaseConfig
sys.path.remove("../isaac_projects/panda-isaac")


class PushConfig(BaseConfig):
    class env(BaseConfig.env):
        seed = 42
        num_envs = 256
        num_observations = 3 * 224 * 224 + 22
        num_actions = 4
        max_episode_length = 50
    
    class obs(BaseConfig.obs):
        type = "pixel"
        im_size = 224
        history_length = 1
    
    class cam(BaseConfig.cam):
        view = "ego"
        fov = 120
    
    class control(BaseConfig.control):
        decimal = 6
        controller = "ik"
        # common_speed = 2.17 * 1
        # common_speed = 1.5 * decimal
    
    class reward(BaseConfig.reward):
        type = "dense"


config = dict(
    env_id="IsaacPandaPushPixel-v0",
    algo="ppo",
    name="ik_dense_256w",
    total_timesteps=int(1e8),
    entry_point=PandaPushEnv,
    env_config=PushConfig(),
    policy_type=("policies.mvp.mvp_policy", "PixelActorCritic"),
    policy=dict(
        image_shape=(3, 224, 224),
        states_shape=(22,),
        actions_shape=(4,),
        initial_std=1.0,
        encoder_cfg=dict(
            model_type="maevit-s16",
            pretrain_dir="policies/mvp/pretrained",
            pretrain_type="hoi",
            freeze=True,
            emb_dim=64,
            state_emb_dim=64),
        policy_cfg=dict(pi_hid_sizes=[64, 64], vf_hid_sizes=[64, 64])
    ),
    train=dict(
      feature_only=True,
      # n_steps=1024,
      n_steps=64,
      nminibatches=16,
      # learning_rate=1e-3,
      learning_rate=2.5e-4,
      cliprange=0.2,
      ent_coef=0.01,
      max_grad_norm=1.0,
      use_wandb=True
    ),
)
