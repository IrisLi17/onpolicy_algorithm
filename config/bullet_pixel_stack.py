from policies.mvp_stacking_policy import MvpStackingPolicy
import numpy as np

config = dict(
    env_id="BulletPixelStack-v1",
    num_workers=64,
    algo="ppo",
    name="base",
    total_timesteps=int(5e7),
    create_env_kwargs=dict(
        kwargs=dict(
            n_object=6, reward_type="sparse", action_dim=7, generate_data=True, primitive=True,
            n_to_stack=np.array([[1, 2, 3]]), name="allow_rotation", use_gpu_render=True,
        ),
    ),
    policy_class=MvpStackingPolicy,
    policy=dict(
        mvp_feat_dim=768, 
        n_primitive=6, 
        act_dim=6, 
        num_bin=32,
        proj_img_dim=128, 
        use_privilege=True,
        state_only_value=True
    ),
    train=dict(
        use_wandb=False,
        n_steps=1024,
        gamma=0.95,
    ),
    warmup_dataset="../stacking_env/warmup_dataset_stacking.pkl",
    save_interval=20,
)
